from collections import Counter
from pathlib import Path

import click
import pydicom


def print_stats(counter: Counter) -> None:
    for key, value in counter.items():
        print(f"\t\t{key} - {value}")


@click.command()
@click.option(
    "-d",
    "--dicom-dir",
    type=click.Path(exists=True, readable=True, path_type=Path),
    required=True,
)
@click.option(
    "-s",
    "--segmentation-dir",
    type=click.Path(exists=True, readable=True, path_type=Path),
    required=True,
)
def main(dicom_dir: Path, segmentation_dir: Path) -> None:
    dicom_patients = [x for x in dicom_dir.iterdir() if x.is_dir()]
    dicom_patients.sort()

    stats = {
        "brain": {
            "number": 0,
            "institution": [],
            "gender": [],
            "age": [],
            "scanner brand": [],
            "scanner model": [],
            "scan date": [],
        },
        "skin": {
            "number": 0,
            "institution": [],
            "gender": [],
            "age": [],
            "scanner brand": [],
            "scanner model": [],
            "scan date": [],
        },
        "tumor": {
            "number": 0,
            "institution": [],
            "gender": [],
            "age": [],
            "scanner brand": [],
            "scanner model": [],
            "scan date": [],
        },
        "ventricles": {
            "number": 0,
            "institution": [],
            "gender": [],
            "age": [],
            "scanner brand": [],
            "scanner model": [],
            "scan date": [],
        },
    }

    for patient in dicom_patients:
        print(patient.name)

        if not (segmentation_dir / patient.name).exists():
            print(f"No segmentation dir found for {patient.name}")
            continue

        if not (patient / "Imaging").exists():
            print(f"No Imaging dir found for {patient.name}")
            continue

        imaging_dir = patient / "Imaging"
        # Find t1 GD image
        images = [x for x in imaging_dir.iterdir() if x.is_dir()]

        t1ce_image = None
        for image in images:
            if "t1" in image.name.lower():
                if "gd" in image.name.lower():
                    print(f"Found T1 GD image: {image.name}")
                    t1ce_image = image
                    break
                elif "contrast" in image.name.lower():
                    print(f"Found T1 GD image: {image.name}")
                    t1ce_image = image
                    break

        if len([x for x in t1ce_image.iterdir() if x.is_file()]) > 0:
            t1ce_image = pydicom.dcmread([x for x in t1ce_image.iterdir() if x.is_file()][0])
        else:
            # Sometimes the images are in a subdirectory
            subdir = [x for x in t1ce_image.iterdir() if x.is_dir()][0]
            t1ce_image = pydicom.dcmread([x for x in subdir.iterdir() if x.is_file()][0])

        segmentations = [x.stem.split("_")[-1] for x in (segmentation_dir / patient.name).iterdir() if x.is_file() and "label" in x.name.lower()]
        segmentations = set(segmentations)

        for segmentation in segmentations:

            try:
                institution = t1ce_image[0x0008_0080].value
            except KeyError:
                institution = "Unknown"

            try:
                age = t1ce_image[0x0010_0030].value
            except KeyError:
                age = "Unknown"

            try:
                gender = t1ce_image[0x0010_0040].value
            except KeyError:
                gender = "Unknown"

            try:
                scanner_brand = t1ce_image[0x0008_0070].value
            except KeyError:
                scanner_brand = "Unknown"

            try:
                scanner_model = t1ce_image[0x0008_1090].value
            except KeyError:
                scanner_model = "Unknown"

            try:
                scan_date = t1ce_image[0x0008_0020].value
            except KeyError:
                scan_date = "Unknown"

            stats[segmentation]["number"] += 1
            stats[segmentation]["institution"].append(institution)
            stats[segmentation]["age"].append(age)
            stats[segmentation]["gender"].append(gender)
            stats[segmentation]["scanner brand"].append(scanner_brand)
            stats[segmentation]["scanner model"].append(scanner_model)
            stats[segmentation]["scan date"].append(scan_date)

    for anatomy in stats.keys():
        print(f"{anatomy} - {stats[anatomy]['number']}")

        institution_counter = Counter(stats[anatomy]["institution"])
        print("\tInstitution:")
        print_stats(institution_counter)

        age_counter = Counter(stats[anatomy]["age"])
        print("\tAge:")
        print_stats(age_counter)

        gender_counter = Counter(stats[anatomy]["gender"])
        print("\tGender:")
        print_stats(gender_counter)

        brand_counter = Counter(stats[anatomy]["scanner brand"])
        print("\tScanner brand:")
        print_stats(brand_counter)

        model_counter = Counter(stats[anatomy]["scanner model"])
        print("\tScanner model:")
        print_stats(model_counter)

        date_counter = Counter(stats[anatomy]["scan date"])
        date_counter = sorted(date_counter.items(), key=lambda x: x[0])
        date_counter = {x[0]: x[1] for x in date_counter}
        print("\tScan date:")
        print_stats(date_counter)


if __name__ == "__main__":
    main()
