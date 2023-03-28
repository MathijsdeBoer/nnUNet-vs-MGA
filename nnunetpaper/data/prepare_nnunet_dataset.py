import json
from pathlib import Path

import click
import numpy as np
from SimpleITK import ReadImage, WriteImage, GetArrayFromImage
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _find_file(path: Path, glob: str) -> Path | None:
    candidates = [x.resolve() for x in path.glob(glob) if x.is_file()]

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        for i, c in enumerate(candidates):
            print(f" {i}: {c.name}")
        selection = int(input(f"Please enter the number of the correct candidate for glob {glob}"))
        return candidates[selection]
    else:
        return None


def _process_set(
        samples: list[Path],
        image_glob: list[str],
        label_glob: str,
        output_base: Path,
        output_dir_suffix: str,
        allow_missing_label: bool,
        as_posix: bool
) -> tuple[dict[str, str], int]:
    paths: list[dict[str, str]] = []
    n_classes = 0

    for sample in tqdm(samples, desc=f"Processing {output_dir_suffix}"):
        image_base = Path(f"images{output_dir_suffix}")
        label_base = Path(f"labels{output_dir_suffix}")

        # Prepare output directories
        if not output_base.exists():
            output_base.mkdir(parents=True)
        if not (output_base / image_base).exists():
            (output_base / image_base).mkdir()
        if not (output_base / label_base).exists():
            (output_base / label_base).mkdir()

        should_skip = False

        # Find image(s)
        images = []
        for i in image_glob:
            image = _find_file(sample, i)
            if image is None:
                print(f"Image could not be found for {i} in {sample.name}, skipping this patient")
                should_skip = True
            else:
                images.append(image)

        # If something is wrong with the images, skip to next patient
        if should_skip:
            continue

        # Find label
        label = _find_file(sample, label_glob)
        if label is None:
            if not allow_missing_label:
                print(f"Label could not be found for {label_glob} in {sample.name}, skipping this patient")
                should_skip = True

        # If something is wrong with the label, and we don't allow missing labels, skip
        if should_skip:
            continue

        # Now that we know our valid images, and label (if it exists)
        # start writing to output!
        # Image(s) first
        for idx, image in enumerate(images):
            output_name = sample.name + f"_{idx:04d}.nii.gz"

            WriteImage(
                ReadImage(image),
                output_base / image_base / output_name
            )

        # Labels second
        if label is not None:
            output_name = sample.name + ".nii.gz"

            label_image = ReadImage(label)
            label_array = GetArrayFromImage(label_image)

            highest_class_no = int(np.max(label_array).item())
            n_classes = max(n_classes, highest_class_no)

            WriteImage(
                label_image,
                output_base / label_base / output_name
            )

        if as_posix:
            paths.append({
                "image": (image_base / (sample.name + ".nii.gz")).as_posix(),
                "label": (label_base / (sample.name + ".nii.gz")).as_posix()
            })
        else:
            paths.append({
                "image": str(image_base / (sample.name + ".nii.gz")),
                "label": str(label_base / (sample.name + ".nii.gz"))
            })

    return paths, n_classes


@click.command()
@click.option(
    "-d",
    "--dataset",
    "datasets",
    required=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
)
@click.option("-i", "--image", "image_glob", multiple=True, required=True, type=str)
@click.option("-l", "--label", "label_glob", required=True, type=str)
@click.option("-s", "--split", required=False, type=float, default=0.2)
@click.option(
    "-p", "--posix", "as_posix", is_flag=True, type=bool, required=False, default=False
)
@click.option(
    "-m",
    "--missing-label",
    "allow_missing_label",
    is_flag=True,
    type=bool,
    required=False,
    default=False,
)
def main(
    datasets: list[Path],
    output: Path,
    image_glob: list[str],
    label_glob: str,
    split: float,
    as_posix: bool,
    allow_missing_label: bool,
):
    # If the user provided more than one source directory, append the samples from each one
    samples = []
    for path in tqdm([Path(x) for x in datasets], "Finding samples"):
        samples += [y.resolve() for y in path.glob("*") if y.is_dir()]

    if split > 0.0:
        train_set, test_set = train_test_split(samples, test_size=split)
    else:
        train_set = samples
        test_set = []

    train_paths, n_classes = _process_set(
        train_set,
        image_glob=image_glob,
        label_glob=label_glob,
        output_base=output,
        output_dir_suffix="Tr",
        allow_missing_label=allow_missing_label,
        as_posix=as_posix
    )
    test_paths, test_n_classes = _process_set(
        test_set,
        image_glob=image_glob,
        label_glob=label_glob,
        output_base=output,
        output_dir_suffix="Ts",
        allow_missing_label=allow_missing_label,
        as_posix=as_posix
    )

    if n_classes == test_n_classes:
        raise RuntimeError("We did not find the same amount of classes in the test and train set")

    data_description = {
        "name": "name",
        "description": "desc",
        "reference": "UMC Utrecht, Utrecht, The Netherlands",
        "license": "None, all rights reserved",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {f"{i}": f"modality {i}" for i in range(len(image_glob))},
        "labels": {f"{i}": f"label {i}" for i in range(n_classes + 1)},
        "numTraining": len(train_paths),
        "numTest": len(test_paths),
        "training": train_paths,
        "test": test_paths,
    }

    with open(output / "dataset.json", "w") as out_file:
        json.dump(data_description, out_file, indent=4)

    print("\nDone!")


if __name__ == '__main__':
    main()
