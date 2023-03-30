from pathlib import Path

import click
import SimpleITK as sitk
from tqdm import tqdm


def _process_patient(image: sitk.Image) -> sitk.Image:
    # Relabel each separate component
    cc_filter = sitk.ConnectedComponentImageFilter()
    image = cc_filter.Execute(image)
    # The RelabelComponent filter can sort by size,
    image = sitk.RelabelComponent(image, sortByObjectSize=True)
    return image == 1


@click.group()
def main():
    ...


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None
)
def directory(input_path: Path, output_path: Path | None = None):
    if output_path is not None and not output_path.exists():
        output_path.mkdir(parents=True)

    skipped_files = []

    patient: Path
    for patient in (
        prog_bar := tqdm([x.resolve() for x in input_path.glob("*") if x.is_file()])
    ):
        prog_bar.set_description(f"Processing {patient.name}")
        try:
            image = sitk.ReadImage(patient)
        except RuntimeError:
            skipped_files.append(patient)
            continue
        if output_path is not None:
            sitk.WriteImage(_process_patient(image), output_path / patient.name)
        else:
            sitk.WriteImage(_process_patient(image), patient)

    print("Skipped:")
    for s in skipped_files:
        print(f"\t{s}")


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def single(input_path: Path, output_path: Path = None):
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    image = sitk.ReadImage(input_path)
    sitk.WriteImage(_process_patient(image), output_path / input_path.name)


if __name__ == "__main__":
    main()
