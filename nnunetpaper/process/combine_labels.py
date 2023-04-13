from pathlib import Path

import click
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def _process_patient(input_files: list[Path], output_file: Path):
    images = [sitk.ReadImage(f) for f in input_files]
    arrays = [sitk.GetArrayFromImage(i) for i in images]

    output = np.zeros_like(arrays[0])
    for idx, im in (
        prog_bar := tqdm(enumerate(arrays), total=len(arrays), leave=False, position=1)
    ):
        prog_bar.set_description(f"{idx}")
        output = np.where(im == 1, idx + 1, output)

    output = sitk.GetImageFromArray(output)
    output.CopyInformation(images[0])
    sitk.WriteImage(output, output_file)


@click.command()
@click.option(
    "-i",
    "--input",
    "input_dirs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
)
def main(input_dirs: list[Path], output: Path):
    if not output.exists():
        output.mkdir(parents=True)

    skipped = []

    files = [x.resolve() for x in input_dirs[0].glob("*") if x.is_file()]
    for file in (prog_bar := tqdm(files)):
        prog_bar.set_description(f"{file.name}")
        try:
            _process_patient([p / file.name for p in input_dirs], output / file.name)
        except RuntimeError:
            skipped.append(file)

    if len(skipped) > 0:
        print("Skipped:")
        for s in skipped:
            print(f" - {s.name}")


if __name__ == "__main__":
    main()
