from pathlib import Path

import click
import SimpleITK as sitk
from tqdm import tqdm


def _process_patient(input_file: Path, output_file: Path):
    image = sitk.ReadImage(input_file)

    # Initial closing pass
    output = sitk.BinaryFillhole(image, fullyConnected=True)
    sitk.WriteImage(output, output_file)


@click.group()
def main():
    ...


@main.command()
@click.option(
    "-i",
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_file",
    required=False,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def file(input_file: Path, output_file: Path = None):
    _process_patient(input_file, output_file)


@main.command()
@click.option(
    "-i",
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
def directory(input_dir: Path):
    skipped = []
    files = [x.resolve() for x in input_dir.glob("*") if x.is_file()]
    for pat in (prog_bar := tqdm(files)):
        prog_bar.set_description(f"{pat.name}")
        try:
            _process_patient(pat, pat)
        except RuntimeError:
            skipped.append(pat)

    if len(skipped) > 0:
        print("Skipped:")
        for s in skipped:
            print(f" - {s.name}")


if __name__ == "__main__":
    main()
