from pathlib import Path

import click
import SimpleITK as sitk


def _process_patient(input_file: Path, output_file: Path, axis: str = "z"):
    image = sitk.ReadImage(input_file)

    # Find the Otsu Threshold
    # We use the multiple threshold function here because keyword arguments
    # work better than the single threshold version, functionally the same
    output: sitk.Image = sitk.OtsuMultipleThresholds(
        image, numberOfThresholds=1, numberOfHistogramBins=256
    )
    output.CopyInformation(image)

    # Initial closing pass
    output = sitk.BinaryMorphologicalClosing(output)

    # Because scans can be in different axis systems
    # we can also manually select which axis we need to do a slice-for-slice
    # binary fill hole in (usually the z axis for axial scans)
    if "x" in axis:
        for x in range(output.GetWidth()):
            output[x] = sitk.BinaryFillhole(
                output[x], foregroundValue=1, fullyConnected=True
            )
    if "y" in axis:
        for y in range(output.GetHeight()):
            output[::, y] = sitk.BinaryFillhole(
                output[::, y], foregroundValue=1, fullyConnected=True
            )
    if "z" in axis:
        for z in range(output.GetDepth()):
            output[..., z] = sitk.BinaryFillhole(
                output[..., z], foregroundValue=1, fullyConnected=True
            )

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
@click.option("-a", "--axis", required=False, default="z", type=str)
def file(input_file: Path, output_file: Path = None, axis: str = "z"):
    if output_file is None:
        output_file = input_file.parent / "label_skin.nii"
    _process_patient(input_file, output_file, axis)


@main.command()
@click.option(
    "-i",
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option("-a", "--axis", required=False, default="z", type=str)
def directory(input_dir: Path, axis: str = "z"):
    for pat in [x.resolve() for x in input_dir.glob("*") if x.is_dir()]:
        _process_patient(pat / "image.nii", pat / "label_skin.nii", axis)


if __name__ == "__main__":
    main()
