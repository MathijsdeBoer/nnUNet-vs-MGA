from pathlib import Path

import click
import SimpleITK as sitk
from tqdm import tqdm


@click.command()
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
    required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
)
def main(input_path: Path, output_path: Path):
    brain_dir = input_path / "brain"
    tumor_dir = input_path / "tumor"
    ventricles_dir = input_path / "ventricles"

    if not output_path.exists():
        output_path.mkdir(parents=True)

    brain: Path
    for brain in (
        prog_bar := tqdm(
            [x.resolve() for x in brain_dir.glob("*.nii.gz") if x.is_file()]
        )
    ):
        prog_bar.set_description(f"Processing {brain.name}")

        brain_image = sitk.ReadImage(brain)
        if (tumor_dir / brain.name).exists():
            tumor_image = sitk.ReadImage(tumor_dir / brain.name)
        else:
            tumor_image = sitk.Image(brain_image.GetSize(), brain_image.GetPixelID())
            tumor_image.CopyInformation(brain_image)

        if (ventricles_dir / brain.name).exists():
            ventricles_image = sitk.ReadImage(ventricles_dir / brain.name)
        else:
            ventricles_image = sitk.Image(
                brain_image.GetSize(), brain_image.GetPixelID()
            )
            ventricles_image.CopyInformation(brain_image)

        output = brain_image - tumor_image - ventricles_image
        output = sitk.Clamp(output, lowerBound=0, upperBound=1)

        sitk.WriteImage(output, output_path / brain.name)


if __name__ == "__main__":
    main()
