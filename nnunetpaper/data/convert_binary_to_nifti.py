import gc
import struct
from pathlib import Path

import click
import nibabel as nib
import numpy as np
from tqdm import tqdm


def _convert_bin(path: Path, reference: Path) -> nib.Nifti1Image:
    ref = nib.load(reference)
    axcodes = nib.aff2axcodes(ref.affine)

    # The original .bin files are stored in LPS order, slice first
    # that is; (z, y, x)
    # Images are not nessecarily stored in that way, so we need to determine
    # image axis order via their axcodes

    shape = [-1, -1, -1]
    order = [-1, -1, -1]
    flip = []
    for i, c in enumerate(axcodes):
        if c in "LR":
            shape[0] = ref.shape[i]
            order[0] = i

            if c == "R":
                flip.append(i)
        elif c in "PA":
            shape[1] = ref.shape[i]
            order[1] = i

            if c == "A":
                flip.append(i)
        elif c in "IS":
            shape[2] = ref.shape[i]
            order[2] = i

            if c == "I":
                flip.append(i)
        else:
            raise RuntimeError(f"Invalid axcode {c}")

    with open(path, mode="rb") as file:
        file_content = file.read()

    # Now that we have the axes in L/R, A/P, I/S order, we can reshape our 1D binary vector
    bin_data: tuple = struct.unpack("H" * (len(file_content) // 2), file_content)
    bin_data: np.ndarray = np.reshape(bin_data, shape[::-1]).astype(np.uint8)

    # And because we need to be in the same voxel order as our reference image
    # transpose and flip as needed
    bin_data = np.transpose(bin_data, axes=order[::-1])
    bin_data = np.flip(bin_data, axis=flip)

    output = nib.Nifti1Image(bin_data, ref.affine)
    del ref
    del bin_data

    return output


@click.group()
def main():
    ...


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(writable=True, path_type=Path),
)
@click.option(
    "-r",
    "--reference",
    "reference_path",
    required=True,
    type=click.Path(writable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    type=click.Path(writable=True, path_type=Path),
)
def directory(input_path: Path, reference_path: Path, output_path: Path = None):
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = [x.resolve() for x in input_path.glob("*.bin") if x.is_file()]
    else:
        raise ValueError(f"{input_path} is not a directory or a file!")

    if reference_path.is_file():
        references = [reference_path]
    elif reference_path.is_dir():
        references = []
        for image in (prog_bar := tqdm(images, "Scanning for references...")):
            stem = image.stem
            prog_bar.set_description(f"Scanning for references... ({stem})")
            ref_candidates = [
                x
                for x in reference_path.iterdir()
                if x.is_file() and stem + "_0000" in x.stem
            ]
            if len(ref_candidates) > 1:
                raise RuntimeError(
                    f"Too many candidate references for {image}:\n{ref_candidates}"
                )
            elif len(ref_candidates) == 0:
                raise RuntimeError(f"No candidate references for {image}")
            else:
                references.append(ref_candidates[0])
    else:
        raise ValueError(f"{reference_path} is not a directory or a file!")

    if output_path is None:
        outputs = [x.parent / (x.stem + ".nii.gz") for x in images]
    elif output_path.is_file():
        outputs = [output_path]
    elif output_path.is_dir():
        outputs = [output_path / (x.stem + ".nii.gz") for x in images]
    else:
        raise ValueError(f"{output_path} is not a directory or a file!")

    for im, ref, out in (
        prog_bar := tqdm(zip(images, references, outputs), total=len(images))
    ):
        prog_bar.set_description(f"Converting {im.name}")
        try:
            result = _convert_bin(im, ref)
            nib.save(result, out)
            del result
            gc.collect()
        except MemoryError:
            print(f"Encountered a memory error for {im.name}")
            continue


@main.command()
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-r",
    "--reference",
    "reference_path",
    required=True,
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
def single(input_path: Path, reference_path: Path, output_path: Path = None):
    if output_path is None:
        output_path = input_path.parent / (input_path.stem + ".nii.gz")

    result = _convert_bin(input_path, reference_path)
    nib.save(result, output_path)


if __name__ == "__main__":
    main()
