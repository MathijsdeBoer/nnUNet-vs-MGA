"""
The mesh-growing algorithm runs all segmentations in one go, so we can simply
copy the times from the first segmentation to all others.

Copying the times is less error-prone than manually copying them for each segmentation.
"""

import json
from pathlib import Path

import click


@click.command()
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, writable=True, path_type=Path),
)
@click.option(
    "-r", "--ref", required=True, type=click.Path(readable=True, path_type=Path)
)
def main(files: list[Path], ref: Path):
    with open(ref, "r") as f:
        ref_data = json.load(f)

    for file in files:
        with open(file, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            data[key]["time"] = ref_data[key]["time"]

        with open(file, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
