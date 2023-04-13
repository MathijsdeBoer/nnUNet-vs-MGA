import json
from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.option(
    "-d",
    "--directory",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-s",
    "--scores",
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
)
@click.option(
    "-m",
    "--model",
    required=True,
    type=int,
)
@click.option(
    "-o",
    "--output",
    required=False,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=True, writable=True, path_type=Path
    ),
    default=None,
)
def main(directory: Path, scores: Path, model: int, output: Path = None):
    if output is None:
        output = scores
    elif output.is_dir():
        output /= "scores.json"

    with open(scores, "r") as f:
        scores = json.load(f)

    subdirs = [x.resolve() for x in directory.iterdir() if x.is_dir()]
    for subdir in (prog_bar := tqdm(subdirs, desc="Processing patients")):
        patient_id = subdir.name.split("_")[0] + ".nii.gz"
        prog_bar.set_description(f"Processing {patient_id}")

        if patient_id not in scores.keys():
            continue

        time = 0
        for run in range(1, 4):
            with open(subdir / f"{run}" / f"{model}" / "prediction_time.txt", "r") as f:
                time += float(f.read())
        time /= 3

        scores[patient_id]["time"] = time

    with open(output, mode="w") as file:
        j = json.dumps(scores, indent=4)
        file.write(j)


if __name__ == "__main__":
    main()
