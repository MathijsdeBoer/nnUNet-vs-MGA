import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


@click.command()
@click.option(
    "-m",
    "--method",
    "methods",
    multiple=True,
    required=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True, file_okay=True, path_type=Path),
)
@click.option(
    "-d",
    "--dataset-file",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
)
def main(methods: list[tuple[str, Path]], output: Path, dataset_file: Path) -> None:
    data = {
        "pt_id": [],
        "method": [],
        "metric": [],
        "metric_name": [],
        "anatomy": [],
    }

    print("Reading dataset file")
    with dataset_file.open("r") as f:
        dataset = json.load(f)
        label_names = dataset["labels"]
        label_names = {
            int(v): k for k, v in label_names.items()
        }

    print("Reading scores")
    for method, path in methods:
        with path.open("r") as f:
            scores = json.load(f)

        for pt_id, pt_scores in scores.items():
            for score in pt_scores:
                for name in ["dice", "hd95"]:
                    data["method"].append(method)
                    data["metric"].append(score[name])
                    data["metric_name"].append(name)
                    data["anatomy"].append(label_names[score["class"]])
                    data["pt_id"].append(pt_id)

    data = pd.DataFrame(data)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    print(data.head())

    print("Plotting")
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=3)
    p = sns.catplot(
        data=data,
        x="anatomy",
        y="metric",
        hue="method",
        kind="box",
        col="metric_name",
        col_wrap=1,
        sharey=False,
        palette=sns.color_palette("colorblind"),
        legend=True,
        height=16,
        aspect=2,
    )

    p.set_titles(col_template="{col_name}", row_template="{row_name}")

    for col, ax in p.axes_dict.items():
        if col in ["dice", "iou"]:
            ax.set_ylabel("Score")
            # ax.set_ylim((0.0, 1.1))
        elif col in ["hd95", "assd"]:
            ax.set_ylabel("Distance (voxels)")
            ax.set_yscale("log")
            # ax.set_ylim((None, None))

    sns.despine(trim=True, left=True)

    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=300)


if __name__ == "__main__":
    main()
