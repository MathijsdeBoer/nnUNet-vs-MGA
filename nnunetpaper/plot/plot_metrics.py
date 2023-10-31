"""
Plot Metrics
Author: Mathijs de Boer

This script is used to create plots and run statistical tests of the collected metrics.
To use this script, run collect_metrics.py on your segmentations first.
Note that this script is not quite universal, and is written with our dataset in mind.
That is, we manually check for UMCU and USZ labels, while your dataset might not have them.
"""
from itertools import combinations
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from nnunetpaper._utils import get_multi_method_dataframe
from nnunetpaper.data import read_json


@click.group()
def main():
    ...


@main.command()
@click.argument("files", nargs=-1, type=click.Path(readable=True, path_type=Path))
@click.option(
    "-o", "--output", required=True, type=click.Path(writable=True, path_type=Path)
)
def metrics(files: list[Path], output: Path):
    data = read_json(files)

    sns.set_style("whitegrid")
    sns.set_context("paper")
    g = sns.catplot(
        data=data[data["center"] != "All"],
        x="anatomy",
        y="metric",
        hue="center",
        col="metric_name",
        col_wrap=2,
        kind="box",
        sharey=False,
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    for col, ax in g.axes_dict.items():
        if col in ["dice", "iou"]:
            ax.set_ylabel("Score")
            # ax.set_ylim((0.0, 1.1))
        elif col in ["hd95", "assd"]:
            ax.set_ylabel("Distance (voxels)")
            ax.set_yscale("log")
            # ax.set_ylim((None, None))

    sns.despine(trim=True, left=True)

    plt.savefig(output, dpi=300)


@main.command()
@click.argument("files", nargs=-1, type=click.Path(readable=True, path_type=Path))
@click.option(
    "-o", "--output", required=True, type=click.Path(writable=True, path_type=Path)
)
def volume(files: list[Path], output: Path):
    data = read_json(files)
    data["segment_volume"] = data["segment_volume"].div(1_000)

    sns.set_style("whitegrid")
    sns.set_context("paper")
    p = sns.FacetGrid(
        data=data,
        col="anatomy",
        row="metric_name",
        sharey=False,
        sharex=False,
        margin_titles=True,
    )
    p.map_dataframe(sns.regplot, x="segment_volume", y="metric")
    p.set_titles(col_template="{col_name}", row_template="{row_name}")
    p.add_legend()
    p.set_xlabels(label="Segmentation Volume ($cm^3$)")

    for col, ax in p.axes_dict.items():
        if col[0] in ["dice", "iou"]:
            ax.set_ylabel("Score")
            # ax.set_ylim((None, 1.0))
        elif col[0] in ["hd95", "assd"]:
            ax.set_ylabel("Distance (voxels)")
            ax.set_yscale("log")
            # ax.set_ylim((None, None))

    plt.savefig(output, dpi=300)


@main.command()
@click.option(
    "-m",
    "--method",
    "methods",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True, file_okay=True, path_type=Path),
)
@click.option(
    "-a",
    "--auto-collect-anatomies",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    show_default=True,
)
def method(
    methods: list[tuple[str, Path]], output: Path, auto_collect_anatomies: bool = False
):
    data = get_multi_method_dataframe(methods, auto_collect_anatomies)
    data["method_and_center"] = data["methods"] + " " + data["center"]

    anatomies = data["anatomy"].unique()
    for anatomy in anatomies:
        print(f"Plotting {anatomy}")
        sns.set_style("whitegrid")
        sns.set_context(
            "paper", font_scale=2, rc={"lines.linewidth": 3, "figure.figsize": (32, 32)}
        )
        g = sns.catplot(
            data=data[data["anatomy"] == anatomy],
            x="methods",
            y="metric",
            hue="center",
            col="metric_name",
            col_wrap=2,
            kind="box",
            sharey=False,
            palette=sns.color_palette("colorblind"),
            legend=True,
        )
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        plt.suptitle(anatomy)

        for col, ax in g.axes_dict.items():
            if col in ["dice", "iou"]:
                ax.set_ylabel("Score")
                # ax.set_ylim((0.0, 1.1))
            elif col in ["hd95", "assd"]:
                ax.set_ylabel("Distance (voxels)")
                ax.set_yscale("log")
                # ax.set_ylim((None, None))

        sns.despine(trim=True, left=True)

        if not output.exists():
            output.mkdir(parents=True)
        plot_output = output / f"methods_plot_{anatomy}.png"
        plt.savefig(plot_output, dpi=300)


@main.command()
@click.argument("files", nargs=-1, type=click.Path(readable=True, path_type=Path))
@click.option(
    "-o", "--output", required=True, type=click.Path(writable=True, path_type=Path)
)
def correlation(files: list[Path], output: Path):
    if output.is_dir():
        output /= "correlation_plot.png"

    data = read_json(files)

    # We just want all the data once
    data = data[data["center"] == "All"]
    data = pd.pivot_table(
        data, values="metric", index=["pt_id", "anatomy"], columns="metric_name"
    )
    data.reset_index(drop=False, inplace=True)

    print(data.head())

    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.pairplot(
        data=data,
        hue="anatomy",
        vars=["dice", "iou", "hd95", "assd"],
        markers=["o", "s", "D", "P"],
        kind="scatter",
        diag_kind="auto",
    )
    plt.savefig(output, dpi=300)


@main.command()
@click.option(
    "-m",
    "--method",
    "methods",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True, file_okay=True, path_type=Path),
)
def blandaltman(methods: list[tuple[str, Path]], output: Path):
    data = get_multi_method_dataframe(methods)
    data["method_and_center"] = data["methods"] + " " + data["center"]

    data.loc[data["metric_name"] == "dice", "metric_name"] = "DSC"
    data.loc[data["metric_name"] == "iou", "metric_name"] = "IoU"
    data.loc[data["metric_name"] == "hd95", "metric_name"] = "HD95"
    data.loc[data["metric_name"] == "assd", "metric_name"] = "ASSD"

    combos = []
    for a, b in combinations(data["methods"].unique(), 2):
        combos.append((a, b))
        data[f"{a}_vs_{b}"] = (
            data.loc[data["methods"] == a, "metric"]
            - data.loc[data["methods"] == b, "metric"]
        )

    sns.set_style("whitegrid")
    row_key = "anatomy"
    col_key = "metric_name"

    rows = data[row_key].unique()
    cols = data[col_key].unique()
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(24, 24))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Bland-Altman Plot", fontsize=24)

    for row, anatomy in enumerate(rows):
        for col, metric in enumerate(cols):
            for a, b in combos:
                print(f"{a} vs {b} - {anatomy}: {metric}")
                print(f"\tRow: {row}, Col: {col}")

                m1 = []
                m2 = []

                for pt_id in data["pt_id"].unique():
                    try:
                        m1.append(
                            data.loc[
                                (data["pt_id"] == pt_id)
                                & (data["anatomy"] == anatomy)
                                & (data["metric_name"] == metric)
                                & (data["methods"] == a),
                                "metric",
                            ].values[0]
                        )
                        m2.append(
                            data.loc[
                                (data["pt_id"] == pt_id)
                                & (data["anatomy"] == anatomy)
                                & (data["metric_name"] == metric)
                                & (data["methods"] == b),
                                "metric",
                            ].values[0]
                        )
                    except IndexError:
                        pass

                sm.graphics.mean_diff_plot(
                    m1=np.array(m1),
                    m2=np.array(m2),
                    ax=ax[row, col],
                )

                ax[0, col].set_title(f"{metric}")
                ax[row, col].set_xlabel(f"{a} - {b}")
                ax[row, col].set_ylabel(f"{metric}")

    fig.tight_layout()
    plt.savefig(output, dpi=300)


if __name__ == "__main__":
    main()
