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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

from nnunetpaper._utils import get_multi_method_dataframe
from nnunetpaper.data import read_json


def metric_formatter(metric_name: str) -> str:
    if metric_name == "dice":
        return "DSC"
    elif metric_name == "iou":
        return "IoU"
    elif metric_name == "hd95":
        return "HD95"
    elif metric_name == "assd":
        return "ASSD"
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")


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
    type=click.Path(writable=True, file_okay=False, path_type=Path),
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
@click.option(
    "-c",
    "--plot-centers",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    show_default=True,
)
def method(
    methods: list[tuple[str, Path]], output: Path, auto_collect_anatomies: bool = False, plot_centers: bool = False
):
    data = get_multi_method_dataframe(methods, auto_collect_anatomies)
    data["method_and_center"] = data["methods"] + " " + data["center"]

    if not plot_centers:
        data = data[data["center"] == "All"]

    anatomies = data["anatomy"].unique()
    metric_names = data["metric_name"].unique()

    sns.set_style("whitegrid")
    sns.set_context(
        "paper",
        font_scale=0.5,
        rc={
            "figure.figsize": (32, 32),
            "lines.linewidth": 0.5,
        }
    )

    fig = plt.figure()
    outer_gs = GridSpec(1, 2, figure=fig, wspace=0.2, hspace=0.25)
    overlap_gs = GridSpecFromSubplotSpec(len(anatomies), len(metric_names) - 2, subplot_spec=outer_gs[0, 0], wspace=0.2)
    distance_gs = GridSpecFromSubplotSpec(len(anatomies), len(metric_names) - 2, subplot_spec=outer_gs[0, 1], wspace=0.2)

    axs = []
    for row, anatomy in enumerate(anatomies):
        for col, metric in enumerate(metric_names):
            if metric in ["dice", "iou"]:
                axs.append(fig.add_subplot(overlap_gs[row, col]))
            elif metric in ["hd95", "assd"]:
                axs.append(fig.add_subplot(distance_gs[row, col - 2]))

    for row, anatomy in enumerate(anatomies):
        for col, metric in enumerate(metric_names):
            print(f"{anatomy} - {metric}: {row} - {col}")
            ax = axs[row * len(metric_names) + col]
            sns.boxplot(
                data=data[(data["anatomy"] == anatomy) & (data["metric_name"] == metric)],
                x="methods",
                y="metric",
                hue="center",
                fliersize=2.5,
                ax=ax,
                legend=False,
                linewidth=1.0,
            )

            ax.tick_params(axis="both", which="major", pad=-2.5)
            ax.tick_params(axis="both", which="minor", pad=-0.5)

            # Set the metric names to the top row
            if row == 0:
                ax.set_title(f"$\\bf{{{metric_formatter(metric)}}}$")

            # Set the methods label to the bottom row
            if row != len(anatomies) - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Method")

            # Set the y-axis to log for distance metrics
            if metric in ["hd95", "assd"]:
                ax.set_yscale("log")

                formatter = FuncFormatter(lambda y, _: "{:.16g}".format(y))
                locator = LogLocator(subs=[1.0, 2.5, 5.0])

                ax.yaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_locator(locator)

                ax.yaxis.set_minor_formatter(NullFormatter())

            # Turn off y-axis labels for all but the first column
            # in a given metric type
            if col == 0:
                ax.set_ylabel("Score")
            elif col == 2:
                ax.set_ylabel("Distance (mm)")
            elif col == 3:
                print("Setting anatomy label")
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"$\\bf{{{anatomy}}}$", rotation=-90, labelpad=10)
            else:
                ax.set_ylabel("")

    if not output.exists():
        output.mkdir(parents=True)

    if plot_centers:
        # Add the legend for the entire figure
        legend_elements = [
            Patch(facecolor="C0", edgecolor="k", label="All"),
            Patch(facecolor="C1", edgecolor="k", label="Center A"),
            Patch(facecolor="C2", edgecolor="k", label="Center B"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="center right",
        )

    plt.tight_layout()
    plot_output = output / f"methods_plot.png"
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
