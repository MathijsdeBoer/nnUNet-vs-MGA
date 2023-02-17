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
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, shapiro, mannwhitneyu

from nnunetpaper.plot.utils import read_json


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

    anatomies = [x.parent.name for x in files]
    metric_names = data["metric_name"].unique()

    sns.set_style("whitegrid")
    sns.set_context("paper")
    g = sns.catplot(
        data=data,
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
            ax.set_ylim((0.0, 1.1))
        elif col in ["hd95", "assd"]:
            ax.set_ylabel("Distance (voxels)")
            ax.set_yscale("log")
            ax.set_ylim((None, None))

    sns.despine(trim=True, left=True)

    plt.savefig(output, dpi=300)

    for anatomy in anatomies:
        print(anatomy)
        for name in metric_names:
            all_centers = data.loc[
                (data["metric_name"] == name)
                & (data["anatomy"] == anatomy)
                & (data["center"] == "All"),
                "metric",
            ]
            umcu = data.loc[
                (data["metric_name"] == name)
                & (data["anatomy"] == anatomy)
                & (data["center"] == "UMCU"),
                "metric",
            ]
            usz = data.loc[
                (data["metric_name"] == name)
                & (data["anatomy"] == anatomy)
                & (data["center"] == "USZ"),
                "metric",
            ]

            print(f"\t{name} | All:  {all_centers.mean():#.3g} ($\\pm$ {all_centers.std():#.3g})")
            print(
                f"\t{len(name) * ' '} | UMCU: {umcu.mean():#.3g} ($\\pm$ {umcu.std():#.3g})"
            )
            print(f"\t{len(name) * ' '} | USZ:  {usz.mean():#.3g} ($\\pm$ {usz.std():#.3g})")

            mannwhitney = mannwhitneyu(
                umcu,
                usz
            )
            print("\tMann-Whitney U:")
            if mannwhitney.pvalue < 0.0125:
                print(f"\t\t{mannwhitney.statistic:#.3g} ($\\mathbf{{p = {mannwhitney.pvalue:#.3g}}}$)")
            else:
                print(f"\t\t{mannwhitney.statistic:#.3g} ($p = {mannwhitney.pvalue:#.3g}$)")


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
            ax.set_ylim((None, 1.0))
        elif col[0] in ["hd95", "assd"]:
            ax.set_ylabel("Distance (voxels)")
            ax.set_yscale("log")
            ax.set_ylim((None, None))

    plt.savefig(output, dpi=300)


@main.command()
@click.option(
    "-m",
    "--method",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-o", "--output", required=True, type=click.Path(writable=True, path_type=Path)
)
def methods(method: list[tuple[str, Path]], output: Path):
    if output.is_dir():
        output /= "methods_plot.png"

    method_dict: dict[str, list[Path]] = {}
    for method_name, scores_path in method:
        method_dict[method_name] = method_dict.get(method_name, [])
        method_dict[method_name].append(scores_path)

    method_data: dict[str, pd.DataFrame] = {}
    for method_name in method_dict.keys():
        method_data[method_name] = read_json(method_dict[method_name])
        method_data[method_name]["method"] = method_name
    data = pd.concat(method_data.values(), ignore_index=True)

    data["method_and_center"] = data["method"] + " " + data["center"]
    print(data.head())

    sns.set_style("whitegrid")
    sns.set_context("paper")
    g = sns.catplot(
        data=data,
        x="anatomy",
        y="metric",
        hue="method_and_center",
        col="metric_name",
        col_wrap=2,
        kind="box",
        sharey=False,
        palette=[
            "darkorange",
            "orange",
            "gold",
            "royalblue",
            "cornflowerblue",
            "skyblue",
            "olivedrab",
            "yellowgreen",
            "greenyellow",
            "mediumvioletred",
            "palevioletred",
            "hotpink",
        ],
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    for col, ax in g.axes_dict.items():
        if col in ["dice", "iou"]:
            ax.set_ylabel("Score")
            ax.set_ylim((0.0, 1.1))
        elif col in ["hd95", "assd"]:
            ax.set_ylabel("Distance (voxels)")
            ax.set_yscale("log")
            ax.set_ylim((None, None))

    sns.despine(trim=True, left=True)

    plt.savefig(output, dpi=300)

    test_data = data[data["center"] == "All"]
    for anatomy in test_data["anatomy"].unique():
        print(f"{anatomy}")
        anatomy_data = test_data[test_data["anatomy"] == anatomy]

        for metric in anatomy_data["metric_name"].unique():
            print(f"\t{metric}")
            metric_data = anatomy_data[anatomy_data["metric_name"] == metric]

            for combo in combinations(metric_data["method"].unique(), 2):
                test = mannwhitneyu(
                    x=metric_data.loc[metric_data["method"] == combo[0], "metric"],
                    y=metric_data.loc[metric_data["method"] == combo[1], "metric"],
                )

                print(f"\t\tTesting {combo[0]} vs {combo[1]}")
                report = f"\t\t{test.statistic:#.3g}, "
                if test.pvalue < 0.05:
                    report += f"($\\mathbf{{p = {test.pvalue:#.3g}}}$)"
                else:
                    report += f"($p = {test.pvalue:#.3g}$)"
                print(report)


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


if __name__ == "__main__":
    main()
