from itertools import combinations
from pathlib import Path

import click
from scipy.stats import mannwhitneyu

from nnunetpaper._utils import get_multi_method_dataframe
from nnunetpaper.data import read_json


@click.group()
def main():
    ...


@main.command()
@click.argument("files", nargs=-1, type=click.Path(readable=True, path_type=Path))
def center(files: list[Path]):
    data = read_json(files)

    anatomies = [x.parent.name for x in files]
    metric_names = data["metric_name"].unique()

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

            print(
                f"\t{name} | All:  {all_centers.mean():#.3g} ($\\pm$ {all_centers.std():#.3g})"
            )
            print(
                f"\t{len(name) * ' '} | UMCU: {umcu.mean():#.3g} ($\\pm$ {umcu.std():#.3g})"
            )
            print(
                f"\t{len(name) * ' '} | USZ:  {usz.mean():#.3g} ($\\pm$ {usz.std():#.3g})"
            )

            mannwhitney = mannwhitneyu(umcu, usz)
            print("\tMann-Whitney U:")
            if mannwhitney.pvalue < 0.0125:
                print(
                    f"\t\t{mannwhitney.statistic:#.3g} ($\\mathbf{{p = {mannwhitney.pvalue:#.3g}}}$)"
                )
            else:
                print(
                    f"\t\t{mannwhitney.statistic:#.3g} ($p = {mannwhitney.pvalue:#.3g}$)"
                )


@main.command()
@click.option(
    "-m",
    "--methods",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-c", "--center", "center_to_check", required=False, type=str, default="All"
)
def method(methods: list[tuple[str, Path]], center_to_check: str = "All"):
    data = get_multi_method_dataframe(methods)

    test_data = data[data["center"] == center_to_check]
    for anatomy in test_data["anatomy"].unique():
        print(f"{anatomy}")
        anatomy_data = test_data[test_data["anatomy"] == anatomy]

        for metric in anatomy_data["metric_name"].unique():
            print(f"\t{metric}")
            metric_data = anatomy_data[anatomy_data["metric_name"] == metric]
            method_names = metric_data["methods"].unique()

            for combo in combinations(method_names, 2):
                x = metric_data.loc[metric_data["methods"] == combo[0], "metric"].to_numpy()
                y = metric_data.loc[metric_data["methods"] == combo[1], "metric"].to_numpy()

                test = mannwhitneyu(
                    x=x,
                    y=y,
                )

                print(f"\t\tTesting {combo[0]} vs {combo[1]}")
                report = f"\t\t\t{test.statistic:#.3g}, "
                if test.pvalue < 0.001:
                    report += f"($\\mathbf{{p < 0.001}}$)"
                elif test.pvalue < 0.0125:
                    report += f"($\\mathbf{{p = {test.pvalue:#.3g}}}$)"
                else:
                    report += f"($p = {test.pvalue:#.3g}$)"
                print(report)

            for method_name in method_names:
                scores = metric_data.loc[
                    metric_data["methods"] == method_name, "metric"
                ]
                print(
                    f"\t\t{method_name}: {scores.mean():#.3g} ($\\pm$ {scores.std():#.3g})"
                )


if __name__ == "__main__":
    main()
