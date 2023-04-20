from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nnunetpaper._utils import get_multi_method_dataframe


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-m",
    "--method",
    "methods",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-s",
    "--sum-method",
    "sum_methods",
    multiple=True,
    type=str,
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True, file_okay=True, path_type=Path),
)
def times(methods: list[tuple[str, Path]], sum_methods: list[str], output: Path):
    data = get_multi_method_dataframe(methods)

    time_data = {
        "method": [],
        "time": [],
    }

    data = data[data["metric_name"] == "dice"]
    data = data[data["center"] == "All"]

    for method in data["methods"].unique():
        print(method)
        if method in sum_methods:
            values = data[data["methods"] == method].groupby(
                ["pt_id"]
            )["time"].sum()

            print(f"{'mean':<10} {'std':<10} {'5th':<10} {'95th':<10}")
            print(
                f"{values.mean():<10.5g} {values.std():<10.5g} "
                f"{values.quantile(0.025):<10.5g} {values.quantile(0.975):<10.5g}"
            )
        else:
            values = data.loc[
                (data["methods"] == method)
                & (data["anatomy"] == data["anatomy"].unique()[0]),
                "time",
            ]

        for value in values:
            time_data["method"].append(method)
            time_data["time"].append(value)

    data = pd.DataFrame(time_data)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    g = sns.catplot(
        data=data,
        x="method",
        y="time",
        kind="box",
    )
    g.set_xlabels("Method")
    g.set_ylabels("Time (s)")
    sns.despine(trim=True, left=True)

    plt.suptitle("Prediction time")
    plt.savefig(output, dpi=300)


if __name__ == "__main__":
    main()
