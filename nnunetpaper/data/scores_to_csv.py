from pathlib import Path

import click
import pandas as pd

from nnunetpaper._utils import get_multi_method_dataframe


@click.command()
@click.option(
    "-m",
    "--methods",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option(
    "-o", "--output", required=True, type=click.Path(writable=True, path_type=Path)
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
def main(methods: list[tuple[str, Path]], output: Path, auto_collect_anatomies: bool = False):
    if output.is_dir():
        output /= "scores.csv"

    data = get_multi_method_dataframe(methods, auto_collect_anatomies)
    print(data.head(10))

    # Filter out the repeated data
    data = data[data["center"] != "All"]

    data = pd.pivot_table(
        data,
        values="metric",
        index=["pt_id", "anatomy", "methods", "center"],
        columns="metric_name",
    )
    # data.reset_index(drop=False, inplace=True)
    print(data.head(10))

    data.to_csv(output)


if __name__ == "__main__":
    main()
