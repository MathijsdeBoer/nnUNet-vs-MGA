from pathlib import Path

import click

from nnunetpaper.data import read_json


@click.command()
@click.argument("file", type=click.Path(readable=True, path_type=Path))
@click.option("-m", "--metric-name", required=False, default=None)
@click.option("-v", "--value", required=False, default=None)
def main(file: Path, metric_name: str | None = None, value: str | None = None):
    if metric_name and value:
        raise ValueError("Only one of metric_name or value can be specified.")
    elif not metric_name and not value:
        raise ValueError("One of metric_name or value must be specified.")

    data = read_json([file])

    table = {}

    for center in data["center"].unique():
        center_data = data[data["center"] == center]
        if metric_name:
            values = center_data.loc[
                center_data["metric_name"] == metric_name, "metric"
            ]
        else:
            values = center_data.loc[
                center_data["metric_name"] == center_data["metric_name"].unique()[0],
                value,
            ]

        print(f"duplicates: {center_data.duplicated().any()}")
        table[center] = values

    print(
        f"{'Center':<10} | {'Mean':<10} | {'Std':<10} | {'Min':<10} |"
        f" {'Max':<10} | {'Median':<10} | {'5th':<10} | {'95th':<10} | {'N':<10}"
    )
    for center, values in table.items():
        print(
            f"{center:<10} | {values.mean():<10.5g} | {values.std():<10.3g} | {values.min():<10.3g} |"
            f" {values.max():<10.3g} | {values.median():<10.3g} | {values.quantile(0.05):<10.5g} |"
            f" {values.quantile(0.95):<10.5g} | {len(values):<10}"
        )


if __name__ == "__main__":
    main()
