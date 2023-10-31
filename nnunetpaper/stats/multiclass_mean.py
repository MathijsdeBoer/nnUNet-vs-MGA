import json
from pathlib import Path

import click
import pandas as pd


def read_json(file: Path) -> pd.DataFrame:
    with file.open('r') as f:
        d: dict = json.load(f)

    transformed_dict = {
        "metric": [],
        "metric_name": [],
        "class": [],
    }

    for k in d.keys():
        for c in d[k]:
            dsc = c["dice"]
            if not (dsc == float("nan") or dsc == float("inf") or dsc == float("-inf")):
                transformed_dict["metric"].append(c["dice"])
                transformed_dict["metric_name"].append("dice")
                transformed_dict["class"].append(c["class"])

            hd95 = c["hd95"]
            if not (hd95 == float("nan") or hd95 == float("inf") or hd95 == float("-inf")):
                transformed_dict["metric"].append(c["hd95"])
                transformed_dict["metric_name"].append("hd95")
                transformed_dict["class"].append(c["class"])

    return pd.DataFrame(transformed_dict)


@click.command()
@click.argument("file", type=click.Path(readable=True, path_type=Path))
@click.option(
    "-d",
    "--dataset-file",
    required=True,
    type=click.Path(exists=True, readable=True, path_type=Path),
)
def main(file: Path, dataset_file: Path):
    data = read_json(file)

    with dataset_file.open("r") as f:
        dataset = json.load(f)
        label_names = dataset["labels"]
        label_names = {
            int(v): k for k, v in label_names.items()
        }

    classes = data["class"].unique()
    classes.sort()

    for c in classes:
        print(label_names[c])
        for name in ["dice", "hd95"]:
            class_data = data.loc[
                (data["metric_name"] == name)
                & (data["class"] == c),
                "metric",
            ]
            n_valid = f"{len(class_data)}"

            print(f"\t{name} | mean: {class_data.mean():#.3g} (std: {class_data.std():#.3g})")
            print(f"\tn={n_valid}{' ' * (len(name) - len(n_valid) - 2)} | median: {class_data.median():#.3g} "
                  f"(95CI: [{class_data.quantile(0.025):#.3g}, {class_data.quantile(0.975):#.3g}])")


if __name__ == "__main__":
    main()
