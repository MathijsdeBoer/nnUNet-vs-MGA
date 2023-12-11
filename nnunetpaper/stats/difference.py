from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from nnunetpaper._utils import get_multi_method_dataframe


@click.command()
@click.option(
    "-m",
    "--methods",
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, readable=True, path_type=Path)]),
)
@click.option("-n", "--metric-name", required=True, type=str)
@click.option("-a", "--auto-collect-anatomies", required=False, type=bool, is_flag=True, default=False, show_default=True)
def main(methods: list[tuple[str, Path]], metric_name: str, auto_collect_anatomies: bool = False):
    data = get_multi_method_dataframe(methods, auto_collect_anatomies)

    # Pre-filter the data
    data = data[data["metric_name"] == metric_name]

    # For each patient, get the difference between the two methods
    patients = data["pt_id"].unique()
    anatomies = data["anatomy"].unique()
    methods = data["methods"].unique()
    centers = data["center"].unique()

    anatomies = sorted(anatomies)

    if len(methods) != 2:
        raise ValueError("Only two methods can be compared.")

    results = {
        center: {
            x: [] for x in anatomies
        } for center in centers
    }

    for anatomy in tqdm(anatomies, desc="Anatomies", leave=False, position=0):
        for patient in tqdm(patients, desc="Patients", leave=False, position=1):
            for center in centers:
                patient_data = data[(data["pt_id"] == patient) & (data["anatomy"] == anatomy) & (data["center"] == center)]

                method1_score = patient_data[patient_data["methods"] == methods[0]]["metric"].to_numpy()
                method2_score = patient_data[patient_data["methods"] == methods[1]]["metric"].to_numpy()

                # Check NaN or not empty
                if len(method1_score) == 0 or len(method2_score) == 0 or np.isnan(method1_score) or np.isnan(method2_score):
                    continue

                results[center][anatomy].append(method1_score - method2_score)

    # Print the results
    print(f"Results for {metric_name}")
    print(f"{methods[0]} - {methods[1]}")
    for anatomy in anatomies:
        print(f"{anatomy}")
        for center in results:
            print(f"\t{center:<10}: N={len(results[center][anatomy]):<3} {np.mean(results[center][anatomy]):#.2g} [{np.quantile(results[center][anatomy], 0.025):#.2g}, {np.quantile(results[center][anatomy], 0.975):#.2g}]")


if __name__ == "__main__":
    main()
