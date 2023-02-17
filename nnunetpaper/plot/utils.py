import json
from pathlib import Path

import pandas as pd


def read_json(files: list[Path]) -> pd.DataFrame:
    transformed_dict = {
        "metric": [],
        "metric_name": [],
        "center": [],
        "anatomy": [],
        "segment_volume": [],
        "pt_id": [],
    }
    for file in files:
        with open(file, mode="r") as f:
            d: dict = json.load(f)

        for k in d.keys():
            for metric in d[k].keys():
                # Manual skip
                if metric == "volume" or metric == "segment_volume":
                    continue

                transformed_dict["metric"].append(d[k][metric])
                transformed_dict["metric_name"].append(metric)
                transformed_dict["center"].append("All")
                transformed_dict["anatomy"].append(file.parent.stem)
                transformed_dict["segment_volume"].append(d[k]["segment_volume"])
                transformed_dict["pt_id"].append(k)

                transformed_dict["metric"].append(d[k][metric])
                transformed_dict["metric_name"].append(metric)
                transformed_dict["center"].append(k.split(" ")[0])
                transformed_dict["anatomy"].append(file.parent.stem)
                transformed_dict["segment_volume"].append(d[k]["segment_volume"])
                transformed_dict["pt_id"].append(k)

    return pd.DataFrame(transformed_dict)
