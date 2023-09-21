from pathlib import Path

import pandas as pd

from nnunetpaper.data import read_json


def get_multi_method_dataframe(methods: list[tuple[str, Path]], auto_collect_anatomies: bool = False) -> pd.DataFrame:
    if auto_collect_anatomies:
        tmp = []
        for name, path in methods:
            print(f"Collecting anatomies for {name}")
            subdirs = [x for x in path.iterdir() if x.is_dir()]
            for subdir in subdirs:
                p = subdir / "scores.json"
                if p.exists():
                    print(f"\t{subdir.name}")
                    tmp.append((name, subdir / "scores.json"))
                else:
                    print(f"\t{subdir.name} (skipped)")

        methods = tmp

    method_dict: dict[str, list[Path]] = {}
    for method_name, scores_path in methods:
        method_dict[method_name] = method_dict.get(method_name, [])
        method_dict[method_name].append(scores_path)

    method_data: dict[str, pd.DataFrame] = {}
    for method_name in method_dict.keys():
        method_data[method_name] = read_json(method_dict[method_name])
        method_data[method_name]["methods"] = method_name
    return pd.concat(method_data.values(), ignore_index=True)
