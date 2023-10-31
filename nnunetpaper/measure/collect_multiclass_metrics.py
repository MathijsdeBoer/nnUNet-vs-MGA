import json
from math import prod
from pathlib import Path

import click
import numpy as np
import SimpleITK as sitk
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    MeanIoU,
    SurfaceDistanceMetric,
)
from torch import Tensor, tensor
from tqdm import tqdm


@click.command()
@click.option(
    "-p",
    "--preds",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-r",
    "--refs",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    required=False,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=True, writable=True, path_type=Path
    ),
)
@click.option("-c", "--n-classes", required=False, type=int, default=None)
def main(preds: Path, refs: Path, output: Path | None = None, n_classes: int | None = None):
    skipped = []
    metrics = {}

    if output is None:
        output = preds / "scores.json"
    elif output.is_dir():
        output /= "scores.json"

    use_background: bool = False
    dsc = DiceMetric(include_background=use_background)
    iou = MeanIoU(include_background=use_background)
    hd95 = HausdorffDistanceMetric(include_background=use_background, percentile=95)
    assd = SurfaceDistanceMetric(include_background=use_background, symmetric=True)

    pred: Path
    for pred in (
        progress_bar := tqdm(
            [x.resolve() for x in preds.glob("*") if x.is_file()],
            desc="Processing patients",
            position=0,
        )
    ):
        progress_bar.set_description(f"Processing {pred.name}")

        try:
            pred_image: sitk.Image = sitk.ReadImage(pred)
            pred_size = pred_image.GetSize()
            pred_spacing = pred_image.GetSpacing()
            pred_image: np.ndarray = sitk.GetArrayFromImage(pred_image)
        except RuntimeError:
            skipped.append(f"Read error: {pred}")
            continue

        try:
            ref_image: np.ndarray = sitk.GetArrayFromImage(
                sitk.ReadImage(refs / pred.name)
            )
        except RuntimeError:
            skipped.append(f"Refs error: {pred}")
            continue

        if n_classes is None:
            current_n_classes = np.max(pred_image)
        else:
            current_n_classes = n_classes

        for check_class in (
            sub_bar := tqdm(
                range(1, current_n_classes + 1),
                desc="Processing classes",
                leave=False,
                position=1,
            )
        ):
            sub_bar.set_description(f"Class {check_class}/{current_n_classes}")
            class_image: np.ndarray = np.where(pred_image == check_class, 1, 0)
            class_ref: np.ndarray = np.where(ref_image == check_class, 1, 0)

            current_metrics = {
                "dice": None,
                "iou": None,
                "hd95": None,
                "assd": None,
                "class": check_class,
                "volume": (prod(pred_size) * prod(pred_spacing)),
                "segment_volume": np.count_nonzero(class_ref) * prod(pred_spacing),
            }

            # Edge cases: empty prediction and/or empty reference
            if np.count_nonzero(class_image) == 0:
                if np.count_nonzero(class_ref) == 0:
                    current_metrics["dice"] = 1.0
                    current_metrics["iou"] = 1.0
                    current_metrics["hd95"] = 0.0
                    current_metrics["assd"] = 0.0
                else:
                    current_metrics["dice"] = 0.0
                    current_metrics["iou"] = 0.0
                    current_metrics["hd95"] = np.inf
                    current_metrics["assd"] = np.inf
            else:
                if np.count_nonzero(class_ref) == 0:
                    current_metrics["dice"] = 0.0
                    current_metrics["iou"] = 0.0
                    current_metrics["hd95"] = np.inf
                    current_metrics["assd"] = np.inf
                else:
                    class_image: Tensor = tensor(class_image[np.newaxis, np.newaxis, ...])
                    class_ref: Tensor = tensor(class_ref[np.newaxis, np.newaxis, ...])

                    sub_bar.set_description(f"Class {check_class}/{current_n_classes}: DSC")
                    current_metrics["dice"] = dsc(class_image, class_ref).item()

                    sub_bar.set_description(f"Class {check_class}/{current_n_classes}: IOU")
                    current_metrics["iou"] = iou(class_image, class_ref).item()

                    sub_bar.set_description(f"Class {check_class}/{current_n_classes}: HD95")
                    current_metrics["hd95"] = hd95(class_image, class_ref).item()

                    sub_bar.set_description(f"Class {check_class}/{current_n_classes}: ASSD")
                    current_metrics["assd"] = assd(class_image, class_ref).item()

            metrics[pred.name] = metrics.get(pred.name, []) + [current_metrics]

    with open(output, mode="w") as file:
        j = json.dumps(metrics, indent=4)
        file.write(j)

    print("Skipped:")
    for s in skipped:
        print(f"\t- {s}")


if __name__ == "__main__":
    main()
