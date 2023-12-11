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
@click.option("-c", "--class", "check_class", required=False, type=int, default=1)
def main(preds: Path, refs: Path, output: Path = None, check_class: int = 1):
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

        pred_image: np.ndarray = np.where(pred_image == check_class, 1, 0)
        ref_image: np.ndarray = np.where(ref_image == check_class, 1, 0)

        current_metrics = {
            "dice": None,
            "iou": None,
            "hd95": None,
            "assd": None,
            "volume": (prod(pred_size) * prod(pred_spacing))
            / 1_000_000,  # Image volume in liters
            "segment_volume": np.count_nonzero(ref_image) * prod(pred_spacing),
        }

        pred_image: Tensor = tensor(pred_image[np.newaxis, np.newaxis, ...])
        ref_image: Tensor = tensor(ref_image[np.newaxis, np.newaxis, ...])

        progress_bar.set_description(f"Processing {pred.name}: DSC")
        current_metrics["dice"] = dsc(pred_image, ref_image).item()

        progress_bar.set_description(f"Processing {pred.name}: IOU")
        current_metrics["iou"] = iou(pred_image, ref_image).item()

        progress_bar.set_description(f"Processing {pred.name}: HD95")
        current_metrics["hd95"] = hd95(pred_image, ref_image, spacing=pred_spacing).item()

        progress_bar.set_description(f"Processing {pred.name}: ASSD")
        current_metrics["assd"] = assd(pred_image, ref_image, spacing=pred_spacing).item()

        metrics[pred.name] = current_metrics

    with open(output, mode="w") as file:
        j = json.dumps(metrics, indent=4)
        file.write(j)

    print("Skipped:")
    for s in skipped:
        print(f"\t- {s}")


if __name__ == "__main__":
    main()
