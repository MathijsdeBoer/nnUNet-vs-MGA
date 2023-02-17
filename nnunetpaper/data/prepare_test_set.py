from pathlib import Path

from SimpleITK import ReadImage, WriteImage
from tqdm import tqdm

input_path: Path = Path("M:/Dataset/Loki")
output_path: Path = Path("S:/data/Loki/ref")
label_names: list[str] = ["tumor", "brain", "skin", "ventricles"]

patients = [x.resolve() for x in input_path.glob("*") if x.is_dir()]

patient: Path
for patient in (progress_bar := tqdm(patients)):
    progress_bar.set_description(f"{patient.name}")

    for name in label_names:
        progress_bar.set_description(f"{patient.name}: {name}")

        candidates: list[Path] = [
            x for x in patient.glob(f"label_{name}*") if x.is_file()
        ]
        if len(candidates) == 1:
            label = candidates[0]
        elif len(candidates) == 0:
            # No available segmentation for this class
            continue
        else:
            raise RuntimeError(
                f"Multiple candidates found for {patient.name}: {candidates=}"
            )

        if not (output_path / name).exists():
            (output_path / name).mkdir(parents=True)

        WriteImage(ReadImage(label), output_path / name / f"{patient.name}.nii.gz")

print("Done")
