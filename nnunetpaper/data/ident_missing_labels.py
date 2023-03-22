from pathlib import Path

import click


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path))
def main(path: Path):
    train_images = path / "imagesTr"
    train_labels = path / "labelsTr"
    test_images = path / "imagesTs"
    test_labels = path / "labelsTs"

    train_missing = []
    if train_images.exists():
        # nnUNet images are formatted as such:
        # [name]_[channelno].nii.gz
        # we want the [name], as appending .nii.gz to that will get us the label name
        # so split on _
        files = [x.name.split("_")[0] for x in train_images.glob("*") if x.is_file()]

        for f in files:
            if not (train_labels / f"{f}.nii.gz").exists():
                train_missing.append(f)

    test_missing = []
    if test_images.exists():
        files = [x.name.split("_")[0] for x in test_images.glob("*") if x.is_file()]

        for f in files:
            if not (test_labels / f"{f}.nii.gz").exists():
                test_missing.append(f)

    if len(train_missing) > 0:
        print(f"Training\nFound {len(train_missing)} missing labels")
        for m in train_missing:
            print(f" - {m}")

    if len(test_missing) > 0:
        print(f"Testing\nFound {len(train_missing)} missing labels")
        for m in test_missing:
            print(f" - {m}")


if __name__ == '__main__':
    main()
