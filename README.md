# NnU-Net versus mesh-growing algorithm as a tool for the robust and timely segmentation of neurosurgical 3D images in contrast enhanced T1 MRI scans

This repository contains the custom code written to support our article.
Mainly, this includes:

- The code used to convert datasets to the format expected by nnU-Net
- The code to measure the performance of the segmentation algorithms
- The code to perform the statistical analyses
- The code to generate the plots

While the implementation of this code is specific to our use case, it can be used as a starting point for other projects.
The code to train the nnU-Nets can be found in the [nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

After we completed our analyses, nnU-Net V2 was released.
We used V1.7.0, which is what our code is designed to work with.
Unfortunately, V2 is not backwards compatible with V1.
We have not tested our code with V2, but it should be possible to adapt it to work with V2.
Finally, no statistically significant differences were found between the performance of the two versions of nnU-Net.

## Requirements

We make use of Python Poetry to manage the dependencies of this project.
Install Poetry by following the guide [on their website](https://python-poetry.org/docs/#installation).
To install the dependencies, clone the repository, use your preferred terminal application, navigate to the root of this repository and run:

```bash
poetry install
```

This will install all the dependencies in a virtual environment, which will allow you to run the code without polluting your system with dependencies.
Activating the virtual environment is done by running:

```bash
poetry shell
```

## Usage

As mentioned, this code is not designed to be generally applicable.
However, much of the processes should translate to any other project.
Each module will have scripts meant for a specific aspect of our project.

- `data` contains scripts that handle file management, conversion and generation
- `measure` contains scripts that measure the performance of the segmentation algorithms
- `plot` contains scripts that generate the plots used in the article
- `process` contains scripts that process the data in our files to generate the data used in the article
- `stats` contains scripts that perform the statistical analyses

## Files

### Scores

`scores.csv` contains the scores of the segmentation algorithms as used in the article.
