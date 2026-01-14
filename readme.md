# Lunar-G2R
Official implementation of **Lunar-G2R: Geometry-to-Reflectance Learning for High-Fidelity Lunar BRDF Estimation**
 
![Lunar-G2R teaser](assets/teasing.png)

## Overview

High-fidelity rendering of lunar surfaces is essential for simulation, perception, and vision-based navigation (VBN), yet current pipelines often rely on simplified or spatially uniform reflectance models (e.g., Hapke).  
We introduce **Lunar-G2R**, a neural framework that estimates **spatially varying BRDF parameters directly from lunar digital elevation models (DEMs)**. The predicted per-pixel reflectance maps enable physically based renderings that more closely match real orbital observations than classical analytical models.  
Lunar-G2R is trained using real lunar imagery through a differentiable rendering formulation, allowing reflectance to be learned from geometry alone.

## Table of Contents

- [News](#news)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Checkpoints](#checkpoints)
- [Training Dataset](#training-dataset)
- [Re-training on New Lunar Models](#re-training-on-new-lunar-models)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## News

- January 2026: Code release.

## Repository Structure
...
Lunar-G2R/
├── BRDFGenerator/
│ ├── data/
│ │ └── best-model-parameters-val_loss.pt
│ ├── models/
│ ├── inference/
│ └── utils/
├── ImageSimulator/
│ ├── surrender_pipeline/
│ └── rendering_utils/
├── src/
│ ├── training/
│ ├── evaluation/
│ ├── dataset_construction/
│ └── metrics/
├── assets/
│ ├── teasing.png
│ └── dataset.png
└── README.md
...
## Architecture

### BRDFGenerator

`BRDFGenerator` contains the network used to predict spatial BRDF parameter maps from DEM patches.

- Input: DEM patches (128 x 128 pixels)
- Output: per-pixel BRDF parameters for a low-order polynomial BRDF model (3 parameters)
- Training region: Tycho crater
- Ground sampling distance: 5 m/px

![Inference example — BRDFGenerator](assets/inference.png)

After training, the model can be applied to any DEM patch to infer reflectance parameters, enabling BRDF-aware image synthesis without photometric inputs.

### ImageSimulator

`ImageSimulator` enables physically based rendering from:
- DEM geometry
- predicted BRDF parameter maps
- illumination geometry (Sun direction)
- viewing geometry (camera pose)

Rendering relies on **SurRender** (Airbus Defence and Space), a physically based ray-tracing engine widely used in space applications.

Note: SurRender is available for academic purpose. Users without access can implement their own ray-tracing pipeline or use the predicted BRDF maps independently within other simulation frameworks.

### src

`src` contains:
- training, validation, and testing scripts
- dataset construction utilities
- evaluation metrics used in the paper

The full-resolution Tycho DEM used in the study is not publicly distributable (obtained via Airbus / Pixel Factory).  
However, the training dataset released with this repository is public and sufficient to reproduce and extend the approach.

## Checkpoints

Pre-trained checkpoint for the 3-parameter polynomial BRDF model:

BRDFGenerator/data/best-model-parameters-val_loss.pt

This checkpoint corresponds to the model reported in the paper and can be used directly for inference.

## Training Dataset

We release the Lunar-G2R training dataset, which can also serve other computer vision and planetary perception studies.

![Training dataset preview](assets/dataset.png)

Dataset characteristics:
- 83,614 DEM-image pairs
- Patch size: 128 x 128 pixels
- Ground sampling distance: 5 m/px
- Region: Tycho crater

Split:
- 66,662 training samples
- 8,615 validation samples
- 8,337 test samples

Each sample includes:
- a DEM patch
- an orthorectified LRO image
- acquisition metadata (Sun direction, viewing geometry, camera parameters, georeferencing)

## Re-training on New Lunar Models

Although the exact Tycho DEM used in the paper cannot be redistributed, the full training pipeline is reproducible.

Using:
- a new lunar DEM
- real or simulated imagery
- the provided dataset construction scripts
- a differentiable renderer (e.g., SurRender)

users can retrain Lunar-G2R on new lunar regions or adapt the method to other planetary bodies.

## Citation

If you find our work useful, please consider citing:
<!-- 
```bibtex
@inproceedings{Grethen2026LunarG2R,
  title     = {Lunar-G2R: Geometry-to-Reflectance Learning for High-Fidelity Lunar BRDF Estimation},
  author    = {Grethen, Cl{\'e}mentine and Gasparini, Simone and Morin, G{\'e}raldine and Lebreton, J{\'e}r{\'e}my and Marti, Lucas and Sanchez-Gestido, Manuel},
  booktitle = {Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
} -->


## Acknowledgements 

This work was supported by the European Space Agency (ESA) under contract 4000140461/23/NL/GLC/my.