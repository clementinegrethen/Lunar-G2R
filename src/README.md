# BRDF GENERATOR

## Dependencies
surrender : feat-vbnkernel-tools branch

## Sources description

```
.
├── .gitmodules
├── README.md
├── pyproject.toml
├── src
│   └── brdfgen
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── dig.py
│       ├── eval.py
│       ├── metrics.py
│       ├── model.py
│       ├── mydata
│       │   ├── __init__.py
│       │   ├── materials
│       │   │   ├── hapke_diff.brdf
│       │   │   ├── lambertian_diff.brdf
│       │   │   ├── polynomial_2p.brdf
│       │   │   ├── polynomial_3p_cos2g.brdf
│       │   │   ├── polynomial_3p_cos2i.brdf
│       │   │   ├── polynomial_3p_cst.brdf
│       │   │   ├── polynomial_4p_cos2g.brdf
│       │   │   ├── polynomial_4p_cos2g_cos2i.brdf
│       │   │   ├── polynomial_4p_cos2i.brdf
│       │   │   └── polynomial_4p_halfvec.brdf
│       │   └── textures
│       │       ├── geo
│       │       │   └── equirectangular_globals.txr
│       │       └── procedural
│       │           ├── tile.geo
│       │           └── tile.txr
│       ├── render.py
│       ├── train.py
│       └── utils.py



src/brdfgen/data.py
    -> Generate dataset + everything related to data formats
src/brdfgen/model.py
    -> Deep learning models and layers definitions
src/brdfgen/render.py
    -> Everything related to SurRender
src/brdfgen/train.py
    -> Train model, dataset split
src/brdfgen/eval.py
    -> Evaluate model
src/brdfgen/generator.py
    -> 'BRDF Generator' itself that uses trained model
src/brdfgen/metrics.py
    -> Metrics and losses definitions
src/brdfgen/config.py
    -> Global configuration and variables
src/brdfgen/utils.py
    -> Useful tools that aren't too specific

src/brdfgen/mydata/materials/*
    -> SuMoL BRDF models
src/brdfgen/mydata/textures/procedural/tile.txr
    -> SuMoL procedural texture that allows reading only a portion (tile) of a texture
src/brdfgen/mydata/textures/procedural/tile.geo
    -> Geo file that tells tile.txr to use a projection model
src/brdfgen/mydata/textures/geo/equirectangular_globals.txr
    -> Equirectangular projection model updated for Tycho use + globals params for on-the-fly changes during simulations from client
```
