# Robust moon renderer User Manual

## Content
```
USER_MANUAL.md
data/materials/polynomial_3p_cst.brdf
data/textures/geo/equirectangular_tycho.txr
data/tycho_conf.json
data/tycho_traj.csv
descentimagegenerator/
geotiff_to_geo.py
simulator.py
```

The Robust moon renderer is an ensemble of Python scripts that commands the Python client API of SurRender software, in order to generate images and data of realistic Moon simulations, that can use the BRDF models that have been generated thanks to SW2.

The entry point is the Python script `simulator.py`, that takes as an input a configuration file (json), that describes the simulation parameters.
This Python script rests on the helper package `descentimagegenerator`, named the *Descent Image Generator* (*DIG*), which is a Python package which purpose is to ease the use of SurRender Python client API for descent and landing simulation use-cases.

The Python script `geotiff_to_geo.py` is a simple helper tool for SurRender, that reads the projection parameters in a GeoTIFF input image, and generates the corresponding `.geo` file that is needed for SurRender to project the image during rendering.

In the `data` directory, the `tycho_conf.json` is the *DIG* configuration file that can be used as input to `simulator.py`. The associated file `tycho_traj.csv` is a trajectory example, referenced in `tycho_conf.json`, of different camera positions in geodetic Moon frame.
The file `equirectangular_tycho.txr` is an up-to-date version of the equirectangular projection used in SurRender to project data, written in SuMoL language, needed for Tycho data that uses a reference latitude != 0.
Finally, `polynomial_3p_cst.brdf` is the BRDF model described in TN2 that was used for training the deep learning model, also written in SuMoL language for SurRender. It will be used at inference time with BRDF parameters generated thanks to SW2 to render the Moon surface. The model outputs the BRDF parameters `a`, `b` and `c` in a GeoTIFF image relative to this formula:
```
BRDF = LeakyReLU(a) * cos(theta_i) + LeakyReLU(b) * cos(theta_p) + LeakyReLU(c)

theta_i is the incident angle
theta_p is the phase angle
```
Note that the LeakyReLU is not needed at inference time, but the provided SuMoL BRDF file `polynomial_3p_cst.brdf` is the exact one that was used at training time.

## Installation
The simulator was tested with Python version 3.8, which is the minimum version required.
The simulator requires to have the *Descent Image Generator* (*DIG*) installed, which itself requires the SurRender Python client interface, for instance in a Python virtual environment.
*DIG* requires the following dependencies: `dataclasses-json, matplotlib, numpy, opencv-python, pandas, scipy, surrender`.
The other dependencies needed by Robust moon renderer are `rasterio, tqdm`.
Installation can be done with pip:
```
pip install dataclasses-json matplotlib numpy opencv-python pandas scipy rasterio tqdm
pip install surrender/src/interfaces/python/surrender
pip install descentimagegenerator/
```
Since Robust moon renderer entry point is a single script (`simulator.py`), it does not need to be installed and can be runned directly with Python CLI.

## Limitations
See limitations of SW2 for the use of BRDF parameters.
SurRender 10 is required to use BRDF model, do not use versions below 10 for renderings with `polynomial_3p_cst.brdf`.

## Usage
### Preparation steps
Before using `simulator.py`, a few steps must be done in order to prepare data and configure simulation.

1. Generate a `.geo` file for Tycho DEM that is needed by SurRender tools to get projection parameters.

```
python geotiff_to_geo.py tycho-rgl.tif
```
That outputs a file named `tycho-rgl.geo` at the same location than GeoTIFF.

2. Conversion of Tycho DEM from GeoTIFF to SurRender heightmap and conemap BIG files.

The BIG format is a SurRender-optimized format that manages several level of details. The Conemap is a pre-computed structure that accelerates the ray tracing and the rendering.
The Tycho DEM was first delivered as a version with 'no data' areas, however this version CANNOT BE used to generated the Heightmap and Conemap files. Instead, a second version delivered with the TN2 with the 'no data' areas filled with interpolation data MUST be used to generate the heightmap and conemap.
In order to convert the Tycho DEM to heightmap and conemap, use the following command (binary provided with SurRender):
```
build_conemap -i tycho-rgl.tif -o tycho.dem
```
The processing can takes several minutes.
It outputs the following files:
```
tycho.dem
tycho_heightmap.big
tycho_conemap.big
```

`tycho.dem` file may need some adaptations that are necessary for SurRender renderings. This manual process will not be needed in SurRender next versions, but because of ongoing debug and integration this still needs to be done manually with the provided SurRender version. The line `TYPE = "CUBEMAP";` must be added to the file, and the `*_AXIS_RADIUS` variable must be set to the Moon Radius, `1737400`.
The `tycho.dem` file should contain the following:
```
TYPE = "CUBEMAP";
LINES = 90000;
LINE_SAMPLES = 95000;
VALID_MINIMUM = -3610.60009765625;
VALID_MAXIMUM = 1898.300048828125;
MAP_SCALE = 1;
EASTERNMOST_LONGITUDE = 180;
WESTERNMOST_LONGITUDE = -180;
MINIMUM_LATITUDE = -90;
MAXIMUM_LATITUDE = 90;
A_AXIS_RADIUS = 1737400;
B_AXIS_RADIUS = 1737400;
C_AXIS_RADIUS = 1737400;
HEIGHTMAP = "tycho_heightmap.big";
CONEMAP = "tycho_conemap.big";
```

3. Use of SW2 BRDF generator to output the BRDF parameters out of a part of Tycho DEM.

The selected area, between 348.3° and 348.8° of longitude and -42.5° and -43° of latitude, is an area that was part of the validation dataset, thus not seen during training (see TN2).
```
python SW2/generator.py -m SW2/data/best-model-parameters-val_loss.pt \
                        -t SW2/data/training_metadata.json \
                        -i tycho-rgl.tif \
                        --crop 348.3 -42.5 348.8 -43
```
This outputs a GeoTIFF file `tycho-rgl_BRDF_params.tif` that contains the 3 BRDF coefficients at a resolution of 5 meters by pixel that corresponds to the Tycho area passed as an input.

4. Generate the `.geo` file for BRDF parameters.

```
python geotiff_to_geo.py tycho-rgl_BRDF_params.tif
```
That outputs a file named `tycho-rgl_BRDF_params.geo` at the same location than GeoTIFF.

6. Create the `DEM/` and `texture/` directory needed for SurRender resource path directory structure in `data/`. Move all the necessary data to the `data/` directory tree that will be used as a SurRender resource path. Copy Tycho DEM GeoTIFF `.geo` file that was generated before (in 1.) for use with generated Heightmap and Conemap BIG files.
Symbolic links can also be used instead of moving or copying the files in SurRender resource path directory.

```
mkdir SW3/data/textures/ SW3/data/DEM/
mv tycho.dem SW3/data/DEM/
mv tycho_heightmap.big SW3/data/textures/
mv tycho_conemap.big SW3/data/textures/
mv tycho-rgl_BRDF_params.* SW3/data/textures/
cp tycho-rgl.geo SW3/data/textures/tycho_heightmap.geo
cp tycho-rgl.geo SW3/data/textures/tycho_conemap.geo
```

7. Run SurRender server (default port is 5151).

```
<path/to/surrender/directory>/bin/surrender --headless [--port <port>]
```

### Simulation configuration
The simulation configuration provided as a base for Tycho simulation using generated BRDF parameters is located in `data/tycho_conf.json`.
It is already configured to use Tycho DEM, generated BRDF parameters for the polynomial 3-parameters BRDF model specified in TN2.

Tunable parameters:
- `trajectory_config.sun_trajectory_config.position` is the Sun position in Moon Centered Moon Fixed (MCMF) frame. It has been set to the Sun position of one of the LROC NAC that was taken at this location, but can be changed to see the effect on BRDF model.
- `simulation_config` can be tuned to change the number of rays per pixel to balance between simulation time and image quality. Some sensor parameters like FoV, image size and PSF can also be tuned.
- `server_config.hostname` and `port` should be changed if SurRender server does not run on the same host than client Python API calls.
- `server_config.resource_path` should point to `SW3/data` where all the DEM, BRDF model and parameters lie. Be careful, this path is read and is relative to **SurRender server** side, so either SurRender server must be launched from `SW3` directory, either an absolute path should be provided.
- `output_config.output_directory`, similarly to `resource_path` is relative to **SurRender server** and should point to a directory that will be automatically created, that will contains script outputs such as rendered images.

The `data/tycho_traj.csv` file can also be tuned to change camera position and attitude. Note that the trajectory can also be defined in MCMF with attitude as quaternion: see `descentimagegenerator` package.

For other examples of configurations usages, check `descentimagegenerator` package and examples.

### Run simulation
Run the Robust moon renderer with:

```
python simulator.py -c data/tycho_conf.json [-d]
```

The output images will be generated according to the parameter `output_config.output_directory` specified in `tycho_conf.json`.
