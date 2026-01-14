# BRDF Generator User Manual

## Content
```
USER_MANUAL.md
data/best-model-parameters-val_loss.pt
data/training_metadata.json
generator.py
```

The generator is a Python script `generator.py` that takes a Digital Elevation Map (DEM) file as a GeoTIFF image, and outputs the corresponding BRDF parameters.
In `data` directory, there is a PyTorch `state_dict` file `best-model-parameters-val_loss.pt`, containing the weights of the trained model that outputs BRDF parameters. There is also a metadata file `training_metadata.json` associated with the `state_dict` file, that contains some parameters of the dataset that was used during training and validation, and some parameters of the training.
The model makes predictions at a resolution of 5 meters by pixel, and outputs the BRDF parameters `a`, `b` and `c` in a GeoTIFF image relative to this formula:
```
BRDF = a * cos(theta_i) + b * cos(theta_p) + c

theta_i is the incident angle
theta_p is the phase angle
```

## Installation
The generator was tested with Python version 3.8, which is the minimum version required.
The generator requires the following dependencies: `dataclasses-json, numpy, opencv-python, rasterio, torch, tqdm`.
Installation can be done with pip:
```
pip install dataclasses-json numpy opencv-python rasterio torch tqdm
```
Since BRDF generator is a single script (`generator.py`), it does not need to be installed and can be runned directly with Python CLI.

## Limitations
The model weights have been trained and validated on orthorectified LROC NAC images covering the Tycho crater, and the dataset characteristics make the model work exclusively on this area.

The current version of `generator.py` convert the input DEM to the resolution of 5 meters by pixels, splits it and makes predictions for every crop. It does not deal completely with edge effects because it was decided that this was not part of the added value of the study, and it can be done with several techniques depending on the use case. A basic mitigation of edge effects is done in `generator.py` by adding a margin to avoid taking the borders of each prediction, and the input normalization is done by taking the mean of a blurred version of the input, in order to smooth out the BRDF parameters.

If the input DEM data has 'no data' values, these pixels will be masked and as a result, the crops containing `no data` will output NaN BRDF parameters because of the convolutional nature of the model. It is advised to generate BRDF parameters for input elevation data that has not 'no data' values, that was filled with interpolation values for example.

## Usage
Once installed, this is BRDF generator usage:

```
python generator.py -m <path to state_dict trained model weights> \
                    -t <path to metadata json file associated with state_dict> \
                    -i <path to input DEM GeoTIFF file> \
                    [-c <left longitude> <top latitude> <right longitude> <bottom lattitude>] \
                    [-b <batch size>] \
                    [-o <pixel size overlap between tiles>] \
                    [--gen-visu] \
                    [--verbose]
```

For example, a nominal usage of BRDF generator would be:
```
python generator.py -m ./data/best-model-parameters-val_loss.pt \
                    -t ./data/training_metadata.json \
                    -i elevation_map_geo.tif
```
This outputs a file named `elevation_map_geo_BRDF_params.tif`, which is a float32 GeoTIFF file with 4 channels that contains the BRDF parameters, at a resolution of 5m/px.
The current provided model uses 3 parameters, so the 4th channel is filled with zeros.
In order to output individual files for each BRDF parameter to help visualization, the option `--gen-visu` can be added.

One can generate BRDF parameters for a smaller portion of the input DEM with the `--crop` option, which takes the left and right longitudes and the top and bottom lattitude in degree as input. As a result, only the area between this boundaries will be used for generating BRDF parameters.
```
python generator.py -m ./data/best-model-parameters-val_loss.pt \
                    -t ./data/training_metadata.json \
                    -i elevation_map_geo.tif \
                    --crop 348 -42.5 349 -43.5
```

The `--batch` option controls the batch size that is passed to the model at inference, it can be adjusted to the amount of GPU memory available.
The `--overlap` option controls the number of pixels on the border of the predicted crops that are not used, only the center area is used to generate the BRDF parameters.
