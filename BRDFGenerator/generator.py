import logging
from pathlib import Path
from argparse import ArgumentParser
import itertools
from typing import Tuple
from copy import deepcopy

import cv2
from torch import nn
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import rasterio
import rasterio.io
import rasterio.windows
import rasterio.transform
from rasterio.enums import Resampling
import numpy as np
from tqdm import tqdm
import torch


logger = logging.getLogger(__name__)

GDAL_CONFIG = dict(
    GDAL_NUM_THREADS="ALL_CPUS",
    GDAL_TIFF_INTERNAL_MASK=True,
    GDAL_TIFF_OVR_BLOCKSIZE="128",
)


def get_parser():
    parser = ArgumentParser(description='BRDF parameter generator, takes as input pytorch weights of deep learning model (associated to a BRDF model) and outputs a single TIF file with BRDF parameters in different channels')
    parser.add_argument('-m', '--model', type=str, help="Model trained weights file path", required=True)
    parser.add_argument('-t', '--training-metadata', type=str, help="Training metadata config json file", required=True)
    parser.add_argument('-i', '--input', type=str, help="Input DEM file path", required=True)
    parser.add_argument('-b', '--batch', type=int, help="Batch size for inference", default=32)
    parser.add_argument('-o', '--overlap', type=float, help="Overlap of DEM crops borders in pixels to limit edge effects", default=32)
    parser.add_argument('-c', '--crop', type=float, help="Crop to left longitude, top latitude, right longitude, bottom latitude", nargs=4)
    parser.add_argument('-g', '--gen-visu', action="store_true", help="Also outputs BRDF parameters in individual TIF files for visualization")
    parser.add_argument('-v', '--verbose', action="store_true", help="Activate debug logs")
    return parser


@dataclass_json
@dataclass
class Proj:
    radius: float
    lambda_0: float
    phi_ts: float
    phi_0: float
    scale: Tuple[float, float]
    translation: Tuple[float, float]

    @staticmethod
    def from_tiff(input_file: Path):
        if input_file.suffix.lower() != '.tif':
            raise Exception(f'{str(input_file)} is not a tiff file')

        with rasterio.open(input_file) as dataset:
            return Proj(radius=dataset.crs.data['R'],
                        lambda_0=np.deg2rad(dataset.crs.data['lon_0']),
                        phi_ts=np.deg2rad(dataset.crs.data['lat_ts']),
                        phi_0=np.deg2rad(dataset.crs.data['lat_0']),
                        scale=(dataset.transform.a, dataset.transform.e),
                        translation=(dataset.transform.xoff, dataset.transform.yoff))



@dataclass_json
@dataclass
class DatasetMetadata:
    gsd: float
    crop_size: Tuple[int, int]
    dem_std: float


@dataclass_json
@dataclass
class TrainingMetadata:
    channels: int
    dataset_metadata: DatasetMetadata


class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down1 = UNetDownLayer(in_c, 16, 0.1)
        self.down2 = UNetDownLayer(16, 32, 0.1)
        self.down3 = UNetDownLayer(32, 64, 0.2)
        self.down4 = UNetDownLayer(64, 128, 0.2)
        self.midle = UNetConvLayer(128, 256, 0.3)
        self.up1   = UNetUpLayer(256, 128, 0.2)
        self.up2   = UNetUpLayer(128, 64, 0.2)
        self.up3   = UNetUpLayer(64, 32, 0.1)
        self.up4   = UNetUpLayer(32, 16, 0.1)
        self.final = nn.Conv2d(16, out_c, kernel_size=1)

    def forward(self, x):
        t1, x = self.down1(x)
        t2, x = self.down2(x)
        t3, x = self.down3(x)
        t4, x = self.down4(x)
        x = self.midle(x)
        x = self.up1(x, t4)
        x = self.up2(x, t3)
        x = self.up3(x, t2)
        x = self.up4(x, t1)
        return self.final(x)

    def load_state_dict(self, state_dict, *args, **kwargs):
        unet_state_dict = dict()
        for k, v in state_dict.items():
            unet_state_dict[k[5:]] = v
        super().load_state_dict(unet_state_dict, *args, **kwargs)
        return


class UNetConvLayer(nn.Module):
    def __init__(self, in_c, out_c, drop_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(out_c)
        self.drop  = nn.Dropout(p=drop_rate)
        self.bnorm2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = nn.functional.leaky_relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        return nn.functional.leaky_relu(x)


class UNetDownLayer(nn.Module):
    def __init__(self, in_c, out_c, drop_rate):
        super().__init__()
        self.conv = UNetConvLayer(in_c, out_c, drop_rate)

    def forward(self, x):
        x = self.conv(x)
        return x, nn.functional.max_pool2d(x, kernel_size=2)


class UNetUpLayer(nn.Module):
    def __init__(self, in_c, out_c, drop_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, dilation=1, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(out_c)
        self.conv2 = UNetConvLayer(out_c * 2, out_c, drop_rate)

    def forward(self, x, skip_tensor):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = nn.functional.leaky_relu(x)
        x = torch.cat([x, skip_tensor], dim=1)
        return self.conv2(x)


def lonlat_to_xy(lon_deg: float, lat_deg: float, proj: Proj):
    R = proj.radius
    lambda0 = (proj.lambda_0 + 2*np.pi) % (2*np.pi)
    lon = (np.deg2rad(lon_deg) + 2*np.pi) % (2*np.pi)
    lat = np.deg2rad(lat_deg)
    scale = proj.scale
    translation = proj.translation
    x_m = R * np.cos(proj.phi_ts) * (lon - lambda0) - translation[0]
    y_m = R * (lat - proj.phi_0) - translation[1]
    return x_m/scale[0], y_m/scale[1]


def main(args):
    if not Path(args.model).exists() or not Path(args.model).is_file():
        logging.error(f'{args.model} is not a valid model file weights')
        exit(1)
    if not Path(args.training_metadata).exists() or Path(args.training_metadata).suffix != '.json':
        logging.error(f'{args.training_metadata} is not a valid training metadata json file')
        exit(1)
    if not Path(args.input).exists() or not Path(args.input).is_file():
        logging.error(f'{args.input} is not a valid input DEM file')
        exit(1)

    # log settings
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger('rasterio').setLevel(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger('rasterio').setLevel(level=logging.WARNING)

    # check GPU availability
    if not torch.cuda.is_available():
        logger.error('GPU not available')
        exit(1)
    device = torch.device("cuda:0")

    # Read dataset metadata which was used for training and extract relevant information
    with open(Path(args.training_metadata), 'r') as f:
        metadata = TrainingMetadata.from_json(f.read())

    # overlap in pixels on the edges of each inference
    overlap = args.overlap
    crop_size = metadata.dataset_metadata.crop_size
    batch_size = args.batch

    # get input DEM scale to compute scale of output
    proj = Proj.from_tiff(Path(args.input))
    logger.info(f'{proj=}')
    f = proj.scale[0] / metadata.dataset_metadata.gsd
    logger.info(f'Input DEM rescale factor {f}')
    out_proj = deepcopy(proj)
    out_proj.scale = [s/f for s in proj.scale]

    # load model UNet part and load weights
    model = UNet(in_c=1, out_c=metadata.channels)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.model, weights_only=True), strict=False)

    # open and read input DEM
    with rasterio.open(args.input) as dem_ds:
        c, height, width = dem_ds.count, dem_ds.height, dem_ds.width

        window = None
        boundless = False
        if args.crop:
            overlap_before_rescale = int(overlap/f)
            col_off, row_off = lonlat_to_xy(args.crop[0], args.crop[1], proj)
            col_end, row_end = lonlat_to_xy(args.crop[2], args.crop[3], proj)
            col_off = max(0, min(width, col_off))
            col_end = max(0, min(width, col_end))
            row_off = max(0, min(height, row_off))
            row_end = max(0, min(height, row_end))
            logger.info(f'Crop between pixels {col_off=} {col_end=} and {row_off=} {row_end=}')
            if col_off == col_end or row_off == row_end:
                logger.error("Wanted crop area is outside of provided DEM")
                dem_ds.close()
                exit(1)
            if 0 in [col_off, col_end, row_off, row_end]:
                logger.warning("Wanted crop area is on the edge of provided DEM and may have been reduced")
            out_proj.translation = (proj.translation[0] + col_off * proj.scale[0],
                                    proj.translation[1] + row_off * proj.scale[1])
            col_off -= overlap_before_rescale
            row_off -= overlap_before_rescale
            col_end += overlap_before_rescale
            row_end += overlap_before_rescale
            height = row_end - row_off
            width = col_end - col_off
            boundless = True if any([col_off < 0, row_off < 0, col_end > width, row_end > height]) else False
            window = rasterio.windows.Window(row_off=row_off, height=height, col_off=col_off, width=width)

        # open input DEM image (resample at the same time) and read size and raster profile
        dem = dem_ds.read(masked=True,
                          window=window,
                          out_shape=(c, int(height * f), int(width * f)),
                          boundless=boundless,
                          resampling=Resampling.bilinear)
        dem[dem.mask] = np.nan
        dem_blur = cv2.GaussianBlur(dem.squeeze(), (15,15), 0)
        c, height, width = dem.shape

        # create output BRDF parameters files for visualisation
        transform = rasterio.transform.Affine(out_proj.scale[0], 0, out_proj.translation[0],
                                              0, out_proj.scale[1], out_proj.translation[1])
        profile = dict(height=height - overlap*2,
                       width=width - overlap*2,
                       count=4,
                       dtype=np.float32,
                       driver="GTiff",
                       interleave="pixel",
                       transform=transform,
                       crs=dem_ds.crs)
        brdf_params_file = rasterio.open(f'{Path(args.input).stem}_BRDF_params.tif', 'w', **profile)
        if args.gen_visu:
            out_file = []
            profile.update(count=1)
            for ci in range(metadata.channels):
                out_file.append(rasterio.open(f'{Path(args.input).stem}_BRDF_param_{ci:02}.tif', 'w', **profile))
        logger.debug(profile)

        # DEM size is below crop size
        if height < crop_size[0]:
            crop_size[0] = height
            logger.warning(f"crop size lowered to {height}")
        if width < crop_size[1]:
            crop_size[1] = width
            logger.warning(f"crop size lowered to {width}")

        # create model inputs tensor
        inputs = torch.zeros((batch_size, 1, *crop_size), dtype=torch.float32)
        inputs = inputs.to(device)
        window_coord = np.zeros((batch_size, 4), dtype=np.uint32)
        brdf_params = np.zeros((batch_size, 4, *crop_size), dtype=np.float32)

        # for each crop in input DEM image, infer BRDF parameters
        b = 0
        it_rows = np.arange(0, height, crop_size[0] - overlap*2)
        it_cols = np.arange(0, width,  crop_size[1] - overlap*2)
        for i, (y, x) in tqdm(enumerate(itertools.product(it_rows, it_cols)), total = len(it_rows)*len(it_cols)):

            # deal with last row/col elements
            if y + crop_size[0] > height:
                y = height - crop_size[0]
            if x + crop_size[1] > width:
                x = width - crop_size[1]

            logger.debug(f'{x=} {y=} {crop_size=} {dem.shape=}')

            # adapt DEM input for inference
            dem_crop = dem[:, y:y+crop_size[0], x:x+crop_size[1]]
            mean = np.mean(dem_blur[y:y+crop_size[0], x:x+crop_size[1]])
            dem_crop = (dem_crop - mean)/metadata.dataset_metadata.dem_std

            # fill inputs tensor with DEM crop
            inputs[b] = torch.from_numpy(dem_crop)
            # save windows for each sample of batch
            window_coord[b] = np.array([y, crop_size[1] - overlap*2, x, crop_size[0] - overlap*2])
            b += 1

            if b == batch_size or i + 1 == len(it_rows) * len(it_cols):
                brdf_params[:,:metadata.channels,...] = model(inputs).cpu().detach().numpy()
                for b, coord in enumerate(window_coord):
                    if coord[1] > 0 and coord[3] > 0:
                        window = rasterio.windows.Window(row_off=coord[0], height=coord[1], col_off=coord[2], width=coord[3])
                        logger.debug(f'{b=} {window=}')
                        brdf_params_file.write(brdf_params[b, :, overlap:-overlap, overlap:-overlap],
                                               window=window)
                        if args.gen_visu:
                            for ci, f in enumerate(out_file):
                                f.write(brdf_params[b, ci:ci+1, overlap:-overlap, overlap:-overlap], window=window)
                b = 0
                inputs = torch.zeros((batch_size, 1, *crop_size), dtype=torch.float32)
                inputs = inputs.to(device)
                window_coord = np.zeros((batch_size, 4), dtype=np.uint32)
                brdf_params = np.zeros((batch_size, 4, *crop_size), dtype=np.float32)

        logger.info(f'Outputted {str(Path(brdf_params_file.files[0]))}')
        brdf_params_file.close()
        if args.gen_visu:
            for f in out_file:
                logger.info(f'Outputted {str(Path(f.files[0]))} (for visualization)')
                f.close()

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
