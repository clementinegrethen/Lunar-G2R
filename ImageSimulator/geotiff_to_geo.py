from pathlib import Path
from typing import Tuple
import sys

import rasterio
import numpy as np
from dataclasses import dataclass


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

    def to_geo(self, geo_filepath: Path):
        with geo_filepath.open('w') as f:
            f.write(fr'''projection = "equirectangular_tycho.txr"
params = {{
    MOON_AVG_RADIUS = {self.radius},
    lambda_0 = {self.lambda_0},
    phi_1 = {self.phi_ts},
    phi_0 = {self.phi_0},
    inv_scale = {{{1/self.scale[0]},{1/self.scale[1]}}},
    south = {'true' if self.phi_ts < 0 else 'false'},
    translation = {{{self.translation[0]}, {self.translation[1]}}}
}}''')

proj = Proj.from_tiff(Path(sys.argv[1]))
proj.to_geo(Path(sys.argv[1]).with_suffix('.geo'))
