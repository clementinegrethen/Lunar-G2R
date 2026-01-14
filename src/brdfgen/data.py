from brdfgen.metrics import DEMStats

from typing import Union, List, Tuple, Dict
from pathlib import Path
import logging
import json
import re
from argparse import ArgumentParser
from shutil import copyfile

#from shapely.geometry import Point
from surrender_data_tools.find_tile_sun_positions import init_spice, find_sun_position, find_moon_tile_body_position
from surrender_data_tools.find_my_lro_nac import get_product_metadata, PdsProduct
import rasterio as rio
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import spiceypy as sp


ua = 1.4959787070000e11  # m
logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class Proj:
    projection: str
    params: Dict

    @classmethod
    def from_file(cls, geo_filepath: str):
        with Path(geo_filepath).open('r') as f:
            from lupa.lua51 import LuaRuntime
            lua = LuaRuntime()
            lua.execute(f.read())
            projection = lua.globals()['projection']
            params = dict(lua.globals()['params'])

            if Path(projection).name != 'equirectangular.txr':
                raise NotImplementedError()

            for k, e in params.items():
                if type(e) == float:
                    params[k] = float(e)
                elif type(e) == int:
                    params[k] = int(e)
                elif type(e) == bool:
                    params[k] = bool(e)
                else:
                    params[k] = list(dict(e).values())
            return Proj(projection=projection, params=params)

    def to_file(self, geo_filepath: str):
        print(self)
        with Path(geo_filepath).open('w') as f:
            kv = ''.join([f'    {k} = {v},\n' for k, v in self.params.items()]).replace('[', '{').replace(']', '}')
            f.write(f'projection = "{self.projection}"\nparams = {{\n{kv}}}')


@dataclass_json
@dataclass
class DatasetMetadata:
# TODO replace str by floats?
    gsd: str
    crop_size: Tuple[int, int]
    dem_file: str
    resource_path: Union[str, List[str]]
    coverage_factor: str
    texture_nb: int
    sun_power: str
    dem_std: str
    dem_min: str
    dem_max: str
    dem_mean: str


@dataclass_json
@dataclass
class TexturePositions:
    texture_file: str
    position: List[Tuple[int, int]]


@dataclass_json
@dataclass
class DatasetPositions:
    texture_positions: List[TexturePositions]


@dataclass_json
@dataclass
class SampleMetadata:
    texture_file: str
    fov: float
    size_wh: Tuple[int, int]
    proj: Proj
    center_xy: Tuple[int, int]
    cam_pos: Tuple[float, float, float]
    cam_att: Tuple[float, float, float, float]
    sun_pos: Tuple[float, float, float]
    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    pds_metadata: PdsProduct = None
    rot90: int = 0 # TODO delete - done in loss


@dataclass_json
@dataclass
class TrainingMetadata:
    brdf_name: str
    channels: int
    dataset_dir: str
    epochs: int
    batch_size: int
    dataset_metadata: DatasetMetadata


class BrdfGeneratorDataset(Dataset):

    def __init__(self,
                 dataset_dir: str,
                 channels: int,
                 subset_file: str = '',
                 augment: bool = False,
                 rng: np.random.Generator = np.random.default_rng()):

        with open(Path(dataset_dir) / 'metadata.json', 'r') as f:
            self.metadata = DatasetMetadata.from_json(f.read())
        self.channels = channels
        logger.debug(f'Dataset global metadata: {self.metadata}, {channels=}, {subset_file=}, {augment=}, {rng=}')

        if subset_file:
            with open(subset_file, 'r') as f:
                samples = f.readlines()
            self.dem_files = [str((Path(dataset_dir) / f'dem/dem_{sample.strip()}.tif').resolve()) for sample in samples]
            self.render_files = [str((Path(dataset_dir) / f'render/render_{sample.strip()}.tif').resolve()) for sample in samples]
            self.metadata_files = [str((Path(dataset_dir) / f'metadata/metadata_{sample.strip()}.json').resolve()) for sample in samples]
        else:
            self.dem_files = [str(x.resolve()) for x in sorted((Path(dataset_dir) / 'dem').glob('**/*'))]
            self.render_files = [str(x.resolve()) for x in sorted((Path(dataset_dir) / 'render').glob('**/*'))]
            self.metadata_files = [str(x.resolve()) for x in sorted((Path(dataset_dir) / 'metadata').glob('**/*'))]

        if len(self.dem_files) != len(self.render_files) or len(self.metadata_files) != len(self.dem_files):
            raise ValueError('Incoherent dataset')

        if augment:
            self.rng = rng

    def __len__(self):
        return len(self.dem_files)

    def __getitem__(self, idx):
        dem = cv2.imread(self.dem_files[idx], cv2.IMREAD_UNCHANGED)
        dem = torch.from_numpy(dem.reshape((1, *dem.shape)))
        mean = torch.mean(dem)
        std = float(self.metadata.dem_std)
        dem = (dem - mean)/std
        minv, maxv = torch.min(dem), torch.max(dem)
        logger.debug(f'Fetching data with DEM={Path(self.dem_files[idx]).name} normalized to min={minv} max={maxv}')

        render = cv2.imread(self.render_files[idx], cv2.IMREAD_UNCHANGED).astype(np.float64)
        render = torch.from_numpy(render).reshape((1, *render.shape))
        render = torch.cat((*[render] * self.channels, torch.zeros(4 - self.channels, *render.shape[-2:])))

        with open(self.metadata_files[idx], 'r') as f:
            metadata = SampleMetadata.from_json(f.read())

        # dataset augmentation: random 90° rotations
        if hasattr(self, "rng"):
            k = self.rng.integers(low=0, high=4, dtype=np.uint8)
            dem    = torch.rot90(dem, k=k, dims=[1, 2])
            render = torch.rot90(render, k=k, dims=[1, 2])
            metadata.rot90 = k

        return dem, render, metadata


# given a resource (file) name and a list of paths, find this file in these paths and return absolute path to file
def get_resource_absolute_path(
        resource: str,
        resource_path: Union[str, List[str]],
    ) -> str:

    if type(resource_path) is not list:
        resource_path = [resource_path]
    for path in resource_path:
        abspath = Path(path) / resource
        if abspath.exists() and abspath.is_file():
            return str(abspath.resolve())
    raise Exception(f'{resource} not found in {resource_path}')


def get_texture_scale(
        texture_file: str,
        resource_path: Union[str, List[str]],
    ) -> float:

    texture_abspath = get_resource_absolute_path(texture_file, resource_path)
    texture_geofile = str(Path(texture_abspath).with_suffix('.geo'))
    texture_proj    = Proj.from_file(texture_geofile)
    return 1/texture_proj.params['inv_scale'][0]


def get_pds_product(
        pds_product_name: str,
        cumindex_file: str = '/data/sharedata/VBN/moon/cumindex.sqlite3', # TODO configurable
    ) -> PdsProduct:
    logger.debug(f'Get PDS product {pds_product_name}...')
    product = get_product_metadata(db_fname=cumindex_file, name=f'{pds_product_name}%', im_type='nac')
    assert(len(product) == 1)
    logger.debug(product)
    return PdsProduct(**product[0])


#def overwrite_sample_metadata(metadata_file: str, regex, dataset, heightmap_scale):
#    with open(metadata_file, 'r') as f:
#        metadata = SampleMetadata.from_json(f.read())
#
#    # get texture name, x and y position from file name
#    match = regex.match(Path(metadata_file).stem)
#    texture_file = f'textures/{match.group(1)}.tif'
#    x = int(match.group(2))
#    y = int(match.group(3))
#
#    # get texture PDS product metadata for sun direction and polygon
#    # TODO pds product name derived from file to be more generic (avoir looking for underscore)
##    product = get_pds_product(Path(texture_file).stem[:Path(texture_file).stem.index('_')-1])#
##    sun_pos, _, _ = find_sun_position(product.start_time, is_mars=False)
##    logger.debug(f'Found sun position {sun_pos}')
#
#    texture_abspath = get_resource_absolute_path(texture_file, resource_path)
#
#    # get texture projection, sun direction and size
#    proj = Proj.from_file(str(Path(texture_abspath).with_suffix('.geo')))
#    with rio.open(texture_abspath) as ds:
#        size_wh = ds.shape[::-1]
#
#    texture_scale = get_texture_scale(texture_file, dataset.metadata.resource_path)
#    texture_scale_factor = float(dataset.metadata.gsd) / texture_scale
#
#    halfside = round(float(dataset.metadata.crop_size[0]) * texture_scale_factor/2)
#    from brdfgen.render import xy_to_lonlat
#    top_left_lon, top_left_lat, _, _ = xy_to_lonlat(x - halfside, y - halfside, proj)
#    top_right_lon, top_right_lat, _, _ = xy_to_lonlat(x + halfside, y - halfside, proj)
#    bottom_left_lon, bottom_left_lat, _, _ = xy_to_lonlat(x - halfside, y + halfside, proj)
#    bottom_right_lon, bottom_right_lat, _, _ = xy_to_lonlat(x + halfside, y + halfside, proj)
#
#    metadata = SampleMetadata(texture_file=texture_file,
#                              fov=fov,
#                              size_wh=size_wh,
#                              proj=proj,
#                              center_xy=(int(x), int(y)),
#                              cam_pos=metadata.cam_pos, # property not updated
#                              cam_att=metadata.cam_att, # property not updated
#                              sun_pos=metadata.sun_pos,
#                              top_left=(top_left_lon, top_left_lat),
#                              top_right=(top_right_lon, top_right_lat),
#                              bottom_left=(bottom_left_lon, bottom_left_lat),
#                              bottom_right=(bottom_right_lon, bottom_right_lat))
#    logger.debug(metadata)
#
#    with open(metadata_file, 'w') as f:
#        json.dump(metadata.to_dict(), f, indent=2)
#
#
#def update_sample_metadata(
#        dataset: BrdfGeneratorDataset,
#    ):
#
##    init_spice('/mnt/20To/sharedata/VBN/SpiceKernels') # TODO configurable
#    regex = re.compile(pattern=f'^metadata_(.*)_([0-9]*)_([0-9]*)')
#
#    # TODO plutôt lire le .dem et la variable HEIGHTMAP
#    heightmap_file = f'textures/{Path(dataset.metadata.dem_file).stem}_heightmap.big'
#    heightmap_scale = get_texture_scale(heightmap_file, dataset.metadata.resource_path)
#
#    from functools import partial
#    overwrite_sample_metadata_it = partial(overwrite_sample_metadata, regex=regex, dataset=dataset, heightmap_scale=heightmap_scale)
#
#    from tqdm.contrib.concurrent import process_map
#    process_map(overwrite_sample_metadata_it, dataset.metadata_files, chunksize=2)


# Generate ground truth images (SurRender rendering with orthorectified texture) giving a list of valid positions (x, y) in image
def generate_dataset_from_positions(
        dem_file: str,
        heightmap_file: str,
        texture_file: str,
        resource_path: Union[str, List[str]],
        positions: List[Tuple[int, int]],
        image_size: int,
        output_dir: str,
        texture_scale_factor: float,
        demstats: DEMStats,
        rays: int,
        sun_power: float,
        serverhost: str = 'localhost',
        serverport: int = 5151,
        debug_show: int = 0,
        dem_only: bool = False,
    ) -> DEMStats:

    if not Path(output_dir).exists() or not Path(output_dir).is_dir():
        raise ValueError(f'{output_dir} is not a directory or does not exists')

    texture_abspath = get_resource_absolute_path(texture_file, resource_path)

    # get texture PDS product metadata for sun direction and polygon
    regex = re.compile(pattern=f'^.*(M[0-9]+[A-Z]?).*$')
    name = regex.match(Path(texture_file).stem).group(1)
    if not (name.endswith('L') or name.endswith('R')):
        name += 'L'
    product = get_pds_product(name)
    sun_pos, _, _, = find_sun_position(product.start_time, is_mars=False)
    lro_start_pos, _, _ = find_moon_tile_body_position('LRO', product.start_time)
    lro_stop_pos, _, _ = find_moon_tile_body_position('LRO', product.stop_time)
    logger.debug(f'Found sun position {sun_pos}')
    logger.debug(f'Found LRO start position {lro_start_pos}')
    logger.debug(f'Found LRO stop position {lro_stop_pos}')

    lroc_fov = 2.8502
    if name.endswith('R'):
        lroc_fov = 2.8412
    fov = texture_scale_factor * lroc_fov * image_size/5064
    logger.debug(f'{fov=}')

    # get texture projection, sun direction and size
    proj = Proj.from_file(str(Path(texture_abspath).with_suffix('.geo')))
    with rio.open(texture_abspath) as ds:
        size_wh = ds.shape[::-1]

    # TODO deal with cubemap
    if heightmap_file:
        heightmap_abspath = get_resource_absolute_path(heightmap_file, resource_path)
        with rio.open(heightmap_abspath) as dem_ds:
            dem = dem_ds.read().squeeze()

    # create output dirs for inputs and ground truths
    (Path(output_dir) / 'dem').mkdir(exist_ok=True)
    (Path(output_dir) / 'render').mkdir(exist_ok=True)
    (Path(output_dir) / 'metadata').mkdir(exist_ok=True)

    if dem_only:
        texture_file = 'default.png'

    from brdfgen.render import init_surrender, xy_to_lonlat, set_camera_pose_from_lro
    with init_surrender(
                dem_file=dem_file,
                texture_file=texture_file,
                resource_path=resource_path,
                image_size=image_size,
                channels=1,
                fov=fov,
                rays=rays,
                sun_power=sun_power,
                serverhost=serverhost,
                serverport=serverport,
        ) as s:

        s.setObjectPosition("sun", sun_pos)

        for i, (x, y) in enumerate(tqdm(positions)):
            # unique identifier for each data
            ID = f'{Path(texture_abspath).stem}_{x:05}_{y:05}'

            # crop and resample DEM to appropriate location and scale
            if heightmap_file:
                #TODO should be improved like in get valid position function: not texture_scale_factor because taken at the centre of texture
                halfside = round(image_size*texture_scale_factor/2)
                dem_crop = cv2.resize(dem[y-halfside:y+halfside, x-halfside:x+halfside], (image_size, image_size), interpolation=cv2.INTER_AREA)

                # extract DEM associated with texture (model input)
                cv2.imwrite(f'{output_dir}/dem/dem_{ID}.tif', dem_crop)

                # udpate statistics
                demstats.update(dem_crop)
                logger.debug(f'DEM statistics update {demstats.get_min()=} {demstats.get_max()=} {demstats.get_mean()=} {demstats.get_std()=}')

            # render with SurRender (ground truth)
            v = y/size_wh[1]
            pos, att, lon, lat, alt = set_camera_pose_from_lro(
                s=s,
                lro_start_pos=lro_start_pos,
                lro_stop_pos=lro_stop_pos,
                v=v,
                proj=proj,
                x_center=x,
                y_center=y,
                heightmap_abspath=heightmap_abspath,
            )
            s.render()
#            with open('record.py', 'w') as f:
#                print(s.generateReplay(), file=f)
#                exit(1)
            target = s.getImage()
            cv2.imwrite(f'{output_dir}/render/render_{ID}.tif', target[:,:,0])

#            halfside = round(image_size*texture_scale_factor/2)
            top_left_lon, top_left_lat, _, _ = xy_to_lonlat(x - halfside, y - halfside, proj)
            top_right_lon, top_right_lat, _, _ = xy_to_lonlat(x + halfside, y - halfside, proj)
            bottom_left_lon, bottom_left_lat, _, _ = xy_to_lonlat(x - halfside, y + halfside, proj)
            bottom_right_lon, bottom_right_lat, _, _ = xy_to_lonlat(x + halfside, y + halfside, proj)

            # dump metadata used for rendering
            metadata = SampleMetadata(texture_file=texture_file,
                                      fov=fov,
                                      size_wh=size_wh,
                                      proj=proj,
                                      center_xy=(int(x), int(y)),
                                      cam_pos=pos,
                                      cam_att=att,
                                      sun_pos=sun_pos,
                                      top_left=(top_left_lon, top_left_lat),
                                      top_right=(top_right_lon, top_right_lat),
                                      bottom_left=(bottom_left_lon, bottom_left_lat),
                                      bottom_right=(bottom_right_lon, bottom_right_lat),
                                      pds_metadata=product,
                                      )
            logger.debug(metadata)
            nans = np.sum(np.isnan(target[...,0]))
            if nans > 0:
                logger.warning(f'{nans} NaNs found. {metadata}')

            with open(f'{output_dir}/metadata/metadata_{ID}.json', 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            if debug_show > 0 and i % (len(positions) // debug_show - 1) == 0:
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'{texture_file}\n{lon=:.3f}°, {lat=:.3f}°, {alt=:.3f}m\n{x=}px, {y=}px')
                im = ax[1].imshow(target[:,:,0], cmap='gray')
                ax[1].set_title("rendered (SurRender)")
                fig.colorbar(im)
                im = ax[0].imshow(dem_crop, cmap='gray')
                ax[0].set_title("input (DEM)")
                fig.colorbar(im)
                plt.show()

        s.close()

    return demstats


# Generate a random list of tuple (x,y) of valid positions in texture & DEM (valid means there is no NaN pixel)
def generate_valid_positions(
        texture_file: str,
        resource_path: Union[str, List[str]],
        image_size: int,
        gsd: float,
        rng: np.random.Generator,
        coverage_factor: float = 1.0,
        heightmap_file: str = '',
        heightmap_mask_file: str = '',
        debug_show: int = 0,
    ) -> List[Tuple[int, int]]:

# "TILE" mode for evaluate / for inference
#    with Image.open(texture_abspath) as im:
#        w, h = im.size
#    return [i for i in itertools.product(range(-(w//-image_size)), range(-(h//-image_size)))]
    logger.info(f'Generating valid positions for {texture_file}...')

    # read texture
    texture_abspath = get_resource_absolute_path(texture_file, resource_path)
    texture_ds = rio.open(texture_abspath)

    # read DEM
    if heightmap_file:
        heightmap_abspath = get_resource_absolute_path(heightmap_file, resource_path)
        heightmap_ds = rio.open(heightmap_abspath)
        assert(texture_ds.shape == heightmap_ds.shape)

    # read DEM mask file if exists
    heightmap_mask = np.ones(texture_ds.shape, dtype=np.uint8)
    if heightmap_mask_file:
        heightmap_mask_abspath = get_resource_absolute_path(heightmap_mask_file, resource_path)
        with rio.open(heightmap_mask_abspath) as heightmap_mask_ds:
            heightmap_mask = heightmap_mask_ds.read().squeeze()
        logger.info(f'Read {heightmap_mask_file}')

    assert(texture_ds.shape == heightmap_mask.shape)

    # read texture mask & heightmap
    if heightmap_file:
        heightmap = heightmap_ds.read().squeeze()
        logger.info(f'Read {heightmap_file}')
    texture = texture_ds.read_masks().squeeze()
    logger.info(f'Read {texture_file}')

    # mark border pixels in texture mask
    texture[:,0] = 0
    texture[0,:] = 0
    texture[:,-1] = 0
    texture[-1,:] = 0
#    halfside = round(image_size*texture_scale_factor/2)
#    texture[:,0:halfside] = 0
#    texture[0:halfside,:] = 0
#    texture[:,-halfside:-1] = 0
#    texture[-halfside:-1,:] = 0

    # create a mask that will generate only valid crops of size 'image_size' in any position taken
    mask = (texture) & (heightmap_mask)
    if heightmap_file:
        mask = mask & (~np.isnan(heightmap))

    # compute margin to erode mask with the approximate minimum scale in m/px on texture
    texture_scale = get_texture_scale(texture_file, resource_path)
    alt_at_center = heightmap[int(heightmap.shape[0]/2.), int(heightmap.shape[1]/2.)]
    alt_max = np.nanmax(heightmap)
    tan_half_fov = np.tan(np.deg2rad(2.85)/2.)
    lro_dist = (5064 * texture_scale) / (2 * tan_half_fov)
    lro_dist_min = lro_dist - (alt_max - alt_at_center)
    scale_min = tan_half_fov * lro_dist_min * 2 / 5064
    logger.debug(f'{texture_scale=} {alt_at_center=} {alt_max=} {tan_half_fov=} {lro_dist=} {lro_dist_min=} {scale_min=} {gsd=} {image_size=}')

    mask = cv2.erode(mask.astype(np.uint8), np.ones((int(np.ceil(image_size * gsd / scale_min)),) * 2))
    logger.info(f'Mask created')
    if debug_show > 0:
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(texture_file)
        ax[0].imshow(texture)
        ax[0].set_title('Texture valid data')
        if heightmap_file:
            ax[1].imshow(heightmap_mask & (~np.isnan(heightmap)))
        else:
            ax[1].imshow(heightmap_mask)
        ax[1].set_title('DEM valid data')
        ax[2].imshow(mask)
        ax[2].set_title('Combination of all masks with erode')
        plt.show()

    if heightmap_file:
        heightmap_ds.close()
    texture_ds.close()

    all_y, all_x = mask.nonzero() # cv2 reads in H, W order
    # no value to pick
    if all_x.shape[0] == 0:
        logger.warning(f'Image {texture_file} does not have any valid position')
        return list()

    # compute numbers of crops to generate:
    # coverage_factor * texture valid surface in m² / crop surface in m²
    texture_scale_factor = gsd / texture_scale
    crops_nb = int(coverage_factor * all_x.shape[0] / (image_size * image_size * texture_scale_factor * texture_scale_factor))

    indexes = rng.choice(all_x.shape[0], size=crops_nb)
    logger.info(f'{crops_nb} random positions found among {all_x.shape[0]}')

    return list(np.transpose(np.concatenate((np.array([all_x[indexes]]), np.array([all_y[indexes]])), axis=0)))


def generate_heightmap_relative_to_texture(
        heightmap_file: str,
        texture_file: str,
        resource_path: Union[str, List[str]],
        output_dir: str,
        overwrite: bool = True,
    ) -> str:

    texture_abspath = get_resource_absolute_path(texture_file, resource_path)
    heightmap_abspath = get_resource_absolute_path(heightmap_file, resource_path)
    output_dem_file = f"textures/{Path(texture_abspath).stem}_{Path(heightmap_abspath).stem}.tif"

    try:
        get_resource_absolute_path(output_dem_file, resource_path)
        output_found = True
    except:
        output_found = False

    (Path(output_dir) / 'textures').mkdir(exist_ok=True, parents=True)

    if not output_found or overwrite:
        logger.info(f'Generating DEM crop from DEM {heightmap_abspath} with texture {texture_abspath} to {output_dem_file}')
        # TODO format moins gourmand ? (uint8 normalisé ?)
        # TODO en fait on pourrait carrément tout faire avec rasterio sur le gros DEM?
        # WARNING needs surrender VBN library (uses ouroborous)
        import os
        os.environ['VBN_LIBS'] = "libVBN_Kernel.so:libSurRender_tools.so"
        import VBN as vbn
        vbn.crop_input_to_reference(heightmap_abspath, texture_abspath, str(Path(output_dir) / output_dem_file))

    return output_dem_file


def generate_geo_file(
        texture_file: str,
        resource_path: Union[str, List[str]],
    ):

    texture_abspath = get_resource_absolute_path(texture_file, resource_path)

    with rio.open(texture_abspath) as ds:
        params = ds.crs.data
        params['x_translation'] = ds.transform.xoff
        params['y_translation'] = ds.transform.yoff
        params['x_scale'] = ds.transform.a
        params['y_scale'] = ds.transform.e
        # TODO avoir un equirectangular avec un autre nom dans le package python pour pointer dessus
        projection_txr_map = {
            "stere" : "stereo_slow.txr",
            "eqc" : "equirectangular.txr"
        }

        logger.info(f'Generating geo file for {texture_abspath}')
        with open(Path(texture_abspath).with_suffix('.geo'), 'w') as geofile:
            geofile.write(fr"""projection = "{projection_txr_map[params['proj']]}"
params = {{
    MOON_AVG_RADIUS = {params['R']},
    lambda_0 = {np.deg2rad(params['lon_0'])}, -- {params['lon_0']}
    phi_1 = {np.deg2rad(params['lat_ts'])}, -- {params['lat_ts']}
    phi_0 = {np.deg2rad(params['lat_0'])}, -- {params['lat_0']}
    inv_scale = {{1/{params['x_scale']},1/{params['y_scale']}}},
    south = {'true' if params['lat_ts'] < 0 else 'false'},
    translation = {{{params['x_translation']}, {params['y_translation']}}}
}}""")


def convert_texture(
        texture_file: str,
        resource_path: Union[str, List[str]],
        working_dir: str = '/tmp',
        overwrite: bool = False,
    ) -> str:

    working_dirp = Path(working_dir)
    (working_dirp / 'textures').mkdir(exist_ok=True)

    texture_file_float = texture_file
    texture_abspath = get_resource_absolute_path(texture_file, resource_path)
    with rio.open(texture_abspath) as dsi:
        if 'int16' in dsi.dtypes:
            # destination float file
            texture_file_float = str(Path(f'textures/{Path(texture_file).stem}_Y8').with_suffix('.tif'))
            try:
                get_resource_absolute_path(texture_file_float, resource_path)
                texture_found = True
            except:
                texture_found = False

            # convert file if it does not exist yet or if overwrite=True
            if overwrite or not texture_found:
                logger.info(f'Convert int16 image {texture_file} to float image {texture_file_float}')
                mimg = np.ma.masked_array(dsi.read(), mask=~dsi.read_masks(1))
                img_float = np.where(mimg.mask, 0, mimg.data / 32767.0).astype(np.float32)
                texture_file = texture_file_float
                with rio.open(working_dirp / texture_file,
                              mode='w',
                              driver='GTiff',
                              height=dsi.shape[0],
                              width=dsi.shape[1],
                              count=1,
                              dtype=np.float32,
                              crs=dsi.crs,
                              transform=dsi.transform,
                              nodata=0) as dso:
                    dso.write(img_float)

    return texture_file_float



def generate_dataset(
        output_dir: str,
        dem_file: str,
        texture_files: List[str],
        resource_path: Union[str, List[str]],
        image_size: int,
        coverage_factor: float,
        rays: int = 1,
        sun_power: float = ua*ua*np.pi,
        dem_mask_file: str = '',
        overwrite: bool = False,
        gsd: float = 1.,
        hot_working_dir: str = '/tmp',
        cold_working_dir: str = '/tmp',
        rng: np.random.Generator = np.random.default_rng(),
        serverhost: str = 'localhost',
        serverport: int = 5151,
        debug_show: int = 0,
        dem_only: bool = False,
        ds_positions: DatasetPositions = DatasetPositions(texture_positions=[]),
    ):

    # TODO also generate a metadata.json with info (resource_path, image_size, etc)

    # create directories (error if already exists to avoid overwriting a dataset)
    Path(output_dir).mkdir(exist_ok=overwrite)

    demstats = DEMStats()

    compute_positions = False
    if ds_positions.texture_positions:
        assert(len(texture_files) == len(ds_positions.texture_positions))
    else:
        compute_positions = True

    for tex_id, texture_file in enumerate(texture_files):

        # convert (if needed) texture to normalized uint8
        texture_file = convert_texture(texture_file, resource_path, hot_working_dir, overwrite)

        # generate geo file for texture
        generate_geo_file(texture_file, resource_path)

        # multiply wanted scale factor by the ratio between heightmap scale and texture scale
        # TODO deal with cubemap
        heightmap_file = f'textures/{Path(dem_file).stem}_heightmap.big'
        heightmap_scale = get_texture_scale(heightmap_file, resource_path)
        texture_scale = get_texture_scale(texture_file, resource_path)
        texture_scale_factor = gsd / texture_scale
        logger.info(f'{gsd=} {heightmap_scale=} {texture_scale=} {texture_scale_factor=}')

        # generate heightmap file relative to texture in working directory
        # TODO if texture scale != heightmap scale (often true), a resampling is done here in surrender image tool to have the same scale than texture -> do we want that ?
        if heightmap_file:
            heightmap_file = generate_heightmap_relative_to_texture(heightmap_file, texture_file, resource_path, cold_working_dir, overwrite)
            texture_abspath = get_resource_absolute_path(texture_file, resource_path)
            heightmap_abspath = get_resource_absolute_path(heightmap_file, resource_path)
            copyfile(Path(texture_abspath).with_suffix('.geo'), Path(heightmap_abspath).with_suffix('.geo'))

        # generate mask file relative to texture in working directory
        mask_file = ''
        if dem_mask_file:
            # TODO same remark than before except it is less important for mask
            mask_file = generate_heightmap_relative_to_texture(dem_mask_file, texture_file, resource_path, cold_working_dir, overwrite)

        # generate N valid positions for this texture
        if compute_positions:
            positions = generate_valid_positions(
                    heightmap_file=heightmap_file,
                    texture_file=texture_file,
                    resource_path=resource_path,
                    image_size=image_size,
                    gsd=gsd,
                    coverage_factor=coverage_factor,
                    rng=rng,
                    heightmap_mask_file = mask_file,
                    debug_show=debug_show)

            # append texture valid chosen positions and write it to file
            ds_positions.texture_positions.append(TexturePositions(texture_file=texture_file, position=[(int(x), int(y)) for x, y in positions]))
            with open(f'{output_dir}/positions.json', 'w') as f:
                json.dump(ds_positions.to_dict(), f)

        # generate dataset files for these positions
        generate_dataset_from_positions(
                dem_file=dem_file,
                heightmap_file=heightmap_file,
                texture_file=texture_file,
                resource_path=resource_path,
                positions=ds_positions.texture_positions[tex_id].position,
                image_size=image_size,
                output_dir=output_dir,
                texture_scale_factor=texture_scale_factor,
                demstats=demstats,
                rays=rays,
                sun_power=sun_power,
                serverhost=serverhost,
                serverport=serverport,
                debug_show=debug_show,
                dem_only=dem_only)

        # update dataset global metadata (do it every image)
        ds_metadata = DatasetMetadata(
            gsd=str(gsd),
            crop_size=(image_size, image_size),
            dem_file=dem_file,
            resource_path=resource_path,
            coverage_factor=str(coverage_factor),
            texture_nb=tex_id+1,
            sun_power=str(sun_power),
            dem_std=str(demstats.get_std()),
            dem_min=str(demstats.get_min()),
            dem_max=str(demstats.get_max()),
            dem_mean=str(demstats.get_mean()),
        )
        logger.debug(ds_metadata)
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(ds_metadata.to_dict(), f, indent=2)


def get_parser():
    parser = ArgumentParser(description='Gen dataset')
    parser.add_argument('-g', '--gsd', type=float, help="GSD of dataset", default=5.0)
    parser.add_argument('-c', '--coverage', type=float, help="Coverage factor", default=1.0)
    parser.add_argument('--overwrite', help="Overwrite eval directory", action='store_true')
    parser.add_argument('--debug', help="Activate debug logs", action='store_true')
    return parser


def main(args):

#    dem_file = 'DEM/NAC_DTM_CHANGE3.dem'
#    dem_mask_file = ''
    dem_file = 'DEM/tycho_v2.dem'
    dem_mask_file = "textures/tycho-msk.tif"
    textures_source_dir = '/data/sharedata/VBN/pxfactory'
    hot_working_dir = '/mnt/20To/sharedata/VBN/pxfactory_float'
    cold_working_dir = '/mnt/20To/sharedata/VBN/pxfactory_workingdir'
    resource_path = [
        textures_source_dir,
        hot_working_dir,
        cold_working_dir,
#        "/mnt/20To/sharedata/VBN/", # for change2 DEM
#        "/imagechain/data/space_cv/2021_EL3/DEM_LOLA_tiles_5m/change3/", # for change3 dem
#        '/data/sharedata/VBN/moon/', # for change3 ortho
    ]

    # log settings
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger('PIL').setLevel(level=logging.WARNING)
    logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
    logging.getLogger('rasterio').setLevel(level=logging.WARNING)

    # get textures list
    texture_files = [f'textures/{f.name}' for f in Path(textures_source_dir).glob('**/M1*.tif') if f.is_file()]

    # init RNGs
    seed = 123456789123456789
    rng = np.random.default_rng(seed)

    rng.shuffle(texture_files)
#    texture_files = ["textures/NAC_DTM_CHANGE3_M1144922100_5M.tiff"]
#    texture_files = ['textures/M106935800RC_Y8.tif']
#    texture_files = texture_files[:2]

    # create working_dirs
    Path(hot_working_dir).mkdir(exist_ok=True)
    Path(cold_working_dir).mkdir(exist_ok=True)

    # dataset configuration
    image_size = 128
    coverage_factor = args.coverage
    output_dir = f'/data/nmenga/prj/lunarsw/ds_v5_crop{image_size}_gsd{args.gsd}_f{coverage_factor}'

    # read tycho positions
#    import re
#    regex = re.compile(pattern='^metadata_(.*)_([0-9]*)_([0-9]*)$')
#    for m in Path('demo/tycho_2rays/metadata/').glob('**/*'):
#        print(regex.match(m.stem))
#    exit(1)

    logger.info('init spice')
    # TODO configurable
    init_spice('/mnt/20To/sharedata/VBN/SpiceKernels',
               '/mnt/20To/sharedata/VBN/projects/Argonaut/ISIS/data/lro/kernels/spk',
               '/mnt/20To/sharedata/VBN/lro_spicek_recent')

    only_update_metadata = False
    if only_update_metadata:
        update_sample_metadata(BrdfGeneratorDataset(output_dir, channels=1))
    else:
        generate_dataset(
            output_dir=output_dir,
            dem_file=dem_file,
            texture_files=texture_files,
            resource_path=resource_path,
            image_size=image_size,
            coverage_factor=coverage_factor,
            dem_mask_file=dem_mask_file,
            gsd=args.gsd,
            rng=rng,
            hot_working_dir=hot_working_dir,
            cold_working_dir=cold_working_dir,
            serverport = 5221,
            debug_show = 0,
            dem_only = False,
            overwrite=args.overwrite,
            )


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)