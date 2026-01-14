from brdfgen.data import Proj, SampleMetadata
from brdfgen.utils import removeprefix, debug_plot

from typing import Union, List, Tuple, Dict
import shutil
from pathlib import Path
import logging

from surrender.geometry import look_at
from dataclasses_json import dataclass_json
from dataclasses import dataclass
import matplotlib.pyplot as plt
from surrender.surrender_client import surrender_client
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


sun_radius = 6.96342000e8; # m
ua = 1.4959787070000e11  # m
logger = logging.getLogger(__name__)


# Initialise SurRender client. Init everything but camera pose
def init_surrender(
        dem_file: str,
        resource_path: Union[str, List[str]],
        image_size: int,
        texture_file: str = '',
        channels: int = 4,
        fov: float = 3.0,
        sun_power: float = ua*ua*np.pi,
        rays: Union[int, None] = None,
        texture_diff: Union[str, List[str]] = "textures/texture_diff.tif",
        serverhost: str = 'localhost',
        serverport: int = 5151,
        brdf_file: str = "hapke_test.brdf",
        for_backprop: bool = False,
    ) -> surrender_client:

    # check that when not backpropagating, there is a texture
    assert(for_backprop or texture_file)

    s = surrender_client()
    s.connectToServer(serverhost, 5151)
    s.closeViewer()
#    s.record()
    if type(resource_path) is not list:
        resource_path = [resource_path]
    # add automatically mydata directory from repository to resource_path
    import brdfgen
    resource_path.append(str(Path(brdfgen.__file__).parent / "mydata"))
    for path in resource_path:
        s.cd(path)
        s.pushResourcePath()
    s.enableRaytracing(True)
    s.enablePathTracing(False)
    s.enableIrradianceMode(False)
    s.setConventions(s.SCALAR_XYZ_CONVENTION, s.Z_FRONTWARD)
    s.setImageSize(image_size, image_size)
    s.setCameraFOVDeg(fov, fov)
    s.createBRDF('sun', 'sun.brdf', {})
    s.createShape('sun', 'sphere.shp', {'radius': sun_radius})
    s.createBody('sun', 'sun', 'sun', [])
    s.setSunPower([*[sun_power] * channels , *[0] * (4 - channels)])

    if for_backprop:
        if rays is None:
            rays = 2
        s.setNbSamplesPerPixel(rays)
        if type(texture_diff) is list:
            raise NotImplementedError()

#        logger.debug(f'Creating textures/proc_texture with file proc_tile.txr with parameter {texture_diff}')
        # TODO automatically generate tile.geo with wanted projection with globals (equirectangular, stereo..)
#        s.loadTextureObject(texture_diff, 'procedural/tile.txr', { 'fulltex': removeprefix(texture_diff, "textures/")})
        s.createUserDataTexture(texture_diff, image_size, image_size, False)
        s.loadTextureObject("textures/tiled_texture", 'procedural/tile.txr', { 'fulltex': removeprefix(texture_diff, "textures/")})
#        s.createBRDF("sphere_brdf", "raw.brdf", {})
#        logger.debug(f'Creating spherical DEM {dem_file} with texture proc_texture')
#        s.createSphericalDEM('sphere', removeprefix(dem_file, "DEM/"), 'sphere_brdf', removeprefix(texture_diff, "textures/"))
#        s.createSphericalDEM('sphere', removeprefix(dem_file, "DEM/"), 'sphere_brdf', 'tiled_texture')
        s.createBRDF("sphere_brdf", brdf_file, {'tex': 'tiled_texture'})
        s.createSphericalDEM('sphere', removeprefix(dem_file, "DEM/"), 'sphere_brdf', 'default.png')
        s.enableTextureGradient(texture_diff, False)
#        texture_fp = s.mapTextureAsNumpyArray(texture_diff)
#        s.createBRDF("sphere_brdf", "raw.brdf", {})
#        s.createSphericalDEM('sphere', removeprefix(dem_file, "DEM/"), 'sphere_brdf', 'tiled_texture')
    else:
        if rays is None:
            rays = 1
        if texture_file == 'default.png':
            brdf = "hapke.brdf"
            params = {"albedo": [3.5,] * 4 }
        # for generating ground truth
        else:
            brdf = "raw.brdf"
            params = {}
        s.setNbSamplesPerPixel(rays)
        s.createBRDF("sphere_brdf", brdf, params)
        s.createSphericalDEM('sphere', removeprefix(dem_file, "DEM/"), 'sphere_brdf', removeprefix(texture_file, "textures/"))

    return s


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-16:
        return v
    else:
        return v / n


def yawpitchroll_to_cartesian_BCBF_attitude(
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        pos: Tuple[float, float, float]
    ) -> Tuple[float, float, float, float]:
    """
    Compute local cartesian attitude quaternion from yaw, pitch, roll and position

    Args:
        yaw_deg (float): yaw angle in degrees
        pitch_deg (float): pitch angle in degrees
        roll_deg (float): roll angle in degrees
        pos (Tuple[float, float, float]): X,Y,Z position

    Returns:
        (w, x, y, z) attitude quaternion (SurRender format)
    """
    # build NED frame unit vectors
    up = normalize(np.array(pos))
    east = np.cross([0, 0, 1], up)
    if np.linalg.norm(east) < 1e-16:
        east = np.array([0, 1, 0])
    else:
        east = normalize(east)
    north = np.cross(up, east)
    # BCBF to NED rotation matrix
    Rned = np.hstack((north[:, np.newaxis], east[:, np.newaxis], -up[:, np.newaxis]))

    # yaw pitch roll rotation matrix
    Rypr = Rotation.from_euler(
        "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
    ).as_matrix()

    att = Rotation.from_matrix(Rned @ Rypr).as_quat()
    att = (att[3], *att[0:3],)  # (x, y, z, w) scipy format to (w, x, y, z) SurRender format
    return att


def geodetic_to_cartesian_BCBF_position(
        lon_deg: float,
        lat_deg: float,
        alt_m: float,
        a_m: float,
        b_m: float
    ) -> Tuple[float, float, float]:
    """
    Args:
        lon_deg (float): longitude angle in degrees
        lat_deg (float): latitude angle in degrees
        alt_m (float): altitude in metres, 0 => body center
        a_m (float): ellipsoid a (m)
        b_m (float): ellipsoid b (m)

    Returns:
        (x, y, z) BCBF position
    """
    # allows using [-180;180] or [0;360] longitudes
    lambda_ = np.deg2rad(((lon_deg + 180) % 360) - 180)
    phi = np.deg2rad(lat_deg)
    # https://en.wikipedia.org/wiki/Geodetic_coordinates
    N = a_m ** 2 / (
        np.sqrt(a_m ** 2 * np.cos(phi) ** 2.0 + b_m ** 2.0 * np.sin(phi) ** 2.0)
    )
    X = (N + alt_m) * np.cos(phi) * np.cos(lambda_)
    Y = (N + alt_m) * np.cos(phi) * np.sin(lambda_)
    Z = ((b_m ** 2.0 / a_m ** 2.0 * N) + alt_m) * np.sin(phi)
    return X, Y, Z


def xy_to_lonlat(
        x: float,
        y: float,
        proj: Proj,
    )-> Tuple[float, float, float, np.ndarray]:

    R = proj.params['MOON_AVG_RADIUS']
    phi0 = proj.params['phi_0']
    phits = proj.params['phi_1']
    lambda0 = (proj.params['lambda_0'] + 2*np.pi) % (2*np.pi)
    scale = 1/np.array(proj.params['inv_scale'])
    translation = np.array(proj.params['translation'])
    lon = (scale[0] * x + translation[0]) / (R * np.cos(phits)) + lambda0
    lat = (scale[1] * y + translation[1]) / R + phi0
    return np.rad2deg(lon), np.rad2deg(lat), R, scale


def set_camera_pose_from_lro(
        s: surrender_client,
        lro_start_pos,
        lro_stop_pos,
        v,
        proj: Proj,
        x_center: float,
        y_center: float,
        heightmap_abspath: Path,
    ):

    # inverse projection to longitude, latitude & other params
    lon, lat, R, scale = xy_to_lonlat(x_center, y_center, proj)
    pos = geodetic_to_cartesian_BCBF_position(lon, lat, 0, R, R)

    # get value of DEM at center of crop (altitude)
    from surrender_data_tools.sumol import get_texture_value_using_surrender
    pos_alts = get_texture_value_using_surrender(
        surrender_client_handle=s,
        texture_path=heightmap_abspath,
        xyz=[pos,],
    )
    alt = pos_alts[0][1][0]
    # TODO here : get alt_at_center to recompute FOV (set fov here) adapted for this rendering altitude (recompute average local scale)
    logger.debug(f'real altitude at crop center {alt}m')

    # get up vector
    lonup, latup, _, _ = xy_to_lonlat(x_center, 0, proj)
    up = normalize(np.array(geodetic_to_cartesian_BCBF_position(lonup, latup, alt, R, R)) - np.array(pos))

    center_pos = geodetic_to_cartesian_BCBF_position(lon, lat, alt, R, R)
    logger.debug(f'center position of crop: {center_pos}')
    lro_pos = lro_start_pos * (1 - v) + lro_stop_pos * v
    logger.debug(f'LRO position: {lro_pos}')
    look_at(s, lro_pos, center_pos, up)

#    lro_dist = s.intersectScene([(lro_pos, normalize(np.array(center_pos - lro_pos)))])[0]
#    logger.debug(f'Setting camera pose to {lon=}°, {lat=}°, {lro_dist=}m (from x={x_center}, y={y_center}, {scale=}')

    att = s.getObjectAttitude('camera')

    return lro_pos, att, lon, lat, alt

# In SurRender, set position and attitude of camera nadir so that FoV ~= texture
def set_camera_pose(
        s: surrender_client,
        proj: Proj,
        image_size: int,
        x_center: float,
        y_center: float,
        texture_scale_factor: float,
        pixel_margin: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:

    # inverse projection to longitude, latitude & other params
    lon, lat, R, scale = xy_to_lonlat(x_center, y_center, proj)

    # minus N pixels for margin to avoid texture edges to be seen
    target_alt = np.mean(np.absolute(((image_size - pixel_margin) * scale * texture_scale_factor / 2) / np.tan(np.array(s.getCameraFOVRad())/2)))

    # set camera one time in order to measure "real" altitude with intersectScene
    pos = geodetic_to_cartesian_BCBF_position(lon, lat, 10000, R, R)
    att = yawpitchroll_to_cartesian_BCBF_attitude(90, 0, 0, pos) # 90° in yaw allows texture to be aligned with camera
    s.setObjectPosition("camera", pos)
    s.setObjectAttitude("camera", att)

    # if no DEM at center, dist is infinite
    logger.debug(f'{pos=}, intersect={s.intersectScene([(pos, -normalize(np.array(pos)))])}')
    real_alt = s.intersectScene([(pos, -normalize(np.array(pos)))])[0] - 10000
    logger.debug(f'Altitude correction: {real_alt}m. Altitude: {target_alt}m')
    if np.isinf(real_alt):
        raise ValueError('no DEM at center of image')

    # set camera a second time to adjust to "real" altitude
    pos = geodetic_to_cartesian_BCBF_position(lon, lat, target_alt - real_alt, R, R)
    att = yawpitchroll_to_cartesian_BCBF_attitude(90, 0, 0, pos)
    s.setObjectPosition("camera", pos)
    s.setObjectAttitude("camera", att)

    # get up vector
    lonup, latup, _, _ = xy_to_lonlat(x_center, 0, proj)
    up = normalize(np.array(geodetic_to_cartesian_BCBF_position(lonup, latup, 10000, R, R)) - np.array(pos))
    center_pos = geodetic_to_cartesian_BCBF_position(lon, lat, real_alt, R, R)
    look_at(s, np.array(pos), np.array(center_pos), up)

    logger.debug(f'Setting camera pose to {lon=}°, {lat=}°, {target_alt=}m (from x={x_center}, y={y_center}, {scale=}, {texture_scale_factor=})')

    return pos, att, lon, lat, target_alt


# Render a view with backprop
def render_image(
        s: surrender_client,
        texture: Union[np.ndarray, List[np.ndarray]],
        texture_diff: Union[str, List[str]],
        gt: np.ndarray,
        metadata: SampleMetadata,
        crop_size: int,
        texture_scale_factor: float,
        texture_fp: np.ndarray,
        rendered: np.ndarray,
        display: bool = False
    ):

    logger.debug(f'Rendering with metadata {metadata}')
    s.setCameraFOVDeg(metadata.fov, metadata.fov)
    s.setObjectPosition("sun", metadata.sun_pos)
    s.setObjectPosition("camera", metadata.cam_pos)
    s.setObjectAttitude("camera", metadata.cam_att)

    if type(texture) is list or type(texture_diff) is list:
        raise NotImplementedError('multiple texture to differentiate not implemented yet')

    h, w, c = texture.shape
    if c < 4:
        texture = np.concatenate((texture, np.zeros((h, w, 4 - c))), axis=2)

    # configure procedural texture and differentiable textures
    factor = np.array(metadata.size_wh) / (crop_size * texture_scale_factor)
    offset = - np.array(metadata.center_xy) / (crop_size * texture_scale_factor) + np.array([0.5, 0.5])
    s.setGlobalVariables({'factor': factor, "offset": offset, "fullsize": metadata.size_wh})
    s.setGlobalVariables(metadata.proj.params)
    logger.debug(f'Set Surrender globals for procedural texture: {s.getGlobalVariables()}')

#    s.disableTextureGradient(texture_diff)
    #s.setTexture(texture_diff, texture)
    texture_fp[:] = texture
#    s.enableTextureGradient(texture_diff, False)

    s.render()
    s.sync()

    s.initializeGradientsToZero()
    #image = s.getImage()
    if metadata.rot90 != 0:
        rendered = np.rot90(rendered, k=metadata.rot90, axes=(0, 1))

    s.backpropagate(rendered - gt)

    grad = s.getTextureGradient(texture_diff) # gradient is H, W, C=4
    if metadata.rot90 != 0:
        grad = np.rot90(grad, k=metadata.rot90, axes=(0, 1))
    logger.debug(f'nan in gt: {np.sum(np.isnan(gt))}')
    logger.debug(f'nan in image: {np.sum(np.isnan(rendered))}')
    logger.debug(f'nan in texture: {np.sum(np.isnan(texture))}')
    if np.sum(np.isnan(grad)):
        logger.warning('NaN in gradient')
    logger.debug(f'nan in grad: {np.sum(np.isnan(grad))}')
    for i in range(4):
        logger.debug(f'grad C#{i} min={np.min(grad[...,i])} max={np.max(grad[...,i])}')
    logger.debug(f'max grad: {np.nanmax(grad)}')
    logger.debug(f'nonzero grad: {np.count_nonzero(grad)}')

    loss_matrix = (rendered[...,0] - gt[...,0])**2
    if np.isnan(loss_matrix).any():
        loss_matrix = np.nan_to_num(loss_matrix, copy=False)
#        logger.warning('NaNs found in loss -> replaced with zero')
#        for i in range(4):
#            debug_plot(title=str(i), gt=gt[:,:,i], img=rendered[:,:,i], grad=grad[:,:,i], tex=texture[:,:,i])
#        plt.show()
#    loss = s.computeL2LossFromTargetAndBackpropagate(gt)
    loss = np.sum(loss_matrix)
    logger.debug(f'{loss=}')

#    with open('record.py', 'w') as f:
#        print(s.generateReplay(), file=f)
    # debug show TODO flag?
    if display:
        for i in range(4):
            debug_plot(title=f'{metadata.pds_metadata.name}_{metadata.center_xy[0]:05}_{metadata.center_xy[1]:05}_{i}.png', gt=gt[:,:,i], img=rendered[:,:,i], grad=grad[:,:,i], tex=texture[:,:,i])
        plt.show()

    return texture, grad[...,:c], loss


if __name__ == '__main__':
    # TEST PIXEL FACTORY
#    dem_file = "DEM/NAC_DTM_TYCHOPK.dem"
#    dem_file = "DEM/change2_20m.dem"
    dem_file = "DEM/tycho_v2.dem"

    texture_file = "textures/tycho-msk.tif"
    texture_files = [
		"textures/M144688460LC_Y8.tif",
		"textures/M1448032122RC_Y8.tif",
		"textures/M1450390707LC_Y8.tif",
		"textures/M1450404748RC_Y8.tif",
		"textures/M1458649145LC_Y8.tif",
		"textures/M1458670191RC_Y8.tif",
		"textures/M1463341437LC_Y8.tif",
		"textures/M1463348450LC_Y8.tif",
		"textures/M1463348450RC_Y8.tif",
		"textures/M1463355452RC_Y8.tif",
		"textures/M1463362477RC_Y8.tif",
		"textures/M155300912RC_Y8.tif",
		"textures/M157654872LC_Y8.tif",
		"textures/M157661677LC_Y8.tif",
		"textures/M157675223RC_Y8.tif",
		"textures/M168266349LC_Y8.tif",
		"textures/M168273094LC_Y8.tif",
		"textures/M168273094RC_Y8.tif",
		"textures/M168279907LC_Y8.tif",
		"textures/M170627837LC_Y8.tif",
		"textures/M170627837RC_Y8.tif",
    ]
    resource_path = [
            "/data/sharedata/VBN/pxfactory_Y8",
            "/data/sharedata/VBN/pxfactory",
            "/data/sharedata/VBN/moon",
            "/data/nmenga/surrender_data",
            "/mnt/20To/sharedata/VBN/",
            "/home/nmenga/workspace/repositories/lunar-software/data/",
            "/tmp",
            ]
    image_size=1024

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('PIL').setLevel(level=logging.WARNING)
    logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
    logging.getLogger('rasterio').setLevel(level=logging.WARNING)

    s = init_surrender(dem_file=dem_file,
                       texture_file=texture_file,
                       resource_path=resource_path,
                       image_size=image_size,
                       channels=1,
                       serverport=5221)

    from brdfgen.data import get_resource_absolute_path, get_sun_pos_from_pds_product, Proj
    texture_abspath = get_resource_absolute_path(texture_file, resource_path)

    texture_geofile = Path(texture_abspath).with_suffix('.geo')
    proj = Proj.from_file(str(texture_geofile))

    # SET SUN POSITION for texture
    sun_dir = get_sun_pos_from_pds_product(Path(texture_file).stem[:Path(texture_file).stem.index('_')-1])
    s.setObjectPosition('sun', sun_dir)

    # GET texture full size
    import rasterio as rio
    with rio.open(texture_abspath) as tex:
        w, h = tex.shape[::-1] # shape is H, W but I need x, y

    n = 1
    scale_factor = 1/tex.transform.a + 50
    plt.figure(1)
    i = 0
    for y, x in zip(np.linspace(0, h, n), np.linspace(0, w, n)):
        lon, lat, alt = set_camera_pose(s,
                                        proj=Proj.from_file(str(texture_geofile)),
                                        image_size=image_size,
                                        x_center=x,
                                        y_center=y,
                                        texture_scale_factor=texture_scale_factor)
        logger.info(f'Set camera pos to {lon=}, {lat=}, {alt=}')
        s.render()
        target = s.getImage()[...,0]
        minv, maxv = np.nanmin(target), np.nanmax(target)
        plt.imshow(target,cmap='gray')
        cv2.imwrite(f'comp/{removeprefix(dem_file, "DEM/")}+mask_{i:06}.png', (target-minv)*255/(maxv-minv))
        i += 1
        plt.show()
#    with open('record.py', 'w') as f:
#        print(s.generateReplay(), file=f)