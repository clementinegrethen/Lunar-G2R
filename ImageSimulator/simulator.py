from pathlib import Path
from argparse import ArgumentParser
from shutil import copy2, SameFileError

import numpy as np
import cv2
from tqdm import tqdm

from descentimagegenerator.config import DIGConfig
from descentimagegenerator.trajectoryrenderer import TrajectoryRenderer, ImageFormat
from descentimagegenerator.display import DisplayVideo


if __name__ == "__main__":
    argument_parser = ArgumentParser(prog='Descent and landing simulator', description='Simulator for absnav3D sequences on celestial bodies',)
    argument_parser.add_argument("-c", "--config_file", required=True)
    argument_parser.add_argument("-d", "--display", action='store_true')
    argument_parser.add_argument("-t", "--traj", action='store_true', help="Gen only traj.csv")
    args = argument_parser.parse_args()

    moon_absnav_config = DIGConfig.load_from_file(args.config_file)
    renderer = TrajectoryRenderer(moon_absnav_config)

    if args.display:
        float_display = DisplayVideo("float image")

    start_index = moon_absnav_config.output_config.image_range[0]
    end_index = moon_absnav_config.output_config.image_range[1]
    end_index = end_index if end_index > 0 else 2**32 - 1

    if end_index > renderer.get_trajectory_length():
        end_index = renderer.get_trajectory_length()
        print(f"Rendering to end index {end_index}")

    Path(moon_absnav_config.output_config.output_directory).mkdir(exist_ok=True)
    if moon_absnav_config.output_config.copy_trajectory_file:
        copy2(moon_absnav_config.scene_config.trajectory_path, moon_absnav_config.output_config.output_directory)
    # copy DIG config
    try:
        copy2(args.config_file, f'{Path(moon_absnav_config.output_config.output_directory)}/{Path(args.config_file).name}')
    except SameFileError:
        pass

    frame_range = range(start_index, end_index, moon_absnav_config.output_config.step)
    bar = tqdm(total=end_index)
    bar.update(start_index)

    n_pos = []
    n_att = []

    outdir_name = "./"
    if moon_absnav_config.output_config.output_directory:
        outdir_name = f"{moon_absnav_config.output_config.output_directory}/"

    sunPowerChanged = False;
    for i in frame_range:

        render_ret = renderer.render_frame(i)
        float_image = renderer.get_visible_frame(ImageFormat.Gray32F)
        pos = renderer.client.getObjectPosition('camera')
        def normalize(v):
            n = np.linalg.norm(v)
            if n < 1e-16:
                return v
            else:
                return v/n
        if not render_ret:
            print(f"Could not render frame {i}")
            break

        pos_n = normalize(pos)
        print(f'alt={renderer.client.intersectScene([(pos, -normalize(pos))])[0]}, lat={np.rad2deg(np.arcsin(pos_n[2]))}, lon={np.rad2deg(np.arctan2(pos_n[1],pos_n[0]))}')

        if args.traj:
            # update traj
            n_pos.append(renderer.client.getObjectPosition('camera'))
            n_att.append(renderer.client.getObjectAttitude('camera'))
        else:
            # normalized 8bit image
            vmin, vmax = np.nanmin(float_image), np.nanmax(float_image)
            if vmin < vmax:
                cv2.imwrite(f"{Path(outdir_name)}/image{i:07d}.png", 255.0*(float_image - vmin)/(vmax - vmin))
            else:
                print('black image')
                cv2.imwrite(f"{Path(outdir_name)}/image{i:07d}.png", float_image)

        # update tqdm
        bar.update(moon_absnav_config.output_config.step)

        if args.display:
            float_display(float_image, renderer.get_depth_map())

    if args.traj:
        traj = np.concatenate((n_pos, n_att), axis=1)
        np.savetxt(f"{Path(outdir_name).name}/traj.csv", traj, delimiter=",", header="x(m),y(m),z(m),q0,qx,qy,qz", fmt="%.16e", comments="")
