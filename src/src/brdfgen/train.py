from brdfgen.data import BrdfGeneratorDataset, get_resource_absolute_path, Proj, SampleMetadata, TrainingMetadata
from brdfgen.render import xy_to_lonlat
from brdfgen.model import BrdfGenerator
from brdfgen.eval import evaluation, eval_plot
import brdfgen.config as config
from brdfgen.metrics import AverageMeter, SurrenderMSELoss, RegularisationLoss, RotationLoss

import json
import logging
from pathlib import Path
from itertools import product
from argparse import ArgumentParser
from typing import Union, List, Tuple, Dict

from shapely.geometry import Point, Polygon, MultiPoint, GeometryCollection
from shapely import voronoi_polygons
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
import torch
from torch.utils.data import random_split, DataLoader, Subset, default_collate
from torch import nn


logger = logging.getLogger(__name__)


def train_one_epoch(
    device,
    model: nn.Module,
    optimizer,
    dataloader: DataLoader,
    channels: int,
    texture_diff: str,
    has_rot: bool,
    has_regul: bool,
    rot_coef: float,
    regul_coef: float,
    debug_show: int = 0
    ) -> Dict[str, AverageMeter]:

    model.to(device)
    model.train()

    # init losses and metrics
    mse_loss_fn = SurrenderMSELoss()
    mse = AverageMeter()
    metrics = {"mse": mse}
    if has_rot:
        rot_loss_fn = RotationLoss(channels)
        rot = AverageMeter()
        metrics["rot"] = rot
    if has_regul:
        regul_loss_fn = RegularisationLoss()
        regul = AverageMeter()
        metrics["regul"] = regul
    total = AverageMeter()
    metrics["loss"] = total

    progress = tqdm(dataloader)

    for i, (dem, render, metadata) in enumerate(progress):
        dem = dem.to(device)
        render = render.to(device)
        # render (ground truth) is passed as an input to model because actual loss is computed in SurRender layer
        Y_pred = model(dem, render, metadata)

        # compute losses
#        with torch.autograd.detect_anomaly(): # for debug
        mse_loss = mse_loss_fn(Y_pred, render)
        mse.update(mse_loss.item())
        loss = mse_loss
        postfix_str = f'MSE:{mse.avg}'
        if has_rot:
            rot_loss = rot_loss_fn(dem, Y_pred, model.unet)
            loss += rot_coef * rot_loss
            rot.update(rot_loss.item())
            postfix_str += f', Rot:{rot.avg}'
        if has_regul:
            regul_loss = regul_loss_fn(Y_pred)
            loss += regul_coef * regul_loss
            regul.update(regul_loss.item())
            postfix_str += f', Regul:{regul.avg}'
        total.update(loss.item())
        postfix_str += f', Loss:{total.avg}'
        progress.set_postfix_str(postfix_str)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if debug_show > 0 and i % ((len(progress) // debug_show)) == 0:
            eval_plot(model, (dem, render, metadata), Y_pred, channels, texture_diff)

    return metrics


def train(
    device,
    model: nn.Module,
    optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    channels: int,
    texture_diff: str,
    output_dir: str,
    has_rot: bool = False,
    has_regul: bool = False,
    rot_coef: float = 0.1,
    regul_coef: float = 0.1,
    reprise: bool = False,
    debug_show: int = 0,
    gen_cpu: torch.Generator = torch.default_generator,
    ):

    Path(output_dir).mkdir(exist_ok=True)

    best_loss = 1e6
    best_val_loss = 1e6

    history = {"epochs": [], "mse": [], "loss": [], "val_mse": [], "val_loss": []}
    if has_rot:
        history['rot'] = []
        history['val_rot'] = []
    if has_regul:
        history['regul'] = []
        history['val_regul'] = []
    epoch_start = 0
    if reprise:
        model.load_state_dict(torch.load(Path(output_dir) / f"last-model-parameters.pt"), strict=False)
        with open(Path(output_dir) / 'history.csv', 'r') as f:
            epoch_start = len(f.readlines()) - 1
    else:
        with open(Path(output_dir) / 'history.csv', 'w') as f:
            f.write('; '.join([k for k in history.keys()]) + '\n')

    for epoch in range(epoch_start, epochs):

        # TRAINING
        metrics = train_one_epoch(device=device,
                        model=model,
                        optimizer=optimizer,
                        dataloader=train_dataloader,
                        channels=channels,
                        texture_diff=texture_diff,
                        has_rot=has_rot,
                        has_regul=has_regul,
                        rot_coef=rot_coef,
                        regul_coef=regul_coef,
                        debug_show=debug_show)

        # VALIDATION
        val_metrics = evaluation(device=device,
                           model=model,
                           dataloader=val_dataloader,
                           channels=channels,
                           texture_diff=texture_diff,
                           has_rot=has_rot,
                           has_regul=has_regul,
                           rot_coef=rot_coef,
                           regul_coef=regul_coef,
                           debug_show=debug_show)

        # save metrics to history and to file
        history["epochs"].append(epoch)
        [history[k].append(v.avg) for k, v in {**metrics, **val_metrics}.items()]
        logger.debug(history)
        with open(Path(output_dir) / 'history.csv', 'a') as f:
            f.write('; '.join([str(v[-1]) for v in history.values()]) + '\n')

        # SAVE BEST MODEL for loss
        if metrics['mse'].avg < best_loss:
            best_loss = metrics['mse'].avg
            logger.info(f'Saving model weights with {best_loss=}')
            torch.save(model.state_dict(), Path(output_dir) / f"best-model-parameters-loss.pt")
        if val_metrics['val_mse'].avg < best_val_loss:
            best_val_loss = val_metrics['val_mse'].avg
            logger.info(f'Saving model weights with {best_val_loss=}')
            torch.save(model.state_dict(), Path(output_dir) / f"best-model-parameters-val_loss.pt")
        torch.save(model.state_dict(), Path(output_dir) / f"last-model-parameters.pt")

        logger.info(f'epoch {epoch+1} mse={metrics["mse"].avg} val_mse={val_metrics["val_mse"].avg}')

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(history['mse'])
    ax[0].set_ylim(0, 200)
    ax[0].set_title('Loss (MSE)')
    ax[1].plot(history['val_mse'])
    ax[1].set_ylim(0, 200)
    ax[1].set_title('Validation loss (MSE)')
    dpi = 100  # DPI (dots per inch) pour la résolution
    width, height = 1920, 512
    from matplotlib.figure import Figure
    fig: Figure
    fig.set_size_inches(width / dpi, height / dpi, forward=False)
    fig.savefig(Path(output_dir) / 'loss_training.png', dpi=dpi)
    plt.close(fig)

    # Generate samples plots for illustration
#    samples_nb = 10
#    logger.info(f'Generate plots for {samples_nb} samples in training and validation dataset...')
#    for dem, render, metadata in DataLoader(Subset(train_dataloader.dataset, range(samples_nb)), generator=gen_cpu, shuffle=True):
#        dem = dem.to(device)
#        Y_pred = model(dem, render, metadata)
#        eval_plot(model, (dem, render, metadata), Y_pred, channels, texture_diff, str(Path(output_dir) / 'train'))
#    for dem, render, metadata in DataLoader(Subset(val_dataloader.dataset, range(samples_nb)), generator=gen_cpu, shuffle=True):
#        dem = dem.to(device)
#        Y_pred = model(dem, render, metadata)
#        eval_plot(model, (dem, render, metadata), Y_pred, channels, texture_diff, str(Path(output_dir) / 'val'))

def is_sample_val(val_regions, metadata_file: str) -> bool:
    with open(metadata_file, 'r') as f:
        metadata = SampleMetadata.from_json(f.read()).to_dict()

        texture_polygon = Polygon([
                Point(metadata['top_left']),
                Point(metadata['top_right']),
                Point(metadata['bottom_right']),
                Point(metadata['bottom_left']),
                ])

        return texture_polygon.intersects(val_regions).any()


def geographic_split(
        dem_file: str,
        dataset: Union[BrdfGeneratorDataset, Subset],
        resource_path: Union[str, List[str]],
        rng: np.random.Generator,
        val_frac: float = 0.1,
        tst_frac: float = 0.1,
        plot: bool = False,
    ) -> Tuple[Subset, Subset]:

    # get DEM size in pixels
    # get DEM size in pixels
    dem_abspath = "/home/cgrethen/Documents/surrender/surrender-10.0/DEM/tycho_v2.dem"
    with Path(dem_abspath).open('r') as f:
        from lupa.lua51 import LuaRuntime
        lua = LuaRuntime()
        lua.execute(f.read())
        height = lua.globals()['LINES']
        width = lua.globals()['LINE_SAMPLES']
        logger.debug(f'Found size for DEM {height=}, {width=}')

    # get DEM boundaries in lon, lat
    heightmap_file = f'{Path(dem_file).stem}_heightmap.big'
    heightmap_abspath = get_resource_absolute_path(heightmap_file, resource_path)
    heightmap_geofile = Path(heightmap_abspath).with_suffix('.geo')
    proj = Proj.from_file(str(heightmap_geofile))
    top_left_lon, top_left_lat, _, _ = xy_to_lonlat(0, 0, proj)
    top_right_lon, top_right_lat, _, _ = xy_to_lonlat(width, 0, proj)
    bottom_left_lon, bottom_left_lat, _, _ = xy_to_lonlat(0, height, proj)
    bottom_right_lon, bottom_right_lat, _, _ = xy_to_lonlat(width, height, proj)

    # get DEM as polygon
    dem_polygon = Polygon([
        Point(top_left_lon, top_left_lat),
        Point(top_right_lon, top_right_lat),
        Point(bottom_right_lon, bottom_right_lat),
        Point(bottom_left_lon, bottom_left_lat),
        ])
    logger.debug(f'Found lon, lat polygon for DEM {dem_polygon=}')

    # place points to cut grid with voronoi function
    grid_width = 15
    grid_height = int(grid_width*height/width)
    width_points = np.add(np.linspace(0, width, grid_width, endpoint=False), width/(grid_width*2))
    height_points = np.add(np.linspace(0, height, grid_height, endpoint=False), height/(grid_height*2))
    grid_points_tex = product(width_points, height_points)
    grid_points = []
    for x, y in grid_points_tex:
        lon, lat, _, _ = xy_to_lonlat(x, y, proj)
        grid_points.append(Point(lon, lat))
    regions = voronoi_polygons(MultiPoint(grid_points), extend_to=dem_polygon)

    # randomly select regions that will be part of validation dataset
    region_nb = len(regions.geoms)
    val_region_nb = int(val_frac*region_nb)
    tst_region_nb = int(tst_frac*region_nb)
    val_tst_regions = rng.choice(regions.geoms, size=val_region_nb + tst_region_nb, replace=False)
    val_regions = val_tst_regions[:val_region_nb]
    tst_regions = val_tst_regions[val_region_nb:]
    assert(len(tst_regions) == tst_region_nb)

    if type(dataset) == Subset:
        metadata_files = np.array(dataset.dataset.metadata_files)[dataset.indices]
    else:
        metadata_files = dataset.metadata_files

    logger.info(f'Split dataset geographically ({val_region_nb}/{region_nb} val, {tst_region_nb}/{region_nb} test)')
    from functools import partial
    from tqdm.contrib.concurrent import process_map
    is_val = np.array(process_map(partial(is_sample_val, val_regions), metadata_files, chunksize=2))
    is_tst = np.array(process_map(partial(is_sample_val, tst_regions), metadata_files, chunksize=2))
    idx_val, = np.nonzero(is_val)
    idx_tst, = np.nonzero(is_tst)
    idx_train, = np.nonzero(~(np.logical_or(is_val,is_tst)))

    if plot:
        img = plt.imread('tychodem.png')
        figsize = img.shape[1] / 80.0, img.shape[0] / 80.0
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, extent=[min(top_left_lon, bottom_left_lon),
                               max(top_right_lon, bottom_right_lon),
                               min(bottom_left_lat, bottom_right_lat),
                               max(top_left_lat, top_right_lat)])
        import shapely.plotting as splt
#        splt.plot_polygon(dem_polygon, ax=ax)
        for i, metadata_file in enumerate(tqdm(metadata_files)):
            with open(metadata_file, 'r') as f:
                metadata = SampleMetadata.from_json(f.read()).to_dict()
                texture_polygon = Polygon([
                        Point(metadata['top_left']),
                        Point(metadata['top_right']),
                        Point(metadata['bottom_right']),
                        Point(metadata['bottom_left']),
                        ])
#                for region in val_regions:
#                    splt.plot_polygon(region, add_points=False, ax=ax, facecolor='red', alpha=0.5)
                if texture_polygon.intersects(val_regions).any():
                    splt.plot_polygon(texture_polygon, ax=ax, alpha=0.5, facecolor='red', add_points=False)
                elif texture_polygon.intersects(tst_regions).any():
                    splt.plot_polygon(texture_polygon, ax=ax, alpha=0.5, facecolor='green', add_points=False)
                else:
                    splt.plot_polygon(texture_polygon, ax=ax, alpha=0.5, add_points=False)
#                splt.plot_points(MultiPoint(grid_points), ax=ax)
        plt.show()
        fig.savefig('geographic_split.png')

    return Subset(dataset, idx_train), Subset(dataset, idx_val), Subset(dataset, idx_tst)


'''
data: list of batch size of tuples (dem, render, metadata) yielded by BrdfGeneratorDataset
'''
def custom_collate(batch):
    dems, renders, metadatas = [], [], []
    for dem, render, metadata in batch:
        dems.append(dem)
        renders.append(render)
        metadatas.append(metadata)
    return default_collate(dems), default_collate(renders), metadatas


def load_split_from_csv(dataset, split_dir: str):
    """Load train/val/test split from existing CSV files.
    
    Args:
        dataset: The full dataset
        split_dir: Directory containing train_dataset.csv, val_dataset.csv, tst_dataset.csv
    
    Returns:
        train_dataset, val_dataset, tst_dataset as Subsets
    """
    split_path = Path(split_dir)
    
    # Get metadata files
    if type(dataset) == Subset:
        metadata_files = dataset.dataset.metadata_files
    else:
        metadata_files = dataset.metadata_files
    
    # Create a dict mapping sample names to indices
    name_to_idx = {}
    for idx, metadata_file in enumerate(metadata_files):
        sample_name = Path(metadata_file).stem[9:]  # Remove 'metadata_' prefix
        name_to_idx[sample_name] = idx
    
    # Load indices from CSV files
    train_indices = []
    with open(split_path / 'train_dataset.csv', 'r') as f:
        for line in f:
            sample_name = line.strip()
            if sample_name in name_to_idx:
                train_indices.append(name_to_idx[sample_name])
    
    val_indices = []
    with open(split_path / 'val_dataset.csv', 'r') as f:
        for line in f:
            sample_name = line.strip()
            if sample_name in name_to_idx:
                val_indices.append(name_to_idx[sample_name])
    
    tst_indices = []
    with open(split_path / 'tst_dataset.csv', 'r') as f:
        for line in f:
            sample_name = line.strip()
            if sample_name in name_to_idx:
                tst_indices.append(name_to_idx[sample_name])
    
    logger.info(f'Loaded split from CSV: {len(train_indices)} train, {len(val_indices)} val, {len(tst_indices)} test samples')
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, tst_indices)


def get_parser():
    parser = ArgumentParser(description='Train')
    parser.add_argument('-b', '--brdf', type=str, help="BRDF file name", required=True)
    parser.add_argument('-p', '--params', type=int, help="BRDF number of parameters", required=True)
    parser.add_argument('-d', '--dataset', type=str, help="Directory path to the training dataset", required=True)
    parser.add_argument('-o', '--out', type=str, help="Output directory", required=True)
    parser.add_argument('--rot', type=float, help="Coefficient of rotation invariance term added to loss", default=0.0)
    parser.add_argument('--regul', type=float, help="Coefficient of regularization term added to loss", default=0.0)
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs", default=50)
    parser.add_argument('--batch', type=int, help="Batch size", default=64)
    parser.add_argument('--subset', type=int, help="Specify a subset (number of samples) of the train dataset, useful for quick test iteration", default=0)
    parser.add_argument('--load-split', type=str, help="Load existing train/val/test split from CSV files in specified directory", default=None)
    parser.add_argument('--reprise', help="Continue a training from last checkpoint found in training directory", action='store_true')
    parser.add_argument('--debug', help="Activate debug logs", action='store_true')
    return parser


def main(args):
    textures_source_dir = '/data/sharedata/VBN/pxfactory'
    hot_working_dir = '/mnt/20To/sharedata/VBN/pxfactory_float'
    resource_path= [
      "/media/cgrethen/T9/Papier2/lunar-software_eae4795152d1da9b7e9ca7df9d8b26b279320b5e",
      "/home/cgrethen/Documents/surrender/surrender-10.0",
      "/media/cgrethen/6540a7cd-daf9-4db5-b150-e6e53fe387fa/pour_clementine/pxfactory"
    ]
    
    # training configuration
    dem_file = 'DEM/tycho_v2.dem'
    brdf_name = args.brdf
    channels = args.params
    dataset_dir = args.dataset
    if args.subset != 0:
        subset_size = args.subset
    batch_size = args.batch
    epochs = args.epochs
    texture_diff = "blablabla" # TODO est-ce que c'est utile de customiser ce nom ?

    # check GPU availability
    if not torch.cuda.is_available():
        logger.error('GPU not available')
        exit(1)
    device = torch.device("cuda:0")

    # init RNGs
    seed = 123456789123456789
    rng = np.random.default_rng(seed)
    gen_cpu = torch.Generator().manual_seed(seed)
    gen_cuda = torch.Generator(device).manual_seed(seed)

    # log settings
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger('PIL').setLevel(level=logging.WARNING)
    logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
    logging.getLogger('rasterio').setLevel(level=logging.WARNING)

    # dataset
    dataset = BrdfGeneratorDataset(dataset_dir, channels, augment=False, rng=rng)
    ds_metadata = dataset.metadata
    if 'subset_size' in locals():
        dataset = Subset(dataset, rng.integers(len(dataset), size=subset_size))

    # training output dir
    output_dir = Path(args.out) / f'out_{Path(dataset_dir).name}_{Path(brdf_name).stem}_sz{len(dataset)}_e{epochs}_b{batch_size}_rot{args.rot}_regul{args.regul}'
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    # train metadata
    train_metadata = TrainingMetadata(
        brdf_name = brdf_name,
        channels = channels,
        dataset_dir = dataset_dir,
        epochs = epochs,
        batch_size = batch_size,
        dataset_metadata=ds_metadata,
    )
    logger.debug(train_metadata)
    with open(Path(output_dir) / 'training_metadata.json', 'w') as f:
        json.dump(train_metadata.to_dict(), f, indent=2)

    # train/val dataset split
    # Check if CSV files exist in dataset directory or load-split directory
    split_dir = args.load_split if args.load_split else dataset_dir
    csv_files_exist = (Path(split_dir) / 'train_dataset.csv').exists() and \
                      (Path(split_dir) / 'val_dataset.csv').exists() and \
                      (Path(split_dir) / 'tst_dataset.csv').exists()
    
    if csv_files_exist:
        # Load existing split from CSV files
        logger.info(f'Found existing split CSV files in {split_dir}')
        train_dataset, val_dataset, tst_dataset = load_split_from_csv(dataset, split_dir)
    else:
        # Compute geographic split
        logger.info('No existing split found, computing geographic split...')
        train_dataset, val_dataset, tst_dataset = geographic_split(dem_file, dataset, resource_path, val_frac=0.095, tst_frac=0.095, rng=rng)#, plot=True)
    
    logger.info(f'{len(val_dataset)=} {len(tst_dataset)=} {len(train_dataset)=}')
    logger.info(f'{len(val_dataset)/len(dataset)} real ratio of validation dataset')
    logger.info(f'{len(tst_dataset)/len(dataset)} real ratio of test dataset')

    # create model
    model = BrdfGenerator(dem_file=dem_file,
                          resource_path=resource_path,
                          gsd=float(ds_metadata.gsd),
                          texture_diff=texture_diff,
                          image_size=ds_metadata.crop_size[0],
                          brdf_file=brdf_name,
                          out_c=channels,
                          serverport=5221)

    total_params = 0
    for _, param in model.named_parameters():
        if not param.requires_grad: continue
        total_params += param.numel()
    logger.info(f'Total number of learnable parameters in model: {total_params}')

    # optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # train / val datasets
    if type(dataset) == Subset:
        metadata_files = dataset.dataset.metadata_files
    else:
        metadata_files = dataset.metadata_files
    with open(Path(output_dir) / 'train_dataset.csv', 'w') as f:
        for metadata_file in np.array(metadata_files)[train_dataset.indices]:
            f.write(f'{Path(metadata_file).stem[9:]}\n')
    with open(Path(output_dir) / 'val_dataset.csv', 'w') as f:
        for metadata_file in np.array(metadata_files)[val_dataset.indices]:
            f.write(f'{Path(metadata_file).stem[9:]}\n')
    with open(Path(output_dir) / 'tst_dataset.csv', 'w') as f:
        for metadata_file in np.array(metadata_files)[tst_dataset.indices]:
            f.write(f'{Path(metadata_file).stem[9:]}\n')

    train(device=device,
          model=model,
          optimizer=optimizer,
          train_dataloader=DataLoader(train_dataset, batch_size=batch_size, generator=gen_cpu, shuffle=True, collate_fn=custom_collate),
          val_dataloader=DataLoader(val_dataset, batch_size=batch_size, generator=gen_cpu, collate_fn=custom_collate),
          epochs=epochs,
          channels=channels,
          texture_diff=texture_diff,
          output_dir=output_dir,
          gen_cpu=gen_cpu,
          has_rot=args.rot > 0.0,
          has_regul=args.regul > 0.0,
          reprise=args.reprise,
          debug_show=0)

    del model
    del optimizer
    del dataset
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)