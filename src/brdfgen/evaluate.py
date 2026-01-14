from brdfgen.data import BrdfGeneratorDataset
from brdfgen.model import BrdfGenerator
from brdfgen.eval import evaluation

import logging
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser(description='Train')
    parser.add_argument('-i', '--input', type=str, help="Directory path to the training directory", required=True)
    parser.add_argument('-ds', '--dataset', help="Evaluate training dataset ('train'), validation dataset ('val'), test dataset ('test' - dft) or custom dataset directory", default='test')
    parser.add_argument('-o', '--output', type=str, help="Output directory", default="out_eval")
    parser.add_argument('-n', '--nb', type=int, help="Number of samples of validation dataset for plot generation", default=-1)
    parser.add_argument('--batch', type=int, help="Batch size", default=64)
    parser.add_argument('--debug', help="Activate debug logs", action='store_true')
    return parser


def main(args):
    textures_source_dir = '/data/sharedata/VBN/pxfactory'
    hot_working_dir = '/mnt/20To/sharedata/VBN/pxfactory_float'
    resource_path = [
        "/tmp",
        textures_source_dir,
        hot_working_dir,
    ]

    # training configuration
    dem_file = 'DEM/tycho_v2.dem'
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

    # training output dir
    output_dir = Path(args.output)
    Path(output_dir).mkdir(exist_ok=True)

    # train metadata
    training_dir = args.input
    with open(Path(training_dir) / 'training_metadata.json', 'r') as f:
        from brdfgen.train import TrainingMetadata, custom_collate
        train_metadata = TrainingMetadata.from_json(f.read())

    # dataset for eval
    if args.dataset == 'train':
        dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=f'{training_dir}/train_dataset.csv')
    elif args.dataset == 'val':
        dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=f'{training_dir}/val_dataset.csv')
    elif args.dataset == 'test':
        dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=f'{training_dir}/tst_dataset.csv')
    else:
        dataset = BrdfGeneratorDataset(args.dataset, train_metadata.channels)
    if args.nb > 0:
        dataloader=DataLoader(Subset(dataset, rng.integers(len(dataset), size=args.nb)), batch_size=args.batch, generator=gen_cpu, shuffle=True, collate_fn=custom_collate)
    else:
        dataloader=DataLoader(dataset, batch_size=args.batch, generator=gen_cpu, shuffle=True, collate_fn=custom_collate)
    ds_metadata = dataset.metadata

    # create model
    model = BrdfGenerator(dem_file=dem_file,
                          resource_path=resource_path,
                          gsd=float(ds_metadata.gsd),
                          texture_diff=texture_diff,
                          image_size=ds_metadata.crop_size[0],
                          brdf_file=train_metadata.brdf_name,
                          out_c=train_metadata.channels,
                          serverport=5221)
    model.load_state_dict(torch.load(Path(training_dir) / f"best-model-parameters-val_loss.pt"), strict=False)

    # eval
    eval_metrics = evaluation(device=device,
                       model=model,
                       dataloader=dataloader,
                       channels=train_metadata.channels,
                       texture_diff=texture_diff,
                       has_rot=False,
                       has_regul=False,
                       rot_coef=0.0,
                       regul_coef=0.0,
                       debug_show=0)

    logger.info(f'evaluation of {args.dataset} dataset: mse={eval_metrics["val_mse"].avg}')

    del model
    del dataset
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
