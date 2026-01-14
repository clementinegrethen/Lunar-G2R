#!/usr/bin/env python3
"""
Script pour trouver les images avec des NaN dans le dataset
"""

from brdfgen.data import BrdfGeneratorDataset
from brdfgen.train import TrainingMetadata
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

def find_nan_images(training_dir, dataset_type='test'):
    """
    Trouve tous les images avec des NaN dans le dataset
    
    Args:
        training_dir: répertoire de training
        dataset_type: 'train', 'val', ou 'test'
    """
    
    # Load metadata
    with open(Path(training_dir) / 'training_metadata.json', 'r') as f:
        train_metadata = TrainingMetadata.from_json(f.read())
    
    # Load dataset
    if dataset_type == 'train':
        subset_file = f'{training_dir}/train_dataset.csv'
    elif dataset_type == 'val':
        subset_file = f'{training_dir}/val_dataset.csv'
    elif dataset_type == 'test':
        subset_file = f'{training_dir}/tst_dataset.csv'
    else:
        subset_file = dataset_type
    
    dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=subset_file)
    
    print(f"Searching for NaN images in {dataset_type} dataset ({len(dataset)} samples)...")
    
    nan_indices = []
    nan_files = []
    
    for i, (dem, render, metadata) in enumerate(tqdm(dataset)):
        # Check DEM
        if not np.isfinite(dem.numpy()).all():
            nan_indices.append(i)
            nan_files.append({
                'index': i,
                'texture': Path(metadata.texture_file).stem,
                'center_xy': metadata.center_xy,
                'type': 'DEM has NaN'
            })
            continue
        
        # Check render (GT)
        if not np.isfinite(render.numpy()).all():
            nan_indices.append(i)
            nan_files.append({
                'index': i,
                'texture': Path(metadata.texture_file).stem,
                'center_xy': metadata.center_xy,
                'type': 'Render (GT) has NaN'
            })
            continue
    
    print(f"\n\n===== RÉSULTATS =====")
    print(f"Total NaN images found: {len(nan_indices)} / {len(dataset)}")
    
    if len(nan_files) > 0:
        print(f"\nImages with NaN:")
        for info in nan_files:
            print(f"  Index {info['index']:5d}: {info['texture']:20s} @ ({info['center_xy'][0]:05d}, {info['center_xy'][1]:05d}) - {info['type']}")
        
        # Save to file
        output_file = Path(training_dir) / f'nan_images_{dataset_type}.txt'
        with open(output_file, 'w') as f:
            f.write(f"Total NaN images: {len(nan_indices)} / {len(dataset)}\n\n")
            for info in nan_files:
                f.write(f"Index {info['index']:5d}: {info['texture']:20s} @ ({info['center_xy'][0]:05d}, {info['center_xy'][1]:05d}) - {info['type']}\n")
        print(f"\nSaved to: {output_file}")
    else:
        print("No NaN images found!")

if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Find NaN images in dataset')
    parser.add_argument('-i', '--input', type=str, help="Training directory input", required=True)
    parser.add_argument('-ds', '--dataset', help="Dataset type ('train', 'val', 'test')", default='test')
    
    args = parser.parse_args()
    
    find_nan_images(args.input, args.dataset)
