"""
Script pour détecter les GT corrompues (contenant NaN/Inf) dans le dataset.
Usage: python check_gt.py -i <training_dir> [-ds test|train|val] [--all]
"""

from brdfgen.data import BrdfGeneratorDataset
from brdfgen.train import TrainingMetadata, custom_collate

import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def main():
    parser = ArgumentParser(description='Vérifier les GT corrompues')
    parser.add_argument('-i', '--input', type=str, required=True, help="Training directory")
    parser.add_argument('-ds', '--dataset', default='test', help="Dataset: train, val, test")
    parser.add_argument('--all', action='store_true', help="Vérifier tout le dataset")
    parser.add_argument('-n', '--nb', type=int, default=1000, help="Nombre d'échantillons à vérifier")
    args = parser.parse_args()

    training_dir = args.input

    # Charger les métadonnées du training
    with open(Path(training_dir) / 'training_metadata.json', 'r') as f:
        train_metadata = TrainingMetadata.from_json(f.read())

    # Charger le dataset
    if args.dataset == 'train':
        dataset = BrdfGeneratorDataset(
            train_metadata.dataset_dir, 
            train_metadata.channels, 
            subset_file=f'{training_dir}/train_dataset.csv'
        )
    elif args.dataset == 'val':
        dataset = BrdfGeneratorDataset(
            train_metadata.dataset_dir, 
            train_metadata.channels, 
            subset_file=f'{training_dir}/val_dataset.csv'
        )
    else:  # test
        dataset = BrdfGeneratorDataset(
            train_metadata.dataset_dir, 
            train_metadata.channels, 
            subset_file=f'{training_dir}/tst_dataset.csv'
        )

    print(f"Dataset: {args.dataset}, Total samples: {len(dataset)}")

    # Créer le dataloader
    if args.all:
        n_samples = len(dataset)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=custom_collate
        )
    else:
        n_samples = min(args.nb, len(dataset))
        indices = list(range(n_samples))
        dataloader = DataLoader(
            Subset(dataset, indices),
            shuffle=False,
            collate_fn=custom_collate
        )

    # Listes pour stocker les problèmes
    corrupted_gt = []
    constant_gt = []

    print(f"\nVérification de {n_samples} échantillons...")
    
    for i, (dem, render, metadata) in enumerate(tqdm(dataloader, total=n_samples)):
        gt = render.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
        meta = metadata[0]
        
        # Vérifier NaN/Inf
        if not np.isfinite(gt).all():
            nan_count = np.count_nonzero(~np.isfinite(gt))
            info = {
                'index': i,
                'texture': meta.texture_file,
                'center': meta.center_xy,
                'nan_pixels': nan_count,
                'total_pixels': gt.size,
                'percent': 100 * nan_count / gt.size
            }
            corrupted_gt.append(info)
            print(f"\n[CORRUPTED] i={i}, texture={Path(meta.texture_file).name}, center={meta.center_xy}, nan={nan_count}/{gt.size} ({info['percent']:.1f}%)")
        
        # Vérifier image constante
        gt_range = gt.max() - gt.min()
        if gt_range < 1e-10:
            info = {
                'index': i,
                'texture': meta.texture_file,
                'center': meta.center_xy,
                'value': gt.mean()
            }
            constant_gt.append(info)
            print(f"\n[CONSTANT] i={i}, texture={Path(meta.texture_file).name}, center={meta.center_xy}, value={gt.mean():.6f}")

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Total échantillons vérifiés: {n_samples}")
    print(f"GT corrompues (NaN/Inf): {len(corrupted_gt)}")
    print(f"GT constantes: {len(constant_gt)}")
    
    if corrupted_gt:
        print("\n--- GT CORROMPUES ---")
        # Grouper par texture
        textures = {}
        for info in corrupted_gt:
            tex = Path(info['texture']).name
            if tex not in textures:
                textures[tex] = []
            textures[tex].append(info)
        
        for tex, infos in textures.items():
            print(f"\n{tex}: {len(infos)} images corrompues")
            for info in infos[:5]:  # Afficher les 5 premières
                print(f"  - center={info['center']}, nan={info['nan_pixels']}/{info['total_pixels']}")
            if len(infos) > 5:
                print(f"  ... et {len(infos)-5} autres")

    if constant_gt:
        print("\n--- GT CONSTANTES ---")
        for info in constant_gt[:10]:
            print(f"  i={info['index']}, texture={Path(info['texture']).name}, center={info['center']}")
        if len(constant_gt) > 10:
            print(f"  ... et {len(constant_gt)-10} autres")

    # Sauvegarder la liste
    output_file = Path(training_dir) / 'corrupted_gt_list.txt'
    with open(output_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Total checked: {n_samples}\n")
        f.write(f"Corrupted: {len(corrupted_gt)}\n")
        f.write(f"Constant: {len(constant_gt)}\n\n")
        
        f.write("=== CORRUPTED GT ===\n")
        for info in corrupted_gt:
            f.write(f"{info['index']},{info['texture']},{info['center']},{info['nan_pixels']}\n")
        
        f.write("\n=== CONSTANT GT ===\n")
        for info in constant_gt:
            f.write(f"{info['index']},{info['texture']},{info['center']},{info['value']}\n")
    
    print(f"\nListe sauvegardée dans: {output_file}")


if __name__ == '__main__':
    main()
