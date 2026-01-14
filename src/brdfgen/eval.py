from brdfgen.render import init_surrender
from brdfgen.data import DatasetMetadata, BrdfGeneratorDataset
from brdfgen.model import BrdfGenerator
from brdfgen.utils import removeprefix
from brdfgen.metrics import AverageMeter, SurrenderMSELoss, RegularisationLoss, RotationLoss

import logging
from pathlib import Path
from typing import Union, List, Dict
from argparse import ArgumentParser

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
try:
    import lpips
except ImportError:
    lpips = None


logger = logging.getLogger(__name__)


def evaluation(
    device,
    model: nn.Module,
    dataloader: DataLoader,
    channels: int,
    texture_diff: str,
    has_rot: bool,
    has_regul: bool,
    rot_coef: float,
    regul_coef: float,
    debug_show: int = 0,
    compute_hapke: bool = True,
    dem_file: str = None,
    resource_path: Union[str, List[str]] = None,
    save_debug_images: int = 0,  # Sauvegarder N premières images pour debug (0 = désactivé)
    eval_dir: str = None,
    sun_power: float = None,  # Sun power pour le rendu Hapke
    ) -> Dict[str, AverageMeter]:

    print("[DEBUG] Appel de la fonction evaluation")
    model.to(device)
    model.eval()

    # init losses and metrics
    mse_loss_fn = SurrenderMSELoss()
    mse = AverageMeter()
    metrics = {"val_mse": mse}
    
    # Hapke MSE (calculé directement ici)
    if compute_hapke:
        mse_hapke = AverageMeter()
        mse_hapke_norm = AverageMeter()
        psnr_hapke_norm = AverageMeter()
        ssim_hapke_norm = AverageMeter()
        metrics["hapke_mse"] = mse_hapke
        metrics["hapke_mse_norm"] = mse_hapke_norm
        metrics["hapke_psnr_norm"] = psnr_hapke_norm
        metrics["hapke_ssim_norm"] = ssim_hapke_norm
    
    # Learned PSNR/SSIM
    psnr_learned = AverageMeter()
    ssim_learned = AverageMeter()
    metrics["psnr_learned"] = psnr_learned
    metrics["ssim_learned"] = ssim_learned
    
    # LPIPS metrics
    lpips_hapke_norm = AverageMeter()
    lpips_learned = AverageMeter()
    metrics["lpips_hapke_norm"] = lpips_hapke_norm
    metrics["lpips_learned"] = lpips_learned
    lpips_fn = None
    if compute_hapke and lpips is not None:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    if has_rot:
        rot_loss_fn = RotationLoss(channels)
        rot = AverageMeter()
        metrics["val_rot"] = rot
    if has_regul:
        regul_loss_fn = RegularisationLoss()
        regul = AverageMeter()
        metrics["val_regul"] = regul
    total = AverageMeter()
    metrics["val_loss"] = total

    progress = tqdm(dataloader)

    for i, (dem, render, metadata) in enumerate(progress):
        dem = dem.to(device)
        render = render.to(device)
        Y_pred = model(dem, render, metadata)

        # compute losses
        mse_loss = mse_loss_fn(Y_pred, render)
        mse.update(mse_loss.item())
        loss = mse_loss
        postfix_str = f'val MSE:{mse.avg:.4f}'
        

        if compute_hapke and dem_file and resource_path:
            meta = metadata[0]
            
            with init_surrender(
                dem_file=dem_file,
                texture_file="default.png",  #  uilise hapke.brdf avec albedo=3.5
                resource_path=resource_path,
                image_size=model.render.image_size,
                channels=1,
                rays=model.render.s.getNbSamplesPerPixel(),
                sun_power=sun_power,
                serverhost='localhost',
                serverport=5151,
            ) as s_hapke:
                s_hapke.setCameraFOVDeg(meta.fov, meta.fov)
                s_hapke.setObjectPosition("sun", meta.sun_pos)
                s_hapke.setObjectPosition("camera", meta.cam_pos)
                s_hapke.setObjectAttitude("camera", meta.cam_att)
                s_hapke.render()
                hapke_img = s_hapke.getImage()
            
            #  MSE Hapke RAW
            gt = render.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
            hapke = hapke_img[:, :, 0]
            
            # Skip if NaN detected
            if not np.isfinite(gt).all() or not np.isfinite(hapke).all():
                print(f"[WARNING] Skipping sample {i}: NaN detected in GT or Hapke")
                continue
            
            mse_hapke_val = np.sum((hapke - gt) ** 2)
            mse_hapke.update(mse_hapke_val)
            
            # DEBUG:
            if not np.isfinite(mse_hapke.avg):
                print(f"\n\n!!! NaN DETECTE à i={i} !!!")
                print(f"  mse_hapke_val={mse_hapke_val}")
                print(f"  mse_hapke.sum={mse_hapke.sum}")
                print(f"  mse_hapke.count={mse_hapke.count}")
                print(f"  mse_hapke.avg={mse_hapke.avg}")
                print(f"  hapke: min={hapke.min()}, max={hapke.max()}, mean={hapke.mean()}, dtype={hapke.dtype}")
                print(f"  gt: min={gt.min()}, max={gt.max()}, mean={gt.mean()}, dtype={gt.dtype}")
                print(f"  np.isfinite(hapke).all()={np.isfinite(hapke).all()}")
                print(f"  np.isfinite(gt).all()={np.isfinite(gt).all()}")
                break
            
            #  MSE Hapke NORMALIZED (même moyenne que GT)
            gt_range = gt.max() - gt.min()
            hapke_range = hapke.max() - hapke.min()
            
            aa = gt_range / hapke_range
            bb = np.mean(gt) - np.mean(hapke) * aa
            hapke_norm = hapke * aa + bb
        
            mse_hapke_norm_val = np.sum((hapke_norm - gt) ** 2)
            mse_hapke_norm.update(mse_hapke_norm_val)
            
            # DEBUG: détecter NaN dans hapke_norm
            if not np.isfinite(mse_hapke_norm.avg):
                print(f"\n\n!!! NaN DETECTE dans HAPKE NORM à i={i} !!!")
                print(f"  mse_hapke_norm_val={mse_hapke_norm_val}")
                print(f"  mse_hapke_norm.sum={mse_hapke_norm.sum}")
                print(f"  mse_hapke_norm.avg={mse_hapke_norm.avg}")
                print(f"  hapke_norm: min={hapke_norm.min()}, max={hapke_norm.max()}, mean={hapke_norm.mean()}")
                print(f"  aa={aa}, bb={bb}")
                print(f"  gt_range={gt_range}, hapke_range={hapke_range}")
                print(f"  np.isfinite(hapke_norm).all()={np.isfinite(hapke_norm).all()}")
                break
            
            #  PSNR et SSIM pour Hapke NORM
            gt_range = gt.max() - gt.min()
            if gt_range > 1e-10:
                psnr_h = peak_signal_noise_ratio(gt, hapke_norm, data_range=gt_range)
                ssim_h = structural_similarity(gt, hapke_norm, data_range=gt_range)
                psnr_hapke_norm.update(psnr_h)
                ssim_hapke_norm.update(ssim_h)

            #  PSNR et SSIM pour Learned
            learnt_img = model.render.rendered[:, :, 0]
            if gt_range > 1e-10:
                psnr_l = peak_signal_noise_ratio(gt, learnt_img, data_range=gt_range)
                ssim_l = structural_similarity(gt, learnt_img, data_range=gt_range)
                psnr_learned.update(psnr_l)
                ssim_learned.update(ssim_l)
            
            #  LPIPS pour Hapke NORM et Learned
            if lpips_fn is not None and gt_range > 1e-10:
                # Prepare images for LPIPS
                gt_lpips = torch.from_numpy(gt / (gt.max() + 1e-8)).float().to(device)
                hapke_norm_lpips = torch.from_numpy(hapke_norm / (hapke_norm.max() + 1e-8)).float().to(device)
                learnt_lpips = torch.from_numpy(learnt_img / (learnt_img.max() + 1e-8)).float().to(device)
                
                # Add batch and channel dimensions (B, C, H, W)
                gt_lpips = gt_lpips.unsqueeze(0).unsqueeze(0)
                hapke_norm_lpips = hapke_norm_lpips.unsqueeze(0).unsqueeze(0)
                learnt_lpips = learnt_lpips.unsqueeze(0).unsqueeze(0)
                
                # Expand to 3 channels for AlexNet
                gt_lpips = gt_lpips.expand(-1, 3, -1, -1)
                hapke_norm_lpips = hapke_norm_lpips.expand(-1, 3, -1, -1)
                learnt_lpips = learnt_lpips.expand(-1, 3, -1, -1)
                
                lpips_h = lpips_fn(hapke_norm_lpips, gt_lpips).item()
                lpips_l = lpips_fn(learnt_lpips, gt_lpips).item()
                lpips_hapke_norm.update(lpips_h)
                lpips_learned.update(lpips_l)
            
            if save_debug_images > 0 and i < save_debug_images and eval_dir:
                debug_dir = Path(eval_dir) / 'debug_hapke'
                debug_dir.mkdir(exist_ok=True, parents=True)
                
                learnt_img = model.render.rendered[:, :, 0]
                
                meta = metadata[0]
                prefix = f'{Path(meta.texture_file).stem}_{meta.center_xy[0]:05}_{meta.center_xy[1]:05}'
                cv2.imwrite(str(debug_dir / f'gt_{prefix}.tif'), gt.astype(np.float32))
                cv2.imwrite(str(debug_dir / f'hapke_{prefix}.tif'), hapke.astype(np.float32))
                cv2.imwrite(str(debug_dir / f'hapke_norm_{prefix}.tif'), hapke_norm.astype(np.float32))
                cv2.imwrite(str(debug_dir / f'learnt_{prefix}.tif'), learnt_img.astype(np.float32))
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'{prefix}\nMSE: Learned={mse_loss.item():.4f}, Hapke={mse_hapke_val:.2f}, HapkeNorm={mse_hapke_norm_val:.4f}')
                
                im0 = axes[0, 0].imshow(gt, cmap='gray')
                axes[0, 0].set_title(f'GT (min={gt.min():.4f}, max={gt.max():.4f})')
                plt.colorbar(im0, ax=axes[0, 0])
                
                im1 = axes[0, 1].imshow(hapke, cmap='gray')
                axes[0, 1].set_title(f'Hapke RAW (min={hapke.min():.4f}, max={hapke.max():.4f})')
                plt.colorbar(im1, ax=axes[0, 1])
                
                im2 = axes[0, 2].imshow(learnt_img, cmap='gray')
                axes[0, 2].set_title(f'Learned (min={learnt_img.min():.4f}, max={learnt_img.max():.4f})')
                plt.colorbar(im2, ax=axes[0, 2])
                
                im3 = axes[1, 0].imshow(hapke_norm, cmap='gray')
                axes[1, 0].set_title(f'Hapke NORM (min={hapke_norm.min():.4f}, max={hapke_norm.max():.4f})')
                plt.colorbar(im3, ax=axes[1, 0])
                
                im4 = axes[1, 1].imshow(np.abs(hapke_norm - gt), cmap='hot')
                axes[1, 1].set_title(f'|Hapke NORM - GT|')
                plt.colorbar(im4, ax=axes[1, 1])
                
                im5 = axes[1, 2].imshow(np.abs(learnt_img - gt), cmap='hot')
                axes[1, 2].set_title(f'|Learned - GT|')
                plt.colorbar(im5, ax=axes[1, 2])
                
                plt.tight_layout()
                fig.savefig(debug_dir / f'compare_{prefix}.png', dpi=150)
                plt.close(fig)
                
                print(f'[DEBUG] Images sauvegardées dans {debug_dir}')
            
            postfix_str += f', Hapke:{mse_hapke.avg:.2f}, HapkeNorm:{mse_hapke_norm.avg:.2f}, PSNR_L:{psnr_learned.avg:.2f}, SSIM_L:{ssim_learned.avg:.3f}'
        
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
        postfix_str += f', Loss:{total.avg:.4f}'
        progress.set_postfix_str(postfix_str)

        if debug_show > 0 and i % ((len(progress) // debug_show)) == 0:
            eval_plot(model, (dem, render, metadata), Y_pred, channels, texture_diff)

    stats = []
    stats.append(f"MSE Learned moyenne : {metrics['val_mse'].avg}")
    stats.append(f"Nombre d'échantillons (N) : {metrics['val_mse'].count}")
    if 'hapke_mse' in metrics:
        stats.append(f"MSE Hapke RAW moyenne : {metrics['hapke_mse'].avg}")
        stats.append(f"MSE Hapke NORM moyenne : {metrics['hapke_mse_norm'].avg}")
        stats.append(f"PSNR Hapke NORM moyenne : {metrics['hapke_psnr_norm'].avg:.4f}")
        stats.append(f"SSIM Hapke NORM moyenne : {metrics['hapke_ssim_norm'].avg:.4f}")
    stats.append(f"PSNR Learned moyenne : {metrics['psnr_learned'].avg:.4f}")
    stats.append(f"SSIM Learned moyenne : {metrics['ssim_learned'].avg:.4f}")
    if 'lpips_hapke_norm' in metrics and metrics['lpips_hapke_norm'].count > 0:
        stats.append(f"LPIPS Hapke NORM moyenne : {metrics['lpips_hapke_norm'].avg:.6f}")
        stats.append(f"LPIPS Learned moyenne : {metrics['lpips_learned'].avg:.6f}")
    if 'val_rot' in metrics:
        stats.append(f"Rotation Loss moyenne : {metrics['val_rot'].avg}")
    if 'val_regul' in metrics:
        stats.append(f"Regularisation Loss moyenne : {metrics['val_regul'].avg}")
    stats.append(f"Loss totale moyenne : {metrics['val_loss'].avg}")

    print(f"[DEBUG] Nombre d'échantillons traités : {metrics['val_mse'].count}")
    print("\n===== STATISTIQUES TEST =====")
    for line in stats:
        print(line)

    eval_dir = None
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'subset_file'):
        subset_path = Path(dataloader.dataset.subset_file)
        eval_dir = subset_path.parent.parent / 'eval' / subset_path.parent.name
    else:
        eval_dir = Path('.')
    try:
        print(f"[DEBUG] Sauvegarde des stats dans : {eval_dir}")
        eval_dir.mkdir(exist_ok=True, parents=True)
        stats_path = eval_dir / 'eval_stats.txt'
        with open(stats_path, 'w') as f:
            f.write('\n'.join(stats) + '\n')
        print(f"[INFO] Statistiques sauvegardées dans : {stats_path.resolve()}")
    except Exception as e:
        print(f"[WARNING] Impossible de sauvegarder les stats : {e}")
    return metrics
import lpips
import torch
    
    # Initialize LPIPS model (VGG-based perceptual loss)
def compare_with_hapke(
    device,
    training_dir: str,
    dataloader: DataLoader,
    ds_metadata: DatasetMetadata,
    channels: int,
    texture_diff: str,
    output_dir: str,
    brdf_name: str,
    dem_file: str,
    resource_path: Union[str, List[str]],
    serverhost: str = 'localhost',
    serverport: int = 5151,
    regen_gt: bool = True,
    regen_model: bool = True,
    ):

    Path(output_dir).mkdir(exist_ok=True)

    # learned metrics
    rmse_l = []
    psnr_l = []
    ssim_l = []

    # hapke RAW
    rmse_h_raw = []
    psnr_h_raw = []
    ssim_h_raw = []

    # hapke NORMALIZED
    rmse_h_norm = []
    psnr_h_norm = []
    ssim_h_norm = []
    mse_h_norm = []
    mse_h_hapke = []
    mse_l = []
    
    # LPIPS metrics
    lpips_h_raw = []
    lpips_h_norm = []
    lpips_l = []
    
    gsd = float(ds_metadata.gsd)
    image_size = ds_metadata.crop_size[0]
    sun_power = float(ds_metadata.sun_power)

    #####################################################################
    # 1) GENERATE GT & HAPKE IMAGES IF NEEDED
    #####################################################################
    if regen_gt:
        # generate hapke
        for _, _, metadata in dataloader:
            with init_surrender(
                dem_file=dem_file,
                texture_file="default.png",
                resource_path=resource_path,
                image_size=image_size,
                channels=1,
                rays=128,
                sun_power=sun_power,
                serverhost=serverhost,
                serverport=serverport,
            ) as s:
                s.setCameraFOVDeg(metadata[0].fov, metadata[0].fov)
                s.setObjectPosition("sun", metadata[0].sun_pos)
                s.setObjectPosition("camera", metadata[0].cam_pos)
                s.setObjectAttitude("camera", metadata[0].cam_att)
                s.render()
                hapke = s.getImage()[...,0]
                cv2.imwrite(str(Path(output_dir) /
                    f'hapke_{Path(metadata[0].texture_file).stem}_{metadata[0].center_xy[0]:05}_{metadata[0].center_xy[1]:05}.tif'), hapke)

        # generate gt
        for _, _, metadata in dataloader:
            with init_surrender(
                dem_file=dem_file,
                texture_file=metadata[0].texture_file,
                resource_path=resource_path,
                image_size=image_size,
                channels=1,
                rays=128,
                sun_power=sun_power,
                serverhost=serverhost,
                serverport=serverport,
            ) as s:
                s.setCameraFOVDeg(metadata[0].fov, metadata[0].fov)
                s.setObjectPosition("sun", metadata[0].sun_pos)
                s.setObjectPosition("camera", metadata[0].cam_pos)
                s.setObjectAttitude("camera", metadata[0].cam_att)
                s.render()
                gt = s.getImage()[...,0]
                cv2.imwrite(str(Path(output_dir) /
                    f'gt128_{Path(metadata[0].texture_file).stem}_{metadata[0].center_xy[0]:05}_{metadata[0].center_xy[1]:05}.tif'), gt)

    #####################################################################
    # 2) GENERATE LEARNED IMAGES IF NEEDED
    #####################################################################
    if regen_model:
        model = BrdfGenerator(
            dem_file=dem_file,
            resource_path=resource_path,
            gsd=gsd,
            texture_diff=texture_diff,
            image_size=image_size,
            brdf_file=brdf_name,
            rays=128,
            out_c=channels,
            serverhost=serverhost,
            serverport=serverport
        )
        model.to(device)
        model.eval()

        weights_file = Path(training_dir) / 'best-model-parameters-val_loss.pt'
        model.load_state_dict(torch.load(weights_file))

        for dem, render, metadata in tqdm(dataloader):
            dem = dem.to(device)
            _ = model(dem, render, metadata)
            img = model.render.rendered

            cv2.imwrite(str(Path(output_dir) /
                f'learnt_{Path(metadata[0].texture_file).stem}_{metadata[0].center_xy[0]:05}_{metadata[0].center_xy[1]:05}.tif'),
                img[:,:,0])

            eval_plot(model, (dem, render, metadata), img, channels,
                      texture_diff, str(Path(output_dir) / 'plot'))

        model.close()

    #####################################################################
    # 3) LOAD FILES AND COMPARE
    #####################################################################
    gts = list(Path(output_dir).glob('gt128_*.tif'))
    if len(gts) == 0:
        raise Exception("Run first with --regen-gt and --regen-model")
    
    # Initialize LPIPS if available
    lpips_fn = None
    if lpips is not None:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    skipped_count = 0
    for gt_file in tqdm(gts):

        image_name = removeprefix(gt_file.name, 'gt128_')
        gt = cv2.imread(str(gt_file), cv2.IMREAD_UNCHANGED)
        hapke = cv2.imread(str(Path(output_dir)/f"hapke_{image_name}"), cv2.IMREAD_UNCHANGED)
        learnt = cv2.imread(str(Path(output_dir)/f"learnt_{image_name}"), cv2.IMREAD_UNCHANGED)

        # Skip if any image is missing or invalid
        if gt is None or hapke is None or learnt is None:
            print(f"[WARNING] Skipping {image_name}: missing file(s)")
            skipped_count += 1
            continue

        # Check for invalid values (NaN, Inf)
        if not np.isfinite(gt).all() or not np.isfinite(hapke).all() or not np.isfinite(learnt).all():
            print(f"[WARNING] Skipping {image_name}: contains NaN or Inf")
            skipped_count += 1
            continue

        hapke_raw = hapke.copy()

        # Check for zero range (constant image)
        gt_range = gt.max() - gt.min()
        hapke_range = np.max(hapke_raw) - np.min(hapke_raw)
        
        if gt_range < 1e-10 or hapke_range < 1e-10:
            print(f"[WARNING] Skipping {image_name}: constant image (gt_range={gt_range}, hapke_range={hapke_range})")
            skipped_count += 1
            continue

        print("GT min/max/mean:", np.min(gt), np.max(gt), np.mean(gt))
        print("Hapke RAW min/max/mean:", np.min(hapke_raw), np.max(hapke_raw), np.mean(hapke_raw))

        # Normalisation hapke pour display ONLY
        minv, maxv = np.min(gt), np.max(gt)
        aa = (maxv - minv) / hapke_range
        bb = - np.mean(hapke_raw) * aa + np.mean(gt)
        hapke_norm = hapke_raw * aa + bb
        print("Hapke NORM min/max/mean:", np.min(hapke_norm), np.max(hapke_norm), np.mean(hapke_norm))

        #################################################################
        # METRICS RAW HAPKE
        #################################################################
        rmse_h_raw.append(np.sqrt(np.mean((hapke_raw - gt) ** 2)))
        mse_h_hapke.append(np.sum((hapke_raw - gt)**2))
        psnr_h_raw.append(peak_signal_noise_ratio(gt, hapke_raw,
            data_range=gt_range))
        ssim_h_raw.append(structural_similarity(gt, hapke_raw,
            data_range=gt_range))

        #################################################################
        # METRICS NORMALIZED HAPKE
        #################################################################
        rmse_h_norm.append(np.sqrt(np.mean((hapke_norm - gt) ** 2)))
        mse_h_norm.append(np.sum((hapke_norm - gt)**2))

        psnr_h_norm.append(peak_signal_noise_ratio(gt, hapke_norm,
            data_range=gt_range))
        ssim_h_norm.append(structural_similarity(gt, hapke_norm,
            data_range=gt_range))

        #################################################################
        # METRICS LEARNED
        #################################################################
        rmse_l.append(np.sqrt(np.mean((learnt - gt) ** 2)))
        mse_l.append(np.sum((learnt - gt)**2))
        psnr_l.append(peak_signal_noise_ratio(gt, learnt,
            data_range=gt_range))
        ssim_l.append(structural_similarity(gt, learnt,
            data_range=gt_range))
        
        #################################################################
        # LPIPS METRICS
        #################################################################
        if lpips_fn is not None:
            # Prepare images for LPIPS (normalize to [-1, 1] and add batch/channel dims)
            gt_lpips = torch.from_numpy(gt / (gt.max() + 1e-8)).float().to(device)
            hapke_raw_lpips = torch.from_numpy(hapke_raw / (hapke_raw.max() + 1e-8)).float().to(device)
            hapke_norm_lpips = torch.from_numpy(hapke_norm / (hapke_norm.max() + 1e-8)).float().to(device)
            learnt_lpips = torch.from_numpy(learnt / (learnt.max() + 1e-8)).float().to(device)
            
            # Add batch and channel dimensions (B, C, H, W)
            gt_lpips = gt_lpips.unsqueeze(0).unsqueeze(0)
            hapke_raw_lpips = hapke_raw_lpips.unsqueeze(0).unsqueeze(0)
            hapke_norm_lpips = hapke_norm_lpips.unsqueeze(0).unsqueeze(0)
            learnt_lpips = learnt_lpips.unsqueeze(0).unsqueeze(0)
            
            # Expand to 3 channels for AlexNet
            gt_lpips = gt_lpips.expand(-1, 3, -1, -1)
            hapke_raw_lpips = hapke_raw_lpips.expand(-1, 3, -1, -1)
            hapke_norm_lpips = hapke_norm_lpips.expand(-1, 3, -1, -1)
            learnt_lpips = learnt_lpips.expand(-1, 3, -1, -1)
            
            lpips_h_raw.append(lpips_fn(hapke_raw_lpips, gt_lpips).item())
            lpips_h_norm.append(lpips_fn(hapke_norm_lpips, gt_lpips).item())
            lpips_l.append(lpips_fn(learnt_lpips, gt_lpips).item())
        #################################################################
        # PLOTS COMPARATIFS
        #################################################################
        # Plot 1: Comparaison simple (clim 0-0.2)
        fig, ax = plt.subplots(3, 1)
        from matplotlib.figure import Figure
        fig: Figure
        dpi = 100
        width, height = 1920, 1080
        fig.set_size_inches(width / dpi, height / dpi, forward=False)
        fig.suptitle(Path(image_name).stem)
        #enregitsrer hapke

        im = ax[1].imshow(learnt, cmap="gray")
        im.set_clim(0, 0.2)
        ax[1].axis('off')

        im = ax[2].imshow(gt, cmap='gray')
        im.set_clim(0, 0.2)
        ax[2].axis('off')

        im = ax[0].imshow(hapke_norm, cmap='gray')
        im.set_clim(0, 0.2)
        ax[0].axis('off')

        (Path(output_dir) / 'hapke').mkdir(exist_ok=True)
        fig.savefig(Path(output_dir) / f'hapke/{Path(image_name).stem}.png', dpi=dpi)
        plt.close(fig)
        
        # Sauvegarder les images float32 en TIF (valeurs précises)
        tif_dir = Path(output_dir) / 'tif_float32'
        tif_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(tif_dir / f'gt_{Path(image_name).stem}.tif'), gt.astype(np.float32))
        cv2.imwrite(str(tif_dir / f'hapke_norm_{Path(image_name).stem}.tif'), hapke_norm.astype(np.float32))
        cv2.imwrite(str(tif_dir / f'learnt_{Path(image_name).stem}.tif'), learnt.astype(np.float32))

        # Plot 2: HIGH CONTRAST
        fig, ax = plt.subplots(3, 1)
        fig: Figure
        dpi = 100
        width, height = 1920, 1080
        fig.set_size_inches(width / dpi, height / dpi, forward=False)
        fig.suptitle(Path(image_name).stem)

        luminosity = -20
        contrast = 50
        minv, maxv = np.min((gt, hapke_norm, learnt)), np.max((gt, hapke_norm, learnt))
        
        def adjust(a):
            a = 255 * (a - minv) / (maxv - minv)
            a = np.clip((100 + contrast) / 100 * a + 255 * luminosity / 100, 0, 255)
            return a

        im = ax[2].imshow(adjust(gt), cmap='gray')
        im.set_clim(0, 255)
        ax[2].axis('off')
        
        im = ax[0].imshow(adjust(hapke_norm), cmap='gray')
        im.set_clim(0, 255)
        ax[0].axis('off')
        
        im = ax[1].imshow(adjust(learnt), cmap="gray")
        im.set_clim(0, 255)
        ax[1].axis('off')

        (Path(output_dir) / 'hapke_high_contrast').mkdir(exist_ok=True)
        fig.savefig(Path(output_dir) / f'hapke_high_contrast/{Path(image_name).stem}.png', dpi=dpi)
        plt.close(fig)

    #####################################################################
    # 4) FINAL STATS (UNE SEULE FOIS)
    #####################################################################
    if len(rmse_l) == 0:
        print("[ERROR] Aucun échantillon valide pour calculer les métriques!")
        print(f"[INFO] Total skipped: {skipped_count} / {len(gts)}")
        return

    print(f"\n[INFO] Processed: {len(rmse_l)} / {len(gts)} samples (skipped: {skipped_count})")
    
    final_stats = []

    final_stats.append("==== RAW Hapke (NO NORMALIZATION) ====")
    final_stats.append(f"RMSE Hapke RAW mean: {np.mean(rmse_h_raw):.6f}")
    final_stats.append(f"PSNR Hapke RAW mean: {np.mean(psnr_h_raw):.4f}")
    final_stats.append(f"SSIM Hapke RAW mean: {np.mean(ssim_h_raw):.4f}\n")
    final_stats.append(f"MSE Hapke RAW mean: {np.mean(mse_h_hapke):.6f}")

    final_stats.append("==== NORMALIZED Hapke (DISPLAY ONLY) ====")
    final_stats.append(f"RMSE Hapke NORM mean: {np.mean(rmse_h_norm):.6f}")
    final_stats.append(f"PSNR Hapke NORM mean: {np.mean(psnr_h_norm):.4f}")
    final_stats.append(f"SSIM Hapke NORM mean: {np.mean(ssim_h_norm):.4f}\n")
    final_stats.append(f"MSE Hapke NORM mean: {np.mean(mse_h_norm):.6f}")

    final_stats.append("==== LEARNED BRDF ====")
    final_stats.append(f"RMSE Learned mean: {np.mean(rmse_l):.6f}")
    final_stats.append(f"PSNR Learned mean: {np.mean(psnr_l):.4f}")
    final_stats.append(f"SSIM Learned mean: {np.mean(ssim_l):.4f}")
    final_stats.append(f"MSE Learned mean: {np.mean(mse_l):.6f}")
    
    if len(lpips_h_raw) > 0:
        final_stats.append("\n==== LPIPS METRICS ====")
        final_stats.append(f"LPIPS Hapke RAW mean: {np.mean(lpips_h_raw):.6f}")
        final_stats.append(f"LPIPS Hapke NORM mean: {np.mean(lpips_h_norm):.6f}")
        final_stats.append(f"LPIPS Learned mean: {np.mean(lpips_l):.6f}")

    with open(Path(output_dir) / "summary_stats.txt", "w") as f:
        f.write("\n".join(final_stats))

    print("\n".join(final_stats))

def eval_plot(model: nn.Module,
              inputs,
              outputs,
              channels: int,
              texture_diff: str,
              output_dir: str = ''):

    dem, render, metadata = inputs

    for c in range(channels):
        fig, ax = plt.subplots(2, 3)
        from matplotlib.figure import Figure
        fig: Figure
    #    plt.get_current_fig_manager().full_screen_toggle()
        dpi = 100  # DPI (dots per inch) pour la résolution
        width, height = 1920, 1080  # Résolution de l'écran en pixels
        fig.set_size_inches(width / dpi, height / dpi, forward=False)
        x, y = metadata[0].center_xy
        texture_file = Path(metadata[0].texture_file).stem
        fig.suptitle(f'{texture_file} {x=} {y=} Parameter #{c}')

        img = outputs
        im = ax[0,0].imshow(img[:,:,0], cmap = "gray")
        ax[0,0].set_title("rendered (SurRender)")
#        im.set_clim(0, 1)
        fig.colorbar(im)

        gt = render.permute((0,2,3,1)).numpy()
        im = ax[1,0].imshow(gt[0,:,:,0], cmap = "gray")
        ax[1,0].set_title("ground truth (SurRender)")
#        im.set_clim(0, 1)
        fig.colorbar(im)

        im = ax[1,1].imshow((img[:,:,0] - gt[0,:,:,0]))
        ax[1,1].set_title("error")
        fig.colorbar(im)

        dem_input = dem.permute((0,2,3,1)).cpu().detach().numpy()
        im = ax[0,1].imshow(dem_input[0,:,:,:], cmap = "gray")
        ax[0,1].set_title(f'input (DEM)')
        fig.colorbar(im)

        tex = model.render.texture_fp
    #    im = ax[1,1].imshow(tex[10:-10,10:-10,:channels])
    #    ax[1,1].set_title(f'learnt texture ZOOMED (surrender input)')
        im = ax[0,2].imshow(tex[...,c], cmap = 'gray')
        ax[0,2].set_title(f'BRDF parameter #{c}')
        #im.set_clim(0, 1)
        fig.colorbar(im)

        grad = model.render.s.getTextureGradient(texture_diff)
        im = ax[1,2].imshow(grad[:,:,c])
        ax[1,2].set_title(f'gradient (parameter #{c})')
        fig.colorbar(im)

        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)
            fig.savefig(Path(output_dir) / f'{texture_file}_{x:05}_{y:05}_p{c}.png', dpi=dpi)

    if not output_dir:
        plt.show()
    plt.close('all')


def get_parser():
    parser = ArgumentParser(description='Evaluate')
    parser.add_argument('-i', '--input', type=str, help="Training directory input", required=True)
    parser.add_argument('-n', '--nb', type=int, help="Number of samples of validation dataset for plot generation", default=30)
    parser.add_argument('--all', help="Evaluate on the entire dataset (ignores -n)", action='store_true')
    parser.add_argument('-ds', '--dataset', help="Generate plots for training dataset ('train'), validation dataset ('val'), test dataset ('test', dft) or custom dataset directory", default='test')
    parser.add_argument('--regen-gt', help="Regenerate hapke image and ground truths with 128 rays", action='store_true')
    parser.add_argument('--regen-model', help="Regenerate image predicted with model with 128 rays", action='store_true')
    parser.add_argument('--overwrite', help="Overwrite eval directory", action='store_true')
    parser.add_argument('--save-debug', type=int, default=0, help="Save N first debug images comparing GT/Hapke/Learned (default: 0 = disabled)")
    parser.add_argument('--debug', help="Activate debug logs", action='store_true')
    return parser


def main(args):
    # check GPU availability
    print("helloworld")
    if not torch.cuda.is_available():
        logger.error('GPU not available')
        exit(1)
    device = torch.device("cuda:0")
    print("hello")

    training_dir = args.input

    # get data from training metadata
    with open(Path(training_dir) / 'training_metadata.json', 'r') as f:
        from brdfgen.train import TrainingMetadata, custom_collate
        train_metadata = TrainingMetadata.from_json(f.read())

    texture_diff = "blablabla" # pas besoin ?!!!!TODO
    print("hello")

    # log settings
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger('PIL').setLevel(level=logging.WARNING)
    logging.getLogger('matplotlib').setLevel(level=logging.WARNING)
    logging.getLogger('rasterio').setLevel(level=logging.WARNING)
    print("hello")

    # init RNGs
    seed = 123456789123456789
    rng = np.random.default_rng(seed)
    gen_cpu = torch.Generator().manual_seed(seed)
    gen_cuda = torch.Generator(device).manual_seed(seed)
    print("hello")

    if args.dataset == 'train':
        
        print("hello")

        dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=f'{training_dir}/train_dataset.csv')
        eval_dir = Path(training_dir) / 'eval' / 'train'
    elif args.dataset == 'val':
        print("hello")

        dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=f'{training_dir}/val_dataset.csv')
        eval_dir = Path(training_dir) / 'eval' / 'val'
    elif args.dataset == 'test':
        print("hello")

        dataset = BrdfGeneratorDataset(train_metadata.dataset_dir, train_metadata.channels, subset_file=f'{training_dir}/tst_dataset.csv')
        print("hellbo")

        eval_dir = Path(training_dir) / 'eval' / 'test'
    else:
        dataset = BrdfGeneratorDataset(args.dataset, train_metadata.channels)
        eval_dir = Path(training_dir) / 'eval' / Path(args.dataset).name
    print("hello")

    # eval output dir
    eval_dir.mkdir(exist_ok=args.overwrite, parents=True)

    ds_metadata = dataset.metadata
    print("hello")

    # Charger le modèle pour l'évaluation
    model = BrdfGenerator(
        dem_file=ds_metadata.dem_file,
        resource_path=ds_metadata.resource_path,
        gsd=float(ds_metadata.gsd),
        texture_diff=texture_diff,
        image_size=ds_metadata.crop_size[0],
        brdf_file=train_metadata.brdf_name,
        rays=128,
        out_c=train_metadata.channels,
        serverhost='localhost',
        serverport=5151
    )
    weights_file = Path(training_dir) / 'best-model-parameters-val_loss.pt'
    print(weights_file)
    model.load_state_dict(torch.load(weights_file))

    # Créer le dataloader pour l'évaluation
    if args.all:
        print(f"[INFO] Mode --all: évaluation sur tout le dataset ({len(dataset)} échantillons)")
        eval_dataloader = DataLoader(
            dataset,
            generator=gen_cpu,
            shuffle=False,
            collate_fn=custom_collate
        )
    else:
        print(f"[INFO] Évaluation sur {args.nb} échantillons aléatoires")
        eval_dataloader = DataLoader(
            Subset(dataset, rng.integers(len(dataset), size=args.nb)),
            generator=gen_cpu,
            shuffle=True,
            collate_fn=custom_collate
        )

    # Appeler la fonction d'évaluation pour générer et sauvegarder les stats
    evaluation(
        device=device,
        model=model,
        dataloader=eval_dataloader,
        channels=train_metadata.channels,
        texture_diff=texture_diff,
        has_rot=False,
        has_regul=False,
        rot_coef=0.0,
        regul_coef=0.0,
        debug_show=0,
        compute_hapke=True,
        dem_file=ds_metadata.dem_file,
        resource_path=ds_metadata.resource_path,
        save_debug_images=args.save_debug,
        eval_dir=str(eval_dir),
        sun_power=float(ds_metadata.sun_power),
    )

    # Générer les images et plots seulement si demandé
    if args.regen_gt or args.regen_model:
        compare_with_hapke(
            device=device,
            training_dir=training_dir,
            dataloader=eval_dataloader,
            ds_metadata=ds_metadata,
            channels=train_metadata.channels,
            texture_diff=texture_diff,
            output_dir=str(eval_dir.resolve()),
            brdf_name=train_metadata.brdf_name,
            dem_file=ds_metadata.dem_file,
            resource_path=ds_metadata.resource_path,
            serverport=5151,
            regen_gt=args.regen_gt,
            regen_model=args.regen_model,
        )
    else:
        print("[INFO] Pas de génération d'images (utiliser --regen-gt et/ou --regen-model si besoin)")


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
