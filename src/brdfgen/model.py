from brdfgen.render import init_surrender, render_image
from brdfgen.data import  get_resource_absolute_path, Proj, get_texture_scale, SampleMetadata
import brdfgen.config as config

from typing import List, Union, Tuple, Dict
import logging

import rasterio as rio
import torch
from pathlib import Path
from torch import nn
from surrender.surrender_client import surrender_client
import numpy as np


logger = logging.getLogger(__name__)


class BrdfGenerator(nn.Module):
    def __init__(self,
                 dem_file: str,
                 resource_path: Union[str, List[str]],
                 gsd: float,
                 texture_diff: Union[str, List[str]] = "tile_init.tif",
                 in_c = 1,
                 out_c = 1,
                 image_size: int = 128,
                 brdf_file: str = "hapke_diff.brdf",
                 rays: int = 2,
                 grad_alpha: float = 0.5,
                 serverhost: str = 'localhost',
                 serverport: int = 5151,
                 ):
        super().__init__()
        self.unet = UNet(in_c, out_c)
        self.render = RenderLayer(
            dem_file=dem_file,
            resource_path=resource_path,
            image_size=image_size,
            brdf_file=brdf_file,
            grad_alpha=grad_alpha,
            texture_diff=texture_diff,
            channels=out_c,
            rays=rays,
            gsd=gsd,
            serverhost=serverhost,
            serverport=serverport,
        )

    def forward(self,
                x: torch.Tensor,
                gt: torch.Tensor,
                metadata: List[SampleMetadata],
                ) -> torch.Tensor:
        x = self.unet(x)
        x = self.render(x, gt, metadata)
        return x

    def close(self):
        self.render.s.close()


class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down1 = UNetDownLayer(in_c, 16, 0.1)
        self.down2 = UNetDownLayer(16, 32, 0.1)
        self.down3 = UNetDownLayer(32, 64, 0.2)
        self.down4 = UNetDownLayer(64, 128, 0.2)
        self.midle = UNetConvLayer(128, 256, 0.3)
#        self.down5 = UNetDownLayer(128, 256, 0.3)
#        self.midle = UNetConvLayer(256, 512, 0.4)
#        self.up0   = UNetUpLayer(512, 256, 0.3)
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
#        t5, x = self.down5(x)
        x = self.midle(x)
#        x = self.up0(x, t5)
        x = self.up1(x, t4)
        x = self.up2(x, t3)
        x = self.up3(x, t2)
        x = self.up4(x, t1)
        return self.final(x)


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


class RenderLayer(nn.Module):
    def __init__(self,
                 dem_file: str,
                 resource_path: Union[str, List[str]],
                 image_size: int,
                 brdf_file: str,
                 grad_alpha: float,
                 texture_diff: Union[str, List[str]],
                 channels: int,
                 gsd: float,
                 rays: int,
                 serverhost: str,
                 serverport: int,
                 ):
        super().__init__()
        self.s = init_surrender(
            dem_file=dem_file,
            resource_path=resource_path,
            image_size=image_size,
            channels=channels,
            texture_diff=texture_diff,
            brdf_file=brdf_file,
            rays=rays,
            for_backprop=True,
            serverhost=serverhost,
            serverport=serverport
        )
        self.fn = RenderFunction.apply
        self.image_size = image_size
        self.grad_alpha = grad_alpha
        self.texture_diff = texture_diff
        self.gsd = gsd
        self.resource_path = resource_path
        self.texture_fp = self.s.mapTextureAsNumpyArray(texture_diff)
        self.rendered = self.s.mapTextureAsNumpyArray("")

    def forward(self,
                x: torch.Tensor,
                gt: torch.Tensor,
                metadata: List[SampleMetadata],
                ):
        return self.fn(x,
                       self.s,
                       self.image_size,
                       self.grad_alpha,
                       self.texture_diff,
                       gt,
                       metadata,
                       self.gsd,
                       self.resource_path,
                       self.texture_fp,
                       self.rendered)


grad_sigma_last = None

class RenderFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                s: surrender_client,
                image_size: int,
                grad_alpha: float,
                texture_diff: str,
                gt: np.ndarray,
                metadata: List[SampleMetadata],
                gsd: float,
                resource_path: Union[str, List[str]],
                texture_fp: np.ndarray,
                rendered: np.ndarray,
                ):

        b, c, h, w = x.shape
        textures = np.zeros((b, image_size, image_size, 4), dtype=np.double)
        grads  = np.zeros((b, h, w, c), dtype=np.double)
        losses = np.zeros((b,), dtype=np.double)
        texture = x.permute(0,2,3,1).cpu().numpy()
        gt = gt.permute(0,2,3,1).cpu().numpy()

        # for every sample in batch
        for i, md in enumerate(metadata):

            texture_abspath = get_resource_absolute_path(md.texture_file, resource_path)
            texture_scale = get_texture_scale(f'textures/{Path(texture_abspath).name}', resource_path)
            texture_scale_factor = gsd / texture_scale

            textures[i,...], grads[i,...], losses[i,...] = render_image(s=s,
                                                                      texture=texture[i,...],
                                                                      texture_diff=texture_diff,
                                                                      gt=gt[i,...],
                                                                      metadata=md,
                                                                      crop_size=image_size,
                                                                      texture_scale_factor=texture_scale_factor,
                                                                      texture_fp=texture_fp,
                                                                      rendered=rendered,
                                                                      display=False,
                                                                      )
        # save gradient for backward
        if np.max(grads) == np.min(grads):
            logger.error("constant gradient !")
        global grad_sigma_last
        if grad_sigma_last:
            grad_sigma = grad_sigma_last * (1 - grad_alpha) + grad_alpha * np.nanstd(grads)
        else:
            grad_sigma = np.nanstd(grads)
        grad_sigma_last = grad_sigma
        logger.debug(f'Batch gradients divided by {grad_sigma}, losses: {losses}')
        ctx.save_for_backward(torch.from_numpy(grads/grad_sigma).permute(0,3,1,2).to(x.get_device()))
        config.batch_loss = np.mean(losses)
        return torch.from_numpy(textures).permute(0,3,1,2).to(x.get_device())

    @staticmethod
    def backward(ctx, grad_output):
        # ignore grad_output and return gradient saved in forward
        grads, = ctx.saved_tensors
        return grads, None, None, None, None, None, None, None, None, None, None