import matplotlib.pyplot as plt
import numpy as np

# for python < 3.9
def removeprefix(txt, prefix):
    return txt[txt.startswith(prefix) and len(prefix):]

def debug_plot(
        title: str = "",
        dem: np.ndarray = np.zeros((1,1)),
        gt: np.ndarray = np.zeros((1,1)),
        img: np.ndarray = np.zeros((1,1)),
        tex: np.ndarray = np.zeros((1,1)),
        grad: np.ndarray = np.zeros((1,1)),
    ):
    fig, ax = plt.subplots(2, 3)
    fig.suptitle(title)

    im = ax[0,0].imshow(gt, cmap = "gray")
    ax[0,0].set_title("ground truth (surrender)")
#    im.set_clim(0, 1)
    fig.colorbar(im)

    im = ax[0,1].imshow(img, cmap = "gray")
    ax[0,1].set_title("image (surrender)")
#    im.set_clim(0, 1)
    fig.colorbar(im)

    im = ax[0,2].imshow(img - gt)
    ax[0,2].set_title("error (image - gt)")
    fig.colorbar(im)

    im = ax[1,0].imshow(dem, cmap = "gray")
    ax[1,0].set_title(f'input (DEM)')
    fig.colorbar(im)

    im = ax[1,1].imshow(tex[10:-10,10:-10,...])
    ax[1,1].set_title(f'learnt texture ZOOMED (surrender input)')
#    im.set_clim(0, 1)
    fig.colorbar(im)

    im = ax[1,2].imshow(grad)
    ax[1,2].set_title(f'computed gradient in surrender')
    fig.colorbar(im)
    #fig.savefig(title)
