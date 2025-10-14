import numpy as np
from multiclean import clean_array

import asyncio
import matplotlib.pyplot as plt
import numpy as np
from src.models import SegmentationModels, ClassificationModels
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



def plot_masks(results: dict, num_classes: int = 16):
    """
    Plota máscaras de segmentação com colormap discreto.
    
    Args:
        results (dict): {nome_modelo: mask_array}
        num_classes (int): número de classes para colormap
    """
    cmap = plt.cm.get_cmap("tab20", num_classes)

    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, mask) in zip(axes, results.items()):
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
        ax.set_title(name, fontsize=12)
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_label("Classes")

    plt.tight_layout()
    plt.show()


def run_segmentation(path: str) -> dict:
    """
    Executa segmentação de imagem usando diferentes modelos.
    
    Args:
        path (str): caminho da imagem
    
    Returns:
        dict: {nome_modelo: mask_array}
    """
    segmentation = SegmentationModels(img_path=path)

    segformer = segmentation.segformer.predict()
    deeplab = segmentation.deeplab.predict()
    unet = segmentation.unet.predict()

    return {
        "segformer": segformer,
        "deeplab": deeplab,
        "unet": unet
    }


masks = run_segmentation("dataset/Amostra_piloto_1_3.tif")
array = masks["segformer"]


# Clean with default parameters
cleaned = clean_array(array)

# Custom parameters
cleaned = clean_array(
    array,
    class_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    smooth_edge_size=2,
    min_island_size=100,
    connectivity=8,
    max_workers=4,
    fill_nan=True   # substitui NaN automaticamente
)


# Criar um colormap com até 16 cores distintas
colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3"
]
cmap = ListedColormap(colors)

# Plotando lado a lado
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Array original
im0 = axes[0].imshow(array, cmap=cmap, vmin=0, vmax=15)
axes[0].set_title("Original")
axes[0].axis("off")

# Array cleaned
im1 = axes[1].imshow(cleaned, cmap=cmap, vmin=0, vmax=15)
axes[1].set_title("Cleaned")
axes[1].axis("off")

# Barra de cores
cbar = fig.colorbar(im1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
cbar.set_label("Classes (0 a 15)")

plt.tight_layout()
plt.show()