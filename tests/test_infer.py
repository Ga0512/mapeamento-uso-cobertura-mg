from utils.sys_env import set_env

set_env()

import warnings
warnings.filterwarnings("ignore")

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiclean import clean_array
from src.models.models import SegmentationModels, ClassificationModels


# === Colormap padrão ===
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3"
]
CMAP = ListedColormap(COLORS)


def plot_masks(results: dict, num_classes: int = 16):
    """
    Plota máscaras de segmentação com colormap discreto.
    
    Args:
        results (dict): {nome_modelo: mask_array}
        num_classes (int): número de classes para colormap
    """
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, mask) in zip(axes, results.items()):
        im = ax.imshow(mask, cmap=CMAP, vmin=0, vmax=num_classes - 1)
        ax.set_title(name, fontsize=12)
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_ticks(np.arange(num_classes))
    cbar.set_label("Classes (0 a 15)")

    plt.tight_layout()
    plt.show()


async def run_segmentation(path: str) -> dict:
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


async def run_classification(path: str, labels: list):
    """
    Executa classificação usando o modelo ViT.
    
    Args:
        path (str): caminho da imagem
        labels (list): lista de classes
    
    Returns:
        dict: resultados do modelo ViT
    """
    classification = ClassificationModels(path, labels=labels)
    vit_result = classification.vit.predict()
    return vit_result


def clean_and_plot(array: np.ndarray):
    """
    Limpa e plota a máscara antes e depois da limpeza.
    
    Args:
        array (np.ndarray): máscara de segmentação
    """
    cleaned = clean_array(
        array,
        class_values=list(range(16)),
        smooth_edge_size=2,
        min_island_size=100,
        connectivity=8,
        max_workers=4,
        fill_nan=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axes[0].imshow(array, cmap=CMAP, vmin=0, vmax=15)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im1 = axes[1].imshow(cleaned, cmap=CMAP, vmin=0, vmax=15)
    axes[1].set_title("Cleaned")
    axes[1].axis("off")

    cbar = fig.colorbar(im1, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Classes (0 a 15)")

    plt.tight_layout()
    plt.show()
    return cleaned


def main(image_path_seg: str, image_path_class: str):
    # Rodar segmentação
    segmentation_results = asyncio.run(run_segmentation(image_path_seg))
    for name, mask in segmentation_results.items():
        print(f"{name}: {type(mask)}, {mask.shape}")

    # Plotar todas as máscaras com colormap fixo
    plot_masks(segmentation_results)

    # Limpar e plotar o resultado do segformer
    segformer_mask = segmentation_results["segformer"]
    cleaned = clean_and_plot(segformer_mask)

    # Rodar classificação
    vit_results = asyncio.run(run_classification(image_path_class, labels=list(range(10))))
    print("Resultados da classificação:", vit_results)


if __name__ == "__main__":
    main("dataset/Amostra_piloto_1_3.tif", "dataset/Amostra_1_fes.tif")
