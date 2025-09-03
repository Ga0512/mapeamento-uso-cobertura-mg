import asyncio
import matplotlib.pyplot as plt
import numpy as np
from src.models import SegmentationModels, ClassificationModels

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

def main(image_path_seg: str, image_path_class: str):
    segmentation_results = asyncio.run(run_segmentation(image_path_seg))
    for name, mask in segmentation_results.items():
        print(f"{name}: {type(mask)}, {mask.shape}")

    plot_masks(segmentation_results)

    vit_results = asyncio.run(run_classification(image_path_class, labels=list(range(10))))
    print(vit_results)



if __name__ == "__main__":
    # seg, class
    main("dataset/Amostra_piloto_1_3.tif", "dataset/Amostra_1_fes.tif")
