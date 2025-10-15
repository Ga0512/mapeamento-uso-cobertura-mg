**documentação desatualizada**

# Análise e Classificação de Imagens de Sensoriamento Remoto
Este projeto oferece um conjunto de ferramentas para análise e classificação de imagens de sensoriamento remoto, com foco em modelos de segmentação semântica e classificação. Ele foi projetado para ser modular e flexível, permitindo o treinamento e a inferência com diferentes arquiteturas de modelos, como U-Net, DeepLab e SegFormer.

### Instalação e Preparação do Ambiente

Para começar, você precisa clonar o repositório e instalar as bibliotecas necessárias.

1.  **Clone o Repositório:**

    ```bash
    git clone https://github.com/ge21gt-repo/mapeamento-uso-cobertura-mg
    cd mapeamento-uso-cobertura-mg
    ```

2.  **Instale as Dependências:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare os Modelos:**

    Os modelos pré-treinados estão em um Google Drive. Baixe-os e organize-os na pasta `./models/` com a seguinte estrutura:

    ```
    +-- models/
    |   +-- deeplab/
    |   +-- segformer/
    |   +-- unet/
    |   +-- tree_torch/
    |       +-- rf_torch.pth
    ```

-----

### Treinamento de Modelos

O treinamento de todos os modelos é feito usando o script `train/train.py`. Você pode especificar o modelo desejado e os caminhos para seus dados de treinamento.

#### Modelos de Segmentação (Deep Learning)

Estes modelos são treinados com imagens **TIF de 128x128 pixels**.

  - **U-Net**

    ```python
    from train.train import unet

    if __name__ == "__main__":
        unet(
            image_dir="dataset/imagens_satelite_128x128",
            mask_dir="dataset/mascaras_128x128",
            num_epochs=100,
            batch_size=16
        ).train()
    ```

  - **SegFormer**

    ```python
    from train.train import segformer

    if __name__ == "__main__":
        segformer(
            image_dir="dataset/imagens_satelite_128x128",
            mask_dir="dataset/mascaras_128x128",
            num_epochs=100,
            batch_size=8
        ).train()
    ```

  - **DeepLab**

    ```python
    from train.train import deeplab

    if __name__ == "__main__":
        deeplab(
            image_dir="dataset/imagens_satelite_128x128",
            mask_dir="dataset/mascaras_128x128",
            num_epochs=100,
            batch_size=16
        ).train()
    ```

#### Modelo de Segmentação (Random Forest)

Este modelo é treinado com amostras extraídas de imagens grandes.

```python
from tree_torch.module import ParallelRandomForest
import torch
import numpy as np

# Exemplo com dados aleatórios.
# Substitua por um script que extraia dados de treinamento do seu raster.
X_train = np.random.rand(1000, 12)
y_train = np.random.randint(0, 16, size=1000)

rf = ParallelRandomForest(
    n_trees=500,
    n_classes=16,
    max_depth=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

rf.fit(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
torch.save(rf, "models/tree_torch/rf_torch.pth")
```

-----

### Inferência (Classificação e Segmentação)

A inferência utiliza os modelos que você baixou ou treinou.

#### Segmentação (Deep Learning) e Classificação (ViT)

A inferência para os modelos **U-Net**, **SegFormer**, **DeepLab** e **ViT** é executada pelo script `main.py`, que já está configurado para usar os arquivos de exemplo. O script executa a inferência para todos os modelos de segmentação e também para o modelo de classificação ViT.

**Como executar:**

```bash
python main.py
```

**Trecho de código de exemplo:**

No arquivo `main.py`, a inferência é encapsulada em funções assíncronas.

```python
# main.py

import asyncio
from src.models import SegmentationModels, ClassificationModels

async def run_segmentation(path: str) -> dict:
    """Executa segmentação de imagem com diferentes modelos."""
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
    """Executa a classificação com o modelo ViT."""
    classification = ClassificationModels(path, labels=labels)
    vit_result = classification.vit.predict()
    return vit_result

def main(image_path_seg: str, image_path_class: str):
    # Executa a segmentação
    segmentation_results = asyncio.run(run_segmentation(image_path_seg))
    print("Resultados da Segmentação:", segmentation_results.keys())

    # Executa a classificação
    vit_results = asyncio.run(run_classification(image_path_class, labels=list(range(10))))
    print("Resultados da Classificação:", vit_results)

if __name__ == "__main__":
    main("dataset/Amostra_piloto_1_3.tif", "dataset/Amostra_1_fes.tif")
```

#### Segmentação (Random Forest)

Para segmentar imagens grandes usando o Random Forest, utilize o script `tree_torch/inference_tiles.py`. Ele processa a imagem em blocos, otimizando o uso de memória.

**Como executar:**

```bash
python tree_torch/inference_tiles.py
```

**Trecho de código de exemplo:**

O script carrega o modelo, a imagem e processa a classificação em blocos.

```python
# tree_torch/inference_tiles.py

import torch
from osgeo import gdal
from tree_torch.module import ParallelRandomForest

# --- Parâmetros ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "models/tree_torch/rf_torch.pth"
img_RS = r"caminho/para/sua/imagem/grande.tif"
classification_image = "./resultado_rf.tif"
tile_size = 1024

# Carrega o modelo
rf = torch.load(model_path, weights_only=False, map_location=device)
rf.eval()

# Carrega a imagem e processa em blocos
img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)
# ... (código para processamento por blocos e escrita no arquivo de saída)
print("Classificação salva em:", classification_image)

# ... (código para avaliação opcional)
```

Antes de rodar, lembre-se de **ajustar os caminhos** para a imagem de entrada (`img_RS`) e para o modelo (`model_path`) diretamente no script.