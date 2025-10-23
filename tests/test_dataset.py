from utils.sys_env import set_env

set_env()

import warnings
warnings.filterwarnings("ignore")

import pytest
import numpy as np
import os
from src.data.dataset import RemoteSensingDataset
from transformers import SegformerFeatureExtractor  # ou outro que você usa

# Caminhos mínimos
IMG_DIR = "dataset/Augmented/Images"
MASK_DIR = "dataset/Augmented/Masks"


@pytest.fixture(scope="module")
def dataset():
    """Cria um dataset pequeno para teste."""
    # Coleta caminhos de teste
    image_paths = sorted([
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.endswith(".tif")
    ])
    mask_paths = sorted([
        os.path.join(MASK_DIR, f)
        for f in os.listdir(MASK_DIR)
        if f.endswith(".tif")
    ])

    if not image_paths or not mask_paths:
        pytest.skip("⚠️ Não há imagens/máscaras de teste nos diretórios definidos")

    # Mock simples de feature extractor
    feature_extractor = SegformerFeatureExtractor(do_normalize=False, do_resize=False)

    ds = RemoteSensingDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        feature_extractor=feature_extractor,
        num_classes=5,
        mode="dlab",  # ou "seg" conforme o pipeline que quiser testar
    )
    return ds


def test_dataset_length(dataset):
    """Verifica se o dataset contém amostras."""
    assert len(dataset) > 0, "Dataset está vazio."


def test_dataset_item_structure(dataset):
    """Verifica se cada item retorna imagem e máscara com shapes válidos."""
    img, mask = dataset[0]
    assert isinstance(img, np.ndarray), "Imagem não é um numpy array"
    assert isinstance(mask, np.ndarray), "Máscara não é um numpy array"
    assert img.ndim == 3, f"Imagem deve ter 3 dimensões (HWC), mas tem {img.ndim}"
    assert mask.ndim in (2, 3), "Máscara deve ter 2 ou 3 dimensões"


def test_mask_value_range(dataset):
    """Confere se as máscaras contêm valores válidos (>=0 e < num_classes)."""
    for i in range(min(3, len(dataset))):
        _, mask = dataset[i]
        valid = np.all((mask >= -1) & (mask < dataset.num_classes))
        assert valid, f"Máscara da amostra {i} contém valores fora do intervalo válido"


def test_no_nan(dataset):
    """Certifica que não há NaN nas imagens ou máscaras."""
    for i in range(min(3, len(dataset))):
        img, mask = dataset[i]
        assert not np.isnan(img).any(), f"Imagem {i} contém NaN"
        assert not np.isnan(mask).any(), f"Máscara {i} contém NaN"


def test_debug_mask_values(dataset, capsys):
    """Mostra os valores únicos nas máscaras para depuração."""
    print("\nDebug de valores nas máscaras:")
    for i in range(min(3, len(dataset))):
        _, mask = dataset[i]
        unique, counts = np.unique(mask, return_counts=True)
        print(f"Amostra {i+1}: valores únicos {unique}, contagens {dict(zip(unique, counts))}")
        assert len(unique) > 0, "Máscara vazia detectada"

    captured = capsys.readouterr()
    assert "valores únicos" in captured.out
