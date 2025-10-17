# tests/test_segmentation_models.py

import os
import numpy as np
import pytest
from src.models import SegmentationModels

@pytest.mark.parametrize("model_name", ["deeplab", "segformer", "unet"])
def test_model_prediction_shape(model_name):
    """
    Testa se cada modelo de segmentação retorna uma máscara válida.
    """

    img_path = "./dataset/Augmented/Images/Amostra_piloto_6_2_aug_cls4_002.tif"
    assert os.path.exists(img_path), f"Imagem não encontrada: {img_path}"

    model = SegmentationModels()
    predictor = getattr(model, model_name)

    mask = predictor.predict(img_path)

    # Verifica se a saída é um numpy array
    assert isinstance(mask, np.ndarray), f"Saída do modelo {model_name} não é um ndarray"
    # Verifica se tem 2 dimensões (H, W)
    assert mask.ndim in (2, 3), f"Máscara do modelo {model_name} tem formato inesperado: {mask.shape}"
    # Verifica se não está vazia
    assert mask.size > 0, f"Máscara do modelo {model_name} está vazia"

    print(f"✅ {model_name} → máscara: {mask.shape}")
