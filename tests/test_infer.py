import os
import numpy as np
import pytest

# 🔹 Mock leve se o ambiente for CI
CI_MODE = os.getenv("CI", "false").lower() == "true"

if CI_MODE:
    class MockPredictor:
        def predict(self, *args, **kwargs):
            # Retorna máscara fake só pra validar forma
            return np.zeros((128, 128), dtype=np.uint8)

    class SegmentationModels:
        def __init__(self):
            self.deeplab = MockPredictor()
            self.segformer = MockPredictor()
            self.unet = MockPredictor()
else:
    from src.models import SegmentationModels


@pytest.mark.parametrize("model_name", ["deeplab", "segformer", "unet"])
def test_model_prediction_shape(model_name):
    """
    Testa se cada modelo de segmentação retorna uma máscara válida.
    No CI, usa mock; localmente roda com modelo real.
    """
    img_path = "./tests/Amostra_piloto_6_2_aug_cls4_002.tif"

    if CI_MODE:
        # Cria imagem fake pra não depender de arquivos
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(b"fake")

    model = SegmentationModels()
    predictor = getattr(model, model_name)
    mask = predictor.predict(img_path)

    assert isinstance(mask, np.ndarray), f"Saída do modelo {model_name} não é ndarray"
    assert mask.ndim in (2, 3), f"Máscara {model_name} tem forma inválida: {mask.shape}"
    assert mask.size > 0, f"Máscara {model_name} está vazia"
