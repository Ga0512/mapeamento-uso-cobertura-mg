import numpy as np

class _FakePredictor:
    def predict(self, *args, **kwargs):
        # Retorna uma lista simulando uma máscara
        return [np.zeros((128, 128), dtype=np.uint8)]

class MockModel:
    def __init__(self):
        # Simula os submodelos
        self.segformer = _FakePredictor()
        self.deeplab = _FakePredictor()
        self.unet = _FakePredictor()
