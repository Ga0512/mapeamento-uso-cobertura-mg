import numpy as np

class MockModel:
    def predict(self, *args, **kwargs):
        # Simula uma predição simples (sem GPU, sem carregar pesos)
        return [np.zeros((128, 128), dtype=np.uint8)]
