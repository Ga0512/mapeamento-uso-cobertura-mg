import torch

class BaseWrapper:
    """Classe base para wrappers de modelos."""
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device or torch.device('cpu')

    def predict(self, img_path: str):
        raise NotImplementedError("Subclasse deve implementar o método predict(img_path).")
