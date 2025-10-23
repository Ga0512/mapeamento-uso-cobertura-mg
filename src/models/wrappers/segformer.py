import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from src.data.loader import Data
from .base import BaseWrapper

class SegformerWrapper(BaseWrapper):
    def __init__(self, model_path, device):
        model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device).eval()
        feature_extractor = SegformerFeatureExtractor.from_pretrained(model_path)
        super().__init__(model, device)
        self.feature_extractor = feature_extractor

    def predict(self, img_path: str):
        data = Data(img_path)
        img = data.array()
        h, w, _ = img.shape
        inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        return pred_mask
