import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from src.data.loader import Data
from .base import BaseWrapper

class SegformerWrapper(BaseWrapper):
    def __init__(self, model_path, device):
        model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device).eval()
        processor = SegformerImageProcessor.from_pretrained(model_path)
        super().__init__(model, device)
        self.processor = processor

    def predict(self, img_path: str):
        data = Data(img_path)
        img = data.array()
        h, w, _ = img.shape
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            upsampled = nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            mask = upsampled.argmax(dim=1)[0].cpu().numpy()
        return mask
