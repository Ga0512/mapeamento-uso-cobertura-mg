import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torchvision import models
import torch.nn as nn
from keras.models import load_model
import numpy as np
from src.data.loader import Data
from torchvision.transforms.v2 import RandomResizedCrop
from torchvision.transforms import Normalize
import timm
import rasterio
from utils.logger import get_logger
import warnings

warnings.filterwarnings("ignore")
logger = get_logger("MLProject")

# /models

class SegformerWrapper:
    def __init__(self, model, feature_extractor, device, data):
        self.model = model
        self.fe = feature_extractor
        self.device = device
        self.data = data

    def predict(self):
        img = self.data.array()
        h, w, _ = img.shape
        inputs = self.fe(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=(h, w), mode='bilinear', align_corners=False
            )
            pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        return pred_mask


class DeeplabWrapper:
    def __init__(self, model, device, data):
        self.model = model
        self.device = device
        self.data = data

    def predict(self):
        tensor, _ = self.data.tensor()
        with torch.no_grad():
            output = self.model(tensor.to(self.device))['out']
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return pred_mask


class UnetWrapper:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def predict(self):
        img = self.data.multiband(12)
        img_input = img[None, ...]
        pred = self.model.predict(img_input, verbose=0)
        return np.argmax(pred[0], axis=-1).astype(np.uint8)


class SegmentationModels:
    def __init__(self, img_path: str, segformer_path="output/segformer_latest/final_model",
                 deeplab_path="output/deeplab_latest/best_model.pth", unet_path='output/unet_lates/model_unet_best.keras'):

        self.NUM_CLASSES = 16
        self.NUM_CHANNELS = 12
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = Data(img_path)

        # Segformer
        seg_model = SegformerForSemanticSegmentation.from_pretrained(segformer_path).to(self.DEVICE).eval()
        seg_fe = SegformerImageProcessor.from_pretrained(segformer_path)

        # Deeplab
        deeplab = models.segmentation.deeplabv3_resnet50(weights=None)
        deeplab.backbone.conv1 = nn.Conv2d(self.NUM_CHANNELS, 64, 7, 2, 3, bias=False)
        deeplab.classifier[4] = nn.Conv2d(256, self.NUM_CLASSES, 1)
        if deeplab.aux_classifier is not None:
            deeplab.aux_classifier[4] = nn.Conv2d(256, self.NUM_CLASSES, 1)
        deeplab.load_state_dict(torch.load(deeplab_path, map_location=self.DEVICE))
        deeplab.to(self.DEVICE).eval()

        # Unet
        unet = load_model(unet_path, compile=False)

        # Wrappers
        self.segformer = SegformerWrapper(seg_model, seg_fe, self.DEVICE, self.data)
        self.deeplab = DeeplabWrapper(deeplab, self.DEVICE, self.data)
        self.unet = UnetWrapper(unet, self.data)



class VitWrapper:
    def __init__(self, model, device, data, val, labels):
        self.model = model
        self.device = device
        self.img_path = data
        self.val_tf = val
        self.labels = labels

    def predict(self):
        with rasterio.open(self.img_path) as src:
            img = src.read().astype('float32')
        logger.debug(f"Shape da imagem original: {img.shape}")

        img = torch.from_numpy(img)
        img = self.val_tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img)
        pred_idx = logits.softmax(dim=1).argmax().item()

        logger.info(f"Inferência ViT finalizada, predição: {self.labels[pred_idx]}")
        return pred_idx
    

class ClassificationModels:
    def __init__(self, img_path: str, labels: list, vit_path: str = "./models/classification/vit.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = 224
        self.img_path = img_path
        self.labels = labels

        logger.info("Carregando modelo de classificação ViT")
        # Transformação para validação
        
        self.val_tf = nn.Sequential(
            RandomResizedCrop((self.img_size, self.img_size), antialias=True),
            Normalize(mean=[0.5]*12, std=[0.2]*12)
        )

        vit_model = timm.create_model('vit_large_patch14_224', pretrained=False,
                                       in_chans=12, num_classes=len(self.labels))
        vit_model.load_state_dict(torch.load(vit_path, map_location=self.device))
        vit_model.to(self.device)
        vit_model.eval()
        logger.info("Modelo de classificação carregado com sucesso")

        self.vit = VitWrapper(vit_model, self.device, self.img_path, self.val_tf, self.labels)

