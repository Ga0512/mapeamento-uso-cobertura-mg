import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


import numpy as np
from keras.models import load_model
from src.data.loader import Data
from .base import BaseWrapper
import tensorflow as tf
import gc

class UnetWrapper(BaseWrapper):
    def __init__(self, model_path):
        model = load_model(model_path, compile=False)
        super().__init__(model)

    def predict(self, img_path: str):
        data = Data(img_path)
        img = data.multiband(12)
        pred = self.model.predict(img[None, ...], verbose=0)

        tf.keras.backend.clear_session()  # 🔧 evita crash
        gc.collect()
        return np.argmax(pred[0], axis=-1).astype(np.uint8)
