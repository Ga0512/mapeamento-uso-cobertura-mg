from osgeo import gdal
import numpy as np
import cv2
import torch

class Data:
    def __init__(self, path: str, size=(128, 128)):
        self.path = path
        self.size = size
        self._raw = None 
        self._rgb = None 
        self._tensor = None 
        self._multiband = None
    
    def _read(self):
        if self._raw is None:
            ds = gdal.Open(self.path)
            if ds is None:
                raise FileNotFoundError(f"Não foi possível abrir {self.path}")
            self._raw = ds.ReadAsArray().astype(np.float32)
            ds = None
        return self._raw
    
    def array(self):
        """
        Carrega a imagem RGB (3 bandas) redimensionada e normalizada.
        Retorna: np.ndarray [H, W, 3]
        """
        if self._rgb is not None:
            return self._rgb
        
        arr = self._read()
        
        if arr.ndim < 3 or arr.shape[0] < 3:
            raise ValueError(f"Imagem {self.path} não possui ao menos 3 bandas RGB.")
        
        arr = arr[:3]  # Pega só RGB
        
        # Normaliza cada banda
        for i in range(3):
            band = arr[i]
            mn, mx = band.min(), band.max()
            if mx - mn > 1e-8:
                arr[i] = (band - mn) / (mx - mn)
            else:
                arr[i] = 0
        
        # Transforma para HxWxC
        arr = np.transpose(arr, (1, 2, 0))
        
        # Redimensiona
        self._rgb = cv2.resize(arr, self.size, interpolation=cv2.INTER_LINEAR)
        return self._rgb
    
    def tensor(self):
        """
        Processa a imagem multispectral.
        Retorna: (torch.Tensor [1, C, H, W], np.ndarray [C, H, W])
        """
        if self._tensor is not None:
            return self._tensor, self._raw
        
        img = self._read()
        
        for i in range(img.shape[0]):
            band = img[i]
            mn, mx = band.min(), band.max()
            if mx - mn > 1e-8:
                img[i] = (band - mn) / (mx - mn)
            else:
                img[i] = 0
        
        self._tensor = torch.from_numpy(img).unsqueeze(0).float()
        return self._tensor, img

    def multiband(self, num_bands):
        if self._multiband is not None:
            return self._multiband, self._raw
       
        ds = gdal.Open(self.path)
        bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(min(num_bands, ds.RasterCount))]
        img_array = np.stack(bands, axis=-1).astype(np.float32)
        per_band_min = np.min(img_array, axis=(0, 1))
        per_band_max = np.max(img_array, axis=(0, 1))
        
        self._multiband = (img_array - per_band_min) / (per_band_max - per_band_min + 1e-6)

        return self._multiband
