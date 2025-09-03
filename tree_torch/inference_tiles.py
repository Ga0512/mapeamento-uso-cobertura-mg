import os
import numpy as np
import torch
from osgeo import gdal, gdal_array, ogr
from module import ParallelRandomForest
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------------
# Parâmetros
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "models/tree_torch/rf_torch.pth"

img_RS = r"C:\Users\gabriel.cicotoste\Downloads\Mosaico_s2_Regional_Piloto.tif"
classification_image = "./result/resultado.tif"

# Opcional: shapefile de validação
validation = r"C:\Users\gabriel.cicotoste\OneDrive - GE21 Consultoria\SAMARCO - Regional_Piloto\Old\Amostras\Amostras_Regional_Piloto_S2_validacao.shp"
attribute = 'Class_id'

# --------------------------------------------------------
# Carrega imagem
img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)
rows, cols, bands = img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount

# --------------------------------------------------------
# Extrair amostras do shapefile (para saber n_classes e validação)
def extract_samples(shapefile, attribute, img_ds):
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('', img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_UInt16)
    mem_raster.SetProjection(img_ds.GetProjection())
    mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    ds = ogr.Open(shapefile)
    layer = ds.GetLayer()
    err = gdal.RasterizeLayer(mem_raster, [1], layer, None, None, [1], ["ATTRIBUTE=" + attribute, "ALL_TOUCHED=TRUE"])
    assert err == gdal.CE_None

    roi = mem_raster.ReadAsArray()
    X = []
    y = []
    for b in range(bands):
        band_data = img_ds.GetRasterBand(b + 1).ReadAsArray()
        if b == 0:
            mask = roi > 0
        if len(X) == 0:
            X = band_data[mask][:, None]
        else:
            X = np.hstack((X, band_data[mask][:, None]))
    y = roi[mask]
    return X, y

if validation:
    X_val, y_val = extract_samples(validation, attribute, img_ds)
    n_classes = int(y_val.max()) + 1
else:
    n_classes = 2  # default

# --------------------------------------------------------
# Inicializa RF e carrega pesos
rf = ParallelRandomForest(
    n_trees=500,
    n_classes=n_classes,
    max_depth=10,
    min_samples_split=2,
    feature_ratio=np.sqrt(bands)/bands,
    device=device
).to(device)

rf = torch.load(model_path, weights_only=False, map_location=device)
rf.eval()

# --------------------------------------------------------
# Criar dataset de saída
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(classification_image, cols, rows, 1, gdal.GDT_Byte,
                       options=["COMPRESS=LZW", "TILED=YES"])
out_ds.SetProjection(img_ds.GetProjection())
out_ds.SetGeoTransform(img_ds.GetGeoTransform())
out_band = out_ds.GetRasterBand(1)
out_band.SetNoDataValue(0)

# --------------------------------------------------------
# Processar por blocos
tile_size = 1024  # ajuste conforme RAM/GPU
batch_size = 100000

import time
start_time = time.time()  

print("Classificando imagem em blocos...")
for y in range(0, rows, tile_size):
    for x in range(0, cols, tile_size):
        win_x = min(tile_size, cols - x)
        win_y = min(tile_size, rows - y)

        # Lê tile (bandas no último eixo)
        tile = np.zeros((win_y, win_x, bands), dtype=np.float32)
        for b in range(bands):
            tile[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray(x, y, win_x, win_y)

        tile_reshaped = tile.reshape(-1, bands)

        preds = []
        with torch.no_grad():
            for i in range(0, tile_reshaped.shape[0], batch_size):
                batch = tile_reshaped[i:i+batch_size]
                batch_tensor = torch.from_numpy(np.nan_to_num(batch)).float().to(device)
                pred_batch = rf.predict(batch_tensor)
                preds.append(pred_batch.cpu().numpy())

        classified_tile = np.concatenate(preds).reshape(win_y, win_x)

        # Escreve direto no arquivo
        out_band.WriteArray(classified_tile, xoff=x, yoff=y)

        print(f"Tile ({x}:{x+win_x}, {y}:{y+win_y}) processado.")

out_ds.FlushCache()
out_ds = None
print("Classificação salva em:", classification_image)

# --------------------------------------------------------
# Avaliação opcional
if validation:
    X_val_tensor = torch.from_numpy(np.nan_to_num(X_val)).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device)
    with torch.no_grad():
        y_pred_tensor = rf.predict(X_val_tensor)

    y_pred = y_pred_tensor.cpu().numpy()
    y_true = y_val_tensor.cpu().numpy()

    print("Avaliação na validação:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))


end_time = time.time()  # <-- fim da contagem
elapsed = end_time - start_time
print(f"Tempo total de inferência: {elapsed/60:.2f} minutos ({elapsed:.2f} segundos)")