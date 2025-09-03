import os
import numpy as np
import torch
from osgeo import gdal, gdal_array
from module import ParallelRandomForest
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------------
# Parâmetros
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "tree_torch/rf_torch.pth"

img_RS = r"C:\Users\gabriel.cicotoste\Downloads\Mosaico_s2_Regional_Piloto.tif"
classification_image = "./result/resultado.tif"

# Opcional: shapefile de validação
validation = r"C:\Users\gabriel.cicotoste\OneDrive - GE21 Consultoria\SAMARCO - Regional_Piloto\Old\Amostras\Amostras_Regional_Piloto_S2_validacao.shp"
attribute = 'Class_id'

# --------------------------------------------------------
# Carrega imagem
img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)
img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

rows, cols, bands = img.shape

# --------------------------------------------------------
# Inicializa RF e carrega pesos
# precisa informar n_classes e band_number = bands
# aqui vamos inferir n_classes a partir do shapefile de validação se existir
from osgeo import ogr

def extract_samples(shapefile, attribute, img_ds, img):
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
    X = img[roi > 0, :]
    y = roi[roi > 0]
    return X, y

if validation:
    X_val, y_val = extract_samples(validation, attribute, img_ds, img)
    n_classes = int(y_val.max()) + 1
else:
    n_classes = 2  # default

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
# Classificação da imagem inteira
print("Classificando imagem completa...")
img_reshaped = img.reshape(-1, bands)

batch_size = 100000
preds = []
with torch.no_grad():
    for i in range(0, img_reshaped.shape[0], batch_size):
        batch = img_reshaped[i:i+batch_size]
        batch_tensor = torch.from_numpy(np.nan_to_num(batch)).float().to(device)
        pred_batch = rf.predict(batch_tensor)
        preds.append(pred_batch.cpu().numpy())

classified = np.concatenate(preds).reshape(rows, cols)

# Salvar GeoTIFF
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(classification_image, cols, rows, 1, gdal.GDT_Byte)
out_ds.SetProjection(img_ds.GetProjection())
out_ds.SetGeoTransform(img_ds.GetGeoTransform())
out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(classified)
out_band.SetNoDataValue(0)
out_ds.FlushCache()
out_ds = None
print("Classificação salva em:", classification_image)

# --------------------------------------------------------
# Avaliação opcional se houver shapefile de validação
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
