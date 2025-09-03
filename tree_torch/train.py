import os
import numpy as np
import torch
from osgeo import gdal, ogr, gdal_array
from module import ParallelRandomForest
from sklearn.metrics import classification_report, accuracy_score
import datetime

# --------------------------------------------------------
# Configurações GDAL
os.unsetenv('PROJ_LIB')
os.environ['PROJ_LIB'] = "../venv/Lib/site-packages/osgeo/data/proj"
os.environ['GDAL_DATA'] = "../venv/Lib/site-packages/osgeo/data/gdal"

gdal.UseExceptions()
gdal.AllRegister()
gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')

# --------------------------------------------------------
# Parâmetros
est = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_RS = r"C:\Users\gabriel.cicotoste\Downloads\Mosaico_s2_Regional_Piloto.tif"
training = r"C:\Users\gabriel.cicotoste\OneDrive - GE21 Consultoria\SAMARCO - Regional_Piloto\Old\Amostras\Amostras_Regional_Piloto_S2_treinamento.shp"
attribute = 'Class_id'
model_path = "tree_torch/rf_torch.pth"

# --------------------------------------------------------
# Carrega imagem
img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)
img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

rows, cols, band_number = img.shape

# --------------------------------------------------------
# Função para extrair amostras
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

# --------------------------------------------------------
# Treino
X_train, y_train = extract_samples(training, attribute, img_ds, img)
X_train_tensor = torch.from_numpy(np.nan_to_num(X_train)).float().to(device)
y_train_tensor = torch.from_numpy(y_train).long().to(device)
n_classes = int(y_train_tensor.max().item()) + 1

rf = ParallelRandomForest(
    n_trees=est,
    n_classes=n_classes,
    max_depth=10,
    min_samples_split=2,
    feature_ratio=np.sqrt(band_number)/band_number,
    device=device
).to(device)

print("Treinando modelo...")
rf.fit(X_train_tensor, y_train_tensor)
print("Treino finalizado.")

# --------------------------------------------------------
# Avaliação no treino
with torch.no_grad():
    y_pred_tensor = rf.predict(X_train_tensor)

y_pred = y_pred_tensor.cpu().numpy()
y_true = y_train_tensor.cpu().numpy()

print("Avaliação no treino:")
print(classification_report(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))

# --------------------------------------------------------
# Salva modelo
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(rf, model_path)
print(f"Modelo salvo em {model_path}")
