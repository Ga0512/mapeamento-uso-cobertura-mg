from multiclean import clean_array
import rasterio
from rasterio.windows import Window

# Arquivo de entrada e saída
input_path = "dataset/Classificada_RF_Planet_Regional1_DO2_Filtrada_v2_200pixels.tif"
output_path = "dataset/Classificada_RF_Planet_Regional1_DO2_Filtrada_v2_200pixels_cleaned.tif"

# Tamanho de cada tile (ajuste conforme sua RAM)
tile_size = 2000  

with rasterio.open(input_path) as src:
    profile = src.profile.copy()

    # Converter para uint8 porque temos só classes 0–18
    profile.update(dtype="uint8")

    # Criar o raster de saída vazio
    with rasterio.open(output_path, "w", **profile) as dst:
        width = src.width
        height = src.height

        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                # Criar a janela (tile)
                window = Window(j, i, min(tile_size, width - j), min(tile_size, height - i))

                # Ler o tile
                landcover_tile = src.read(1, window=window).astype("uint8")

                # Limpar com multiclean
                cleaned_tile = clean_array(
                    landcover_tile,
                    class_values=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18],  # classes 0–18
                    smooth_edge_size=1,
                    min_island_size=25,
                    connectivity=8,
                    fill_nan=False,
                    max_workers=1   # evita explosão de RAM
                )

                # Gravar no arquivo de saída
                dst.write(cleaned_tile.astype("uint8"), 1, window=window)

print("✅ Raster limpo salvo em:", output_path)
