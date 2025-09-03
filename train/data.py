import os
import glob
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
from keras.utils import to_categorical


class RemoteSensingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, feature_extractor, num_classes, mode):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "seg":
            return self.__segitem__(idx)
        elif self.mode == "dlab":
            return self.__dlabitem__(idx)
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

    def __len__(self):
        return len(self.image_paths)

    def __segitem__(self, idx):
        # Carrega imagem (12 bandas)
        img_ds = gdal.Open(self.image_paths[idx])
        img = img_ds.ReadAsArray().astype(np.float32)
        img_ds = None

        # Seleciona bandas RGB
        if img.shape[0] >= 3:
            img = img[:3]
        else:
            raise ValueError("A imagem não possui ao menos 3 bandas RGB.")

        # Normalização por banda
        for i in range(img.shape[0]):
            band = img[i]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 1e-8:
                img[i] = (band - min_val) / (max_val - min_val)
            else:
                img[i] = band * 0

        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        # Carrega máscara
        mask_ds = gdal.Open(self.mask_paths[idx])
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_ds = None

        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        # Pré-processamento
        inputs = self.feature_extractor(
            images=img,
            segmentation_maps=mask,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __dlabitem__(self, idx):
        # Carrega imagem
        img_ds = gdal.Open(self.image_paths[idx])
        img = img_ds.ReadAsArray().astype(np.float32)
        img_ds = None

        # Normalização por banda
        for i in range(img.shape[0]):
            band = img[i]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 1e-8:
                img[i] = (band - min_val) / (max_val - min_val)
            else:
                img[i] = band * 0  # Caso banda constante

        # Carrega máscara
        mask_ds = gdal.Open(self.mask_paths[idx])
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_ds = None

        # Processamento da máscara:
        # - Valores fora do intervalo 0-15 são marcados como -1 (ignorar)
        new_mask = np.full_like(mask, -1)  # Inicializa tudo como -1
        valid_mask = (mask >= 0) & (mask < self.num_classes)
        new_mask[valid_mask] = mask[valid_mask]

        # Garante que máscara tem dimensões corretas
        if len(new_mask.shape) == 3:
            new_mask = new_mask.squeeze(0)

        return img, new_mask

    def plot_sample(self, idx):
        """Plota uma imagem RGB normalizada e a máscara correspondente."""
        # Carrega imagem e máscara originais
        img_ds = gdal.Open(self.image_paths[idx])
        img = img_ds.ReadAsArray().astype(np.float32)
        img_ds = None

        if img.shape[0] >= 3:
            img = img[:3]
        else:
            raise ValueError("A imagem não possui ao menos 3 bandas RGB.")

        for i in range(img.shape[0]):
            band = img[i]
            min_val = band.min()
            max_val = band.max()
            if max_val - min_val > 1e-8:
                img[i] = (band - min_val) / (max_val - min_val)
            else:
                img[i] = band * 0

        img = np.transpose(img, (1, 2, 0))  # (H, W, C)

        mask_ds = gdal.Open(self.mask_paths[idx])
        mask = mask_ds.ReadAsArray().astype(np.int64)
        mask_ds = None
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)

        # Plotagem
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].imshow(img)
        axs[0].set_title("Imagem RGB")
        axs[0].axis("off")

        cmap = plt.get_cmap("tab20", self.num_classes)
        im = axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=16)
        axs[1].set_title("Máscara de Segmentação")
        axs[1].axis("off")

        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
        cbar.set_ticks(list(range(self.num_classes)))
        cbar.set_label("Classes (0 a 15)")

        plt.tight_layout()
        plt.show()


class AugmentationGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, augmenter, num_classes, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_imgs = self.images[batch_indexes]
        batch_masks = self.masks[batch_indexes]

        X_aug = []
        Y_aug = []

        for img, mask in zip(batch_imgs, batch_masks):
            transformed = self.augmenter(image=img, mask=np.argmax(mask, axis=-1))  # input mask shape: (H, W)
            img_aug = transformed['image']
            mask_aug = to_categorical(transformed['mask'], num_classes=self.num_classes)

            X_aug.append(img_aug)
            Y_aug.append(mask_aug)

        return np.array(X_aug), np.array(Y_aug)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def prepare_data(img_dir, mask_dir):
    print("Preparando dados...")
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_files = []

    for img_path in image_files:
        base_name = os.path.basename(img_path).replace('.tif', '')
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.tif")

        if not os.path.exists(mask_path):
            # Tenta encontrar máscara com padrão alternativo
            alt_mask = glob.glob(os.path.join(mask_dir, f"{base_name}*_mask.tif"))
            if alt_mask:
                mask_path = alt_mask[0]

        if os.path.exists(mask_path):
            mask_files.append(mask_path)
        else:
            print(f"Aviso: Máscara não encontrada para {img_path}")

    # Filtra imagens sem máscaras correspondentes
    valid_pairs = [(img, mask) for img, mask in zip(image_files, mask_files) if os.path.exists(mask)]
    image_files, mask_files = zip(*valid_pairs) if valid_pairs else ([], [])

    # Divisão treino/validação
    train_img, val_img, train_mask, val_mask = train_test_split(
        list(image_files), list(mask_files), test_size=0.25, random_state=42
    )

    print(f"Total de amostras: {len(image_files)}")
    print(f"Treino: {len(train_img)} | Validação: {len(val_img)}")
    return train_img, train_mask, val_img, val_mask


def get_train_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-10, 10), shear=0, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.CoarseDropout(num_holes_range=(8, 8), hole_height_range=(8, 8), hole_width_range=(8, 8), fill=0.0, p=0.3),
    ])


def load_image_mask_pairs(images_dir, labels_dir, num_bands, num_classes, mask_suffix='_mask'):
    label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir) if f.lower().endswith(('.tif', '.tiff'))}

    X, Y = [], []

    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.lower().endswith(('.tif', '.tiff')):
            continue

        img_name = os.path.splitext(img_file)[0]  # Ex: image_001
        label_name = img_name + mask_suffix       # Ex: image_001_mask

        if label_name not in label_files:
            continue

        try:
            img_path = os.path.join(images_dir, img_file)
            lbl_path = os.path.join(labels_dir, label_files[label_name])

            #Load image
            img_ds = gdal.Open(img_path)
            if img_ds is None:
               
                continue

            bands = []
            per_band_min = []
            per_band_max = []

            for b in range(min(num_bands, img_ds.RasterCount)):
                band = img_ds.GetRasterBand(b + 1)

                #Calculate band stats
                stats = band.GetStatistics(0, 1)
                if stats is None:
                    raise RuntimeError(f'No available stats for band {b+1} of {img_file}')
                min_val, max_val = stats[0], stats[1]
                per_band_min.append(min_val)
                per_band_max.append(max_val)

                band_array = band.ReadAsArray()
                bands.append(band_array)

            #Stack data
            img_array = np.stack(bands, axis=-1).astype(np.float32)

            #Normalize data
            per_band_min = np.array(per_band_min, dtype=np.float32)
            per_band_max = np.array(per_band_max, dtype=np.float32)
            img_array = (img_array - per_band_min[np.newaxis, np.newaxis, :]) / \
                        (per_band_max - per_band_min + 1e-6)[np.newaxis, np.newaxis, :]
            img_array = np.clip(img_array, 0, 1)

            #Load mask
            lbl_ds = gdal.Open(lbl_path)
            if lbl_ds is None:
        
                continue

            lbl_array = lbl_ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
            lbl_array = np.clip(lbl_array, 0, num_classes)
            lbl_onehot = to_categorical(lbl_array, num_classes=num_classes)

            X.append(img_array)
            Y.append(lbl_onehot)

        except Exception as e:
           raise (f'Error processing {img_file} & {label_files.get(label_name)}: {e}')

    X, Y = np.array(X), np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25)

    return X_train, X_test, Y_train, Y_test