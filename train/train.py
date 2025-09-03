import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchvision import models
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, Trainer, TrainingArguments
from utils.metrics import compute_metrics, categorical_focal_loss, mean_iou_metric
from train.data import RemoteSensingDataset, prepare_data, AugmentationGenerator, get_train_augmentation, load_image_mask_pairs
import numpy as np
import time
from keras import layers, models, mixed_precision
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, mixed_precision
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam



class SegmentationModel:
    def __init__(self, model_type, image_dir, mask_dir, num_classes=16, batch_size=16, num_epochs=160, patience=100,
                 learning_rate=6e-5, output_dir="./output", img_size=128, num_channels=12, device=None, save_steps=500):
        self.model_type = model_type.lower()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.img_size = img_size
        self.num_channels = num_channels
        self.save_steps = save_steps
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_augmentation = True
        self.model = None
        self.feature_extractor = None

    def create_model(self, model_name="nvidia/mit-b1"):
        if self.model_type == "deeplab":
            self.model = models.segmentation.deeplabv3_resnet50(weights=None)
            self.model.backbone.conv1 = nn.Conv2d(
                self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            if self.model.aux_classifier is not None:
                self.model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            self.model.to(self.device)

        elif self.model_type == "segformer":
            self.feature_extractor = SegformerFeatureExtractor(
                do_random_flip=True,
                do_resize=False,
                do_normalize=False,
                do_reduce_labels=False,
                size=self.img_size
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                reshape_last_stage=True
            )
            self.model.to(self.device)

        elif self.model_type == "unet":
            inputs = layers.Input(shape=(self.img_size, self.img_size, self.num_channels))

            # Encoder
            c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
            c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
            p1 = layers.MaxPooling2D()(c1)

            c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
            c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
            p2 = layers.MaxPooling2D()(c2)

            c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
            c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
            p3 = layers.MaxPooling2D()(c3)

            # Bottleneck
            c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
            c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)

            # Decoder
            u5 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c4)
            u5 = layers.concatenate([u5, c3])
            c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
            c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

            u6 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
            u6 = layers.concatenate([u6, c2])
            c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
            c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

            u7 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
            u7 = layers.concatenate([u7, c1])
            c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
            c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

            outputs = layers.Conv2D(self.num_classes, 1, activation='softmax', dtype='float32')(c7)

            self.model = models.Model(inputs, outputs)

        else:
            raise ValueError("Modelo desconhecido. Use 'segformer', 'deeplab' ou 'unet'.")

    @staticmethod
    def debug_mask_values(dataset, num_samples=3):
        print("\nDebug de valores nas máscaras:")
        for i in range(min(num_samples, len(dataset))):
            _, mask = dataset[i]
            unique, counts = np.unique(mask, return_counts=True)
            print(f"Amostra {i+1}:")
            print(f"  Valores únicos: {unique}")
            print(f"  Contagens: {dict(zip(unique, counts))}")
            print(f"  Shape: {mask.shape}")

    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(loader, desc="Treinando"):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            main_output = outputs['out'] if isinstance(outputs, dict) else outputs
            loss = criterion(main_output, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        return running_loss / len(loader.dataset)

    def validate_epoch(self, loader, criterion):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Validando"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                main_output = outputs['out'] if isinstance(outputs, dict) else outputs
                loss = criterion(main_output, masks)
                running_loss += loss.item() * images.size(0)
        return running_loss / len(loader.dataset)

    def train(self):
        # Preparar dados
        train_img, train_mask, val_img, val_mask = prepare_data(self.image_dir, self.mask_dir)
        
        train_dataset = RemoteSensingDataset(train_img, train_mask)
        val_dataset = RemoteSensingDataset(val_img, val_mask)

        self.debug_mask_values(train_dataset)
        self.debug_mask_values(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=4, drop_last=False)

        if self.model_type == "deeplab":
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

            train_history, val_history = [], []
            best_val_loss = float('inf')
            start_time = time.time()

            for epoch in range(self.num_epochs):
                print(f"\nÉpoca {epoch+1}/{self.num_epochs}")
                train_loss = self.train_epoch(train_loader, criterion, optimizer)
                val_loss = self.validate_epoch(val_loader, criterion)
                scheduler.step(val_loss)

                train_history.append(train_loss)
                val_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pth"))
                    print(f"Melhor modelo salvo! Val_loss: {val_loss:.4f}")

                if (epoch+1) % 10 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{epoch+1}.pth"))

                print(f"Treino: {train_loss:.4f} | Validação: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Salva modelo final
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "final_model.pth"))

            # Plot de perdas
            plt.figure(figsize=(10,5))
            plt.plot(train_history, label='Treino')
            plt.plot(val_history, label='Validação')
            plt.title('Perda por Época')
            plt.xlabel('Época')
            plt.ylabel('Perda')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'loss_curve.png'))
            plt.close()

            print(f"\nTreinamento completo! Tempo total: {(time.time()-start_time)/60:.2f} minutos")
            print(f"Melhor perda de validação: {best_val_loss:.4f}")
            print(f"Modelos salvos em: {self.output_dir}")

        elif self.model_type == "segformer":
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                save_total_limit=3,
                eval_strategy="steps",
                save_strategy="steps",
                save_steps=self.save_steps,
                logging_steps=100,
                eval_steps=self.save_steps,
                load_best_model_at_end=True,
                metric_for_best_model="mean_iou",
                greater_is_better=True,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            print("Iniciando treinamento SegFormer...")
            trainer.train()
            trainer.save_model(os.path.join(self.output_dir, "final_model"))
            self.feature_extractor.save_pretrained(os.path.join(self.output_dir, "final_model"))

        elif self.model_type == "unet":
            X_train, X_test, Y_train, Y_test = load_image_mask_pairs(self.image_dir, self.mask_dir, self.img_size, self.num_channels, self.num_classes)

            callbacks = []

            checkpoint_cb = ModelCheckpoint(
                filepath=os.path.join("./checkpoint", 'model_unet_128x128_seg_epoch_{epoch:02d}_val_accuracy{val_accuracy:.4f}_v2.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint_cb)

            early_stopping_cb = EarlyStopping(
                monitor='val_accuracy',   # ou 'val_accuracy'
                patience=self.patience,        # número de épocas sem melhoria antes de parar
                restore_best_weights=True,
                mode='max',
                verbose=1
            )
            callbacks.append(early_stopping_cb)

            focal_loss_fn = categorical_focal_loss(gamma=2.0, alpha=0.25)
            base_optimizer = Adam(learning_rate=1e-4)
            optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)
            #optimizer = Adam(learning_rate=1e-4)
            self.model.compile(optimizer=optimizer, loss=focal_loss_fn, metrics=['accuracy', mean_iou_metric(num_classes=self.num_classes)])


            if self.data_augmentation:
                augmenter = get_train_augmentation()
                
                train_generator = AugmentationGenerator(
                    X_train, Y_train,
                    batch_size=self.batch_size,
                    augmenter=augmenter,
                    num_classes=self.num_classes
                )
                
                history = self.model.fit(
                    train_generator,
                    epochs=self.num_epochs,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks,
                )
            else:
                history = self.model.fit(
                    X_train, Y_train,
                    epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks,

                )
            logging.info('Training finished.')


            
            logging.info('Starting evaluation')
            eval_results  = self.model.evaluate(X_test, Y_test)
            logging.info(f'Evaluation results - Loss: {eval_results[0]:.4f}, Accuracy: {eval_results[1]:.4f}, Mean IoU: {eval_results[2]:.4f}')



def segformer(**kwargs):
    kwargs['model_type'] = 'segformer'
    model = SegmentationModel(**kwargs)
    model.create_model(model_name="nvidia/mit-b1")
    return model


def deeplab(**kwargs):
    kwargs['model_type'] = 'deeplab'
    model = SegmentationModel(**kwargs)
    model.create_model()
    return model


def unet(**kwargs):
    kwargs['model_type'] = 'unet'
    model = SegmentationModel(**kwargs)
    model.create_model()
    return model

