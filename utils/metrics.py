import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


def pixel_accuracy(preds, masks):
    """Acurácia pixel a pixel."""
    correct = (preds == masks).float()
    acc = correct.sum() / correct.numel()
    return acc.item()


def dice_coefficient(preds, masks, smooth=1e-5):
    """Coeficiente de Dice entre duas máscaras."""
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)
    intersection = (preds_flat * masks_flat).sum()
    return ((2. * intersection + smooth) /
            (preds_flat.sum() + masks_flat.sum() + smooth)).item()


import evaluate as hf_evaluate
from torch import nn

def eval_mean_iou(eval_pred, num_classes=16):

    metric = hf_evaluate.load("mean_iou")
    logits, labels = eval_pred

    logits_tensor = torch.from_numpy(logits)
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    pred_labels = logits_tensor.argmax(dim=1).cpu().numpy()

    metrics = metric._compute(
        predictions=pred_labels,
        references=labels,
        num_labels=num_classes,
        ignore_index=0,
        reduce_labels=False,
    )

    per_cat_acc = metrics.pop("per_category_accuracy").tolist()
    per_cat_iou = metrics.pop("per_category_iou").tolist()
    metrics.update({f"accuracy_{i}": v for i, v in enumerate(per_cat_acc)})
    metrics.update({f"iou_{i}": v for i, v in enumerate(per_cat_iou)})

    return metrics


@register_keras_serializable()
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss categórica — robusta a classes desbalanceadas.
    """
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss


@register_keras_serializable()
def mean_iou_metric_keras(num_classes=16):
    """
    Mean IoU nativo para Keras (compatível com model.compile()).
    """
    def mean_iou(y_true, y_pred):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        ious = []
        for i in range(num_classes):
            inter = tf.reduce_sum(tf.cast(tf.logical_and(y_true == i, y_pred == i), tf.float32))
            union = tf.reduce_sum(tf.cast(tf.logical_or(y_true == i, y_pred == i), tf.float32))
            iou = tf.where(tf.equal(union, 0), 0.0, inter / union)
            ious.append(iou)
        return tf.reduce_mean(tf.stack(ious))
    return mean_iou
