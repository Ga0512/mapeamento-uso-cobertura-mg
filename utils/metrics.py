import evaluate
import torch
import tensorflow as tf
import keras.backend as K
from keras.utils import register_keras_serializable

def compute_metrics(eval_pred, num_classes):
    # Carrega métrica mean_iou
    from torch import nn
    metric = evaluate.load("mean_iou")

    # Desempacota logits e labels
    logits, labels = eval_pred
    # Converte logits para tensor e faz upsample para o tamanho das masks
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    # Predições finais (B, H, W)
    pred_labels = logits_tensor.argmax(dim=1).cpu().numpy()

    # Chama método privado para lidar com ignore_index e redução de labels
    metrics = metric._compute(
        predictions=pred_labels,
        references=labels,
        num_labels=num_classes,
        ignore_index=0,
        reduce_labels=False,
    )

    # Extrai métricas por categoria
    per_cat_acc = metrics.pop("per_category_accuracy").tolist()
    per_cat_iou = metrics.pop("per_category_iou").tolist()
    metrics.update({f"accuracy_{i}": v for i, v in enumerate(per_cat_acc)})
    metrics.update({f"iou_{i}": v for i, v in enumerate(per_cat_iou)})

    return metrics


@register_keras_serializable()
def categorical_focal_loss(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = K.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return focal_loss

@register_keras_serializable()
def mean_iou_metric(y_true, y_pred, num_classes=16):
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    ious = []
    for i in range(num_classes):
        intersect = K.sum(K.cast((y_true == i) & (y_pred == i), tf.float32))
        union = K.sum(K.cast((y_true == i) | (y_pred == i), tf.float32))
        iou = tf.where(K.equal(union, 0), 0.0, intersect / union)
        ious.append(iou)
    return K.mean(tf.stack(ious))
