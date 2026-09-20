"""Фильтрация прогнозов по уверенности и удаление дубликатов рамок."""
import numpy as np


def iou_matrix(predicted, reference):
    p = np.asarray(predicted, dtype=float).reshape(-1, 4)
    r = np.asarray(reference, dtype=float).reshape(-1, 4)
    left = np.maximum(p[:, None, :2], r[None, :, :2])
    right = np.minimum(p[:, None, 2:], r[None, :, 2:])
    inter = np.maximum(right - left, 0).prod(axis=-1)
    pa = np.maximum(p[:, 2:] - p[:, :2], 0).prod(axis=-1)
    ra = np.maximum(r[:, 2:] - r[:, :2], 0).prod(axis=-1)
    union = pa[:, None] + ra[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms_indices(boxes, scores, threshold=0.8):
    order = np.argsort(-np.asarray(scores), kind="stable").tolist()
    keep = []
    while order:
        i = order.pop(0); keep.append(i)
        if order:
            ious = iou_matrix([boxes[i]], [boxes[j] for j in order])[0]
            order = [j for j, overlap in zip(order, ious) if overlap <= threshold]
    return keep


def select_predictions(prediction, threshold=0.25, nms_threshold=0.8):
    candidates = [d for d in prediction['detections'] if d['score'] >= threshold]
    indices = nms_indices([d['box'] for d in candidates], [d['score'] for d in candidates], nms_threshold)
    return [candidates[i] for i in indices]
