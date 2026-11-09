"""Фильтрация прогнозов по уверенности и удаление дубликатов рамок."""
import numpy as np


def matched_iou(predicted, reference):
    """Optimal one-to-one IoU sum / max box count; unmatched boxes score zero."""
    from scipy.optimize import linear_sum_assignment

    count = max(len(predicted), len(reference))
    if not count:
        return 1.0
    if not len(predicted) or not len(reference):
        return 0.0
    overlaps = iou_matrix(predicted, reference)
    rows, cols = linear_sum_assignment(overlaps, maximize=True)
    return float(overlaps[rows, cols].sum() / count)


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


def box_edit_counts(initial, final, tolerance=1e-4, match_iou=0.5):
    """Infer edits from two snapshots; no claim about actual editor actions."""
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(initial, dtype=float).reshape(-1, 4)
    b = np.asarray(final, dtype=float).reshape(-1, 4)
    unchanged = adjusted = 0
    if len(a) and len(b):
        same = np.max(np.abs(a[:, None] - b[None, :]), axis=-1) <= tolerance
        rows, cols = linear_sum_assignment(same.astype(int), maximize=True)
        pairs = [(i, j) for i, j in zip(rows, cols) if same[i, j]]
        unchanged = len(pairs)
        a = a[[i for i in range(len(a)) if i not in {p[0] for p in pairs}]]
        b = b[[j for j in range(len(b)) if j not in {p[1] for p in pairs}]]
    if len(a) and len(b):
        overlaps = iou_matrix(a, b)
        valid = overlaps >= match_iou
        # Prefer more valid pairs, then their total IoU.
        scores = valid * (min(len(a), len(b)) + 1 + overlaps)
        rows, cols = linear_sum_assignment(scores, maximize=True)
        adjusted = int(valid[rows, cols].sum())
    return {"initial": len(initial), "unchanged": unchanged, "adjusted": adjusted,
            "removed": len(a) - adjusted, "added": len(b) - adjusted}
