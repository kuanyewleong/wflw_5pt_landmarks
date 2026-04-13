from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def smooth_l1_landmark_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=0.02)


def interocular_distance_from_5pt(pts5: np.ndarray) -> float:
    left_eye = pts5[0]
    right_eye = pts5[1]
    return float(np.linalg.norm(left_eye - right_eye) + 1e-6)


def nme_5pt(pred_abs: np.ndarray, gt_abs: np.ndarray) -> float:
    iod = interocular_distance_from_5pt(gt_abs)
    err = np.linalg.norm(pred_abs - gt_abs, axis=1).mean()
    return float(err / iod)


def decode_predictions_to_abs(pred_norm: torch.Tensor, crop_box: torch.Tensor) -> torch.Tensor:
    pred = pred_norm.view(-1, 5, 2)
    x1y1 = crop_box[:, None, 0:2]
    wh = (crop_box[:, None, 2:4] - crop_box[:, None, 0:2]).clamp(min=1e-6)
    return pred * wh + x1y1