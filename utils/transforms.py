from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np
import torch


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box.astype(np.float32)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def square_box(box: np.ndarray, scale: float = 1.25) -> np.ndarray:
    cx, cy, w, h = xyxy_to_cxcywh(box)
    side = max(w, h) * scale
    return cxcywh_to_xyxy(np.array([cx, cy, side, side], dtype=np.float32))


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    x2 = np.clip(x2, 0, width - 1)
    y2 = np.clip(y2, 0, height - 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def apply_affine_to_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts.astype(np.float32), ones], axis=1)
    out = hom @ M.T
    return out


def image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def draw_landmarks(image_bgr: np.ndarray, pts: np.ndarray, radius: int = 2) -> np.ndarray:
    out = image_bgr.copy()
    for x, y in pts.astype(np.int32):
        cv2.circle(out, (int(x), int(y)), radius, (0, 255, 0), -1)
    return out


class FaceAugmentor:
    def __init__(
        self,
        image_size: int,
        max_rotate: float = 20.0,
        max_shift: float = 0.08,
        max_scale_jitter: float = 0.15,
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
    ) -> None:
        self.image_size = image_size
        self.max_rotate = max_rotate
        self.max_shift = max_shift
        self.max_scale_jitter = max_scale_jitter
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob

    def color_jitter(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.color_jitter_prob:
            return img
        out = img.astype(np.float32)
        alpha = 1.0 + random.uniform(-0.25, 0.25)
        beta = random.uniform(-20, 20)
        out = out * alpha + beta
        if random.random() < 0.5:
            hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= 1.0 + random.uniform(-0.2, 0.2)
            hsv[..., 2] *= 1.0 + random.uniform(-0.15, 0.15)
            hsv[..., 0] += random.uniform(-8, 8)
            hsv[..., 0] = np.mod(hsv[..., 0], 180)
            hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def __call__(self, img: np.ndarray, pts5: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        center = np.array([self.image_size / 2.0, self.image_size / 2.0], dtype=np.float32)
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        scale = 1.0 + random.uniform(-self.max_scale_jitter, self.max_scale_jitter)
        tx = random.uniform(-self.max_shift, self.max_shift) * self.image_size
        ty = random.uniform(-self.max_shift, self.max_shift) * self.image_size

        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, scale)
        M[:, 2] += np.array([tx, ty], dtype=np.float32)

        out_img = cv2.warpAffine(
            img,
            M,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        out_pts = apply_affine_to_points(pts5, M)

        if random.random() < self.hflip_prob:
            out_img = cv2.flip(out_img, 1)
            out_pts[:, 0] = self.image_size - 1 - out_pts[:, 0]
            out_pts[[0, 1]] = out_pts[[1, 0]]
            out_pts[[3, 4]] = out_pts[[4, 3]]
        return out_img, out_pts