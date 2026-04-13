from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import (
    FaceAugmentor,
    apply_affine_to_points,
    clip_box,
    image_to_tensor,
    square_box,
)


LEFT_EYE_INDICES = list(range(60, 68))
RIGHT_EYE_INDICES = list(range(68, 76))
NOSE_TIP_INDEX = 54
LEFT_MOUTH_CORNER_INDEX = 76
RIGHT_MOUTH_CORNER_INDEX = 82


@dataclass
class Sample:
    image_path: str
    landmarks98: np.ndarray
    landmarks5: np.ndarray
    bbox: np.ndarray
    attrs: np.ndarray



def parse_wflw_line(line: str) -> Dict:
    parts = line.strip().split()
    if len(parts) < 196 + 4 + 6 + 1:
        raise ValueError(f"Unexpected WFLW line format, got {len(parts)} tokens")

    coords = np.array(list(map(float, parts[:196])), dtype=np.float32).reshape(98, 2)
    bbox = np.array(list(map(float, parts[196:200])), dtype=np.float32)
    attrs = np.array(list(map(int, parts[200:206])), dtype=np.int64)
    img_rel = parts[206]

    return {
        "landmarks98": coords,
        "bbox": bbox,
        "attrs": attrs,
        "img_rel": img_rel,
    }


def landmarks98_to_5(pts98: np.ndarray) -> np.ndarray:
    left_eye = pts98[LEFT_EYE_INDICES].mean(axis=0)
    right_eye = pts98[RIGHT_EYE_INDICES].mean(axis=0)
    nose_tip = pts98[NOSE_TIP_INDEX]
    left_mouth = pts98[LEFT_MOUTH_CORNER_INDEX]
    right_mouth = pts98[RIGHT_MOUTH_CORNER_INDEX]
    return np.stack([left_eye, right_eye, nose_tip, left_mouth, right_mouth], axis=0).astype(np.float32)


class WFLW5PointDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        image_size: int = 128,
        train: bool = True,
        crop_scale: float = 1.25,
        use_attr_filter: bool = False,
        allowed_attr_mask: Optional[Sequence[int]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.annotation_file = Path(annotation_file)
        self.image_size = image_size
        self.train = train
        self.crop_scale = crop_scale
        self.augment = FaceAugmentor(image_size=image_size) if train else None
        self.samples: List[Sample] = []

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                item = parse_wflw_line(line)
                attrs = item["attrs"]
                if use_attr_filter and allowed_attr_mask is not None:
                    if not all(attrs[i] == allowed_attr_mask[i] for i in range(min(len(attrs), len(allowed_attr_mask)))):
                        continue
                img_path = str(self.data_root / item["img_rel"])
                lmk98 = item["landmarks98"]
                lmk5 = landmarks98_to_5(lmk98)
                self.samples.append(
                    Sample(
                        image_path=img_path,
                        landmarks98=lmk98,
                        landmarks5=lmk5,
                        bbox=item["bbox"],
                        attrs=attrs,
                    )
                )


        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found from {annotation_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        image = cv2.imread(s.image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {s.image_path}")

        h, w = image.shape[:2]
        crop_box = square_box(s.bbox, self.crop_scale)
        crop_box = clip_box(crop_box, w, h)

        x1, y1, x2, y2 = crop_box.astype(np.float32)
        src = np.array([[x1, y1], [x2, y1], [x1, y2]], dtype=np.float32)
        dst = np.array([[0, 0], [self.image_size - 1, 0], [0, self.image_size - 1]], dtype=np.float32)
        M = cv2.getAffineTransform(src, dst)

        cropped = cv2.warpAffine(
            image,
            M,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        pts5_crop = apply_affine_to_points(s.landmarks5, M)

        if self.train and self.augment is not None:
            cropped, pts5_crop = self.augment(cropped, pts5_crop)

        pts5_norm = pts5_crop / float(self.image_size - 1)
        pts5_norm = np.clip(pts5_norm, 0.0, 1.0)

        return {
            "image": image_to_tensor(cropped),
            "target": torch.from_numpy(pts5_norm.reshape(-1)).float(),
            "target_pts": torch.from_numpy(pts5_norm).float(),
            "crop_box": torch.from_numpy(crop_box).float(),
            "orig_size": torch.tensor([w, h], dtype=torch.float32),
            "image_path": s.image_path,
            "landmarks5_abs": torch.from_numpy(s.landmarks5).float(),
        }