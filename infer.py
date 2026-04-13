from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from models import LandmarkNet5
from utils.common import load_yaml
from utils.transforms import apply_affine_to_points, clip_box, cxcywh_to_xyxy, draw_landmarks, image_to_tensor, square_box


def detect_or_center_crop_face(image: np.ndarray, margin_scale: float = 1.25) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    h, w = image.shape[:2]
    if len(faces) > 0:
        areas = [fw * fh for (_, _, fw, fh) in faces]
        i = int(np.argmax(areas))
        x, y, fw, fh = faces[i]
        box = np.array([x, y, x + fw, y + fh], dtype=np.float32)
    else:
        side = min(h, w) * 0.8
        cx, cy = w / 2.0, h / 2.0
        box = cxcywh_to_xyxy(np.array([cx, cy, side, side], dtype=np.float32))

    return clip_box(square_box(box, margin_scale), w, h)



def infer_single(model, image_bgr, image_size, device, face_box=None):
    if face_box is None:
        face_box = detect_or_center_crop_face(image_bgr)

    x1, y1, x2, y2 = face_box.astype(np.float32)
    src = np.array([[x1, y1], [x2, y1], [x1, y2]], dtype=np.float32)
    dst = np.array([[0, 0], [image_size - 1, 0], [0, image_size - 1]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(
        image_bgr,
        M,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    inp = image_to_tensor(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).view(1, 5, 2)

    pred_norm = pred[0].cpu().numpy()
    pred_crop = pred_norm * float(image_size - 1)
    Minv = cv2.invertAffineTransform(M)
    pred_abs = apply_affine_to_points(pred_crop, Minv)
    return pred_abs.astype(np.float32), face_box.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="infer_result.jpg")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml("configs/default.yaml")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or f"{cfg['train']['save_dir']}/best.pt"
    model = LandmarkNet5(num_points=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    pred_abs, box = infer_single(model, image, cfg["data"]["image_size"], device)
    vis = image.copy()
    x1, y1, x2, y2 = box.astype(np.int32)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    vis = draw_landmarks(vis, pred_abs, radius=3)
    cv2.imwrite(args.output, vis)

    print("Predicted landmarks:")
    print(pred_abs)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
