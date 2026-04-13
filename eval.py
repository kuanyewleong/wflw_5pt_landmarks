from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from datasets import WFLW5PointDataset
from models import LandmarkNet5
from train import validate
from utils import load_yaml



def main():
    cfg = load_yaml("configs/default.yaml")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = LandmarkNet5(num_points=5).to(device)
    ckpt = torch.load(f"{cfg['train']['save_dir']}/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = WFLW5PointDataset(
        data_root=cfg["data"]["data_root"],
        annotation_file=cfg["data"]["val_list"],
        image_size=cfg["data"]["image_size"],
        train=False,
        crop_scale=cfg["data"]["crop_scale"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    val_loss, val_nme = validate(model, loader, device)
    print(json.dumps({"val_loss": val_loss, "val_nme": val_nme}, indent=2))


if __name__ == "__main__":
    main()