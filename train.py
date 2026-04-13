from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import WFLW5PointDataset
from models import LandmarkNet5
from utils import decode_predictions_to_abs, ensure_dir, load_yaml, nme_5pt, set_seed, smooth_l1_landmark_loss


@dataclass
class AverageMeter:
    sum: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n


def validate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    nme_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            crop_box = batch["crop_box"].to(device, non_blocking=True)
            gt_abs = batch["landmarks5_abs"].cpu().numpy()

            pred = model(images)
            loss = smooth_l1_landmark_loss(pred, target)
            loss_meter.update(loss.item(), images.size(0))

            pred_abs = decode_predictions_to_abs(pred, crop_box).cpu().numpy()
            for p, g in zip(pred_abs, gt_abs):
                nme_meter.update(nme_5pt(p, g), 1)

    return loss_meter.avg, nme_meter.avg


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_nme, cfg):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_nme": best_nme,
            "config": cfg,
        },
        path,
    )


def main():
    cfg = load_yaml("configs/default.yaml")
    set_seed(cfg["seed"])
    ensure_dir(cfg["train"]["save_dir"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    train_ds = WFLW5PointDataset(
        data_root=cfg["data"]["data_root"],
        annotation_file=cfg["data"]["train_list"],
        image_size=cfg["data"]["image_size"],
        train=True,
        crop_scale=cfg["data"]["crop_scale"],
    )
    val_ds = WFLW5PointDataset(
        data_root=cfg["data"]["data_root"],
        annotation_file=cfg["data"]["val_list"],
        image_size=cfg["data"]["image_size"],
        train=False,
        crop_scale=cfg["data"]["crop_scale"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = LandmarkNet5(num_points=5).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    amp_enabled = cfg["train"]["amp"] and device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)

    best_nme = float("inf")
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(images)
                loss = smooth_l1_landmark_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(train_loss=f"{loss_meter.avg:.4f}")

        scheduler.step()
        val_loss, val_nme = validate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": loss_meter.avg,
            "val_loss": val_loss,
            "val_nme": val_nme,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))

        save_checkpoint(
            os.path.join(cfg["train"]["save_dir"], "last.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_nme,
            cfg,
        )

        if val_nme < best_nme:
            best_nme = val_nme
            save_checkpoint(
                os.path.join(cfg["train"]["save_dir"], "best.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                best_nme,
                cfg,
            )
            print(f"[INFO] New best checkpoint saved. NME={best_nme:.6f}")

    with open(os.path.join(cfg["train"]["save_dir"], "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()