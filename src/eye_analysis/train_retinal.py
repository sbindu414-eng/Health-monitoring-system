"""
Retinal Fundus Diabetic Retinopathy Deep Learning Training Script.

Uses Transfer Learning (MobileNetV3 / ResNet18 / EfficientNet-B0) trained on the
verified retinal fundus dataset (`data/retinal_fundus/train.csv` and `val.csv`).
- Test split (`test.csv`) is strictly excluded from training and checkpoint selection.
- Handles class imbalance via inverse-frequency class weights in CrossEntropyLoss.
- Applies data augmentation strictly to training images.
- Saves trained weights, preprocessing config, class-label mapping, and plots.
"""

import argparse
import json
import os
import time
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from src.eye_analysis.dataset_prep import DR_CLASS_NAMES, RETINAL_DIR, ROOT_DIR, prepare_retinamnist
from src.eye_analysis.preprocessing import (
    get_eval_transforms,
    get_preprocessing_config,
    get_train_transforms,
    preprocess_fundus_rgb,
)

MODELS_DIR = os.path.join(ROOT_DIR, "models")


class RetinalFundusDataset(Dataset):
    """PyTorch Dataset reading real retinal fundus images from split CSV manifests."""

    def __init__(self, csv_path: str, transform=None, image_size: int = 224, apply_clahe: bool = True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size
        self.apply_clahe = apply_clahe

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_abs = os.path.join(ROOT_DIR, str(row["image_path"]))
        with Image.open(img_abs) as im:
            rgb_np = np.asarray(im.convert("RGB"), dtype=np.uint8)

        preprocessed_pil = preprocess_fundus_rgb(
            rgb_np, target_size=self.image_size, apply_clahe=self.apply_clahe
        )
        tensor = self.transform(preprocessed_pil) if self.transform else get_eval_transforms(self.image_size)(preprocessed_pil)
        label = int(row["label"])
        return tensor, label


def build_backbone_model(arch: str = "mobilenet_v3_small", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Build transfer learning architecture using PyTorch/Torchvision pretrained weights."""
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose mobilenet_v3_small, mobilenet_v2, resnet18, or efficientnet_b0.")
    return model


def _compute_class_weights(train_csv: str, num_classes: int = 5) -> torch.Tensor:
    """Compute smoothed inverse-frequency weights to counter DR class imbalance."""
    df = pd.read_csv(train_csv)
    counts = np.bincount(df["label"].astype(int).values, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    # Square-root smoothed inverse class weighting prevents extreme loss instability on tiny classes
    raw_weights = (np.sum(counts) / (num_classes * counts)) ** 0.6
    norm_weights = raw_weights / np.mean(raw_weights)
    return torch.tensor(norm_weights, dtype=torch.float32)


def _plot_training_curves(history: list, out_png: str) -> None:
    """Save clean training/validation loss & macro-F1 curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] * 100.0 for h in history]
    val_f1 = [h["val_macro_f1"] * 100.0 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=1.8)
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="s", linewidth=1.8)
    axes[0].set_title("Cross-Entropy Loss (Train vs Val)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, val_acc, label="Val Accuracy (%)", marker="o", linewidth=1.8)
    axes[1].plot(epochs, val_f1, label="Val Macro F1 (%)", marker="s", linewidth=1.8)
    axes[1].set_title("Validation Accuracy & Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def train_retinal_model(
    arch: str = "mobilenet_v3_small",
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 4e-4,
    image_size: int = 224,
    apply_clahe: bool = True,
) -> Dict:
    """Train transfer-learning retinal DR grading CNN on prepared train/val splits."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_csv = os.path.join(RETINAL_DIR, "train.csv")
    val_csv = os.path.join(RETINAL_DIR, "val.csv")
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        prepare_retinamnist()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train Retinal CNN] Architecture={arch} | Device={device} | Epochs={epochs} | Batch={batch_size}")

    train_ds = RetinalFundusDataset(
        train_csv,
        transform=get_train_transforms(image_size),
        image_size=image_size,
        apply_clahe=apply_clahe,
    )
    val_ds = RetinalFundusDataset(
        val_csv,
        transform=get_eval_transforms(image_size),
        image_size=image_size,
        apply_clahe=apply_clahe,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = _compute_class_weights(train_csv, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = build_backbone_model(arch=arch, num_classes=5, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x_batch.size(0)
            n_train += x_batch.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, n_train)

        # Validation phase (strictly val.csv, never test.csv)
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += float(loss.item()) * x_batch.size(0)
                n_val += x_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(y_batch.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(1, n_val)
        val_acc = float(accuracy_score(val_targets, val_preds))
        val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))

        # Balanced selection score weighting macro F1 and accuracy
        selection_score = 0.60 * val_macro_f1 + 0.40 * val_acc
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
            f"Val Acc={val_acc * 100:.2f}% | Val Macro-F1={val_macro_f1 * 100:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model checkpoint, preprocessing config, and class mapping
    model_path = os.path.join(MODELS_DIR, "retinal_dr_model.pth")
    torch.save(
        {
            "arch": arch,
            "num_classes": 5,
            "image_size": image_size,
            "apply_clahe": apply_clahe,
            "best_epoch": best_epoch,
            "best_val_selection_score": round(best_score, 4),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state_dict": best_state,
        },
        model_path,
    )

    preproc_cfg = get_preprocessing_config(image_size=image_size, apply_clahe=apply_clahe)
    preproc_cfg["backbone_architecture"] = arch
    with open(os.path.join(MODELS_DIR, "preprocessing_config.json"), "w", encoding="utf-8") as f:
        json.dump(preproc_cfg, f, indent=2)

    class_labels_payload = {
        "labels": {str(k): v for k, v in DR_CLASS_NAMES.items()},
        "clinical_scale": "International Clinical Diabetic Retinopathy (ICDR) Severity Scale",
        "referable_dr_grades": [2, 3, 4],
    }
    with open(os.path.join(MODELS_DIR, "class_labels.json"), "w", encoding="utf-8") as f:
        json.dump(class_labels_payload, f, indent=2)

    train_meta = {
        "architecture": arch,
        "best_epoch": best_epoch,
        "epochs_total": epochs,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "class_weights": [round(float(w), 4) for w in class_weights.cpu().numpy()],
        "history": history,
    }
    with open(os.path.join(MODELS_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    _plot_training_curves(history, os.path.join(MODELS_DIR, "training_history.png"))
    print(f"[Train Retinal CNN] Saved best weights from epoch {best_epoch} -> {model_path}")
    return train_meta


def main():
    parser = argparse.ArgumentParser(description="Train Retinal Fundus DR Classification CNN")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v2", "resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()
    train_retinal_model(arch=args.arch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
