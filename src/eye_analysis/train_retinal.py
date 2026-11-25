"""
Retinal Fundus Diabetic Retinopathy Deep Learning Training & Model Selection Script.

Features:
- Multi-architecture transfer learning (`efficientnet_b0`, `resnet34`, `resnet18`,
  `mobilenet_v3_large`, `mobilenet_v3_small`, `efficientnet_b2`, `convnext_tiny`)
- 2-Stage Gradual Unfreezing (Stage 1: classifier head warmup; Stage 2: differential-LR fine-tuning)
- Class-Weighted Focal Cross-Entropy Loss for 5-grade ICDR class imbalance
- Early Stopping + CosineAnnealingLR + Weight Decay + Dropout
- Strict train/val discipline (test.csv & external_messidor2.csv never used in training or tuning)
- Validation Quadratic Weighted Kappa (QWK), Macro-F1, Accuracy tracking & candidate comparison
- Saves model checkpoint with `model_version`, `training_config.json`, `preprocessing_config.json`,
  `class_labels.json`, `training_history.json`, and `model_comparison.json`.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
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
MODEL_VERSION = "2.0.0-multicenter"


class RetinalFundusDataset(Dataset):
    """PyTorch Dataset reading real retinal fundus images from split CSV manifests."""

    def __init__(
        self,
        csv_path: str,
        transform=None,
        image_size: int = 224,
        apply_clahe: bool = True,
        cache_preprocessed: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.cache_preprocessed = cache_preprocessed
        self._cache: Dict[int, Image.Image] = {}

    def preload(self, max_workers: int = 8):
        """Preload all preprocessed fundus images into memory so epochs run purely in RAM."""
        from concurrent.futures import ThreadPoolExecutor

        def _load_one(idx):
            row = self.df.iloc[idx]
            img_abs = os.path.join(ROOT_DIR, str(row["image_path"]))
            with Image.open(img_abs) as im:
                rgb_np = np.asarray(im.convert("RGB"), dtype=np.uint8)
            self._cache[idx] = preprocess_fundus_rgb(
                rgb_np, target_size=self.image_size, apply_clahe=self.apply_clahe
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_load_one, range(len(self.df))))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        if self.cache_preprocessed and idx in self._cache:
            preprocessed_pil = self._cache[idx]
        else:
            img_abs = os.path.join(ROOT_DIR, str(row["image_path"]))
            with Image.open(img_abs) as im:
                rgb_np = np.asarray(im.convert("RGB"), dtype=np.uint8)
            preprocessed_pil = preprocess_fundus_rgb(
                rgb_np, target_size=self.image_size, apply_clahe=self.apply_clahe
            )
            if self.cache_preprocessed:
                self._cache[idx] = preprocessed_pil

        tensor = (
            self.transform(preprocessed_pil)
            if self.transform
            else get_eval_transforms(self.image_size)(preprocessed_pil)
        )
        label = int(row["label"])
        return tensor, label


def build_backbone_model(arch: str = "efficientnet_b0", num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """Build transfer-learning CNN architecture using PyTorch/Torchvision pretrained weights."""
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, num_classes))
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.35), nn.Linear(in_features, num_classes))
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.35), nn.Linear(in_features, num_classes))
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.40), nn.Linear(in_features, num_classes))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=0.35), nn.Linear(in_features, num_classes))
    elif arch == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(p=0.35), nn.Linear(in_features, num_classes))
    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(p=0.35), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    return model


def _split_head_and_backbone(model: nn.Module, arch: str) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Separate classifier head parameters from feature-extractor backbone parameters."""
    if arch.startswith("resnet"):
        head_params = list(model.fc.parameters())
    else:
        head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone_params, head_params


class WeightedFocalLoss(nn.Module):
    """Class-weighted Focal Cross-Entropy Loss for imbalanced multi-grade DR classification."""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 1.5, label_smoothing: float = 0.04):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal_factor = (1.0 - pt) ** self.gamma
        return torch.mean(focal_factor * ce_loss)


def _compute_class_weights(train_csv: str, num_classes: int = 5, power: float = 0.40) -> torch.Tensor:
    """Compute smoothed inverse-frequency weights countering imbalance without starving Grade 0."""
    df = pd.read_csv(train_csv)
    counts = np.bincount(df["label"].astype(int).values, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    raw_weights = (np.sum(counts) / (num_classes * counts)) ** power
    norm_weights = raw_weights / np.mean(raw_weights)
    return torch.tensor(norm_weights, dtype=torch.float32)


def _calibrate_any_dr_threshold(val_targets: np.ndarray, val_probs: np.ndarray) -> float:
    """Select binary Any-DR cutoff strictly on val split balancing Specificity and Sensitivity."""
    y_bin = (val_targets >= 1).astype(int)
    p_any = np.sum(val_probs[:, 1:], axis=1)
    best_thr = 0.50
    best_score = -1.0
    # Threshold search calibrated for balanced screening (sensitivity and specificity)
    for thr in np.linspace(0.40, 0.85, 46):
        preds = (p_any >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (y_bin == 1)))
        tn = int(np.sum((preds == 0) & (y_bin == 0)))
        fp = int(np.sum((preds == 1) & (y_bin == 0)))
        fn = int(np.sum((preds == 0) & (y_bin == 1)))
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal_acc = 0.50 * sens + 0.50 * spec
        if bal_acc > best_score:
            best_score = bal_acc
            best_thr = float(round(thr, 3))
    return best_thr


def predict_batch_tta(model: nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    """Run 4-view flip Test-Time Augmentation and average class probabilities."""
    p0 = torch.softmax(model(x_batch), dim=1)
    p1 = torch.softmax(model(torch.flip(x_batch, dims=[3])), dim=1)
    p2 = torch.softmax(model(torch.flip(x_batch, dims=[2])), dim=1)
    p3 = torch.softmax(model(torch.flip(x_batch, dims=[2, 3])), dim=1)
    return 0.25 * (p0 + p1 + p2 + p3)


def _plot_training_curves(history: list, out_png: str) -> None:
    """Save clean training/validation loss, accuracy, macro-F1, and QWK curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] * 100.0 for h in history]
    val_f1 = [h["val_macro_f1"] * 100.0 for h in history]
    val_qwk = [h["val_qwk"] * 100.0 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=1.8)
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="s", linewidth=1.8)
    axes[0].set_title("Loss Curve (Train vs Validation)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, val_acc, label="Val Accuracy (%)", marker="o", linewidth=1.8)
    axes[1].plot(epochs, val_f1, label="Val Macro F1 (%)", marker="s", linewidth=1.8)
    axes[1].plot(epochs, val_qwk, label="Val QWK (%)", marker="^", linewidth=1.8)
    axes[1].set_title("Validation Accuracy, Macro-F1 & QWK")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def _train_single_candidate(
    arch: str,
    train_ds: Dataset,
    val_ds: Dataset,
    class_weights: torch.Tensor,
    epochs: int = 10,
    warmup_epochs: int = 2,
    patience: int = 4,
    batch_size: int = 32,
    lr: float = 2.5e-4,
    loss_type: str = "focal",
    focal_gamma: float = 1.5,
    image_size: int = 224,
    apply_clahe: bool = True,
) -> Dict:
    """Train one candidate CNN with 2-stage gradual unfreezing and early stopping on val.csv."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cw = class_weights.to(device)
    if loss_type == "focal":
        criterion = WeightedFocalLoss(weight=cw, gamma=focal_gamma, label_smoothing=0.04).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05).to(device)

    model = build_backbone_model(arch=arch, num_classes=5, pretrained=True).to(device)
    backbone_params, head_params = _split_head_and_backbone(model, arch)

    # Stage 1: Freeze backbone, warm up classifier head
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=2e-4)
    scheduler = None

    print(
        f"\n[Train Retinal CNN] Arch={arch} | Device={device} | Epochs={epochs} (Warmup={warmup_epochs}) | "
        f"Loss={loss_type} | Train={len(train_ds)} | Val={len(val_ds)}",
        flush=True,
    )

    best_score = -1.0
    best_metrics = {}
    best_epoch = 0
    best_state = None
    best_val_probs = None
    best_val_targets = None
    epochs_without_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Stage 2 transition: Unfreeze full backbone with differential learning rates
        if epoch == warmup_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr},
                    {"params": head_params, "lr": lr * 2.5},
                ],
                weight_decay=3e-4,
            )
            ft_epochs = max(1, epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=1e-5)
            print(f"  -> Stage 2 Unfrozen Full Fine-Tuning (backbone lr={lr}, head lr={lr * 2.5})", flush=True)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item()) * x_batch.size(0)
            n_train += x_batch.size(0)

        if scheduler is not None:
            scheduler.step()
        train_loss = running_loss / max(1, n_train)

        # Validation phase (strictly val.csv, never test.csv)
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        val_preds, val_targets, val_prob_chunks = [], [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += float(loss.item()) * x_batch.size(0)
                n_val += x_batch.size(0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                val_prob_chunks.append(probs.cpu().numpy())
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(y_batch.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(1, n_val)
        val_acc = float(accuracy_score(val_targets, val_preds))
        val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
        val_qwk = float(cohen_kappa_score(val_targets, val_preds, weights="quadratic"))

        # Composite ordinal + class-balanced selection metric
        selection_score = 0.45 * max(0.0, val_qwk) + 0.35 * val_macro_f1 + 0.20 * val_acc
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"ValAcc={val_acc * 100:.2f}% | ValF1={val_macro_f1 * 100:.2f}% | ValQWK={val_qwk:.4f}",
            flush=True,
        )

        history.append(
            {
                "epoch": epoch,
                "stage": "warmup_head" if epoch <= warmup_epochs else "full_finetune",
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "val_qwk": round(val_qwk, 4),
                "selection_score": round(selection_score, 4),
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs = np.vstack(val_prob_chunks)
            best_val_targets = np.array(val_targets, dtype=int)
            best_metrics = {
                "val_accuracy": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "val_qwk": round(val_qwk, 4),
                "val_loss": round(val_loss, 4),
                "selection_score": round(selection_score, 4),
            }
            epochs_without_improve = 0
        else:
            if epoch > warmup_epochs:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"  [Early Stopping] No validation improvement for {patience} epochs after epoch {best_epoch}.", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        tta_chunks, tta_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                probs_tta = predict_batch_tta(model, x_batch.to(device))
                tta_chunks.append(probs_tta.cpu().numpy())
                tta_targets.extend(y_batch.cpu().numpy().tolist())
        if tta_chunks:
            best_val_probs = np.vstack(tta_chunks)
            best_val_targets = np.array(tta_targets, dtype=int)

    any_dr_threshold = (
        _calibrate_any_dr_threshold(best_val_targets, best_val_probs)
        if best_val_probs is not None
        else 0.52
    )

    return {
        "arch": arch,
        "best_epoch": best_epoch,
        "best_score": round(best_score, 4),
        "best_metrics": best_metrics,
        "any_dr_threshold": any_dr_threshold,
        "state_dict": best_state,
        "history": history,
        "image_size": image_size,
        "apply_clahe": apply_clahe,
    }


def save_winning_model(
    winner: Dict,
    train_ds: Dataset,
    val_ds: Dataset,
    class_weights: torch.Tensor,
    comparison_summary: list,
    loss_type: str,
    focal_gamma: float,
    warmup_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
) -> Dict:
    """Persist winning checkpoint, configs, class mapping, training curves, and comparison table."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    arch = winner["arch"]
    image_size = winner["image_size"]
    apply_clahe = winner["apply_clahe"]
    model_path = os.path.join(MODELS_DIR, "retinal_dr_model.pth")

    torch.save(
        {
            "model_version": MODEL_VERSION,
            "arch": arch,
            "num_classes": 5,
            "image_size": image_size,
            "apply_clahe": apply_clahe,
            "best_epoch": winner["best_epoch"],
            "best_val_selection_score": winner["best_score"],
            "best_val_metrics": winner["best_metrics"],
            "any_dr_threshold": winner["any_dr_threshold"],
            "use_tta": True,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state_dict": winner["state_dict"],
        },
        model_path,
    )

    preproc_cfg = get_preprocessing_config(image_size=image_size, apply_clahe=apply_clahe)
    preproc_cfg["model_version"] = MODEL_VERSION
    preproc_cfg["backbone_architecture"] = arch
    preproc_cfg["any_dr_threshold"] = winner["any_dr_threshold"]
    preproc_cfg["use_tta"] = True
    with open(os.path.join(MODELS_DIR, "preprocessing_config.json"), "w", encoding="utf-8") as f:
        json.dump(preproc_cfg, f, indent=2)

    training_cfg = {
        "model_version": MODEL_VERSION,
        "selected_architecture": arch,
        "loss_function": f"WeightedFocalLoss(gamma={focal_gamma})" if loss_type == "focal" else "WeightedCrossEntropyLoss",
        "optimizer": "AdamW",
        "gradual_unfreezing": True,
        "warmup_head_epochs": warmup_epochs,
        "finetune_backbone_lr": lr,
        "finetune_head_lr": lr * 2.5,
        "weight_decay": 3e-4,
        "scheduler": "CosineAnnealingLR",
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "class_weights": [round(float(w), 4) for w in class_weights.cpu().numpy()],
        "any_dr_threshold": winner["any_dr_threshold"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(MODELS_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_cfg, f, indent=2)

    class_labels_payload = {
        "labels": {str(k): v for k, v in DR_CLASS_NAMES.items()},
        "clinical_scale": "International Clinical Diabetic Retinopathy (ICDR) Severity Scale",
        "referable_dr_grades": [2, 3, 4],
        "model_version": MODEL_VERSION,
    }
    with open(os.path.join(MODELS_DIR, "class_labels.json"), "w", encoding="utf-8") as f:
        json.dump(class_labels_payload, f, indent=2)

    train_meta = {
        "model_version": MODEL_VERSION,
        "architecture": arch,
        "best_epoch": winner["best_epoch"],
        "epochs_total": len(winner["history"]),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_metrics": winner["best_metrics"],
        "class_weights": training_cfg["class_weights"],
        "history": winner["history"],
    }
    with open(os.path.join(MODELS_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    if comparison_summary:
        with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "compared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "selection_criterion": "0.45*Val_QWK + 0.35*Val_MacroF1 + 0.20*Val_Accuracy",
                    "selected_winner": arch,
                    "candidates": comparison_summary,
                },
                f,
                indent=2,
            )

    _plot_training_curves(winner["history"], os.path.join(MODELS_DIR, "training_history.png"))
    print(f"[Train Retinal CNN] Saved {MODEL_VERSION} ({arch}, epoch {winner['best_epoch']}) -> {model_path}", flush=True)
    return train_meta


def train_retinal_model(
    arch: str = "efficientnet_b0",
    compare_archs: Optional[List[str]] = None,
    epochs: int = 10,
    warmup_epochs: int = 2,
    patience: int = 4,
    batch_size: int = 32,
    lr: float = 2.5e-4,
    image_size: int = 224,
    apply_clahe: bool = True,
    loss_type: str = "focal",
    focal_gamma: float = 1.5,
    subsample_train: Optional[int] = None,
    subsample_val: Optional[int] = None,
) -> Dict:
    """Train and optionally compare candidate architectures strictly using train.csv & val.csv."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_csv = os.path.join(RETINAL_DIR, "train.csv")
    val_csv = os.path.join(RETINAL_DIR, "val.csv")
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        prepare_retinamnist()

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

    if subsample_train:
        from src.eye_analysis.dataset_prep import _stratified_cap
        # Crucial clinical screening balance: Ensure Grade 0 (Healthy) represents ~45-50%
        # of training data so the model does NOT over-predict Mild DR on normal healthy eyes
        train_ds.df = _stratified_cap(
            train_ds.df,
            {0: 680, 1: 180, 2: 240, 3: 160, 4: 150},
            seed=42,
            prioritize_sources=("idrid", "aptos2019"),
        )
    if subsample_val:
        from src.eye_analysis.dataset_prep import _stratified_cap
        val_ds.df = _stratified_cap(
            val_ds.df,
            {0: 150, 1: 45, 2: 55, 3: 35, 4: 35},
            seed=42,
            prioritize_sources=("idrid", "aptos2019"),
        )

    print(f"[Train Retinal CNN] Preloading {len(train_ds)} train & {len(val_ds)} val images into memory...", flush=True)
    t0_pre = time.time()
    train_ds.preload(max_workers=8)
    val_ds.preload(max_workers=8)
    print(f"[Train Retinal CNN] Preloaded in {time.time() - t0_pre:.1f}s. All epochs run purely from RAM!", flush=True)

    # Class weights balancing healthy specificity with high-grade disease sensitivity
    class_weights = torch.tensor([1.20, 1.00, 1.00, 1.10, 1.15], dtype=torch.float32)

    candidates = compare_archs if compare_archs else [arch]
    results = []
    comparison_summary = []

    for cand_arch in candidates:
        res = _train_single_candidate(
            arch=cand_arch,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            image_size=image_size,
            apply_clahe=apply_clahe,
        )
        results.append(res)
        comparison_summary.append(
            {
                "arch": cand_arch,
                "best_epoch": res["best_epoch"],
                "any_dr_threshold": res["any_dr_threshold"],
                **res["best_metrics"],
            }
        )

    winner = max(results, key=lambda r: r["best_score"])
    print(f"\n[Model Selection] Winner on Validation Set: {winner['arch']} (score={winner['best_score']}, metrics={winner['best_metrics']})", flush=True)
    return save_winning_model(
        winner=winner,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        comparison_summary=comparison_summary,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        warmup_epochs=warmup_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
    )


def main():
    parser = argparse.ArgumentParser(description="Train & Compare Retinal Fundus DR Classification CNNs")
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet_b0",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v2", "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b2", "convnext_tiny"],
    )
    parser.add_argument("--compare", nargs="+", default=None, help="Optional list of architectures to compare on val.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "ce"])
    parser.add_argument("--subsample-train", type=int, default=None, help="Cap training set size for fast CPU execution")
    parser.add_argument("--subsample-val", type=int, default=None, help="Cap validation set size for fast CPU execution")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--no-clahe", action="store_true")
    args = parser.parse_args()
    train_retinal_model(
        arch=args.arch,
        compare_archs=args.compare,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss,
        focal_gamma=args.gamma,
        apply_clahe=not args.no_clahe,
        subsample_train=args.subsample_train,
        subsample_val=args.subsample_val,
    )


if __name__ == "__main__":
    main()
