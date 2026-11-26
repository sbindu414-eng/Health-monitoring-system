"""
Training, Validation-Based Threshold Calibration & Independent Evaluation
for the Human Retinal Fundus Image Validator Model.

Compares candidate pretrained architectures (e.g., `mobilenet_v3_small` vs `resnet18`)
strictly on `data/retinal_validator/val.csv`, selects the winner balancing validation
ROC-AUC, F1, False Acceptance Rate (FAR), and inference latency, calibrates the
acceptance & uncertain abstention thresholds on `val.csv`, and evaluates once on
held-out `data/retinal_validator/test.csv` + external `external_messidor2_pos.csv`.
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
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from src.eye_analysis.dataset_prep import ROOT_DIR
from src.eye_analysis.preprocessing import get_eval_transforms, get_train_transforms
from src.eye_analysis.retinal_validator import (
    MODELS_DIR,
    VALIDATOR_CFG_PATH,
    VALIDATOR_CMP_PATH,
    VALIDATOR_EVAL_PATH,
    VALIDATOR_MODEL_PATH,
    predict_validator_batch_tta,
    preprocess_validator_rgb,
)
from src.eye_analysis.train_retinal import _split_head_and_backbone, build_backbone_model

VALIDATOR_DATA_DIR = os.path.join(ROOT_DIR, "data", "retinal_validator")
VALIDATOR_VERSION = "1.0.0"


class RetinalValidatorDataset(Dataset):
    """Dataset reading `train/val/test.csv` for binary Retinal Fundus vs Non-Retinal validation."""

    def __init__(
        self,
        csv_path: str,
        transform=None,
        image_size: int = 224,
        cache_preprocessed: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size
        self.cache_preprocessed = cache_preprocessed
        self._cache: Dict[int, Image.Image] = {}

    def preload(self, max_workers: int = 8):
        """Decode and pad all images into RAM once so training epochs run fast."""
        from concurrent.futures import ThreadPoolExecutor

        def _load_one(idx: int):
            row = self.df.iloc[idx]
            abs_p = os.path.join(ROOT_DIR, str(row["image_path"]))
            with Image.open(abs_p) as im:
                rgb_np = np.asarray(im.convert("RGB"), dtype=np.uint8)
            self._cache[idx] = preprocess_validator_rgb(rgb_np, target_size=self.image_size)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_load_one, range(len(self.df))))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.cache_preprocessed and idx in self._cache:
            prep_pil = self._cache[idx]
        else:
            row = self.df.iloc[idx]
            abs_p = os.path.join(ROOT_DIR, str(row["image_path"]))
            with Image.open(abs_p) as im:
                rgb_np = np.asarray(im.convert("RGB"), dtype=np.uint8)
            prep_pil = preprocess_validator_rgb(rgb_np, target_size=self.image_size)
            if self.cache_preprocessed:
                self._cache[idx] = prep_pil

        x = self.transform(prep_pil) if self.transform else get_eval_transforms(self.image_size)(prep_pil)
        y = int(self.df.iloc[idx]["label"])
        return x, y


def _calibrate_thresholds_on_val(y_val: np.ndarray, p_retinal: np.ndarray) -> Tuple[float, float]:
    """
    Determine acceptance cutoff (`valid_threshold`) and abstention lower bound (`uncertain_low`)
    strictly on validation probabilities:
      - Pick the acceptance threshold in [0.65, 0.94] minimizing False Acceptance Rate (FAR)
        with target FAR <= 0.5% while preserving high retinal Recall/F1.
      - Pick `uncertain_low` in [0.25, 0.45] to isolate ambiguous borderline inputs cleanly.
    """
    best_thr = 0.80
    best_obj = -1e9
    for thr in np.linspace(0.65, 0.94, 59):
        preds = (p_retinal >= thr).astype(int)
        tn = int(np.sum((preds == 0) & (y_val == 0)))
        fp = int(np.sum((preds == 1) & (y_val == 0)))
        tp = int(np.sum((preds == 1) & (y_val == 1)))
        fn = int(np.sum((preds == 0) & (y_val == 1)))
        far = fp / max(1, fp + tn)
        rec = tp / max(1, tp + fn)
        f1 = float(f1_score(y_val, preds, zero_division=0))
        # Heavily penalize False Acceptance of non-retinal images (e.g., animal eyes),
        # breaking exact ties in favor of higher acceptance margin up to 0.85
        far_penalty = 8.0 * far if far <= 0.006 else (8.0 * far + 5.0 * (far - 0.006))
        margin_bonus = 0.015 * min(float(thr), 0.85)
        obj = f1 + 0.35 * rec - far_penalty + margin_bonus
        if obj > best_obj:
            best_obj = obj
            best_thr = float(round(thr, 3))

    # Calibrate uncertain_low where negative specificity stays high and ambiguous cases abstain
    best_unc = 0.35
    best_u_score = -1e9
    for u_thr in np.linspace(0.30, min(0.50, best_thr - 0.15), 21):
        preds_u = (p_retinal >= u_thr).astype(int)
        # Minimize false negatives at u_thr while preventing excessive abstention band
        fn_u = int(np.sum((preds_u == 0) & (y_val == 1))) / max(1, int(np.sum(y_val == 1)))
        abstain_rate = float(np.mean((p_retinal >= u_thr) & (p_retinal < best_thr)))
        u_score = -3.0 * fn_u - 0.35 * abstain_rate
        if u_score > best_u_score:
            best_u_score = u_score
            best_unc = float(round(u_thr, 3))

    return best_thr, best_unc


def _train_validator_candidate(
    arch: str,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = 6,
    warmup_epochs: int = 1,
    patience: int = 3,
    batch_size: int = 48,
    lr: float = 2.5e-4,
    image_size: int = 224,
) -> Dict:
    """Train one candidate binary validator CNN with 2-stage gradual unfreezing."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Slightly emphasize Negative class (0) precision so false acceptance of non-retinal eyes stays near 0
    labels_arr = train_ds.df["label"].astype(int).values
    c0 = max(1, int(np.sum(labels_arr == 0)))
    c1 = max(1, int(np.sum(labels_arr == 1)))
    total = c0 + c1
    w0 = (total / (2.0 * c0)) * 1.15
    w1 = (total / (2.0 * c1)) * 1.00
    mean_w = 0.5 * (w0 + w1)
    class_weights = torch.tensor([w0 / mean_w, w1 / mean_w], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03).to(device)

    model = build_backbone_model(arch=arch, num_classes=2, pretrained=True).to(device)
    backbone_params, head_params = _split_head_and_backbone(model, arch)

    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=2e-4)
    scheduler = None

    print(
        f"\n[Train Retinal Validator] Arch={arch} | Device={device} | Train={len(train_ds)} | Val={len(val_ds)}",
        flush=True,
    )

    best_score = -1e9
    best_epoch = 0
    best_state = None
    best_metrics = {}
    history = []
    no_improve = 0

    for epoch in range(1, epochs + 1):
        if epoch == warmup_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [{"params": backbone_params, "lr": lr}, {"params": head_params, "lr": lr * 2.5}],
                weight_decay=3e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-5)

        t0 = time.time()
        model.train()
        running_loss, n_tr = 0.0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item()) * x_b.size(0)
            n_tr += x_b.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss = running_loss / max(1, n_tr)

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        val_probs_list, val_targets_list = [], []
        t_inf0 = time.time()
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss_sum += float(loss.item()) * x_b.size(0)
                n_val += x_b.size(0)
                probs = torch.softmax(logits, dim=1)
                val_probs_list.append(probs[:, 1].cpu().numpy())
                val_targets_list.extend(y_b.cpu().numpy().tolist())
        ms_per_img = ((time.time() - t_inf0) * 1000.0) / max(1, n_val)

        p_ret = np.concatenate(val_probs_list)
        y_val = np.array(val_targets_list, dtype=int)
        preds_05 = (p_ret >= 0.50).astype(int)
        preds_075 = (p_ret >= 0.75).astype(int)

        val_loss = val_loss_sum / max(1, n_val)
        val_acc = float(accuracy_score(y_val, preds_05))
        val_f1 = float(f1_score(y_val, preds_05, zero_division=0))
        val_auc = float(roc_auc_score(y_val, p_ret)) if len(np.unique(y_val)) == 2 else 0.5
        far_075 = float(np.sum((preds_075 == 1) & (y_val == 0)) / max(1, int(np.sum(y_val == 0))))
        frr_075 = float(np.sum((preds_075 == 0) & (y_val == 1)) / max(1, int(np.sum(y_val == 1))))

        # Selection score prioritizes ROC-AUC + F1 + near-zero False Acceptance Rate
        sel_score = 0.45 * val_auc + 0.35 * val_f1 + 0.20 * val_acc - 0.50 * far_075
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | TrLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | ValAcc={val_acc*100:.2f}% | ValAUC={val_auc:.4f} | "
            f"FAR@0.75={far_075*100:.2f}% | FRR@0.75={frr_075*100:.2f}% | {ms_per_img:.1f}ms/img",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_roc_auc": round(val_auc, 4),
            "val_far_at_075": round(far_075, 4),
            "val_frr_at_075": round(frr_075, 4),
            "ms_per_image": round(ms_per_img, 2),
        })

        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(history[-1], selection_score=round(sel_score, 4))
            no_improve = 0
        else:
            if epoch > warmup_epochs:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [Early Stopping] Stopping after epoch {best_epoch}", flush=True)
                    break

    # Final validation calibration with 2-view TTA strictly on val_ds
    model.load_state_dict(best_state)
    model.eval()
    tta_probs, tta_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            p_b = predict_validator_batch_tta(model, x_b.to(device))
            tta_probs.append(p_b[:, 1].cpu().numpy())
            tta_targets.extend(y_b.numpy().tolist())
    val_p_tta = np.concatenate(tta_probs)
    val_y_tta = np.array(tta_targets, dtype=int)
    valid_thr, uncertain_low = _calibrate_thresholds_on_val(val_y_tta, val_p_tta)

    val_preds_cal = (val_p_tta >= valid_thr).astype(int)
    val_cal_far = float(np.sum((val_preds_cal == 1) & (val_y_tta == 0)) / max(1, int(np.sum(val_y_tta == 0))))
    val_cal_frr = float(np.sum((val_preds_cal == 0) & (val_y_tta == 1)) / max(1, int(np.sum(val_y_tta == 1))))
    val_abstain = float(np.mean((val_p_tta >= uncertain_low) & (val_p_tta < valid_thr)))
    print(
        f"  [Val Calibration] arch={arch} | valid_thr={valid_thr} | uncertain_low={uncertain_low} | "
        f"Val FAR={val_cal_far*100:.2f}% | Val FRR={val_cal_frr*100:.2f}% | Val Abstain={val_abstain*100:.2f}%",
        flush=True,
    )

    param_mb = sum(p.numel() for p in model.parameters()) * 4.0 / (1024.0 * 1024.0)
    return {
        "arch": arch,
        "best_epoch": best_epoch,
        "best_score": round(best_score, 4),
        "best_metrics": best_metrics,
        "valid_threshold": valid_thr,
        "uncertain_low": uncertain_low,
        "val_cal_far": round(val_cal_far, 4),
        "val_cal_frr": round(val_cal_frr, 4),
        "param_size_mb": round(param_mb, 2),
        "state_dict": best_state,
        "history": history,
        "image_size": image_size,
    }


def _plot_validator_cm_and_history(cm: np.ndarray, history: List[Dict]) -> None:
    """Save Retinal Validator confusion matrix and training curves plots."""
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    labels = ["Invalid / Non-Retinal (0)", "Valid Retinal Fundus (1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Class",
        xlabel="Predicted Decision",
        title="Retinal Fundus Validator Test Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "retinal_validator_confusion_matrix.png"), dpi=140)
    plt.close(fig)

    epochs = [h["epoch"] for h in history]
    fig2, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="Train Loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="s", label="Val Loss")
    axes[0].set_title("Validator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [h["val_roc_auc"] * 100 for h in history], marker="o", label="Val ROC-AUC (%)")
    axes[1].plot(epochs, [h["val_accuracy"] * 100 for h in history], marker="s", label="Val Acc (%)")
    axes[1].plot(epochs, [h["val_far_at_075"] * 100 for h in history], marker="^", label="Val FAR@0.75 (%)")
    axes[1].set_title("Validator Discriminative Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "retinal_validator_history.png"), dpi=140)
    plt.close(fig2)


def evaluate_validator_on_test(
    model_path: str = VALIDATOR_MODEL_PATH,
    test_csv: str = os.path.join(VALIDATOR_DATA_DIR, "test.csv"),
    external_pos_csv: str = os.path.join(VALIDATOR_DATA_DIR, "external_messidor2_pos.csv"),
) -> Dict:
    """Evaluate trained Retinal Fundus Validator strictly on unseen held-out test splits."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    image_size = int(ckpt.get("image_size", 224))
    valid_thr = float(ckpt.get("valid_threshold", 0.80))
    unc_low = float(ckpt.get("uncertain_low", 0.35))
    use_tta = bool(ckpt.get("use_tta", True))

    model = build_backbone_model(arch=arch, num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = RetinalValidatorDataset(test_csv, transform=get_eval_transforms(image_size), image_size=image_size, cache_preprocessed=False)
    loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=0)

    probs_all, targets_all = [], []
    t0 = time.time()
    with torch.no_grad():
        for x_b, y_b in loader:
            p_b = predict_validator_batch_tta(model, x_b) if use_tta else torch.softmax(model(x_b), dim=1)
            probs_all.append(p_b[:, 1].numpy())
            targets_all.extend(y_b.numpy().tolist())
    elapsed_ms = (time.time() - t0) * 1000.0 / max(1, len(test_ds))

    p_ret = np.concatenate(probs_all)
    y_true = np.array(targets_all, dtype=int)
    y_pred = (p_ret >= valid_thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, p_ret)) if len(np.unique(y_true)) == 2 else 0.5
    far = fp / max(1, fp + tn)
    frr = fn / max(1, fn + tp)

    # Breakdown including abstention / uncertain band [unc_low, valid_thr)
    unc_mask = (p_ret >= unc_low) & (p_ret < valid_thr)
    abstained_total = int(np.sum(unc_mask))
    abstained_neg = int(np.sum(unc_mask & (y_true == 0)))
    abstained_pos = int(np.sum(unc_mask & (y_true == 1)))
    definitive_invalid_pos_fn = int(np.sum((p_ret < unc_low) & (y_true == 1)))

    # Per-subcategory metrics & False Acceptance Rates
    per_subcat = {}
    if "subcategory" in test_ds.df.columns:
        subcats = test_ds.df["subcategory"].astype(str).values
        for sc in sorted(np.unique(subcats)):
            m = subcats == sc
            yt_s = y_true[m]
            pr_s = p_ret[m]
            yp_s = y_pred[m]
            is_pos_cat = bool(np.mean(yt_s) >= 0.5)
            n_s = int(np.sum(m))
            accepted = int(np.sum(yp_s == 1))
            abstained = int(np.sum((pr_s >= unc_low) & (pr_s < valid_thr)))
            rejected = int(np.sum(pr_s < unc_low))
            per_subcat[sc] = {
                "samples": n_s,
                "expected_class": "valid_retinal" if is_pos_cat else "invalid_non_retinal",
                "accepted_as_retinal": accepted,
                "abstained_uncertain": abstained,
                "rejected_non_retinal": rejected,
                "false_acceptance_rate": round(accepted / max(1, n_s), 4) if not is_pos_cat else None,
                "rejection_rate": round((abstained + rejected) / max(1, n_s), 4) if not is_pos_cat else None,
                "retinal_acceptance_rate": round(accepted / max(1, n_s), 4) if is_pos_cat else None,
                "mean_p_retinal": round(float(np.mean(pr_s)), 4),
            }

    # Zero-shot external Messidor-2 positive generalization
    messidor_eval = None
    if os.path.exists(external_pos_csv):
        ext_ds = RetinalValidatorDataset(external_pos_csv, transform=get_eval_transforms(image_size), image_size=image_size, cache_preprocessed=False)
        ext_loader = DataLoader(ext_ds, batch_size=48, shuffle=False, num_workers=0)
        ext_probs = []
        with torch.no_grad():
            for x_b, _ in ext_loader:
                p_b = predict_validator_batch_tta(model, x_b) if use_tta else torch.softmax(model(x_b), dim=1)
                ext_probs.append(p_b[:, 1].numpy())
        ep = np.concatenate(ext_probs)
        n_ext = len(ep)
        acc_ext = int(np.sum(ep >= valid_thr))
        abs_ext = int(np.sum((ep >= unc_low) & (ep < valid_thr)))
        rej_ext = int(np.sum(ep < unc_low))
        messidor_eval = {
            "samples": n_ext,
            "accepted_valid_retinal": acc_ext,
            "abstained_uncertain": abs_ext,
            "rejected_invalid": rej_ext,
            "recall_acceptance_rate": round(acc_ext / max(1, n_ext), 4),
            "hard_false_rejection_rate": round(rej_ext / max(1, n_ext), 4),
        }

    report = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": ckpt.get("model_version", VALIDATOR_VERSION),
        "architecture": arch,
        "valid_threshold": valid_thr,
        "uncertain_low": unc_low,
        "test_samples": int(len(y_true)),
        "avg_inference_ms": round(elapsed_ms, 2),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_acceptance_rate": round(far, 4),
            "false_rejection_rate": round(frr, 4),
            "confusion_matrix_labels": ["Invalid_Non_Retinal(0)", "Valid_Retinal(1)"],
            "confusion_matrix": cm.tolist(),
            "tn": tn,
            "fp_false_accept": fp,
            "fn_rejected_or_uncertain": fn,
            "tp": tp,
        },
        "abstention_analysis": {
            "abstained_total": abstained_total,
            "abstained_negative_caught": abstained_neg,
            "abstained_positive_flagged_uncertain": abstained_pos,
            "definitive_false_rejections_below_uncertain": definitive_invalid_pos_fn,
            "definitive_frr": round(definitive_invalid_pos_fn / max(1, tp + fn), 4),
        },
        "per_subcategory_breakdown": per_subcat,
        "external_messidor2_positive_generalization": messidor_eval,
    }

    with open(VALIDATOR_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== HELD-OUT RETINAL VALIDATOR TEST EVALUATION ===", flush=True)
    print(
        f"Arch={arch} | Acc={acc*100:.2f}% | Prec={prec*100:.2f}% | Rec={rec*100:.2f}% | "
        f"F1={f1*100:.2f}% | AUC={roc_auc:.4f} | FAR={far*100:.2f}% | FRR={frr*100:.2f}%",
        flush=True,
    )
    print("Confusion Matrix [[TN, FP], [FN, TP]]:", cm.tolist(), flush=True)
    return report


def train_and_select_validator(
    compare_archs: Optional[List[str]] = None,
    epochs: int = 6,
    warmup_epochs: int = 1,
    patience: int = 3,
    batch_size: int = 48,
    lr: float = 2.5e-4,
    image_size: int = 224,
) -> Dict:
    """Train, compare candidate backbones on validation split, save winner, and run test evaluation."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_csv = os.path.join(VALIDATOR_DATA_DIR, "train.csv")
    val_csv = os.path.join(VALIDATOR_DATA_DIR, "val.csv")
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        raise FileNotFoundError(f"Validator CSVs missing in {VALIDATOR_DATA_DIR}. Run prepare_validator_dataset.py first.")

    train_ds = RetinalValidatorDataset(train_csv, transform=get_train_transforms(image_size), image_size=image_size)
    val_ds = RetinalValidatorDataset(val_csv, transform=get_eval_transforms(image_size), image_size=image_size)
    print(f"[Train Retinal Validator] Preloading {len(train_ds)} train & {len(val_ds)} val images into RAM...", flush=True)
    train_ds.preload(max_workers=8)
    val_ds.preload(max_workers=8)

    candidates = compare_archs or ["mobilenet_v3_small", "resnet18"]
    results, cmp_rows = [], []
    for arch in candidates:
        res = _train_validator_candidate(
            arch=arch,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            image_size=image_size,
        )
        results.append(res)
        cmp_rows.append({
            "arch": arch,
            "best_epoch": res["best_epoch"],
            "valid_threshold": res["valid_threshold"],
            "uncertain_low": res["uncertain_low"],
            "val_cal_far": res["val_cal_far"],
            "val_cal_frr": res["val_cal_frr"],
            "param_size_mb": res["param_size_mb"],
            **res["best_metrics"],
        })

    winner = max(results, key=lambda r: r["best_score"])
    print(f"\n[Validator Selection] Winner on Validation Split: {winner['arch']} (score={winner['best_score']})", flush=True)

    torch.save({
        "model_version": VALIDATOR_VERSION,
        "arch": winner["arch"],
        "num_classes": 2,
        "image_size": image_size,
        "valid_threshold": winner["valid_threshold"],
        "uncertain_low": winner["uncertain_low"],
        "max_valid_entropy": 0.72,
        "use_tta": True,
        "best_epoch": winner["best_epoch"],
        "best_val_metrics": winner["best_metrics"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_dict": winner["state_dict"],
    }, VALIDATOR_MODEL_PATH)

    with open(VALIDATOR_CMP_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "compared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected_winner": winner["arch"],
            "selection_criterion": "0.45*Val_AUC + 0.35*Val_F1 + 0.20*Val_Acc - 0.50*Val_FAR@0.75",
            "candidates": cmp_rows,
        }, f, indent=2)

    with open(VALIDATOR_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": VALIDATOR_VERSION,
            "architecture": winner["arch"],
            "image_size": image_size,
            "valid_threshold": winner["valid_threshold"],
            "uncertain_low": winner["uncertain_low"],
            "max_valid_entropy": 0.72,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    eval_report = evaluate_validator_on_test(VALIDATOR_MODEL_PATH)
    _plot_validator_cm_and_history(np.array(eval_report["metrics"]["confusion_matrix"]), winner["history"])
    return eval_report


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate Human Retinal Fundus Validator CNN")
    parser.add_argument("--compare", nargs="+", default=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    args = parser.parse_args()
    train_and_select_validator(compare_archs=args.compare, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
