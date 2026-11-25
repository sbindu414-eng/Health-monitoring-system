"""
Independent Held-Out Test Set Evaluation Script.

Evaluates the saved retinal DR classifier strictly on `data/retinal_fundus/test.csv`.
Calculates & saves:
- Overall Accuracy, Precision, Recall, F1-Score (macro & weighted)
- Confusion matrix + heatmap plot
- Multi-class One-vs-Rest ROC-AUC + ROC curves plot
- Per-class Precision, Recall (Sensitivity), F1-score, Specificity, and Support
- Referable Diabetic Retinopathy (Grades 2-4 vs Grades 0-1) Sensitivity & False Negatives
"""

import argparse
import json
import os
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.eye_analysis.dataset_prep import DR_CLASS_NAMES, RETINAL_DIR, ROOT_DIR
from src.eye_analysis.preprocessing import get_eval_transforms
from src.eye_analysis.train_retinal import RetinalFundusDataset, build_backbone_model, predict_batch_tta

MODELS_DIR = os.path.join(ROOT_DIR, "models")


def _plot_confusion_matrix(cm: np.ndarray, class_labels: list, out_png: str) -> None:
    """Save clean annotated confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    short_labels = [
        "0: No DR",
        "1: Mild",
        "2: Moderate",
        "3: Severe",
        "4: Proliferative",
    ][: len(class_labels)]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short_labels,
        yticklabels=short_labels,
        ylabel="True Clinical Grade",
        xlabel="Predicted Clinical Grade",
        title="Retinal Fundus DR Test Set Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, out_png: str) -> None:
    """Plot and save multi-class One-vs-Rest ROC curves."""
    n_classes = y_probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    short_labels = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
    for i in range(n_classes):
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.8, label=f"{short_labels[i]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("One-vs-Rest ROC Curves (Held-Out Test Split)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def evaluate_retinal_model(
    model_path: str = os.path.join(MODELS_DIR, "retinal_dr_model.pth"),
    test_csv: str = os.path.join(RETINAL_DIR, "test.csv"),
) -> Dict:
    """Evaluate trained retinal DR classifier on the unseen held-out test set."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_retinal.py first.")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Held-out test CSV not found at {test_csv}. Run dataset_prep.py first.")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    image_size = int(ckpt.get("image_size", 224))
    apply_clahe = bool(ckpt.get("apply_clahe", True))
    any_dr_threshold = float(ckpt.get("any_dr_threshold", 0.50))
    use_tta = bool(ckpt.get("use_tta", True))

    model = build_backbone_model(arch=arch, num_classes=5, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ds = RetinalFundusDataset(
        test_csv,
        transform=get_eval_transforms(image_size),
        image_size=image_size,
        apply_clahe=apply_clahe,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            probs = predict_batch_tta(model, x_batch) if use_tta else torch.softmax(model(x_batch), dim=1)
            probs_np = probs.numpy()
            for row_p in probs_np:
                p_any = float(np.sum(row_p[1:]))
                if p_any >= any_dr_threshold:
                    pred_cls = int(np.argmax(row_p))
                    if pred_cls == 0:
                        pred_cls = int(1 + np.argmax(row_p[1:]))
                else:
                    pred_cls = 0
                all_preds.append(pred_cls)
            all_probs.append(probs_np)
            all_targets.extend(y_batch.numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_probs = np.vstack(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    # Multi-class OvR ROC-AUC
    present_classes = sorted(list(set(y_true.tolist())))
    try:
        if len(present_classes) == 5:
            roc_auc_ovr_macro = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
            roc_auc_ovr_weighted = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted"))
        else:
            roc_auc_ovr_macro = None
            roc_auc_ovr_weighted = None
    except ValueError:
        roc_auc_ovr_macro = None
        roc_auc_ovr_weighted = None

    # Per-class metrics including Sensitivity (Recall) and Specificity
    per_class = {}
    total_samples = int(np.sum(cm))
    for cls_idx in range(5):
        tp = int(cm[cls_idx, cls_idx])
        fn = int(np.sum(cm[cls_idx, :]) - tp)
        fp = int(np.sum(cm[:, cls_idx]) - tp)
        tn = int(total_samples - tp - fn - fp)
        cls_prec = tp / max(1, tp + fp)
        cls_sens = tp / max(1, tp + fn)
        cls_spec = tn / max(1, tn + fp)
        cls_f1 = (2 * cls_prec * cls_sens / max(1e-9, cls_prec + cls_sens)) if (cls_prec + cls_sens) > 0 else 0.0
        per_class[DR_CLASS_NAMES[cls_idx]] = {
            "grade": cls_idx,
            "support": int(tp + fn),
            "precision": round(float(cls_prec), 4),
            "recall_sensitivity": round(float(cls_sens), 4),
            "specificity": round(float(cls_spec), 4),
            "f1_score": round(float(cls_f1), 4),
            "false_negatives": fn,
        }

    # Screening sensitivity: Referable DR (Grade >= 2 vs Grade <= 1) AND Any-DR (Grade >= 1 vs Grade 0)
    ref_true = (y_true >= 2).astype(int)
    ref_pred = (y_pred >= 2).astype(int)
    ref_prob = np.sum(y_probs[:, 2:], axis=1)
    ref_cm = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    r_tn, r_fp, r_fn, r_tp = [int(v) for v in ref_cm.ravel()]
    ref_sens = r_tp / max(1, r_tp + r_fn)
    ref_spec = r_tn / max(1, r_tn + r_fp)
    ref_auc = float(roc_auc_score(ref_true, ref_prob)) if len(np.unique(ref_true)) == 2 else None

    any_true = (y_true >= 1).astype(int)
    any_pred = (y_pred >= 1).astype(int)
    any_prob = np.sum(y_probs[:, 1:], axis=1)
    any_cm = confusion_matrix(any_true, any_pred, labels=[0, 1])
    a_tn, a_fp, a_fn, a_tp = [int(v) for v in any_cm.ravel()]
    any_sens = a_tp / max(1, a_tp + a_fn)
    any_spec = a_tn / max(1, a_tn + a_fp)
    any_auc = float(roc_auc_score(any_true, any_prob)) if len(np.unique(any_true)) == 2 else None

    _plot_confusion_matrix(cm, list(DR_CLASS_NAMES.values()), os.path.join(MODELS_DIR, "confusion_matrix.png"))
    _plot_roc_curves(y_true, y_probs, os.path.join(MODELS_DIR, "roc_curves.png"))

    report = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone_architecture": arch,
        "best_epoch": int(ckpt.get("best_epoch", 0)),
        "any_dr_threshold": round(any_dr_threshold, 3),
        "test_time_augmentation": use_tta,
        "test_samples": int(len(y_true)),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "roc_auc_ovr_macro": round(roc_auc_ovr_macro, 4) if roc_auc_ovr_macro is not None else None,
            "roc_auc_ovr_weighted": round(roc_auc_ovr_weighted, 4) if roc_auc_ovr_weighted is not None else None,
        },
        "screening_sensitivity_analysis": {
            "any_dr_detection": {
                "sensitivity_recall": round(float(any_sens), 4),
                "specificity": round(float(any_spec), 4),
                "false_negatives": a_fn,
                "false_positives": a_fp,
                "roc_auc": round(any_auc, 4) if any_auc is not None else None,
            },
            "referable_dr_detection_grade_2plus": {
                "sensitivity_recall": round(float(ref_sens), 4),
                "specificity": round(float(ref_spec), 4),
                "false_negatives": r_fn,
                "false_positives": r_fp,
                "roc_auc": round(ref_auc, 4) if ref_auc is not None else None,
            },
        },
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[DR_CLASS_NAMES[i] for i in range(5)],
            zero_division=0,
        ),
        "limitations": (
            f"Evaluated strictly on internal held-out test images (DeepDRiD / RetinaMNIST, n={len(y_true)} after cross-split deduplication). "
            "Internal benchmark accuracy must NOT be interpreted as multi-site prospective clinical validation."
        ),
    }

    report_path = os.path.join(MODELS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== HELD-OUT TEST SET EVALUATION ===")
    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")
    if roc_auc_ovr_macro is not None:
        print(f"OvR ROC-AUC Macro: {roc_auc_ovr_macro:.4f}")
    print(f"Referable DR (>=2) Sensitivity: {ref_sens * 100:.2f}% (False Negatives: {r_fn})")
    print(f"Any DR (>=1)       Sensitivity: {any_sens * 100:.2f}% (False Negatives: {a_fn})")
    print(f"Confusion Matrix saved -> {os.path.join(MODELS_DIR, 'confusion_matrix.png')}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retinal DR Classifier on Held-Out Test Split")
    parser.parse_args()
    evaluate_retinal_model()


if __name__ == "__main__":
    main()
