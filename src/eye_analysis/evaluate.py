"""
Independent Held-Out & Cross-Dataset Test Evaluation Script.

Evaluates the saved retinal DR classifier strictly on unseen test sets:
1. Primary Multi-Center Held-Out Test Split (`data/retinal_fundus/test.csv`)
2. External Zero-Shot Cross-Dataset Validation (`data/retinal_fundus/external_messidor2.csv`)

Computes & saves:
- Accuracy, Macro & Weighted Precision, Recall, F1-Score
- Quadratic Weighted Kappa (QWK)
- Multi-class One-vs-Rest ROC-AUC + ROC Curves plot
- Per-class Sensitivity (Recall), Specificity, Precision, F1, False Negatives
- Per-dataset-source breakdown & Confusion Matrix heatmap
- Any-DR (>= Grade 1) and Referable DR (>= Grade 2) screening metrics
"""

import argparse
import json
import os
import time
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
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


def _plot_confusion_matrix(cm: np.ndarray, class_labels: list, out_png: str, title: str = "Retinal Fundus DR Test Set Confusion Matrix") -> None:
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
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
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


def _evaluate_csv_split(
    model: torch.nn.Module,
    ckpt: Dict,
    csv_path: str,
    save_plots: bool = False,
) -> Dict:
    """Run deterministic evaluation on a single held-out CSV split without leaking into training."""
    image_size = int(ckpt.get("image_size", 224))
    apply_clahe = bool(ckpt.get("apply_clahe", True))
    any_dr_threshold = float(ckpt.get("any_dr_threshold", 0.52))
    use_tta = bool(ckpt.get("use_tta", True))

    test_ds = RetinalFundusDataset(
        csv_path,
        transform=get_eval_transforms(image_size),
        image_size=image_size,
        apply_clahe=apply_clahe,
        cache_preprocessed=False,
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
                raw_am = int(np.argmax(row_p))
                p_any = float(np.sum(row_p[1:]))
                det = bool(
                    (p_any >= any_dr_threshold and raw_am >= 1)
                    or (p_any >= max(0.65, any_dr_threshold))
                )
                if det:
                    pred_cls = raw_am if raw_am >= 1 else int(1 + np.argmax(row_p[1:]))
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
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

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

    # Optional per-dataset-source breakdown if column is present
    per_source = {}
    if "dataset_source" in test_ds.df.columns:
        sources = test_ds.df["dataset_source"].astype(str).values
        for src in sorted(np.unique(sources)):
            m = sources == src
            if int(np.sum(m)) == 0:
                continue
            yt_s, yp_s = y_true[m], y_pred[m]
            at_s, ap_s = (yt_s >= 1).astype(int), (yp_s >= 1).astype(int)
            per_source[src] = {
                "samples": int(np.sum(m)),
                "accuracy": round(float(accuracy_score(yt_s, yp_s)), 4),
                "qwk": round(float(cohen_kappa_score(yt_s, yp_s, weights="quadratic")), 4),
                "any_dr_sensitivity": round(float(recall_score(at_s, ap_s, zero_division=0)), 4),
                "any_dr_specificity": round(float(recall_score(1 - at_s, 1 - ap_s, zero_division=0)), 4),
            }

    if save_plots:
        _plot_confusion_matrix(cm, list(DR_CLASS_NAMES.values()), os.path.join(MODELS_DIR, "confusion_matrix.png"))
        _plot_roc_curves(y_true, y_probs, os.path.join(MODELS_DIR, "roc_curves.png"))

    return {
        "test_samples": int(len(y_true)),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "quadratic_weighted_kappa": round(qwk, 4),
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
        "per_source_metrics": per_source,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[DR_CLASS_NAMES[i] for i in range(5)],
            zero_division=0,
        ),
    }


def evaluate_retinal_model(
    model_path: str = os.path.join(MODELS_DIR, "retinal_dr_model.pth"),
    test_csv: str = os.path.join(RETINAL_DIR, "test.csv"),
    external_csv: Optional[str] = os.path.join(RETINAL_DIR, "external_messidor2.csv"),
) -> Dict:
    """Evaluate trained retinal DR classifier on primary test split & optional external Messidor-2 set."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_retinal.py first.")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Held-out test CSV not found at {test_csv}. Run dataset_prep.py first.")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    model = build_backbone_model(arch=arch, num_classes=5, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    primary_eval = _evaluate_csv_split(model, ckpt, test_csv, save_plots=True)

    external_eval = None
    if external_csv and os.path.exists(external_csv):
        external_eval = _evaluate_csv_split(model, ckpt, external_csv, save_plots=False)

    report = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": ckpt.get("model_version", "1.0.0"),
        "backbone_architecture": arch,
        "best_epoch": int(ckpt.get("best_epoch", 0)),
        "any_dr_threshold": round(float(ckpt.get("any_dr_threshold", 0.52)), 3),
        "test_time_augmentation": bool(ckpt.get("use_tta", True)),
        **primary_eval,
        "external_cross_dataset_validation": external_eval,
        "limitations": (
            f"Evaluated on independent held-out multi-center test scans (n={primary_eval['test_samples']}) "
            + (
                f"and zero-shot external Messidor-2 scans (n={external_eval['test_samples']}). "
                if external_eval
                else ". "
            )
            + "Benchmark accuracy must NOT be interpreted as prospective clinical validation or regulatory clearance."
        ),
    }

    report_path = os.path.join(MODELS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    om = report["overall_metrics"]
    sa = report["screening_sensitivity_analysis"]
    print("\n=== HELD-OUT MULTI-CENTER TEST SET EVALUATION ===", flush=True)
    print(f"Architecture:      {arch} (version={report['model_version']}, epoch={report['best_epoch']})", flush=True)
    print(f"Accuracy:          {om['accuracy'] * 100:.2f}%", flush=True)
    print(f"QWK (Quadratic):   {om['quadratic_weighted_kappa']:.4f}", flush=True)
    print(f"Macro F1-Score:    {om['f1_macro'] * 100:.2f}%", flush=True)
    print(f"Weighted F1-Score: {om['f1_weighted'] * 100:.2f}%", flush=True)
    if om["roc_auc_ovr_macro"] is not None:
        print(f"OvR ROC-AUC Macro: {om['roc_auc_ovr_macro']:.4f}", flush=True)
    print(
        f"Any DR (>=1):      Sens={sa['any_dr_detection']['sensitivity_recall'] * 100:.2f}% | "
        f"Spec={sa['any_dr_detection']['specificity'] * 100:.2f}% | AUC={sa['any_dr_detection']['roc_auc']}",
        flush=True,
    )
    print(
        f"Referable (>=2):   Sens={sa['referable_dr_detection_grade_2plus']['sensitivity_recall'] * 100:.2f}% | "
        f"Spec={sa['referable_dr_detection_grade_2plus']['specificity'] * 100:.2f}% | "
        f"AUC={sa['referable_dr_detection_grade_2plus']['roc_auc']}",
        flush=True,
    )
    if report["per_source_metrics"]:
        print("Per-Source Breakdown:", json.dumps(report["per_source_metrics"], indent=2), flush=True)
    if external_eval:
        em = external_eval["overall_metrics"]
        es = external_eval["screening_sensitivity_analysis"]
        print(
            f"\n=== EXTERNAL ZERO-SHOT CROSS-DATASET (Messidor-2, n={external_eval['test_samples']}) ===",
            flush=True,
        )
        print(
            f"Acc={em['accuracy'] * 100:.2f}% | QWK={em['quadratic_weighted_kappa']:.4f} | "
            f"Macro-F1={em['f1_macro'] * 100:.2f}% | Any-DR Sens={es['any_dr_detection']['sensitivity_recall'] * 100:.2f}% "
            f"Spec={es['any_dr_detection']['specificity'] * 100:.2f}%",
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Retinal DR Classifier on Held-Out & External Splits")
    parser.parse_args()
    evaluate_retinal_model()


if __name__ == "__main__":
    main()
