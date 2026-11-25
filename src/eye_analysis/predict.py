"""
Unified Eye Image & Tabular Diabetes Prediction Engine.

Loads trained PyTorch retinal DR model and trained scikit-learn Pima tabular model
efficiently in memory once (no retraining per request). If a model is not yet trained,
reports this status explicitly rather than inventing random outputs.
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.eye_analysis.dataset_prep import DR_CLASS_NAMES, ROOT_DIR
from src.eye_analysis.iris_pipeline import analyze_iris_image
from src.eye_analysis.preprocessing import (
    assess_image_quality,
    decode_and_validate_upload,
    detect_eye_modality,
    get_eval_transforms,
    preprocess_fundus_rgb,
)
from src.eye_analysis.train_retinal import build_backbone_model

MODELS_DIR = os.path.join(ROOT_DIR, "models")
RETINAL_MODEL_PATH = os.path.join(MODELS_DIR, "retinal_dr_model.pth")
PREPROC_CONFIG_PATH = os.path.join(MODELS_DIR, "preprocessing_config.json")
LABELS_PATH = os.path.join(MODELS_DIR, "class_labels.json")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")
PIMA_MODEL_PATH = os.path.join(MODELS_DIR, "pima_diabetes_model.joblib")

MEDICAL_DISCLAIMER = (
    "EXPERIMENTAL AI SCREENING DISCLAIMER: This automated prediction is strictly for academic "
    "research and preliminary screening demonstration. It is NOT a clinically validated medical "
    "diagnosis and cannot measure blood glucose (mg/dL). Consult a qualified ophthalmologist or physician."
)

GLUCOSE_BOUNDARY_MESSAGE = (
    "Blood glucose cannot be measured from this image using the current model. "
    "Please use a validated blood glucose testing device."
)

DR_CLINICAL_GUIDANCE = {
    0: "No diabetic retinopathy signs seen on this retinal scan. Note: A diabetic patient can still have a normal retina if microvascular eye damage has not developed — blood glucose testing is still required.",
    1: "Mild non-proliferative features (e.g., microaneurysms) suspected. Schedule periodic ophthalmic follow-up.",
    2: "Moderate non-proliferative diabetic retinopathy suspected. Refer to ophthalmologist for dilated exam.",
    3: "Severe diabetic retinopathy risk detected. Prompt specialist ophthalmology referral recommended.",
    4: "Proliferative diabetic retinopathy risk detected. Urgent specialist retinal evaluation recommended.",
}

_RETINAL_MODEL_CACHE = None
_RETINAL_META_CACHE = None
_PIMA_CACHE = None


def load_retinal_bundle() -> Tuple[Optional[torch.nn.Module], Dict]:
    """Load trained retinal CNN weights & configs efficiently once."""
    global _RETINAL_MODEL_CACHE, _RETINAL_META_CACHE
    if _RETINAL_MODEL_CACHE is not None and _RETINAL_META_CACHE is not None:
        return _RETINAL_MODEL_CACHE, _RETINAL_META_CACHE

    if not os.path.exists(RETINAL_MODEL_PATH):
        return None, {"trained": False, "message": "Retinal DR model weights not found. Run training script first."}

    ckpt = torch.load(RETINAL_MODEL_PATH, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_small")
    image_size = int(ckpt.get("image_size", 224))
    apply_clahe = bool(ckpt.get("apply_clahe", True))

    model = build_backbone_model(arch=arch, num_classes=int(ckpt.get("num_classes", 5)), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    labels = DR_CLASS_NAMES
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            raw_lbls = json.load(f).get("labels", {})
            labels = {int(k): v for k, v in raw_lbls.items()}

    eval_summary = {}
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)

    meta = {
        "trained": True,
        "arch": arch,
        "image_size": image_size,
        "apply_clahe": apply_clahe,
        "trained_at": ckpt.get("trained_at"),
        "best_epoch": ckpt.get("best_epoch"),
        "labels": labels,
        "evaluation": eval_summary,
    }
    _RETINAL_MODEL_CACHE = model
    _RETINAL_META_CACHE = meta
    return model, meta


def load_pima_bundle() -> Optional[Dict]:
    """Load trained tabular Pima diabetes classifier once."""
    global _PIMA_CACHE
    if _PIMA_CACHE is not None:
        return _PIMA_CACHE
    if not os.path.exists(PIMA_MODEL_PATH):
        return None
    _PIMA_CACHE = joblib.load(PIMA_MODEL_PATH)
    return _PIMA_CACHE


def get_eye_module_status() -> Dict:
    """Return honest runtime availability & evaluation metrics for models."""
    _, ret_meta = load_retinal_bundle()
    pima_bundle = load_pima_bundle()
    return {
        "retinal_model_trained": bool(ret_meta.get("trained", False)),
        "retinal_architecture": ret_meta.get("arch"),
        "retinal_trained_at": ret_meta.get("trained_at"),
        "retinal_evaluation": ret_meta.get("evaluation", {}),
        "tabular_diabetes_trained": pima_bundle is not None,
        "tabular_diabetes_evaluation": pima_bundle.get("metadata", {}) if pima_bundle else {},
        "iris_cv_pipeline_ready": True,
        "supports_direct_blood_glucose_from_image": False,
        "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
    }


def analyze_uploaded_eye_image(
    file_bytes: bytes,
    filename: str = "upload.png",
    analysis_mode: str = "auto",
) -> Dict:
    """
    Full validated inference workflow for an uploaded image:
    1. File integrity & resolution check
    2. Eye vs non-eye & Retinal Fundus vs Anterior Iris classification
    3. Sharpness / exposure / glare quality gating
    4. Modality-appropriate execution (trained CNN OR Iris OpenCV pipeline)
    """
    rgb_np, file_meta = decode_and_validate_upload(file_bytes, filename=filename)
    modality_info = detect_eye_modality(rgb_np, requested_mode=analysis_mode)
    detected_modality = modality_info["detected_modality"]

    is_fundus_like = detected_modality == "retinal_fundus"
    quality = assess_image_quality(rgb_np, is_fundus_like=is_fundus_like)

    if not modality_info["valid_eye_image"] or detected_modality == "invalid":
        if not quality["is_acceptable"]:
            return {
                "status": "rejected_quality",
                "valid_eye_image": False,
                "analysis_mode": analysis_mode,
                "detected_modality": "rejected",
                "file_meta": file_meta,
                "quality": quality,
                "prediction_label": "Image Quality Unacceptable — Rejected",
                "confidence_pct": 0.0,
                "message": "Image quality failed validation (cannot reliably verify eye anatomy): " + " ".join(quality["issues"]),
                "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
                "disclaimer": MEDICAL_DISCLAIMER,
            }
        return {
            "status": "rejected",
            "valid_eye_image": False,
            "analysis_mode": analysis_mode,
            "detected_modality": "rejected",
            "file_meta": file_meta,
            "quality": quality,
            "prediction_label": "Invalid / Non-Eye Image Rejected",
            "confidence_pct": 0.0,
            "message": modality_info["reason"],
            "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    if not quality["is_acceptable"]:
        return {
            "status": "rejected_quality",
            "valid_eye_image": True,
            "analysis_mode": analysis_mode,
            "detected_modality": detected_modality,
            "file_meta": file_meta,
            "quality": quality,
            "prediction_label": "Image Quality Unacceptable — Rejected",
            "confidence_pct": 0.0,
            "message": "Image quality failed validation: " + " ".join(quality["issues"]),
            "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    if detected_modality == "retinal_fundus":
        model, meta = load_retinal_bundle()
        if model is None:
            return {
                "status": "model_not_trained",
                "valid_eye_image": True,
                "analysis_mode": analysis_mode,
                "detected_modality": "retinal_fundus",
                "file_meta": file_meta,
                "quality": quality,
                "prediction_label": "Retinal DR Model Not Yet Trained",
                "confidence_pct": 0.0,
                "message": meta.get("message"),
                "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        image_size = int(meta.get("image_size", 224))
        apply_clahe = bool(meta.get("apply_clahe", True))
        labels = meta.get("labels", DR_CLASS_NAMES)

        preprocessed_pil = preprocess_fundus_rgb(rgb_np, target_size=image_size, apply_clahe=apply_clahe)
        eval_tf = get_eval_transforms(image_size)
        x_tensor = eval_tf(preprocessed_pil).unsqueeze(0)

        with torch.no_grad():
            logits = model(x_tensor)
            probs_t = torch.softmax(logits, dim=1)[0]
            probs = probs_t.numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        conf_pct = round(float(probs[pred_idx] * 100.0), 2)
        prob_map = {labels[i]: round(float(probs[i] * 100.0), 2) for i in range(len(probs))}
        referable_dr_prob = round(float(np.sum(probs[2:]) * 100.0), 2)

        return {
            "status": "completed",
            "valid_eye_image": True,
            "analysis_mode": analysis_mode,
            "detected_modality": "retinal_fundus",
            "modality_note": modality_info["reason"],
            "file_meta": file_meta,
            "quality": quality,
            "model_name": f"PyTorch Transfer Learning ({meta.get('arch')})",
            "predicted_grade": pred_idx,
            "prediction_label": f"Predicted diabetic retinopathy severity: {pred_label}",
            "severity_label": pred_label,
            "confidence_pct": conf_pct,
            "referable_dr_probability_pct": referable_dr_prob,
            "class_probabilities": prob_map,
            "clinical_guidance": DR_CLINICAL_GUIDANCE.get(pred_idx, ""),
            "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # Anterior eye / iris computer vision pipeline
    iris_res = analyze_iris_image(rgb_np)
    return {
        "status": "completed",
        "valid_eye_image": True,
        "analysis_mode": analysis_mode,
        "detected_modality": "anterior_iris",
        "modality_note": (
            "External anterior eye / iris photograph detected (NOT a retinal fundus scan). "
            "Diabetic retinopathy screening requires an internal retinal fundus scan, and external iris photos cannot diagnose diabetes."
        ),
        "file_meta": file_meta,
        "quality": quality,
        "model_name": "OpenCV Iris Segmentation & CV Feature Extractor (Experimental)",
        "prediction_label": "External Iris Segmented (Cannot Screen Diabetes From Iris Photo)",
        "confidence_pct": quality["quality_score"],
        "iris_detected": iris_res["iris_detected"],
        "segmentation_completed": iris_res["segmentation_completed"],
        "iris_analysis": iris_res,
        "extracted_features": iris_res["extracted_features"],
        "segmentation_overlay_image": iris_res["segmentation_overlay_image"],
        "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def predict_tabular_diabetes(payload: Dict) -> Dict:
    """Run separate tabular clinical diabetes risk inference using trained Pima model."""
    bundle = load_pima_bundle()
    if bundle is None:
        return {
            "status": "model_not_trained",
            "message": "Tabular Pima diabetes model is not trained yet. Run src/eye_analysis/train_tabular.py first.",
        }

    pipeline = bundle["pipeline"]
    meta = bundle["metadata"]
    feature_order = meta["feature_names"]

    # Map clean API keys to Pima dataset column names; 0s in clinical labs converted to NaN for imputer
    values = {
        "Pregnancies": float(payload.get("pregnancies", 1)),
        "Glucose": float(payload.get("glucose", 110)),
        "BloodPressure": float(payload.get("bloodPressure", 72)),
        "SkinThickness": float(payload.get("skinThickness", 24)),
        "Insulin": float(payload.get("insulin", 85)),
        "BMI": float(payload.get("bmi", 26.5)),
        "DiabetesPedigreeFunction": float(payload.get("dpf", 0.38)),
        "Age": float(payload.get("age", 34)),
    }

    row_df = pd.DataFrame([values])[feature_order]
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        row_df[col] = row_df[col].replace(0, np.nan)

    prob_pos = float(pipeline.predict_proba(row_df)[0, 1]) * 100.0
    pred_class = int(pipeline.predict(row_df)[0])
    risk_tier = "High" if prob_pos >= 60.0 else ("Moderate" if prob_pos >= 35.0 else "Low")

    return {
        "status": "completed",
        "model_name": f"Pima Tabular {meta.get('selected_model')}",
        "prediction_label": "Elevated Tabular Diabetes Risk" if pred_class == 1 else "Low Tabular Diabetes Risk",
        "outcome": pred_class,
        "probability_pct": round(prob_pos, 2),
        "risk_tier": risk_tier,
        "inputs": values,
        "test_roc_auc": meta["test_metrics"]["roc_auc"],
        "disclaimer": (
            "Estimated solely from clinical/tabular biometric parameters (Pima Indians Diabetes Database), "
            "independent from eye image predictions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Eye Image Inference from CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal fundus or eye image")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "retinal_fundus", "anterior_iris"])
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()
    res = analyze_uploaded_eye_image(data, filename=os.path.basename(args.image), analysis_mode=args.mode)
    res.pop("segmentation_overlay_image", None)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
