"""
Unified Eye Image & Tabular Diabetes Prediction Engine.

Features:
- Thread-safe lazy checkpoint loading with automatic hot-reload on file update
- Strict modality isolation (Retinal Fundus DR CNN vs. Anterior Iris CV Pipeline)
- Resolution-normalized quality gating (rejects blurry, dark, overexposed, incomplete, non-eye)
- Exact training preprocessing alignment (black-margin crop, square pad, LAB CLAHE)
- 4-view Test-Time Augmentation (TTA) + calibrated uncertainty detection
- Real model-generated Grad-CAM spatial attention heatmap
- Structured ICDR clinical reference report grounded strictly in CNN outputs
- Separate Pima Indians Diabetes tabular screening inference
"""

import argparse
import json
import math
import os
import threading
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from src.eye_analysis.clinical_reference import build_retinal_screening_report
from src.eye_analysis.dataset_prep import DR_CLASS_NAMES, ROOT_DIR
from src.eye_analysis.gradcam import generate_fundus_gradcam
from src.eye_analysis.iris_pipeline import analyze_iris_image
from src.eye_analysis.liveness_pad import (
    MSG_LIVENESS_FAILED,
    MSG_SPOOF_DETECTED,
    PAD_GENUINE_THRESHOLD,
    SESSION_MANAGER,
    UI_MESSAGES,
    PresentationAttackDetector,
    RealTimeEyeDetector,
)
from src.eye_analysis.preprocessing import (
    assess_image_quality,
    decode_and_validate_upload,
    detect_eye_modality,
    get_eval_transforms,
    preprocess_fundus_rgb,
)
from src.eye_analysis.retinal_validator import (
    MSG_INVALID_NON_RETINAL,
    MSG_UNCERTAIN_RETINAL,
    load_validator_bundle,
    validate_human_retinal_fundus,
)
from src.eye_analysis.train_retinal import build_backbone_model, predict_batch_tta

MODELS_DIR = os.path.join(ROOT_DIR, "models")
RETINAL_MODEL_PATH = os.path.join(MODELS_DIR, "retinal_dr_model.pth")
PREPROC_CONFIG_PATH = os.path.join(MODELS_DIR, "preprocessing_config.json")
LABELS_PATH = os.path.join(MODELS_DIR, "class_labels.json")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")
TRAINING_CFG_PATH = os.path.join(MODELS_DIR, "training_config.json")
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

UNCERTAINTY_MESSAGE = (
    "Prediction uncertain — please upload a clearer retinal image or consult an eye-care professional."
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
_RETINAL_MTIME_CACHE = None
_PIMA_CACHE = None
_MODEL_LOAD_LOCK = threading.Lock()


def load_retinal_bundle() -> Tuple[Optional[torch.nn.Module], Dict]:
    """Load trained retinal CNN weights & configs safely once (reloading if checkpoint updated)."""
    global _RETINAL_MODEL_CACHE, _RETINAL_META_CACHE, _RETINAL_MTIME_CACHE

    with _MODEL_LOAD_LOCK:
        if not os.path.exists(RETINAL_MODEL_PATH):
            _RETINAL_MODEL_CACHE = None
            _RETINAL_META_CACHE = None
            _RETINAL_MTIME_CACHE = None
            return None, {"trained": False, "message": "Retinal DR model weights not found. Run training script first."}

        cur_mtime = os.path.getmtime(RETINAL_MODEL_PATH)
        if (
            _RETINAL_MODEL_CACHE is not None
            and _RETINAL_META_CACHE is not None
            and _RETINAL_MTIME_CACHE == cur_mtime
        ):
            return _RETINAL_MODEL_CACHE, _RETINAL_META_CACHE

        ckpt = torch.load(RETINAL_MODEL_PATH, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "efficientnet_b0")
        image_size = int(ckpt.get("image_size", 224))
        apply_clahe = bool(ckpt.get("apply_clahe", True))
        any_dr_threshold = float(ckpt.get("any_dr_threshold", 0.52))
        use_tta = bool(ckpt.get("use_tta", True))

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

        comparison = {}
        if os.path.exists(COMPARISON_PATH):
            with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
                comparison = json.load(f)

        training_cfg = {}
        if os.path.exists(TRAINING_CFG_PATH):
            with open(TRAINING_CFG_PATH, "r", encoding="utf-8") as f:
                training_cfg = json.load(f)

        meta = {
            "trained": True,
            "model_version": ckpt.get("model_version", "2.0.0"),
            "arch": arch,
            "image_size": image_size,
            "apply_clahe": apply_clahe,
            "any_dr_threshold": any_dr_threshold,
            "use_tta": use_tta,
            "trained_at": ckpt.get("trained_at"),
            "best_epoch": ckpt.get("best_epoch"),
            "labels": labels,
            "evaluation": eval_summary,
            "comparison": comparison,
            "training_config": training_cfg,
        }
        _RETINAL_MODEL_CACHE = model
        _RETINAL_META_CACHE = meta
        _RETINAL_MTIME_CACHE = cur_mtime
        return model, meta


def load_pima_bundle() -> Optional[Dict]:
    """Load trained tabular Pima diabetes classifier once."""
    global _PIMA_CACHE
    with _MODEL_LOAD_LOCK:
        if _PIMA_CACHE is not None:
            return _PIMA_CACHE
        if not os.path.exists(PIMA_MODEL_PATH):
            return None
        _PIMA_CACHE = joblib.load(PIMA_MODEL_PATH)
        return _PIMA_CACHE


def get_eye_module_status() -> Dict:
    """Return honest runtime availability & evaluation metrics for models."""
    _, ret_meta = load_retinal_bundle()
    _, val_meta = load_validator_bundle()
    pima_bundle = load_pima_bundle()
    return {
        "retinal_model_trained": bool(ret_meta.get("trained", False)),
        "retinal_model_version": ret_meta.get("model_version"),
        "retinal_architecture": ret_meta.get("arch"),
        "retinal_trained_at": ret_meta.get("trained_at"),
        "retinal_evaluation": ret_meta.get("evaluation", {}),
        "retinal_comparison": ret_meta.get("comparison", {}),
        "retinal_training_config": ret_meta.get("training_config", {}),
        "retinal_validator_trained": bool(val_meta.get("trained", False)),
        "retinal_validator_architecture": val_meta.get("arch"),
        "retinal_validator_version": val_meta.get("model_version"),
        "retinal_validator_evaluation": val_meta.get("evaluation", {}),
        "retinal_validator_comparison": val_meta.get("comparison", {}),
        "retinal_validator_config": val_meta.get("config", {}),
        "tabular_diabetes_trained": pima_bundle is not None,
        "tabular_diabetes_evaluation": pima_bundle.get("metadata", {}) if pima_bundle else {},
        "iris_cv_pipeline_ready": True,
        "gradcam_explainability_ready": True,
        "supports_direct_blood_glucose_from_image": False,
        "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
    }


def _assess_uncertainty(probs: np.ndarray, quality_score: float) -> Dict:
    """Compute normalized Shannon entropy & top-2 probability margin to detect uncertain predictions."""
    sorted_p = np.sort(probs)[::-1]
    top1 = float(sorted_p[0])
    top2 = float(sorted_p[1])
    margin = top1 - top2

    # Normalized entropy in [0, 1] across 5 classes
    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)))
    norm_entropy = entropy / math.log(len(probs))

    # Flag uncertainty when probability is spread thin, top classes tie closely, or quality is marginal
    reasons = []
    if top1 < 0.45:
        reasons.append(f"low top-class probability ({top1 * 100:.1f}%)")
    if norm_entropy > 0.80:
        reasons.append(f"high class entropy ({norm_entropy:.2f})")
    if margin < 0.08:
        reasons.append(f"narrow top-2 margin ({margin * 100:.1f}%)")
    if quality_score < 32.0:
        reasons.append(f"borderline image quality ({quality_score}/100)")

    is_uncertain = len(reasons) > 0
    return {
        "is_uncertain": is_uncertain,
        "normalized_entropy": round(norm_entropy, 3),
        "top2_margin_pct": round(margin * 100.0, 2),
        "uncertainty_reasons": reasons,
        "uncertainty_warning": UNCERTAINTY_MESSAGE if is_uncertain else "",
    }


def _run_pad_on_rgb(rgb_np: np.ndarray) -> Tuple[Optional[Dict], Dict]:
    """Detect eye ROI and run Presentation Attack Detection (PAD) on an RGB image."""
    import cv2

    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    h_img, w_img = bgr_np.shape[:2]
    det_res = None
    eye_box = (0, 0, w_img, h_img)
    face_box = None
    try:
        det_res = RealTimeEyeDetector().detect_eye_region(bgr_np)
        if det_res and det_res.get("eye_box"):
            eye_box = det_res["eye_box"]
            face_box = det_res.get("face_box")
    except Exception:
        det_res = None
    pad_res = PresentationAttackDetector().analyze_frame_pad(bgr_np, eye_box, face_box=face_box)
    return det_res, pad_res


def _spoof_rejection_response(
    analysis_mode: str,
    file_meta: Dict,
    quality: Dict,
    pad_res: Dict,
) -> Dict:
    """Build standardized Presentation Attack / Spoof blocking payload."""
    return {
        "status": "spoof_detected",
        "valid_eye_image": False,
        "analysis_mode": analysis_mode,
        "detected_modality": "anterior_iris",
        "file_meta": file_meta,
        "quality": quality,
        "prediction_label": "Possible Photograph or Screen Detected",
        "confidence_pct": pad_res.get("pad_confidence_pct", 0.0),
        "pad_result": pad_res,
        "message": MSG_SPOOF_DETECTED,
        "ui_message": UI_MESSAGES["spoof_detected"],
        "guidance_message": MSG_SPOOF_DETECTED,
        "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def _liveness_rejection_response(
    analysis_mode: str,
    file_meta: Dict,
    reason: str = "",
    pad_res: Optional[Dict] = None,
) -> Dict:
    """Build standardized liveness verification failure blocking payload."""
    payload = {
        "status": "liveness_failed",
        "valid_eye_image": False,
        "analysis_mode": analysis_mode,
        "detected_modality": "anterior_iris",
        "file_meta": file_meta,
        "quality": {"is_acceptable": False, "quality_score": 0.0, "status": "liveness_unverified"},
        "prediction_label": "Live Human Eye Not Verified",
        "confidence_pct": 0.0,
        "message": MSG_LIVENESS_FAILED,
        "ui_message": UI_MESSAGES["verification_failed"],
        "guidance_message": MSG_LIVENESS_FAILED,
        "auth_reason": reason or MSG_LIVENESS_FAILED,
        "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
        "disclaimer": MEDICAL_DISCLAIMER,
    }
    if pad_res is not None:
        payload["pad_result"] = pad_res
    return payload


def analyze_uploaded_eye_image(
    file_bytes: bytes,
    filename: str = "upload.png",
    analysis_mode: str = "auto",
    include_gradcam: bool = True,
    is_liveness_verified: bool = False,
    liveness_telemetry: Optional[Dict] = None,
    session_id: Optional[str] = None,
    verification_token: Optional[str] = None,
    source: str = "upload",
) -> Dict:
    """
    Full validated inference workflow for an eye image:
    1. File integrity & resolution check
    2. Live-camera backend authorization (validates short-lived single-use session token & PAD)
    3. Eye vs non-eye & Retinal Fundus vs Anterior Iris modality check
    4. Resolution-normalized quality gating (blur, exposure, glare, completeness)
    5. Modality-appropriate inference (Retinal DR CNN + Grad-CAM OR Iris OpenCV pipeline with PAD)
    """
    rgb_np, file_meta = decode_and_validate_upload(file_bytes, filename=filename)

    is_camera_mode = (
        analysis_mode in ("camera", "live_camera")
        or source in ("camera", "live_camera")
        or bool(session_id)
    )

    session_snapshot = None
    snap_det_res = None
    if is_camera_mode:
        # Independently validate short-lived server verification token; never trust raw boolean flags alone
        if liveness_telemetry is not None:
            t_pad = liveness_telemetry.get("pad_analysis") or liveness_telemetry.get("pad_result") or {}
            t_score = float(liveness_telemetry.get("liveness_score_pct", liveness_telemetry.get("confidence", 0.0)) or 0.0)
            if t_pad.get("is_spoof"):
                return _spoof_rejection_response(
                    analysis_mode,
                    file_meta,
                    {"is_acceptable": False, "quality_score": 0.0, "status": "spoof_detected"},
                    t_pad,
                )
            if (not liveness_telemetry.get("is_live")) or t_score <= 0.0:
                return _liveness_rejection_response(analysis_mode, file_meta, "Live telemetry failed threshold.")

        token_ok, token_reason, session_snapshot = SESSION_MANAGER.validate_and_consume_token(
            session_id=session_id,
            verification_token=verification_token,
            consume=True,
        )
        if not token_ok or session_snapshot is None:
            return _liveness_rejection_response(analysis_mode, file_meta, reason=token_reason)

        # Immediately check PAD on the submitted camera snapshot as well (prevents post-verification phone swaps)
        snap_det_res, snap_pad_res = _run_pad_on_rgb(rgb_np)
        snap_sm = snap_pad_res.get("sub_metrics") or {}
        snap_corroborated_screen = (
            float(snap_pad_res.get("moire_score", 1.0)) <= 0.45
            and (
                float(snap_sm.get("screen_region_score", 0.0)) >= 0.35
                or float(snap_sm.get("bezel_score", 0.0)) >= 0.50
            )
        )
        if snap_pad_res.get("is_spoof") or snap_corroborated_screen or snap_pad_res.get("pad_score", 0.0) < PAD_GENUINE_THRESHOLD:
            return _spoof_rejection_response(
                analysis_mode,
                file_meta,
                {"is_acceptable": False, "quality_score": 0.0, "status": "spoof_detected"},
                snap_pad_res,
            )

        is_liveness_verified = True

    # Strict pipeline enforcement:
    # - `anterior_iris` / live webcam routes stay strictly isolated for anterior iris CV pipeline.
    # - `retinal_fundus` and default `auto` upload run the trained Human Retinal Fundus Validator CNN FIRST;
    #   the DR EfficientNet-B0 model NEVER runs unless validator verifies Valid Human Retinal Fundus Image.
    retinal_validation = None
    if is_camera_mode or analysis_mode == "anterior_iris":
        modality_info = detect_eye_modality(rgb_np, requested_mode="anterior_iris")
        detected_modality = modality_info["detected_modality"]
        if is_camera_mode and detected_modality == "invalid" and snap_det_res is not None:
            modality_info = {
                "valid_eye_image": True,
                "detected_modality": "anterior_iris",
                "reason": "Verified live webcam anterior eye/iris ROI detected.",
            }
            detected_modality = "anterior_iris"
    else:
        # Catch pitch-black / severely underexposed blank frames cleanly via image signal quality
        import cv2 as _cv2

        if float(np.mean(_cv2.cvtColor(rgb_np, _cv2.COLOR_RGB2GRAY) > 12)) < 0.08:
            quality = assess_image_quality(rgb_np, is_fundus_like=True)
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

        # Step 3 & 4: Run dedicated Human Retinal Fundus Image Validation Model BEFORE quality/DR CNN
        retinal_validation = validate_human_retinal_fundus(rgb_np)
        if not retinal_validation.get("validator_available", False):
            return {
                "status": "validator_not_trained",
                "valid_eye_image": False,
                "analysis_mode": analysis_mode,
                "detected_modality": "rejected",
                "file_meta": file_meta,
                "quality": {"is_acceptable": False, "quality_score": 0.0, "status": "validator_unavailable"},
                "retinal_validation": retinal_validation,
                "prediction_label": "Retinal Fundus Validator Unavailable",
                "confidence_pct": 0.0,
                "message": retinal_validation.get("message", "Retinal validator model not trained."),
                "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        v_status = retinal_validation.get("validator_status")
        if v_status == "uncertain":
            quality = assess_image_quality(rgb_np, is_fundus_like=False)
            return {
                "status": "uncertain_retinal",
                "valid_eye_image": False,
                "analysis_mode": analysis_mode,
                "detected_modality": "uncertain",
                "file_meta": file_meta,
                "quality": quality,
                "retinal_validation": retinal_validation,
                "prediction_label": "Unable to Verify Retinal Fundus Image",
                "confidence_pct": retinal_validation.get("retinal_fundus_probability_pct", 0.0),
                "message": MSG_UNCERTAIN_RETINAL,
                "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        if not retinal_validation.get("is_valid_retinal", False) or v_status != "valid_retinal":
            quality = assess_image_quality(rgb_np, is_fundus_like=False)
            return {
                "status": "rejected",
                "valid_eye_image": False,
                "analysis_mode": analysis_mode,
                "detected_modality": "rejected",
                "file_meta": file_meta,
                "quality": quality,
                "retinal_validation": retinal_validation,
                "prediction_label": "Invalid / Non-Retinal Image Rejected",
                "confidence_pct": retinal_validation.get("non_retinal_probability_pct", 0.0),
                "message": MSG_INVALID_NON_RETINAL,
                "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        modality_info = {
            "valid_eye_image": True,
            "detected_modality": "retinal_fundus",
            "reason": f"Human Retinal Fundus Validator CNN verified image ({retinal_validation['retinal_fundus_probability_pct']}% confidence).",
        }
        detected_modality = "retinal_fundus"

    # Keep retinal fundus CNN strictly separated from live webcam capture
    if is_camera_mode and detected_modality == "retinal_fundus":
        _, cam_pad = _run_pad_on_rgb(rgb_np)
        return _spoof_rejection_response(
            analysis_mode,
            file_meta,
            {"is_acceptable": False, "quality_score": 0.0, "status": "spoof_detected"},
            cam_pad,
        )

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

    # Strict modality isolation: run Retinal Fundus DR CNN ONLY on validated non-camera retinal_fundus uploads.
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

        arch = meta.get("arch", "efficientnet_b0")
        image_size = int(meta.get("image_size", 224))
        apply_clahe = bool(meta.get("apply_clahe", True))
        any_dr_threshold = float(meta.get("any_dr_threshold", 0.52))
        use_tta = bool(meta.get("use_tta", True))
        labels = meta.get("labels", DR_CLASS_NAMES)

        preprocessed_pil = preprocess_fundus_rgb(rgb_np, target_size=image_size, apply_clahe=apply_clahe)
        eval_tf = get_eval_transforms(image_size)
        x_tensor = eval_tf(preprocessed_pil).unsqueeze(0)

        with torch.no_grad():
            probs_t = predict_batch_tta(model, x_tensor)[0] if use_tta else torch.softmax(model(x_tensor), dim=1)[0]
            probs = probs_t.numpy()

        raw_argmax = int(np.argmax(probs))
        any_dr_prob_raw = float(np.sum(probs[1:]))

        # Flag diabetic retinopathy only when any-DR probability meets the calibrated threshold
        # AND the model's primary prediction is a DR grade (or DR probability is dominant >= 0.65)
        diabetes_detected = bool(
            (any_dr_prob_raw >= any_dr_threshold and raw_argmax >= 1)
            or (any_dr_prob_raw >= max(0.65, any_dr_threshold))
        )
        if diabetes_detected:
            pred_idx = raw_argmax if raw_argmax >= 1 else int(1 + np.argmax(probs[1:]))
        else:
            pred_idx = 0

        pred_label = labels[pred_idx]
        conf_pct = round(float(probs[pred_idx] * 100.0), 2)
        prob_map = {labels[i]: round(float(probs[i] * 100.0), 2) for i in range(len(probs))}
        referable_dr_prob = round(float(np.sum(probs[2:]) * 100.0), 2)
        any_dr_prob = round(any_dr_prob_raw * 100.0, 2)

        uncertainty = _assess_uncertainty(probs, quality["quality_score"])
        gradcam_info = (
            generate_fundus_gradcam(model, arch, x_tensor, preprocessed_pil, target_class=pred_idx)
            if include_gradcam
            else {"available": False}
        )

        if uncertainty["is_uncertain"]:
            binary_label = "⚠ PREDICTION UNCERTAIN — REVIEW RECOMMENDED"
            binary_color = "warning"
            prediction_label = f"Uncertain Retinal Screening (Leaning: {pred_label})"
        elif diabetes_detected:
            binary_label = "⚠ DIABETIC RETINOPATHY SIGNS DETECTED"
            binary_color = "warning"
            prediction_label = f"Predicted diabetic retinopathy severity: {pred_label}"
        else:
            binary_label = "✓ NO DIABETIC RETINOPATHY SIGNS DETECTED"
            binary_color = "success"
            prediction_label = f"Predicted diabetic retinopathy severity: {pred_label}"

        severity_meta = {
            0: ("No DR", "No diabetic retinopathy lesions detected on this retinal fundus scan."),
            1: ("Mild DR", "Mild non-proliferative diabetic retinopathy features detected. Periodic ophthalmic monitoring recommended."),
            2: ("Moderate DR", "Moderate diabetic retinopathy signs detected. Ophthalmologist referral recommended."),
            3: ("Severe DR", "Severe diabetic retinopathy signs detected. Prompt specialist evaluation recommended."),
            4: ("Proliferative DR", "Proliferative diabetic retinopathy signs detected. Urgent specialist retinal evaluation required."),
        }
        severity_tier, severity_description = severity_meta[pred_idx]

        clinical_report = build_retinal_screening_report(
            predicted_grade=pred_idx,
            class_probabilities=prob_map,
            confidence_pct=conf_pct,
            is_uncertain=uncertainty["is_uncertain"],
            quality=quality,
            any_dr_probability_pct=any_dr_prob,
            referable_dr_probability_pct=referable_dr_prob,
            gradcam_info=gradcam_info,
        )

        return {
            "status": "completed",
            "valid_eye_image": True,
            "analysis_mode": analysis_mode,
            "detected_modality": "retinal_fundus",
            "modality_note": modality_info["reason"],
            "file_meta": file_meta,
            "quality": quality,
            "model_name": f"PyTorch Transfer Learning ({arch})",
            "model_version": meta.get("model_version", "2.0.0"),
            "diabetes_detected": diabetes_detected,
            "binary_label": binary_label,
            "binary_color": binary_color,
            "severity_tier": severity_tier,
            "severity_description": severity_description,
            "predicted_grade": pred_idx,
            "prediction_label": prediction_label,
            "severity_label": pred_label,
            "confidence_pct": conf_pct,
            "referable_dr_probability_pct": referable_dr_prob,
            "any_dr_probability_pct": any_dr_prob,
            "any_dr_threshold_pct": round(any_dr_threshold * 100.0, 1),
            "class_probabilities": prob_map,
            "uncertainty": uncertainty,
            "is_uncertain": uncertainty["is_uncertain"],
            "uncertainty_warning": uncertainty["uncertainty_warning"],
            "retinal_validation": retinal_validation,
            "gradcam": gradcam_info,
            "gradcam_overlay_image": gradcam_info.get("overlay_image", ""),
            "clinical_report": clinical_report,
            "clinical_guidance": DR_CLINICAL_GUIDANCE.get(pred_idx, ""),
            "glucose_statement": GLUCOSE_BOUNDARY_MESSAGE,
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # Anterior eye / iris computer vision pipeline ONLY (never run retinal fundus DR CNN on iris photos)
    # Reuse PAD check if already executed on live camera snapshot; otherwise run PAD on uploaded anterior image
    if session_snapshot is not None:
        pad_res = snap_pad_res
    else:
        _, pad_res = _run_pad_on_rgb(rgb_np)

    # Presentation Attack / Spoof Gate: strictly reject any screen, photograph, or presentation attack
    if pad_res.get("is_spoof"):
        return _spoof_rejection_response(analysis_mode, file_meta, quality, pad_res)

    # Live Camera Mode defense-in-depth gate
    if is_camera_mode and not is_liveness_verified:
        return _liveness_rejection_response(analysis_mode, file_meta, pad_res=pad_res)

    cam_eye_box = snap_det_res.get("eye_box") if (session_snapshot is not None and snap_det_res) else None
    iris_res = analyze_iris_image(rgb_np, eye_box=cam_eye_box)
    return {
        "status": "completed",
        "valid_eye_image": True,
        "analysis_mode": analysis_mode,
        "detected_modality": "anterior_iris",
        "modality_note": (
            "External anterior eye / iris photograph detected (NOT an internal retinal fundus scan). "
            "Front-of-eye photos and webcam captures cannot diagnose diabetes or screen retinal vessels, "
            "so retinal DR grading is intentionally skipped."
        ),
        "file_meta": file_meta,
        "quality": quality,
        "model_name": "OpenCV Iris Segmentation & CV Feature Extractor (Non-Diagnostic)",
        "diabetes_detected": None,
        "binary_label": None,
        "binary_color": "info",
        "severity_tier": "Not Applicable (Iris Photo)",
        "severity_description": (
            "External iris segmented & quality-verified. Upload a retinal fundus scan to screen for diabetic retinopathy, "
            "or use clinical risk screening for metabolic diabetes risk."
        ),
        "predicted_grade": None,
        "prediction_label": "External Iris Segmented (Retinal Fundus Scan Required for DR Screening)",
        "confidence_pct": quality["quality_score"],
        "class_probabilities": {},
        "is_uncertain": False,
        "uncertainty_warning": "",
        "pad_result": pad_res,
        "liveness_verified": bool(is_liveness_verified),
        "liveness_telemetry": liveness_telemetry,
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
    parser.add_argument("--no-gradcam", action="store_true")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()
    res = analyze_uploaded_eye_image(
        data,
        filename=os.path.basename(args.image),
        analysis_mode=args.mode,
        include_gradcam=not args.no_gradcam,
    )
    res.pop("segmentation_overlay_image", None)
    res.pop("gradcam_overlay_image", None)
    if "gradcam" in res:
        res["gradcam"].pop("overlay_image", None)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
