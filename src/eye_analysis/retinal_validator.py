"""
Dedicated Human Retinal Fundus Image Validator Model.

Strictly separates Valid Human Retinal Fundus Photographs (Class 1) from
Invalid / Non-Retinal Images (Class 0: cat eyes, dog eyes, animal photos,
human iris/anterior eye photos, human faces, webcam/screen/printed eye images,
non-retinal medical radiographs/lesions, random objects, and landscapes),
with a validation-calibrated Uncertain / Abstain decision band.

Must run BEFORE and independently of the 5-grade EfficientNet-B0 DR classifier.
Never uses image filenames as a classification signal.
"""

import json
import math
import os
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from src.eye_analysis.dataset_prep import ROOT_DIR
from src.eye_analysis.preprocessing import get_eval_transforms, pad_to_square
from src.eye_analysis.train_retinal import build_backbone_model

MODELS_DIR = os.path.join(ROOT_DIR, "models")
VALIDATOR_MODEL_PATH = os.path.join(MODELS_DIR, "retinal_validator_model.pth")
VALIDATOR_CFG_PATH = os.path.join(MODELS_DIR, "retinal_validator_config.json")
VALIDATOR_EVAL_PATH = os.path.join(MODELS_DIR, "retinal_validator_evaluation.json")
VALIDATOR_CMP_PATH = os.path.join(MODELS_DIR, "retinal_validator_comparison.json")

MSG_INVALID_NON_RETINAL = (
    "Invalid Image: Please upload a valid human retinal fundus photograph. "
    "Animal eye images, ordinary eye photographs, and non-retinal images are not supported."
)

MSG_UNCERTAIN_RETINAL = (
    "Unable to verify this as a human retinal fundus image. "
    "Please upload a clearer retinal photograph."
)

VALIDATOR_CLASS_LABELS = {
    0: "Invalid / Non-Retinal Image",
    1: "Valid Human Retinal Fundus Image",
}

_VALIDATOR_MODEL_CACHE: Optional[torch.nn.Module] = None
_VALIDATOR_META_CACHE: Optional[Dict] = None
_VALIDATOR_MTIME_CACHE: Optional[float] = None
_VALIDATOR_LOCK = threading.Lock()


def preprocess_validator_rgb(rgb_np: np.ndarray, target_size: int = 224) -> Image.Image:
    """
    Preserve natural borders, colors, and aspect ratio (`pad_to_square` -> Bicubic resize)
    WITHOUT stripping black margins or applying strong CLAHE, so the retinal validator
    preserves real aperture framing, sclera/fur context, and screen bezels.
    """
    if rgb_np.ndim == 2:
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2RGB)
    elif rgb_np.shape[2] == 4:
        rgb_np = rgb_np[:, :, :3]
    squared = pad_to_square(rgb_np)
    return Image.fromarray(squared).resize((target_size, target_size), Image.Resampling.BICUBIC)


def load_validator_bundle() -> Tuple[Optional[torch.nn.Module], Dict]:
    """Thread-safe lazy loader with automatic hot-reload when model weights update."""
    global _VALIDATOR_MODEL_CACHE, _VALIDATOR_META_CACHE, _VALIDATOR_MTIME_CACHE

    with _VALIDATOR_LOCK:
        if not os.path.exists(VALIDATOR_MODEL_PATH):
            _VALIDATOR_MODEL_CACHE = None
            _VALIDATOR_META_CACHE = None
            _VALIDATOR_MTIME_CACHE = None
            return None, {
                "trained": False,
                "message": "Retinal Fundus Validator model not found. Run src/eye_analysis/train_validator.py first.",
            }

        cur_mtime = os.path.getmtime(VALIDATOR_MODEL_PATH)
        if (
            _VALIDATOR_MODEL_CACHE is not None
            and _VALIDATOR_META_CACHE is not None
            and _VALIDATOR_MTIME_CACHE == cur_mtime
        ):
            return _VALIDATOR_MODEL_CACHE, _VALIDATOR_META_CACHE

        ckpt = torch.load(VALIDATOR_MODEL_PATH, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "resnet18")
        image_size = int(ckpt.get("image_size", 224))
        model = build_backbone_model(arch=arch, num_classes=2, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        eval_summary = {}
        if os.path.exists(VALIDATOR_EVAL_PATH):
            with open(VALIDATOR_EVAL_PATH, "r", encoding="utf-8") as f:
                eval_summary = json.load(f)

        cmp_summary = {}
        if os.path.exists(VALIDATOR_CMP_PATH):
            with open(VALIDATOR_CMP_PATH, "r", encoding="utf-8") as f:
                cmp_summary = json.load(f)

        cfg_summary = {}
        if os.path.exists(VALIDATOR_CFG_PATH):
            with open(VALIDATOR_CFG_PATH, "r", encoding="utf-8") as f:
                cfg_summary = json.load(f)

        meta = {
            "trained": True,
            "model_version": ckpt.get("model_version", "1.0.0"),
            "arch": arch,
            "image_size": image_size,
            "valid_threshold": float(ckpt.get("valid_threshold", 0.80)),
            "uncertain_low": float(ckpt.get("uncertain_low", 0.35)),
            "max_valid_entropy": float(ckpt.get("max_valid_entropy", 0.72)),
            "use_tta": bool(ckpt.get("use_tta", True)),
            "trained_at": ckpt.get("trained_at"),
            "best_epoch": ckpt.get("best_epoch"),
            "evaluation": eval_summary,
            "comparison": cmp_summary,
            "config": cfg_summary,
        }
        _VALIDATOR_MODEL_CACHE = model
        _VALIDATOR_META_CACHE = meta
        _VALIDATOR_MTIME_CACHE = cur_mtime
        return model, meta


def predict_validator_batch_tta(model: torch.nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    """2-view horizontal-flip Test-Time Augmentation for stable binary probabilities."""
    p0 = torch.softmax(model(x_batch), dim=1)
    p1 = torch.softmax(model(torch.flip(x_batch, dims=[3])), dim=1)
    return 0.5 * (p0 + p1)


def _check_retinal_structural_sanity(rgb_np: np.ndarray) -> Tuple[bool, str]:
    """
    Lightweight pixel-level sanity guard catching flat uniform frames, extreme monochrome scans,
    flat vector/solid shapes without microvascular texture, or cool-blue non-retinal backgrounds.
    """
    h, w = rgb_np.shape[:2]
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    non_dark = gray > 12
    fg_ratio = float(np.mean(non_dark))
    if fg_ratio < 0.08:
        return False, "No active illuminated retinal structure visible."

    fg = rgb_np[non_dark]
    r_m = float(np.mean(fg[:, 0]))
    g_m = float(np.mean(fg[:, 1]))
    b_m = float(np.mean(fg[:, 2]))

    # Real fundus imaging captures warm hemoglobin/choroidal spectra (not pure monochrome grey/blue-cyan)
    if np.std(rgb_np.astype(np.float32) - gray[:, :, None].astype(np.float32)) < 2.5 and b_m >= r_m - 2.0:
        return False, "Grayscale / monochrome scan lacks chromatic retinal tissue signatures."
    if b_m > r_m * 1.15 and r_m < 140.0:
        return False, "Blue-dominant illumination inconsistent with human retinal fundus photography."

    # Guard against flat synthetic circles / vector drawings lacking organic choroidal or vascular detail
    ker = max(7, min(h, w) // 16) | 1
    inner = cv2.erode(non_dark.astype(np.uint8), np.ones((ker, ker), np.uint8)) > 0
    if int(np.sum(inner)) < 100:
        inner = non_dark
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    inner_grad = np.hypot(gx, gy)[inner]
    flat_ratio = float(np.mean(inner_grad < 1.5))
    med_grad = float(np.median(inner_grad))
    if flat_ratio > 0.65 and med_grad < 1.5:
        return False, "Uniform flat region lacks organic retinal microvascular and optic-disc texture."

    return True, "OK"


def validate_human_retinal_fundus(rgb_np: np.ndarray) -> Dict:
    """
    Classify an uploaded RGB image strictly as:
      - 'valid_retinal': Accepted to proceed to quality check -> DR EfficientNet-B0
      - 'uncertain': Abstained due to borderline probability / high entropy
      - 'invalid_non_retinal': Rejected as non-retinal (animal eye, iris, face, medical X-ray, natural scene)
    """
    model, meta = load_validator_bundle()
    if model is None:
        return {
            "validator_available": False,
            "is_valid_retinal": False,
            "validator_status": "validator_not_trained",
            "retinal_fundus_probability": 0.0,
            "non_retinal_probability": 1.0,
            "normalized_entropy": 1.0,
            "valid_threshold": 0.80,
            "uncertain_low": 0.35,
            "message": meta.get("message", "Retinal validator model is not trained."),
        }

    image_size = int(meta.get("image_size", 224))
    valid_thr = float(meta.get("valid_threshold", 0.80))
    unc_low = float(meta.get("uncertain_low", 0.35))
    max_ent = float(meta.get("max_valid_entropy", 0.72))
    use_tta = bool(meta.get("use_tta", True))

    prep_pil = preprocess_validator_rgb(rgb_np, target_size=image_size)
    eval_tf = get_eval_transforms(image_size)
    x = eval_tf(prep_pil).unsqueeze(0)

    with torch.no_grad():
        probs_t = predict_validator_batch_tta(model, x)[0] if use_tta else torch.softmax(model(x), dim=1)[0]
        probs = probs_t.numpy()

    p_invalid = float(probs[0])
    p_retinal = float(probs[1])
    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)))
    norm_entropy = entropy / math.log(2.0)

    sane, sanity_reason = _check_retinal_structural_sanity(rgb_np)

    if not sane:
        # Never force grayscale X-rays or pure blue/blank textures through as retinal
        eff_p_retinal = min(p_retinal, 0.20)
        validator_status = "invalid_non_retinal"
        is_valid = False
        msg = MSG_INVALID_NON_RETINAL
    elif p_retinal >= valid_thr and norm_entropy <= max_ent:
        eff_p_retinal = p_retinal
        validator_status = "valid_retinal"
        is_valid = True
        msg = "Valid human retinal fundus photograph verified."
    elif p_retinal >= unc_low:
        eff_p_retinal = p_retinal
        validator_status = "uncertain"
        is_valid = False
        msg = MSG_UNCERTAIN_RETINAL
    else:
        eff_p_retinal = p_retinal
        validator_status = "invalid_non_retinal"
        is_valid = False
        msg = MSG_INVALID_NON_RETINAL

    return {
        "validator_available": True,
        "model_architecture": meta.get("arch"),
        "model_version": meta.get("model_version"),
        "is_valid_retinal": is_valid,
        "validator_status": validator_status,
        "retinal_fundus_probability": round(eff_p_retinal, 4),
        "retinal_fundus_probability_pct": round(eff_p_retinal * 100.0, 2),
        "non_retinal_probability": round(1.0 - eff_p_retinal, 4),
        "non_retinal_probability_pct": round((1.0 - eff_p_retinal) * 100.0, 2),
        "normalized_entropy": round(norm_entropy, 4),
        "valid_threshold": round(valid_thr, 3),
        "uncertain_low": round(unc_low, 3),
        "structural_sanity_ok": sane,
        "structural_sanity_note": sanity_reason,
        "message": msg,
    }
