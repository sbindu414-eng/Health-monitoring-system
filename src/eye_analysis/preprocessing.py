"""
Image Validation, Quality Assessment, Modality Verification, and Preprocessing.

Handles:
- File extension, size, and readable RGB image verification
- Image sharpness (Laplacian variance), brightness, contrast, glare/exposure checks
- Eye image modality discrimination:
  * Retinal fundus photograph vs. Anterior eye/iris photograph vs. Invalid/non-eye image
- Retinal fundus border cropping & LAB-space CLAHE contrast normalization
- Training data augmentation transforms (applied ONLY during training) & eval transforms
"""

import io
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def decode_and_validate_upload(
    file_bytes: bytes,
    filename: str = "upload.png",
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Tuple[np.ndarray, Dict]:
    """Validate uploaded image size, extension, decoding integrity, and dimensions."""
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > max_bytes:
        raise ValueError(
            f"File size ({len(file_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit ({max_bytes // (1024 * 1024)} MB)."
        )

    ext = os.path.splitext(filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            pil_img.verify()
        with Image.open(io.BytesIO(file_bytes)) as pil_img:
            img_format = pil_img.format or "UNKNOWN"
            rgb_img = pil_img.convert("RGB")
            rgb_np = np.asarray(rgb_img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    height, width = rgb_np.shape[:2]
    if width < 64 or height < 64:
        raise ValueError(
            f"Image resolution ({width}x{height}) is too small for reliable eye analysis (minimum 64x64)."
        )

    return rgb_np, {
        "filename": filename,
        "format": img_format,
        "width": int(width),
        "height": int(height),
        "file_size_bytes": int(len(file_bytes)),
    }


def assess_image_quality(rgb_np: np.ndarray, is_fundus_like: bool = False) -> Dict:
    """
    Evaluate focus sharpness (resolution-normalized), luminance exposure, dynamic contrast, glare, and FOV completeness.

    For retinal fundus images, pure black framing borders around the circular FOV
    are cropped/masked before evaluation, and multi-megapixel inputs are normalized
    to a 256px reference scale before Laplacian sharpness so crisp 4K clinical photos
    are not falsely flagged as blurry.
    """
    if is_fundus_like:
        eval_rgb = crop_black_borders(rgb_np)
    else:
        eval_rgb = rgb_np

    eh, ew = eval_rgb.shape[:2]
    max_side = max(eh, ew)
    if max_side > 256:
        scale = 256.0 / float(max_side)
        norm_rgb = cv2.resize(
            eval_rgb,
            (max(32, int(round(ew * scale))), max(32, int(round(eh * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        norm_rgb = eval_rgb

    bgr = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Mask black fundus padding if evaluating fundus FOV interior
    non_black_mask = gray > 12
    active_ratio = float(np.mean(non_black_mask))
    if is_fundus_like and active_ratio > 0.15:
        active_pixels = gray[non_black_mask]
        lap_map = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_raw = float(lap_map[non_black_mask].var())
    else:
        active_pixels = gray.reshape(-1)
        sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness = float(np.mean(active_pixels))
    contrast = float(np.std(active_pixels))
    dark_ratio = float(np.mean(active_pixels < 18))
    bright_glare_ratio = float(np.mean(active_pixels > 245))

    reasons = []
    if is_fundus_like and active_ratio < 0.22:
        reasons.append(f"Incomplete or severely occluded retinal field of view (active area={active_ratio * 100:.1f}% < 22%).")
    min_sharpness = 8.0 if is_fundus_like else 18.0
    if sharpness_raw < min_sharpness:
        reasons.append(f"Image appears overly blurry or lacks edge detail (Laplacian variance={sharpness_raw:.1f} < {min_sharpness}).")
    if brightness < 28.0:
        reasons.append(f"Image is severely underexposed / too dark (mean luminance={brightness:.1f}/255).")
    elif brightness > 235.0:
        reasons.append(f"Image is overexposed / washed out (mean luminance={brightness:.1f}/255).")
    if contrast < 12.0:
        reasons.append(f"Insufficient contrast / flat signal (std={contrast:.1f}).")
    if bright_glare_ratio > 0.35:
        reasons.append(f"Excessive specular glare / white blowout covers {bright_glare_ratio * 100:.1f}% of pixels.")

    # Compute normalized quality score 0..100
    sharp_norm = min(1.0, sharpness_raw / 120.0)
    bright_norm = max(0.0, 1.0 - abs(brightness - 118.0) / 118.0)
    contrast_norm = min(1.0, contrast / 55.0)
    glare_penalty = max(0.0, bright_glare_ratio - 0.10) * 1.5
    quality_score = round(
        max(0.0, min(100.0, (0.45 * sharp_norm + 0.30 * bright_norm + 0.25 * contrast_norm - glare_penalty) * 100.0)),
        1,
    )

    is_acceptable = len(reasons) == 0
    return {
        "is_acceptable": is_acceptable,
        "status": "acceptable" if is_acceptable else "unacceptable",
        "quality_score": quality_score,
        "sharpness_laplacian_var": round(sharpness_raw, 2),
        "mean_brightness": round(brightness, 2),
        "contrast_std": round(contrast, 2),
        "dark_pixel_ratio": round(dark_ratio, 4),
        "glare_ratio": round(bright_glare_ratio, 4),
        "issues": reasons,
    }


def _retinal_fundus_heuristics(rgb_np: np.ndarray) -> Dict:
    """Compute physical/chromatic signatures characteristic of retinal fundus images."""
    h, w = rgb_np.shape[:2]
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

    # Check corner darkness vs center brightness (fundus camera circular aperture)
    ch, cw = max(8, h // 8), max(8, w // 8)
    corners = np.concatenate([
        gray[:ch, :cw].ravel(),
        gray[:ch, -cw:].ravel(),
        gray[-ch:, :cw].ravel(),
        gray[-ch:, -cw:].ravel(),
    ])
    center_crop = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    corner_dark = float(np.mean(corners < 25))
    center_mean = float(np.mean(center_crop))

    # Inside illuminated region: retinal tissue has strong Red > Green >= Blue distribution
    fg_mask = gray > 20
    if float(np.mean(fg_mask)) < 0.10:
        return {"fundus_score": 0.0, "is_fundus": False, "corner_dark_ratio": corner_dark}

    fg_pixels = rgb_np[fg_mask]
    r_mean = float(np.mean(fg_pixels[:, 0]))
    g_mean = float(np.mean(fg_pixels[:, 1]))
    b_mean = float(np.mean(fg_pixels[:, 2]))

    red_dom = (r_mean > g_mean * 1.08) and (r_mean > b_mean * 1.35)
    warm_ratio = (r_mean - b_mean) / max(1.0, r_mean + g_mean + b_mean)

    # Edge / circular dark border symmetry
    score = 0.0
    if corner_dark >= 0.45 and center_mean > 35:
        score += 0.45
    elif corner_dark >= 0.20 and center_mean > 35:
        score += 0.25
    if red_dom:
        score += 0.35
    if warm_ratio > 0.14:
        score += 0.20

    # Extra check: retinal fundus can also be tightly cropped (no dark corners)
    if r_mean > 1.25 * g_mean and r_mean > 1.7 * b_mean and b_mean < 95:
        score = max(score, 0.72)

    return {
        "fundus_score": round(min(1.0, score), 3),
        "is_fundus": score >= 0.55,
        "corner_dark_ratio": round(corner_dark, 3),
        "rgb_means": [round(r_mean, 1), round(g_mean, 1), round(b_mean, 1)],
    }


def _anterior_eye_heuristics(rgb_np: np.ndarray) -> Dict:
    """Detect whether an image contains an anterior eye / iris using OpenCV cascades + circle geometry."""
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    min_side = min(h, w)

    # 1. Haar cascade eye detection (handles both single-eye and two-eye portrait photos)
    eye_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    haar_eyes = []
    if os.path.exists(eye_cascade_path):
        cascade = cv2.CascadeClassifier(eye_cascade_path)
        eq = cv2.equalizeHist(gray)
        detected = cascade.detectMultiScale(
            eq,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(max(20, min_side // 10), max(20, min_side // 10)),
        )
        if len(detected) > 0:
            haar_eyes = [tuple(map(int, box)) for box in detected]

    # 2. Concentric iris/pupil/sclera circle test with outward radial gradient symmetry
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_side // 5),
        param1=65,
        param2=27,
        minRadius=max(14, min_side // 16),
        maxRadius=max(20, int(min_side * 0.48)),
    )

    valid_circle_hit = False
    best_circle = None
    if circles is not None:
        cy_img = h / 2.0
        thetas = np.linspace(0, 2.0 * np.pi, 36, endpoint=False)
        for c in np.round(circles[0, :]).astype(int):
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            if 0 <= cx < w and 0 <= cy < h and r >= 14:
                # Verify radial outward brightness transition around circular rim (eliminates square grids)
                pts, radial_hits = 0, 0
                for th in thetas:
                    px = int(round(cx + r * np.cos(th)))
                    py = int(round(cy + r * np.sin(th)))
                    if 0 <= px < w and 0 <= py < h:
                        pts += 1
                        dot = gx[py, px] * np.cos(th) + gy[py, px] * np.sin(th)
                        if dot > 12.0:
                            radial_hits += 1
                outward_ratio = radial_hits / max(1, pts)

                yy, xx = np.ogrid[:h, :w]
                dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                inner_mask = dist2 <= (r * 0.42) ** 2
                ring_mask = (dist2 > (r * 0.45) ** 2) & (dist2 <= (r * 0.95) ** 2)
                outer_mask = (dist2 > (r * 1.05) ** 2) & (dist2 <= (r * 1.55) ** 2)
                if np.any(inner_mask) and np.any(ring_mask):
                    inner_val = float(np.mean(gray[inner_mask]))
                    ring_val = float(np.mean(gray[ring_mask]))
                    outer_val = float(np.mean(gray[outer_mask])) if np.any(outer_mask) else ring_val
                    is_pupil_dark = inner_val + 8 < ring_val
                    is_iris_in_sclera = (ring_val + 14 < outer_val) and (inner_val < outer_val - 10)
                    if (is_pupil_dark or is_iris_in_sclera) and outward_ratio >= 0.42:
                        vert_offset = abs(cy - cy_img) / max(1.0, h)
                        if vert_offset < 0.36 or len(haar_eyes) > 0:
                            valid_circle_hit = True
                            best_circle = (cx, cy, r)
                            break

    score = 0.0
    if len(haar_eyes) > 0:
        score += 0.55
    if valid_circle_hit:
        score += 0.45

    return {
        "anterior_eye_score": round(min(1.0, score), 3),
        "is_anterior_eye": score >= 0.45,
        "haar_detections": haar_eyes,
        "hough_circle_detected": valid_circle_hit,
        "best_circle": best_circle,
    }


def detect_eye_modality(rgb_np: np.ndarray, requested_mode: str = "auto") -> Dict:
    """
    Validate whether uploaded image depicts a valid eye image and identify modality:
    - 'retinal_fundus'
    - 'anterior_iris'
    - 'invalid'
    """
    fundus_info = _retinal_fundus_heuristics(rgb_np)
    anterior_info = _anterior_eye_heuristics(rgb_np)

    is_fundus = fundus_info["is_fundus"]
    is_anterior = anterior_info["is_anterior_eye"]

    if requested_mode == "retinal_fundus":
        if is_fundus:
            modality = "retinal_fundus"
            valid_eye = True
            reason = "Valid retinal fundus image signature detected."
        elif is_anterior:
            modality = "anterior_iris"
            valid_eye = True
            reason = (
                "Notice: Uploaded image appears to be an external anterior eye/iris photo rather than a "
                "retinal fundus scan. Switching to experimental Iris CV pipeline instead of retinal DR grading."
            )
        else:
            modality = "invalid"
            valid_eye = False
            reason = (
                "No valid retinal fundus or eye structure detected. Please upload a clear retinal fundus photograph "
                "(macula/optic-disc view) or anterior eye photograph."
            )
    elif requested_mode == "anterior_iris":
        if is_anterior:
            modality = "anterior_iris"
            valid_eye = True
            reason = "Valid anterior eye / iris region detected."
        elif is_fundus:
            modality = "retinal_fundus"
            valid_eye = True
            reason = (
                "Notice: Uploaded image has retinal fundus optical characteristics rather than an external iris photo."
            )
        else:
            modality = "invalid"
            valid_eye = False
            reason = (
                "Could not locate an anterior eye / iris region. Ensure the eye and iris are clearly centered and unobscured."
            )
    else:  # auto
        if is_fundus and fundus_info["fundus_score"] >= anterior_info["anterior_eye_score"]:
            modality = "retinal_fundus"
            valid_eye = True
            reason = "Retinal fundus photograph detected automatically."
        elif is_anterior:
            modality = "anterior_iris"
            valid_eye = True
            reason = "Anterior eye / iris photograph detected automatically."
        elif fundus_info["fundus_score"] >= 0.40:
            modality = "retinal_fundus"
            valid_eye = True
            reason = "Retinal fundus characteristics detected (low border contrast)."
        else:
            modality = "invalid"
            valid_eye = False
            reason = (
                "Image rejected: No recognizable retinal fundus or anterior eye/iris anatomy detected."
            )

    return {
        "valid_eye_image": valid_eye,
        "detected_modality": modality,
        "reason": reason,
        "fundus_heuristics": fundus_info,
        "anterior_heuristics": anterior_info,
    }


def crop_black_borders(rgb_np: np.ndarray, tol: int = 10) -> np.ndarray:
    """Trim dark outer camera border margins surrounding a circular retinal fundus FOV."""
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if np.mean(mask) < 0.05:
        return rgb_np
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = rgb_np[y0:y1, x0:x1]
    return cropped if min(cropped.shape[:2]) >= 32 else rgb_np


def pad_to_square(rgb_np: np.ndarray) -> np.ndarray:
    """Pad an RGB fundus image equally to a square using black margins so aspect ratio is preserved."""
    h, w = rgb_np.shape[:2]
    if h == w:
        return rgb_np
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(rgb_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def preprocess_fundus_rgb(rgb_np: np.ndarray, target_size: int = 224, apply_clahe: bool = True) -> Image.Image:
    """
    Standardized clinical preprocessing for retinal fundus images:
    1. Trim dead black outer camera margins
    2. Aspect-ratio-preserving square pad so optic disc & vessel geometry are never stretched
    3. Optional gentle CLAHE contrast equalization on LAB luminance channel
    4. Resize with bicubic interpolation to model input dimensions
    """
    cropped = crop_black_borders(rgb_np)
    squared = pad_to_square(cropped)
    if apply_clahe:
        lab = cv2.cvtColor(squared, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_chan)
        merged = cv2.merge((l_eq, a_chan, b_chan))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        enhanced = squared

    pil_out = Image.fromarray(enhanced).resize((target_size, target_size), Image.Resampling.BICUBIC)
    return pil_out


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training-only lesion-preserving augmentation + normalization pipeline (never applied to val/test)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.86, 1.0), ratio=(0.94, 1.06)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.015),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic validation / test / inference normalization pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_preprocessing_config(image_size: int = 224, apply_clahe: bool = True) -> Dict:
    """Serializable preprocessing config saved alongside trained weights."""
    return {
        "image_size": image_size,
        "color_space": "RGB",
        "crop_black_borders": True,
        "pad_to_square": True,
        "apply_clahe": apply_clahe,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": [8, 8],
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "train_augmentations": [
            "RandomResizedCrop(scale=(0.86, 1.0), ratio=(0.94, 1.06))",
            "RandomHorizontalFlip(p=0.5)",
            "RandomRotation(degrees=15)",
            "ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.015)",
        ],
    }
