"""
Experimental Computer Vision Pipeline for Anterior Eye / Iris Images.

IMPORTANT SCIENTIFIC BOUNDARY:
- Segments the eye region, locates the pupil and iris limbus, masks eyelids and
  specular glare, and extracts quantitative CV texture/color/geometry features.
- Does NOT classify iris color/patterns as diabetes indicators and does NOT
  fabricate blood glucose readings.
"""

import base64
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _encode_png_base64(rgb_np: np.ndarray) -> str:
    """Encode an RGB uint8 numpy image as a data-URI PNG string for dashboard display."""
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _locate_eye_roi(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Locate primary eye bounding box via Haar cascade or centered ROI fallback."""
    h, w = gray.shape
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    if os.path.exists(cascade_path):
        cascade = cv2.CascadeClassifier(cascade_path)
        eq = cv2.equalizeHist(gray)
        boxes = cascade.detectMultiScale(
            eq,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(max(30, min(h, w) // 6), max(30, min(h, w) // 6)),
        )
        if len(boxes) > 0:
            # Choose largest eye detection closest to image center
            cx_img, cy_img = w / 2.0, h / 2.0
            best = max(
                boxes,
                key=lambda b: (b[2] * b[3]) / (1.0 + 0.01 * np.hypot(b[0] + b[2] / 2 - cx_img, b[1] + b[3] / 2 - cy_img)),
            )
            return tuple(map(int, best))
    return (0, 0, w, h)


def analyze_iris_image(rgb_np: np.ndarray) -> Dict:
    """
    Perform computer vision iris localization, pupil/eyelid masking,
    segmentation overlay rendering, and non-diagnostic CV feature extraction.
    """
    h, w = rgb_np.shape[:2]
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

    x, y, rw, rh = _locate_eye_roi(gray)
    roi_gray = gray[y : y + rh, x : x + rw]
    roi_blurred = cv2.GaussianBlur(roi_gray, (7, 7), 1.5)
    min_dim = min(rw, rh)

    # Step 1: Detect pupil / inner circle using intensity thresholding + Hough fallback
    # Focus on central 70% of eye ROI to avoid corners/lashes
    pad_y, pad_x = int(rh * 0.15), int(rw * 0.15)
    center_patch = roi_blurred[pad_y : rh - pad_y, pad_x : rw - pad_x]
    if center_patch.size == 0:
        center_patch = roi_blurred
        pad_y, pad_x = 0, 0

    dark_thresh = np.percentile(center_patch, 14)
    pupil_bin = (center_patch <= max(15, dark_thresh)).astype(np.uint8) * 255
    pupil_bin = cv2.morphologyEx(pupil_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pupil_bin = cv2.morphologyEx(pupil_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(pupil_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_cx, pupil_cy, pupil_r = x + rw // 2, y + rh // 2, max(6, min_dim // 10)
    best_circularity = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        (cx_c, cy_c), r_c = cv2.minEnclosingCircle(cnt)
        if 0.45 <= circularity <= 1.2 and r_c <= min_dim * 0.28:
            if circularity > best_circularity:
                best_circularity = circularity
                pupil_cx = x + pad_x + int(round(cx_c))
                pupil_cy = y + pad_y + int(round(cy_c))
                pupil_r = max(6, int(round(r_c)))

    # Step 2: Detect outer iris boundary concentric to pupil via HoughCircles or geometric prior
    circles = cv2.HoughCircles(
        roi_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_dim // 3),
        param1=65,
        param2=28,
        minRadius=max(int(pupil_r * 1.6), min_dim // 7),
        maxRadius=max(int(pupil_r * 4.5), int(min_dim * 0.46)),
    )

    iris_cx, iris_cy, iris_r = pupil_cx, pupil_cy, min(int(min_dim * 0.38), int(pupil_r * 2.5))
    iris_hough_found = False
    if circles is not None:
        best_dist = float("inf")
        for c in np.round(circles[0, :]).astype(int):
            cx_i = x + int(c[0])
            cy_i = y + int(c[1])
            r_i = int(c[2])
            d = np.hypot(cx_i - pupil_cx, cy_i - pupil_cy)
            if d < min_dim * 0.25 and r_i > pupil_r * 1.35:
                if d < best_dist:
                    best_dist = d
                    iris_cx, iris_cy, iris_r = cx_i, cy_i, r_i
                    iris_hough_found = True

    iris_r = max(iris_r, pupil_r + 6)

    # Step 3: Build annular iris mask, excluding pupil, specular glare, and eyelids
    yy, xx = np.ogrid[:h, :w]
    dist_iris = np.sqrt((xx - iris_cx) ** 2 + (yy - iris_cy) ** 2)
    dist_pupil = np.sqrt((xx - pupil_cx) ** 2 + (yy - pupil_cy) ** 2)

    iris_annulus = (dist_iris <= iris_r) & (dist_pupil >= pupil_r * 1.05)

    # Eyelid masking: exclude top 18% and bottom 15% horizontal strips of iris bounding circle
    upper_lid_y = iris_cy - int(iris_r * 0.64)
    lower_lid_y = iris_cy + int(iris_r * 0.70)
    eyelid_free = (yy >= upper_lid_y) & (yy <= lower_lid_y)

    # Specular reflection mask: remove intense white spots
    glare_mask = gray >= 230
    final_iris_mask = iris_annulus & eyelid_free & (~glare_mask)

    total_annulus_pixels = max(1, int(np.sum(iris_annulus)))
    usable_pixels = int(np.sum(final_iris_mask))
    usable_iris_pct = round(usable_pixels / total_annulus_pixels * 100.0, 1)
    glare_in_annulus_pct = round(float(np.sum(iris_annulus & glare_mask)) / total_annulus_pixels * 100.0, 1)

    # Step 4: Extract Iris Computer Vision Features
    if usable_pixels >= 25:
        hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        iris_hsv = hsv[final_iris_mask]
        iris_rgb = rgb_np[final_iris_mask]
        iris_gray_vals = gray[final_iris_mask]

        mean_hue = float(np.mean(iris_hsv[:, 0]) * 2.0)  # OpenCV hue [0..179] -> [0..358] deg
        mean_sat = float(np.mean(iris_hsv[:, 1]) / 255.0 * 100.0)
        mean_val = float(np.mean(iris_hsv[:, 2]) / 255.0 * 100.0)
        rgb_mean = [round(float(v), 1) for v in np.mean(iris_rgb, axis=0)]

        # Texture metrics via Sobel gradient magnitude & Laplacian energy on masked iris
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

        texture_gradient_energy = round(float(np.mean(grad_mag[final_iris_mask])), 2)
        texture_laplacian_energy = round(float(np.mean(lap[final_iris_mask])), 2)
        intensity_std = round(float(np.std(iris_gray_vals)), 2)
    else:
        mean_hue, mean_sat, mean_val = 0.0, 0.0, 0.0
        rgb_mean = [0.0, 0.0, 0.0]
        texture_gradient_energy = 0.0
        texture_laplacian_energy = 0.0
        intensity_std = 0.0

    # Render clean segmentation visualization overlay
    vis = rgb_np.copy()
    green_overlay = vis.copy()
    green_overlay[final_iris_mask] = (16, 185, 129)  # Emerald highlight on usable iris mask
    vis = cv2.addWeighted(green_overlay, 0.32, vis, 0.68, 0)

    cv2.rectangle(vis, (x, y), (x + rw, y + rh), (99, 102, 241), 2)
    cv2.circle(vis, (iris_cx, iris_cy), iris_r, (16, 185, 129), 2)
    cv2.circle(vis, (pupil_cx, pupil_cy), pupil_r, (34, 211, 238), 2)
    cv2.line(vis, (iris_cx - iris_r, upper_lid_y), (iris_cx + iris_r, upper_lid_y), (245, 158, 11), 1)
    cv2.line(vis, (iris_cx - iris_r, lower_lid_y), (iris_cx + iris_r, lower_lid_y), (245, 158, 11), 1)

    overlay_b64 = _encode_png_base64(vis)

    return {
        "iris_detected": True,
        "segmentation_completed": usable_pixels >= 25,
        "hough_limbus_locked": iris_hough_found,
        "eye_bounding_box": {"x": x, "y": y, "width": rw, "height": rh},
        "pupil": {"cx": pupil_cx, "cy": pupil_cy, "radius_px": pupil_r},
        "iris": {"cx": iris_cx, "cy": iris_cy, "radius_px": iris_r},
        "extracted_features": {
            "pupil_to_iris_ratio": round(pupil_r / max(1, iris_r), 3),
            "usable_iris_area_pct": usable_iris_pct,
            "glare_masked_pct": glare_in_annulus_pct,
            "mean_hsv": {
                "hue_deg": round(mean_hue, 1),
                "saturation_pct": round(mean_sat, 1),
                "value_pct": round(mean_val, 1),
            },
            "mean_rgb": rgb_mean,
            "texture_gradient_energy": texture_gradient_energy,
            "texture_laplacian_energy": texture_laplacian_energy,
            "iris_intensity_std": intensity_std,
        },
        "segmentation_overlay_image": overlay_b64,
        "glucose_statement": (
            "Blood glucose cannot be measured from this image using the current model. "
            "Please use a validated blood glucose testing device."
        ),
        "scientific_note": (
            "Experimental Computer Vision feature only: extracts anatomical iris/pupil geometry "
            "and surface texture metrics. Iris patterns are NOT used to diagnose diabetes."
        ),
    }
