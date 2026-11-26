"""
Real-Time Live Human Eye Verification & Presentation Attack Detection (PAD) Module.

Defends against presentation attacks including:
- Eye photographs displayed on mobile phones (OLED/LCD screen grids, Moiré patterns, high-frequency FFT spikes, device bezels)
- Eye photographs displayed on laptop / desktop computer monitors
- Printed eye photographs on paper (flat diffuse reflection, absence of 3D corneal curvature, halftone texture)
- Replayed eye videos (rigid planar device motion, screen refresh/pixel grid artifacts)
- Static or freeze-frame eye spoofs (lack of physiological ocular micro-tremor, absence of natural blink dynamics)

Enforces strict multi-frame temporal validation directly from live webcam streams.
Includes a live challenge-response system to defeat replayed video attacks.
"""

import base64
import math
import os
import random
import secrets
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Canonical state machine state identifiers
STATE_WAITING_FOR_CAMERA = "WAITING_FOR_CAMERA"
STATE_WAITING_FOR_EYE = "WAITING_FOR_EYE"
STATE_EYE_DETECTED = "EYE_DETECTED"
STATE_VERIFYING_LIVENESS = "VERIFYING_LIVENESS"
STATE_PAD_DETECTED = "PRESENTATION_ATTACK_DETECTED"
STATE_LIVE_VERIFIED = "LIVE_EYE_VERIFIED"
STATE_ANALYSIS_RUNNING = "ANALYSIS_RUNNING"
STATE_ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
STATE_VERIFICATION_FAILED = "VERIFICATION_FAILED"

# Standard backend guidance messages (preserved for API & test compatibility)
MSG_LIVENESS_FAILED = "Live human eye not verified. Please position your real eye directly in front of the camera."
MSG_SPOOF_DETECTED = "Possible photograph or screen detected. Please use a real eye directly in front of the camera."
MSG_WAITING_FOR_EYE = "Waiting for eye. Please look directly into the camera."
MSG_EYE_DETECTED = "Eye detected. Verifying that it is live."
MSG_LIVE_VERIFIED = "Live eye verified. Starting analysis."
MSG_VERIFICATION_FAILED = "Live human eye could not be verified. Please try again."

# Exact user-facing UI messages mapped to verification states
UI_MESSAGES = {
    "waiting_for_camera": "Start your camera to begin.",
    "waiting_for_eye": MSG_WAITING_FOR_EYE,
    "eye_detected": "Eye detected. Verifying that it is live.",
    "verifying_liveness": "Please follow the eye movement instructions.",
    "spoof_detected": "Possible screen or photograph detected. Analysis blocked.",
    "verification_failed": "Live human eye could not be verified. Please try again.",
    "live_verified": "Live eye verified. Starting analysis.",
    "analysis_in_progress": "Analysis in progress...",
    "completed": "Eye analysis completed successfully.",
}

# Validated verification & PAD thresholds
LIVENESS_CONFIDENCE_THRESHOLD = 70.0
TEMPORAL_LIVENESS_THRESHOLD = 0.65
PAD_GENUINE_THRESHOLD = 0.65
SESSION_TOKEN_TTL_SECONDS = 15.0

# Challenge-response system constants
CHALLENGE_TIMEOUT_SECONDS = 12.0
CHALLENGE_STEPS = [
    {"id": "look_straight", "instruction": "Look straight at the camera.",  "type": "fixation"},
    {"id": "blink",         "instruction": "Blink naturally once.",         "type": "blink"},
    {"id": "look_left",     "instruction": "Look to your left.",            "type": "gaze_left"},
    {"id": "look_right",    "instruction": "Look to your right.",           "type": "gaze_right"},
]


# ---------------------------------------------------------------------------
# Helper: Fast Local Binary Patterns (LBP) for Texture Analysis
# ---------------------------------------------------------------------------
def _compute_fast_lbp(gray_patch: np.ndarray) -> np.ndarray:
    """Compute 8-neighbor Local Binary Pattern for fine-grained surface micro-texture."""
    if gray_patch.shape[0] < 3 or gray_patch.shape[1] < 3:
        return np.zeros((16,), dtype=np.float32)
    h, w = gray_patch.shape
    center = gray_patch[1 : h - 1, 1 : w - 1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    shifts = [
        (-1, -1, 0),
        (-1, 0, 1),
        (-1, 1, 2),
        (0, 1, 3),
        (1, 1, 4),
        (1, 0, 5),
        (1, -1, 6),
        (0, -1, 7),
    ]
    for dy, dx, bit in shifts:
        neighbor = gray_patch[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= ((neighbor >= center).astype(np.uint8)) << bit
    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
    total = np.sum(hist)
    prob = hist.astype(np.float32) / (total if total > 0 else 1.0)
    return prob


# ---------------------------------------------------------------------------
# 1. Real-Time Eye & Face Region Detector (Consecutive Frames)
# ---------------------------------------------------------------------------
class RealTimeEyeDetector:
    """
    Detects face and eye regions across consecutive camera frames.
    Tracks bounding boxes temporally to ensure continuity and smooth frame-to-frame jitter.
    """

    def __init__(self):
        cv_data = cv2.data.haarcascades
        face_path = os.path.join(cv_data, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(cv_data, "haarcascade_eye.xml")
        glasses_path = os.path.join(cv_data, "haarcascade_eye_tree_eyeglasses.xml")

        self.face_cascade = cv2.CascadeClassifier(face_path) if os.path.exists(face_path) else None
        self.eye_cascade = cv2.CascadeClassifier(eye_path) if os.path.exists(eye_path) else None
        self.glasses_cascade = cv2.CascadeClassifier(glasses_path) if os.path.exists(glasses_path) else None

        # Temporal tracking memory
        self.tracked_eye_box: Optional[Tuple[int, int, int, int]] = None
        self.tracked_face_box: Optional[Tuple[int, int, int, int]] = None
        self.consecutive_detections = 0
        self.missed_cascade_frames = 0

    def _estimate_pupil_rel_x(self, gray_roi: np.ndarray) -> float:
        """
        Estimate normalized horizontal pupil/iris center offset in [-0.5, +0.5]
        within the detected eye ROI to detect pure iris gaze shifts (left/right).
        """
        if gray_roi.size == 0 or gray_roi.shape[0] < 8 or gray_roi.shape[1] < 8:
            return 0.0
        h, w = gray_roi.shape
        y1, y2 = int(h * 0.25), max(int(h * 0.78), int(h * 0.25) + 2)
        x1, x2 = int(w * 0.12), max(int(w * 0.88), int(w * 0.12) + 2)
        band = cv2.GaussianBlur(gray_roi[y1:y2, x1:x2], (3, 3), 0)
        if band.size == 0:
            return 0.0
        thresh = float(np.percentile(band, 22))
        mask = band <= thresh
        if not np.any(mask):
            return 0.0
        ys, xs = np.where(mask)
        cx_in_roi = x1 + float(np.mean(xs))
        return round((cx_in_roi - (w / 2.0)) / max(1.0, float(w)), 4)

    def detect_eye_region(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        Detect face and primary eye in frame.
        Returns dict with bounding boxes, cropped eye ROI, eye aperture metric, and pupil gaze offset.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Face Detection on raw gray first (OpenCV Haar cascades are trained on raw 8-bit images)
        face_box = None
        if self.face_cascade:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(int(min(h, w) * 0.18), int(min(h, w) * 0.18))
            )
            if len(faces) == 0:
                # Try CLAHE for low-contrast/backlit conditions
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                faces = self.face_cascade.detectMultiScale(
                    clahe, scaleFactor=1.1, minNeighbors=2, minSize=(int(min(h, w) * 0.18), int(min(h, w) * 0.18))
                )
            if len(faces) > 0:
                face_box = max(faces, key=lambda b: b[2] * b[3])
                face_box = tuple(map(int, face_box))

        # 2. Search area: eye band of face if found, else central 84% of image
        if face_box:
            fx, fy, fw, fh = face_box
            search_x = max(0, fx)
            search_y = max(0, fy + int(fh * 0.12))
            search_w = min(w - search_x, fw)
            search_h = min(h - search_y, int(fh * 0.48))
        else:
            search_x = int(w * 0.08)
            search_y = int(h * 0.08)
            search_w = int(w * 0.84)
            search_h = int(h * 0.84)

        search_roi = gray[search_y : search_y + search_h, search_x : search_x + search_w]
        if search_roi.size == 0:
            return None

        # 3. Detect eyes within search area
        eyes = []
        if self.eye_cascade:
            e = self.eye_cascade.detectMultiScale(
                search_roi, scaleFactor=1.06, minNeighbors=2, minSize=(int(min(h, w) * 0.05), int(min(h, w) * 0.05))
            )
            if len(e) > 0:
                eyes.extend(e)

        if len(eyes) == 0 and self.glasses_cascade:
            eg = self.glasses_cascade.detectMultiScale(
                search_roi, scaleFactor=1.06, minNeighbors=2, minSize=(int(min(h, w) * 0.05), int(min(h, w) * 0.05))
            )
            if len(eg) > 0:
                eyes.extend(eg)

        # 4. Fallback if cascades did not detect eye in current frame (bounded to survive brief blinks)
        if len(eyes) == 0:
            self.missed_cascade_frames += 1

            # Fallback 1: Brief continuity during natural blink closure (up to 5 frames ~0.7s when patch still contains eyelid texture)
            max_miss = 5 if face_box is not None else 4
            if self.tracked_eye_box is not None and self.missed_cascade_frames <= max_miss:
                tx, ty, tw, th = self.tracked_eye_box
                patch = gray[ty : ty + th, tx : tx + tw]
                if patch.size > 0 and np.std(patch) > 8.0:
                    self.consecutive_detections += 1
                    eye_crop_bgr = frame_bgr[ty : ty + th, tx : tx + tw]
                    aperture = self._estimate_eye_openness(patch)
                    return {
                        "eye_detected": True,
                        "eye_box": self.tracked_eye_box,
                        "face_box": self.tracked_face_box,
                        "eye_crop_bgr": eye_crop_bgr,
                        "eye_crop_gray": patch,
                        "openness_ratio": aperture,
                        "eye_aspect_ratio": aperture,
                        "pupil_rel_x": self._estimate_pupil_rel_x(patch),
                        "is_open": aperture > 0.22,
                        "consecutive_count": self.consecutive_detections,
                    }

            # Fallback 2: Strict direct macro anterior eye close-up (dark central pupil inside brighter annulus)
            center_x, center_y = w // 2, h // 2
            roi_w, roi_h = int(w * 0.35), int(h * 0.30)
            cx1, cy1 = max(0, center_x - roi_w // 2), max(0, center_y - roi_h // 2)
            center_patch = gray[cy1 : cy1 + roi_h, cx1 : cx1 + roi_w]
            if center_patch.size > 0:
                patch_std = float(np.std(center_patch))
                ph, pw = center_patch.shape
                inner = center_patch[ph // 3 : 2 * ph // 3, pw // 3 : 2 * pw // 3]
                left_w = center_patch[ph // 3 : 2 * ph // 3, : max(1, pw // 5)]
                right_w = center_patch[ph // 3 : 2 * ph // 3, min(pw - 1, 4 * pw // 5) :]
                if (
                    patch_std > 18.0
                    and inner.size > 0
                    and left_w.size > 0
                    and right_w.size > 0
                    and float(np.mean(inner)) + 15.0 < 0.5 * (float(np.mean(left_w)) + float(np.mean(right_w)))
                ):
                    macro_box = (cx1, cy1, roi_w, roi_h)
                    self.tracked_eye_box = macro_box
                    self.missed_cascade_frames = 0
                    self.consecutive_detections += 1
                    eye_crop_bgr = frame_bgr[cy1 : cy1 + roi_h, cx1 : cx1 + roi_w]
                    aperture = self._estimate_eye_openness(center_patch)
                    return {
                        "eye_detected": True,
                        "eye_box": macro_box,
                        "face_box": None,
                        "eye_crop_bgr": eye_crop_bgr,
                        "eye_crop_gray": center_patch,
                        "openness_ratio": aperture,
                        "eye_aspect_ratio": aperture,
                        "pupil_rel_x": self._estimate_pupil_rel_x(center_patch),
                        "is_open": aperture > 0.22,
                        "consecutive_count": self.consecutive_detections,
                    }

            # Eye genuinely disappeared
            if self.missed_cascade_frames >= 3:
                self.tracked_eye_box = None
                self.tracked_face_box = None
                self.consecutive_detections = 0
            return None

        # Eye cascade matched cleanly
        self.missed_cascade_frames = 0

        # Select most prominent eye (largest area closest to optical center)
        cx_target = search_w / 2.0
        best_eye = max(
            eyes,
            key=lambda b: (b[2] * b[3]) / (1.0 + 0.005 * abs((b[0] + b[2] / 2) - cx_target)),
        )

        # Convert back to full-frame coordinates
        ex = int(search_x + best_eye[0])
        ey = int(search_y + best_eye[1])
        ew = int(best_eye[2])
        eh = int(best_eye[3])

        # Smooth bounding box with gentle temporal EMA (alpha=0.55) while preserving real movement
        if self.tracked_eye_box is not None:
            ox, oy, ow, oh = self.tracked_eye_box
            alpha = 0.55
            ex = int(round(alpha * ex + (1 - alpha) * ox))
            ey = int(round(alpha * ey + (1 - alpha) * oy))
            ew = int(round(alpha * ew + (1 - alpha) * ow))
            eh = int(round(alpha * eh + (1 - alpha) * oh))

        # Clamp to frame bounds
        ex = max(0, min(w - 10, ex))
        ey = max(0, min(h - 10, ey))
        ew = max(10, min(w - ex, ew))
        eh = max(10, min(h - ey, eh))

        eye_box = (ex, ey, ew, eh)
        self.tracked_eye_box = eye_box
        self.tracked_face_box = face_box
        self.consecutive_detections += 1

        eye_crop_bgr = frame_bgr[ey : ey + eh, ex : ex + ew]
        eye_crop_gray = gray[ey : ey + eh, ex : ex + ew]
        aperture = self._estimate_eye_openness(eye_crop_gray)

        return {
            "eye_detected": True,
            "eye_box": eye_box,
            "face_box": face_box,
            "eye_crop_bgr": eye_crop_bgr,
            "eye_crop_gray": eye_crop_gray,
            "openness_ratio": aperture,
            "eye_aspect_ratio": aperture,
            "pupil_rel_x": self._estimate_pupil_rel_x(eye_crop_gray),
            "is_open": aperture > 0.22,
            "consecutive_count": self.consecutive_detections,
        }

    def _estimate_eye_openness(self, gray_roi: np.ndarray) -> float:
        """
        Estimate lighting-invariant vertical eye openness ratio (EAR proxy).
        Uses anatomical horizontal sclera-to-iris contrast plus central slit vertical structure.
        Fully open eye: ~0.30 - 0.52; closed / blinking eye: <= 0.20.
        """
        if gray_roi.size == 0:
            return 0.35
        h, w = gray_roi.shape
        if h < 8 or w < 8:
            return 0.35

        p90 = float(np.percentile(gray_roi, 90))
        p10 = float(np.percentile(gray_roi, 10))
        dyn_range = p90 - p10
        if dyn_range < 12.0:
            return 0.12

        # Central eyelid slit band
        y1 = int(h * 0.32)
        y2 = max(int(h * 0.68), y1 + 2)
        left_sclera = gray_roi[y1:y2, int(w * 0.14) : max(int(w * 0.33), int(w * 0.14) + 1)]
        right_sclera = gray_roi[y1:y2, int(w * 0.67) : max(int(w * 0.86), int(w * 0.67) + 1)]
        center_iris = gray_roi[y1:y2, int(w * 0.36) : max(int(w * 0.64), int(w * 0.36) + 1)]

        sclera_vals = [float(np.percentile(x, 80)) for x in (left_sclera, right_sclera) if x.size > 0]
        sclera_hi = float(np.mean(sclera_vals)) if sclera_vals else p90
        iris_lo = float(np.percentile(center_iris, 25)) if center_iris.size > 0 else p10

        horiz_contrast = max(0.0, sclera_hi - iris_lo) / max(28.0, dyn_range)

        # Dark iris vertical aperture & contrast within slit
        col = gray_roi[:, int(w * 0.32) : max(int(w * 0.68), int(w * 0.32) + 1)]
        thresh = iris_lo + 0.35 * max(10.0, sclera_hi - iris_lo)
        dark_rows = float(np.mean(np.min(col, axis=1) < thresh)) if col.size > 0 else 0.0

        slit_patch = gray_roi[y1:y2, int(w * 0.20) : max(int(w * 0.80), int(w * 0.20) + 1)]
        slit_std = float(np.std(slit_patch)) if slit_patch.size > 0 else 0.0
        std_norm = min(1.0, slit_std / max(20.0, dyn_range * 0.42))

        score = 0.45 * min(1.0, horiz_contrast) + 0.35 * std_norm + 0.20 * min(1.0, dark_rows / 0.65)
        ear = float(np.clip(0.10 + score * 0.38, 0.10, 0.55))
        return round(ear, 3)

    def reset(self):
        """Reset temporal tracking state."""
        self.tracked_eye_box = None
        self.tracked_face_box = None
        self.consecutive_detections = 0
        self.missed_cascade_frames = 0


# ---------------------------------------------------------------------------
# 2. Presentation Attack Detector (PAD)
# ---------------------------------------------------------------------------
class PresentationAttackDetector:
    """
    Presentation Attack Detection (PAD) for physical biometric authentication:
    - Analyzes native-resolution 2D FFT for display pixel grid / Moiré harmonics without upsampling artifacts.
    - Analyzes multi-channel Local Binary Patterns (LBP) for screen and print textures.
    - Analyzes specular corneal reflection vs flat screen glare / diffuse matte print reflection.
    - Detects rectangular device borders / screen bezels and emissive display plateaus.
    """

    def analyze_frame_pad(
        self,
        frame_bgr: np.ndarray,
        eye_box: Tuple[int, int, int, int],
        face_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict:
        """
        Analyze a single frame for presentation attack signatures.
        Returns detailed scores and spoof classification.
        """
        ex, ey, ew, eh = eye_box
        h, w = frame_bgr.shape[:2]

        eye_bgr = frame_bgr[ey : ey + eh, ex : ex + ew]
        eye_gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY) if eye_bgr.size > 0 else np.zeros((10, 10), np.uint8)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Extract native-resolution periocular window (min 64px if possible) to avoid small-ROI upsampling artifacts
        eye_cx, eye_cy = ex + ew // 2, ey + eh // 2
        native_side = max(min(h, w, 96), max(ew, eh))
        nx1 = max(0, min(w - native_side, eye_cx - native_side // 2))
        ny1 = max(0, min(h - native_side, eye_cy - native_side // 2))
        native_peri_gray = frame_gray[ny1 : ny1 + native_side, nx1 : nx1 + native_side]

        # 1. Native-Resolution 2D FFT Moiré & Screen Pixel Grid Analysis
        moire_eye, peak_eye = self._analyze_fft_moire(eye_gray)
        moire_peri, peak_peri = self._analyze_fft_moire(native_peri_gray) if native_peri_gray.size > 0 else (moire_eye, peak_eye)
        moire_score = min(moire_eye, moire_peri)
        fft_peak_ratio = max(peak_eye, peak_peri) if moire_score < 0.6 else peak_peri

        # 2. Color Texture & LBP Surface Analysis
        texture_score, lbp_entropy = self._analyze_color_texture_lbp(eye_bgr)

        # 3. Specular Corneal Reflection vs Flat Display Glare vs Diffuse Print
        specular_score, reflection_type = self._analyze_specular_reflection(eye_bgr)

        # 4. Device Bezel / Screen Border Detection
        bezel_detected, bezel_score = self._detect_device_borders(frame_bgr, eye_box, face_box=face_box)

        # 5. Emissive Dynamic Range & Color Gamut Saturation
        color_gamut_score = self._analyze_color_gamut(eye_bgr)

        # 6. Whole-frame emissive screen region detection
        screen_region_detected, screen_region_score = self._detect_emissive_screen_region(
            frame_bgr, eye_box, face_box=face_box
        )

        # Compute composite genuine-eye PAD score (0.0 = definite spoof, 1.0 = bona fide live eye)
        composite_pad_score = (
            0.30 * moire_score
            + 0.20 * texture_score
            + 0.15 * specular_score
            + 0.10 * (0.0 if bezel_detected else 1.0)
            + 0.10 * color_gamut_score
            + 0.15 * (0.0 if screen_region_detected else 1.0)
        )
        composite_pad_score = float(np.clip(composite_pad_score, 0.0, 1.0))

        # Multi-cue spoof classification:
        is_spoof = False
        spoof_type = None
        reasons = []

        # 1. Screen attack: High harmonic subpixel raster grid spike (> 34x median)
        if moire_score < 0.30:
            is_spoof = True
            spoof_type = "mobile_screen"
            reasons.append("High-frequency Moiré/pixel grid pattern detected characteristic of digital displays.")
        # 2. Rectangular phone/monitor border detected enclosing the eye
        elif bezel_detected and bezel_score >= 0.70:
            is_spoof = True
            spoof_type = "mobile_screen"
            reasons.append("Rectangular screen bezel or device border detected around eye region.")
        # 3. Whole-frame emissive display surface detected (>= 2 independent screen cues)
        elif screen_region_detected and screen_region_score >= 0.65:
            is_spoof = True
            spoof_type = "mobile_screen"
            reasons.append("Emissive digital display surface detected in camera view.")
        # 4. Broad planar display glare or saturated emitter combined with screen cues
        elif reflection_type == "broad_screen_glare" and (bezel_detected or screen_region_score >= 0.35 or color_gamut_score < 0.6):
            is_spoof = True
            spoof_type = "mobile_screen"
            reasons.append("Flat emissive screen glare detected across the eye presentation surface.")
        # 5. Paper print attack: diffuse matte reflection without corneal depth AND depressed micro-texture
        elif reflection_type == "diffuse_matte_print" and texture_score <= 0.60:
            is_spoof = True
            spoof_type = "printed_photo"
            reasons.append("Flat diffuse printed paper texture detected without natural 3D corneal highlight.")
        # 6. Low composite score confirming non-organic presentation
        elif composite_pad_score < 0.40:
            is_spoof = True
            spoof_type = "screen_or_photo"
            reasons.append("Multi-factor surface texture and spectral analysis indicates non-organic presentation.")

        # Cap composite genuine score strictly <= 0.15 when spoof attack is detected
        if is_spoof:
            composite_pad_score = min(composite_pad_score, 0.15)

        pad_pct = round(composite_pad_score * 100.0, 1)
        spoof_risk_pct = round((1.0 - composite_pad_score) * 100.0, 1)

        return {
            "is_spoof": is_spoof,
            "spoof_type": spoof_type,
            "is_screen": spoof_type in ("mobile_screen", "laptop_screen", "computer_monitor") or (moire_score < 0.30) or bezel_detected or screen_region_detected,
            "is_printed": spoof_type == "printed_photo" or (reflection_type == "diffuse_matte_print" and texture_score <= 0.60),
            "lbp_entropy": round(float(lbp_entropy), 3),
            "moire_score": round(float(moire_score), 3),
            "pad_score": round(composite_pad_score, 3),
            "pad_confidence_pct": pad_pct,
            "spoof_risk_pct": spoof_risk_pct,
            "pad_label": "Presentation Attack / Spoof" if is_spoof else "Bona Fide (Clean)",
            "reasons": reasons,
            "sub_metrics": {
                "moire_score": round(float(moire_score), 3),
                "fft_peak_ratio": round(float(fft_peak_ratio), 3),
                "texture_score": round(float(texture_score), 3),
                "lbp_entropy": round(float(lbp_entropy), 3),
                "specular_score": round(float(specular_score), 3),
                "reflection_type": reflection_type,
                "bezel_detected": bool(bezel_detected),
                "bezel_score": round(float(bezel_score), 3),
                "gamut_score": round(float(color_gamut_score), 3),
                "screen_region_detected": bool(screen_region_detected),
                "screen_region_score": round(float(screen_region_score), 3),
            },
        }

    def _analyze_fft_moire(self, gray: np.ndarray) -> Tuple[float, float]:
        """
        Detect periodic Moiré patterns and pixel-grid harmonics in 2D Fourier power spectrum
        strictly at native camera resolution (downsampling large crops to 96x96, never upsampling
        tiny crops which causes false bilinear interpolation spikes).
        """
        if gray.size == 0 or gray.shape[0] < 16 or gray.shape[1] < 16:
            return 0.88, 1.0

        gh, gw = gray.shape[:2]
        if min(gh, gw) >= 96:
            patch = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            # Square native-pixel crop (zero upsampling artifacts!)
            side = min(gh, gw)
            cy0, cx0 = gh // 2, gw // 2
            patch = gray[cy0 - side // 2 : cy0 - side // 2 + side, cx0 - side // 2 : cx0 - side // 2 + side].astype(np.float32)

        ph, pw = patch.shape
        patch = patch - np.mean(patch)
        window = np.outer(np.hanning(ph), np.hanning(pw))
        patch_win = patch * window

        f = np.fft.fft2(patch_win)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        cy, cx = ph // 2, pw // 2
        y_grid, x_grid = np.ogrid[:ph, :pw]
        r = np.hypot(x_grid - cx, y_grid - cy)

        # Native-resolution high-frequency radial band strictly below Nyquist
        r_lo = max(3.5, min(ph, pw) * 0.16)
        r_hi = min(ph, pw) * 0.46
        band_mask = (r >= r_lo) & (r <= r_hi)
        band_vals = magnitude[band_mask]

        if len(band_vals) == 0:
            return 0.88, 1.0

        max_val = float(np.max(band_vals))
        median_val = float(np.median(band_vals))
        spike_ratio = float(max_val / max(1e-4, median_val))

        if spike_ratio > 34.0:
            moire_score = 0.15  # Definite screen pixel grid / periodic Moiré
        elif spike_ratio > 28.5:
            moire_score = 0.42  # Suspicious borderline spike
        elif spike_ratio > 22.0:
            moire_score = 0.68  # Mild compression/eyelash harmonics
        else:
            moire_score = 0.92  # Clean organic eye

        return float(moire_score), float(spike_ratio)

    def _analyze_color_texture_lbp(self, bgr: np.ndarray) -> Tuple[float, float]:
        """
        Analyze multi-channel texture regularity using Local Binary Patterns (LBP).
        Organic skin/eye has rich high-entropy micro-texture; screens and prints exhibit
        flat diffuse loss of high-frequency detail.
        """
        if bgr.size == 0 or bgr.shape[0] < 12 or bgr.shape[1] < 12:
            return 0.70, 2.5

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]

        prob_gray = _compute_fast_lbp(gray)
        prob_cr = _compute_fast_lbp(cr)

        p_nz_gray = prob_gray[prob_gray > 1e-6]
        entropy_gray = -np.sum(p_nz_gray * np.log2(p_nz_gray)) if len(p_nz_gray) > 0 else 0.0

        p_nz_cr = prob_cr[prob_cr > 1e-6]
        entropy_cr = -np.sum(p_nz_cr * np.log2(p_nz_cr)) if len(p_nz_cr) > 0 else 0.0

        if entropy_gray < 1.40 or entropy_cr < 0.50 or (entropy_gray < 2.20 and entropy_cr < 0.95):
            texture_score = 0.30
        elif entropy_gray < 1.70 or entropy_cr < 0.85:
            texture_score = 0.58
        else:
            texture_score = 0.90

        return float(texture_score), float(entropy_gray)

    def _analyze_specular_reflection(self, bgr: np.ndarray) -> Tuple[float, str]:
        """
        Analyze corneal specular reflection vs flat glass screen glare vs diffuse matte paper print.
        """
        if bgr.size == 0 or bgr.shape[0] < 12 or bgr.shape[1] < 12:
            return 0.75, "inconclusive"

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_pixels = h * w

        # Check sharp corneal glint peak vs flat matte surface
        p995 = float(np.percentile(gray, 99.5))
        p80 = float(np.percentile(gray, 80.0))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        spec_thresh = max(235, int(np.percentile(gray, 99.0)))
        spec_mask = (gray >= spec_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(spec_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # Absence of high-intensity highlight combined with low highlight headroom / blurred edge acutance indicates matte print
            if (p995 - p80) < 28.0 and lap_var < 65.0:
                return 0.42, "diffuse_matte_print"
            return 0.78, "ambient_diffuse"

        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        area_pct = (area / max(1, total_pixels)) * 100.0

        perim = cv2.arcLength(largest_cnt, True)
        compactness = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0

        if area_pct <= 6.0 and compactness >= 0.25:
            return 0.92, "natural_corneal_purkinje"
        elif area_pct > 25.0:
            return 0.35, "broad_screen_glare"
        else:
            return 0.78, "ambient_natural"

    def _find_screen_frame_edges(
        self,
        edges: np.ndarray,
        w: int,
        h: int,
        ref_x: int,
        ref_y: int,
        ref_w: int,
        ref_h: int,
        face_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[bool, float]:
        """
        Detect matched, finite-extent parallel vertical or horizontal screen bezel pairs
        framing the eye ROI while ignoring open-ended room wall/door/cabinet lines behind a live user.
        """
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=34, minLineLength=int(min(h, w) * 0.28), maxLineGap=12
        )
        if lines is None:
            return False, 0.0

        left_segs, right_segs = [], []
        top_segs, bottom_segs = [], []
        margin_x = max(4, int(w * 0.015))
        margin_y = max(2, int(h * 0.02))

        for line in lines:
            x1, y1, x2, y2 = [int(v) for v in line[0]]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            angle = math.degrees(math.atan2(dy, max(dx, 1)))
            seg_len = math.hypot(dx, dy)
            avg_x = (x1 + x2) / 2.0
            avg_y = (y1 + y2) / 2.0

            if angle > 72 and seg_len >= int(h * 0.28) and (0.04 * w < avg_x < 0.96 * w):
                y_lo, y_hi = min(y1, y2), max(y1, y2)
                if avg_x < ref_x:
                    left_segs.append((avg_x, y_lo, y_hi))
                elif avg_x > ref_x + ref_w:
                    right_segs.append((avg_x, y_lo, y_hi))
            elif angle < 18 and seg_len >= int(w * 0.30) and (margin_y <= avg_y <= h - margin_y):
                # True device/monitor top-bottom bezels stay bounded inside the frame horizontally,
                # whereas room walls, shelves, and ceiling rails extend to x=0 or x=w edges.
                x_lo, x_max = min(x1, x2), max(x1, x2)
                if x_lo > margin_x and x_max < (w - margin_x):
                    if avg_y < ref_y:
                        top_segs.append((avg_y, x_lo, x_max))
                    elif avg_y > ref_y + ref_h:
                        bottom_segs.append((avg_y, x_lo, x_max))

        # Check paired top+bottom horizontal screen edges with aligned X endpoints
        matched_h = 0
        for ty, tx0, tx1 in top_segs:
            for by, bx0, bx1 in bottom_segs:
                if abs(tx0 - bx0) <= 0.16 * w and abs(tx1 - bx1) <= 0.16 * w and (by - ty) >= 0.18 * h:
                    matched_h += 1

        all_horiz = top_segs + bottom_segs

        # Check paired left+right vertical screen edges forming an enclosed or finite Y-aligned screen frame
        matched_v = 0
        edge_pad_y = max(4, int(h * 0.04))
        for lx, ly0, ly1 in left_segs:
            for rx, ry0, ry1 in right_segs:
                span_w = rx - lx
                overlap_y = max(0, min(ly1, ry1) - max(ly0, ry0))
                min_len = max(1, min(ly1 - ly0, ry1 - ry0))
                if not (0.14 * w <= span_w <= 0.86 * w) or (overlap_y / min_len < 0.50):
                    continue

                # Ignore open-ended vertical room furniture/wall lines behind a real face
                # unless bridged by a matching horizontal bezel bar or tightly enclosed in-frame.
                has_h_bridge = any(
                    abs(hx0 - lx) <= 0.15 * w and abs(hx1 - rx) <= 0.15 * w and (hx1 - hx0) >= 0.55 * span_w
                    for _, hx0, hx1 in all_horiz
                )
                is_finite_aligned_pair = (
                    min(ly0, ry0) >= edge_pad_y
                    and max(ly1, ry1) <= (h - edge_pad_y)
                    and abs(ly0 - ry0) <= 0.16 * h
                    and abs(ly1 - ry1) <= 0.16 * h
                    and span_w <= 0.72 * w
                )
                if face_box is not None and not has_h_bridge:
                    fx, _, fw, _ = face_box
                    if lx < fx - 0.08 * fw and rx > fx + fw + 0.08 * fw and span_w > fw * 1.25:
                        continue

                if has_h_bridge or is_finite_aligned_pair or matched_h >= 1:
                    matched_v += 1

        if matched_h >= 1 and matched_v >= 1:
            return True, min(1.0, 0.88 + 0.04 * min(3, matched_v + matched_h))
        if matched_v >= 1:
            return True, min(1.0, 0.78 + 0.08 * min(3, matched_v))
        if matched_h >= 1:
            return True, min(1.0, 0.78 + 0.08 * min(3, matched_h))
        return False, 0.0

    def _detect_device_borders(
        self,
        frame_bgr: np.ndarray,
        eye_box: Tuple[int, int, int, int],
        face_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[bool, float]:
        """
        Detect rectangular mobile phone, tablet, laptop bezel, or handheld screen borders enclosing the eye.
        Handles both full-frame crops and centered eye ROIs.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ex, ey, ew, eh = eye_box

        # If eye_box covers almost entire image (fallback full-frame check), evaluate borders outside central 50%
        if ew > 0.78 * w or eh > 0.78 * h:
            ref_x, ref_y = int(w * 0.25), int(h * 0.25)
            ref_w, ref_h = int(w * 0.50), int(h * 0.50)
        else:
            ref_x, ref_y, ref_w, ref_h = ex, ey, ew, eh

        # 1. Enclosed rectangular contour (tablet / card / phone / laptop bezel)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 110)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        eye_cx, eye_cy = ex + ew // 2, ey + eh // 2
        frame_area = h * w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (0.08 * frame_area < area < 0.94 * frame_area):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                bx, by, bw, bh = cv2.boundingRect(approx)
                aspect = float(max(bw, bh)) / max(1, min(bw, bh))
                if 1.15 <= aspect <= 2.6:
                    if bx <= eye_cx <= bx + bw and by <= eye_cy <= by + bh:
                        return True, 1.0

        # 2. Matched parallel opposite device edges around the eye
        return self._find_screen_frame_edges(
            edges, w, h, ref_x, ref_y, ref_w, ref_h, face_box=face_box
        )

    def _analyze_color_gamut(self, bgr: np.ndarray) -> float:
        """
        Analyze color distribution. Emissive digital displays have narrow spectrum primaries
        causing saturation clipping compared to natural reflected ambient skin/sclera illumination.
        """
        if bgr.size == 0:
            return 0.70
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        clipped_pixels = np.sum((sat > 190) & (val > 230))
        pct_clipped = (clipped_pixels / max(1, sat.size)) * 100.0

        if pct_clipped > 8.0:
            return 0.30
        elif pct_clipped > 4.0:
            return 0.55
        return 0.85

    def _detect_emissive_screen_region(
        self,
        frame_bgr: np.ndarray,
        eye_box: Tuple[int, int, int, int],
        face_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[bool, float]:
        """
        Detect whether an emissive digital screen (phone, tablet, monitor, laptop) is directly
        displaying the eye without falsely triggering on normal sunlit/bright room walls behind a face.
        Requires >= 2 independent corroborating display cues.
        """
        ex, ey, ew, eh = eye_box
        h, w = frame_bgr.shape[:2]
        score = 0.0
        evidence = 0
        eye_cx, eye_cy = ex + ew // 2, ey + eh // 2
        frame_area = h * w

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # --- Cue 1: Native-resolution FFT Moiré on wide periocular window (~2.4x eye) ---
        ctx_side = max(min(h, w, 128), min(h, w, int(max(ew, eh) * 2.4)))
        cx1 = max(0, min(w - ctx_side, eye_cx - ctx_side // 2))
        cy1 = max(0, min(h - ctx_side, eye_cy - ctx_side // 2))
        ctx_patch = gray[cy1 : cy1 + ctx_side, cx1 : cx1 + ctx_side]
        if ctx_patch.shape[0] >= 24 and ctx_patch.shape[1] >= 24:
            _, wide_spike_ratio = self._analyze_fft_moire(ctx_patch)
            if wide_spike_ratio > 34.0:
                score += 0.45
                evidence += 1

        # --- Cue 2: Emissive backlit screen plateau actually enveloping the EYE itself ---
        _, bright_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        peri_bright_ratio = float(np.mean(ctx_patch > 195)) if ctx_patch.size > 0 else 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (0.06 * frame_area <= area <= 0.78 * frame_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            inside_contour = cv2.pointPolygonTest(cnt, (float(eye_cx), float(eye_cy)), False) >= 0
            if inside_contour or (bx <= eye_cx <= bx + bw and by <= eye_cy <= by + bh and peri_bright_ratio >= 0.25):
                fill_ratio = area / max(1, bw * bh)
                if fill_ratio > 0.55 and peri_bright_ratio >= 0.22:
                    score += 0.40
                    evidence += 1
                    break

        # --- Cue 3: LED/OLED Cool-Blue Backlight Cast inside immediate glowing screen region ---
        if ctx_patch.size > 0:
            ctx_bgr = frame_bgr[cy1 : cy1 + ctx_side, cx1 : cx1 + ctx_side]
            bright_ctx_px = ctx_bgr[ctx_patch > 190]
            if len(bright_ctx_px) >= 0.18 * ctx_patch.size:
                b_minus_r = float(np.mean(bright_ctx_px[:, 0].astype(np.float32) - bright_ctx_px[:, 2].astype(np.float32)))
                if b_minus_r > 8.0:
                    score += 0.30
                    evidence += 1

        # --- Cue 4: Matched finite parallel screen frame edges around eye ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 110)
        has_frame, _ = self._find_screen_frame_edges(
            edges, w, h, ex, ey, ew, eh, face_box=face_box
        )
        if has_frame:
            score += 0.35
            evidence += 1

        is_screen = (evidence >= 2) and (score >= 0.65)
        return is_screen, float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Live Human Eye Verifier (Temporal Dynamics: Blink & Micro-Motion)
# ---------------------------------------------------------------------------
class LiveHumanEyeVerifier:
    """
    Analyzes temporal sequences of consecutive webcam frames to verify:
    1. Real-time eye detection and tracking with strict disappearance reset.
    2. Natural physiological blinking dynamics (closure and reopening excursion).
    3. Ocular micro-motion and micro-saccades (rejects static & rigid photos).
    4. Live challenge-response (Look straight -> Blink / Look left / Look right) to defeat replayed videos.
    5. Explicit state machine enforcement from EYE_DETECTED -> VERIFYING_LIVENESS -> LIVE_EYE_VERIFIED.
    """

    def __init__(
        self,
        min_frames: int = 10,
        max_buffer_size: int = 90,
        min_frames_for_decision: Optional[int] = None,
        enable_challenge: bool = True,
        challenge_timeout_seconds: float = CHALLENGE_TIMEOUT_SECONDS,
    ):
        self.min_frames = min_frames_for_decision if min_frames_for_decision is not None else min_frames
        self.max_buffer_size = max_buffer_size
        self.challenge_timeout_seconds = challenge_timeout_seconds
        self.frame_history: List[Dict] = []
        self.detector = RealTimeEyeDetector()
        self.pad = PresentationAttackDetector()

        # Eye absence tracking
        self._missing_eye_streak = 0

        # Session blink memory across the active challenge window
        self._session_blink_confirmed = False
        self._in_blink_excursion = False

        # Challenge-response state
        self.enable_challenge = enable_challenge
        self._challenge_steps = self._build_challenge_steps()
        self._challenge_index = 0
        self._challenge_start_time: Optional[float] = None
        self._challenge_step_frames: List[Dict] = []
        self._challenge_passed = False
        self._baseline_pupil_x: Optional[float] = None
        self._first_gaze_sign: Optional[int] = None  # remembers direction used in first lateral gaze step

    @staticmethod
    def _build_challenge_steps() -> List[Dict]:
        """Start with baseline fixation ('Look straight'), then randomise blink/left/right order."""
        first = CHALLENGE_STEPS[0]
        rest = list(CHALLENGE_STEPS[1:])
        random.shuffle(rest)
        return [first] + rest

    @property
    def blink_count(self) -> int:
        """Count natural eyelid closure/reopening excursions in buffer."""
        if len(self.frame_history) < 3:
            return 0
        openness = [r["openness"] for r in self.frame_history]
        blinks = 0
        in_blink = False
        for opn in openness:
            if opn < 0.24:
                if not in_blink:
                    in_blink = True
                    blinks += 1
            elif opn > 0.30:
                in_blink = False
        return blinks

    # ------------------------------------------------------------------
    # Challenge-response helpers
    # ------------------------------------------------------------------

    @property
    def _current_challenge(self) -> Optional[Dict]:
        """Return the current challenge step, or None if all steps done."""
        if not self.enable_challenge:
            return None
        if self._challenge_index >= len(self._challenge_steps):
            return None
        return self._challenge_steps[self._challenge_index]

    def _challenge_step_timed_out(self) -> bool:
        """True if the current challenge step has exceeded its time limit."""
        if self._challenge_start_time is None:
            return False
        return (time.monotonic() - self._challenge_start_time) > self.challenge_timeout_seconds

    def _start_challenge_step(self) -> None:
        """Begin timing the current challenge step."""
        self._challenge_start_time = time.monotonic()
        self._challenge_step_frames = []

    def _evaluate_challenge_step(self, record: Dict) -> bool:
        """
        Check whether the current challenge step has been completed.
        Returns True when the step succeeds; caller advances _challenge_index.
        """
        step = self._current_challenge
        if step is None:
            return True

        self._challenge_step_frames.append(record)
        step_type = step["type"]

        if step_type == "fixation":
            stable = [r for r in self._challenge_step_frames if r["openness"] > 0.25]
            if len(stable) >= 5:
                self._baseline_pupil_x = float(np.mean([r["pupil_rel_x"] for r in stable]))
                return True
            return False

        elif step_type == "blink":
            openness = [r["openness"] for r in self._challenge_step_frames]
            in_blink, found = False, False
            for opn in openness:
                if opn < 0.23:
                    in_blink = True
                elif opn > 0.29 and in_blink:
                    found = True
                    break
            return found

        elif step_type in ("gaze_left", "gaze_right"):
            if len(self._challenge_step_frames) < 5:
                return False
            early_box = [r["eye_box"][0] + r["eye_box"][2] / 2.0 for r in self._challenge_step_frames[:3]]
            late_box = [r["eye_box"][0] + r["eye_box"][2] / 2.0 for r in self._challenge_step_frames[-3:]]
            delta_box = float(np.mean(late_box) - np.mean(early_box))

            early_pupil = [r["pupil_rel_x"] for r in self._challenge_step_frames[:3]]
            late_pupil = [r["pupil_rel_x"] for r in self._challenge_step_frames[-3:]]
            delta_pupil = float(np.mean(late_pupil) - np.mean(early_pupil))

            # Accept significant iris-in-socket shift OR eye-box head+gaze turn
            moved = (abs(delta_pupil) >= 0.045) or (abs(delta_box) >= 5.5)
            if not moved:
                return False

            cur_sign = 1 if (delta_pupil if abs(delta_pupil) >= 0.045 else delta_box) > 0 else -1
            if self._first_gaze_sign is None:
                # First lateral direction matches camera parity cleanly; record its sign so the opposite step must go opposite
                self._first_gaze_sign = cur_sign
                return True
            return cur_sign != self._first_gaze_sign

        return False

    def _reset_challenge_only(self) -> None:
        """Reset active challenge steps and session blink flag."""
        self._challenge_steps = self._build_challenge_steps()
        self._challenge_index = 0
        self._challenge_start_time = None
        self._challenge_step_frames = []
        self._challenge_passed = False
        self._baseline_pupil_x = None
        self._first_gaze_sign = None
        self._session_blink_confirmed = False
        self._in_blink_excursion = False

    def add_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Process a single live webcam frame through:
        EYE_DETECTED -> VERIFYING_LIVENESS -> LIVE_EYE_VERIFIED (or PRESENTATION_ATTACK_DETECTED / VERIFICATION_FAILED).
        """
        det = self.detector.detect_eye_region(frame_bgr)
        if not det:
            self._missing_eye_streak += 1
            if len(self.frame_history) > 0:
                self.frame_history.pop(0)
            if self._missing_eye_streak >= 3:
                # Eye disappeared -> reset full verification & challenge state
                self.frame_history.clear()
                self._reset_challenge_only()
            else:
                if self.enable_challenge and self._challenge_start_time is not None:
                    self._challenge_start_time = None
                    self._challenge_step_frames = []
            return {
                "status": "waiting_for_eye",
                "state_machine_state": STATE_WAITING_FOR_EYE,
                "is_live": False,
                "confidence_pct": 0.0,
                "liveness_score_pct": 0.0,
                "guidance_message": MSG_WAITING_FOR_EYE,
                "ui_message": UI_MESSAGES["waiting_for_eye"],
                "eye_detected": False,
                "pad_result": None,
                "temporal_metrics": None,
                "consecutive_frames": 0,
                "challenge": None,
            }

        self._missing_eye_streak = 0

        # Track genuine full open->closed->open blink excursion in session
        opn = float(det["openness_ratio"])
        if opn < 0.24:
            self._in_blink_excursion = True
        elif opn > 0.29 and self._in_blink_excursion:
            self._in_blink_excursion = False
            self._session_blink_confirmed = True

        # Run Presentation Attack Detection on current frame
        pad_res = self.pad.analyze_frame_pad(frame_bgr, det["eye_box"], face_box=det.get("face_box"))

        # During a transient mid-blink closure (< 0.25 openness) immediately following clean open-eye frames,
        # horizontal eyelid crease edges can momentarily spike FFT; suppress FFT-only false trigger during
        # the blink itself while still rejecting immediately on any bezel/emissive display cues or open frames.
        sm_cur = pad_res.get("sub_metrics") or {}
        has_recent_clean_open = any(
            (r["openness"] >= 0.27 and not r["is_spoof_frame"]) for r in self.frame_history[-5:]
        )
        is_transient_blink_crease = (
            opn < 0.25
            and has_recent_clean_open
            and pad_res.get("is_spoof")
            and not sm_cur.get("bezel_detected", False)
            and not sm_cur.get("screen_region_detected", False)
        )
        if is_transient_blink_crease:
            pad_res = dict(pad_res)
            pad_res["is_spoof"] = False
            pad_res["spoof_type"] = None
            pad_res["reasons"] = []
            pad_res["pad_score"] = max(float(pad_res.get("pad_score", 0.0)), 0.78)
            pad_res["pad_confidence_pct"] = round(pad_res["pad_score"] * 100.0, 1)

        # Compute incremental optical flow once per frame
        internal_motion = 0.0
        if len(self.frame_history) > 0:
            g1 = self.frame_history[-1]["eye_gray"]
            g2 = det["eye_crop_gray"]
            if g1.size > 0 and g2.size > 0 and g2.shape[0] >= 14 and g2.shape[1] >= 14:
                if g1.shape != g2.shape:
                    g1_cmp = cv2.resize(g1, (g2.shape[1], g2.shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    g1_cmp = g1
                flow = cv2.calcOpticalFlowFarneback(
                    g1_cmp, g2, None, pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                )
                internal_motion = float(np.mean(np.hypot(flow[..., 0], flow[..., 1])))

        record = {
            "eye_box": det["eye_box"],
            "openness": opn,
            "pupil_rel_x": float(det.get("pupil_rel_x", 0.0)),
            "eye_gray": det["eye_crop_gray"],
            "internal_motion": internal_motion,
            "pad_score": pad_res["pad_score"],
            "is_spoof_frame": pad_res["is_spoof"],
            "spoof_type": pad_res["spoof_type"],
            "reasons": pad_res["reasons"],
        }
        self.frame_history.append(record)
        if len(self.frame_history) > self.max_buffer_size:
            self.frame_history.pop(0)

        # Immediate Presentation Attack Detection Gate
        # Require >= 2 corroborating screen signals for instant frame-1 rejection, or 2 recent spoof frames
        recent_spoofs = sum(1 for r in self.frame_history[-5:] if r["is_spoof_frame"])
        is_immediate_screen = pad_res["is_spoof"] and pad_res.get("spoof_type") in ("mobile_screen", "laptop_screen", "computer_monitor", "printed_photo")
        sm = pad_res.get("sub_metrics") or {}
        is_unambiguous_screen = pad_res["is_spoof"] and (
            (pad_res.get("moire_score", 1.0) <= 0.15 and (sm.get("bezel_detected", False) or sm.get("screen_region_detected", False)))
            or (sm.get("bezel_score", 0.0) >= 0.85 and bool(sm.get("screen_region_detected", False)))
        )
        if is_unambiguous_screen or (len(self.frame_history) >= 3 and recent_spoofs >= 2) or (is_immediate_screen and len(self.frame_history) >= 2):
            self._reset_challenge_only()
            return {
                "status": "spoof_detected",
                "state_machine_state": STATE_PAD_DETECTED,
                "is_live": False,
                "confidence_pct": round(pad_res["pad_confidence_pct"], 1),
                "liveness_score_pct": 0.0,
                "guidance_message": MSG_SPOOF_DETECTED,
                "ui_message": UI_MESSAGES["spoof_detected"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": None,
                "consecutive_frames": len(self.frame_history),
                "challenge": None,
            }

        # Explicit EYE_DETECTED initial state before entering multi-frame VERIFYING_LIVENESS
        if len(self.frame_history) <= 2:
            current_step = self._current_challenge
            challenge_info = {
                "step_id": current_step["id"] if current_step else None,
                "instruction": current_step["instruction"] if current_step else "Look straight at the camera.",
                "step_index": self._challenge_index,
                "total_steps": len(self._challenge_steps),
                "passed": False,
                "timed_out": False,
            } if self.enable_challenge else None
            return {
                "status": "eye_detected",
                "state_machine_state": STATE_EYE_DETECTED,
                "is_live": False,
                "confidence_pct": 0.0,
                "liveness_score_pct": 0.0,
                "guidance_message": MSG_EYE_DETECTED,
                "ui_message": UI_MESSAGES["eye_detected"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": None,
                "consecutive_frames": len(self.frame_history),
                "challenge": challenge_info,
            }

        # Collect minimum consecutive frames before full temporal scoring & challenge evaluation
        if len(self.frame_history) < self.min_frames:
            if self.enable_challenge and self._challenge_start_time is None:
                self._start_challenge_step()
            current_step = self._current_challenge
            challenge_info = {
                "step_id": current_step["id"] if current_step else None,
                "instruction": current_step["instruction"] if current_step else "Follow the eye movement instructions.",
                "step_index": self._challenge_index,
                "total_steps": len(self._challenge_steps),
                "passed": self._challenge_passed,
                "timed_out": False,
            } if self.enable_challenge else None
            return {
                "status": "verifying_liveness",
                "state_machine_state": STATE_VERIFYING_LIVENESS,
                "is_live": False,
                "confidence_pct": 0.0,
                "liveness_score_pct": 0.0,
                "guidance_message": UI_MESSAGES["verifying_liveness"],
                "ui_message": UI_MESSAGES["verifying_liveness"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": None,
                "consecutive_frames": len(self.frame_history),
                "challenge": challenge_info,
            }

        # Run multi-frame temporal dynamics
        temporal_eval = self._evaluate_temporal_dynamics()
        mean_pad_score = float(np.mean([r["pad_score"] for r in self.frame_history[-self.min_frames :]]))
        has_any_spoof_frame = any(r["is_spoof_frame"] for r in self.frame_history[-6:])
        is_temporally_live = bool(temporal_eval["is_live"])
        is_pad_clean = (mean_pad_score >= PAD_GENUINE_THRESHOLD) and (not has_any_spoof_frame) and (not pad_res["is_spoof"])

        liveness_score_pct = round(float(temporal_eval["temporal_score"]) * 100.0, 1)
        if not is_pad_clean or pad_res["is_spoof"]:
            overall_confidence = min(15.0, round(mean_pad_score * 100.0, 1))
        else:
            overall_confidence = float(
                np.clip((0.65 * temporal_eval["temporal_score"] + 0.35 * mean_pad_score) * 100.0, 0.0, 100.0)
            )

        # Challenge-response evaluation
        challenge_info = None
        if self.enable_challenge and not self._challenge_passed:
            if self._challenge_start_time is None:
                self._start_challenge_step()

            # Timeout -> fail & reset verification process (VERIFYING_LIVENESS -> VERIFICATION_FAILED)
            if self._challenge_step_timed_out():
                failed_step = self._current_challenge
                failed_idx = self._challenge_index
                self.frame_history.clear()
                self._reset_challenge_only()
                return {
                    "status": "verification_failed",
                    "state_machine_state": STATE_VERIFICATION_FAILED,
                    "is_live": False,
                    "confidence_pct": 0.0,
                    "liveness_score_pct": 0.0,
                    "guidance_message": MSG_VERIFICATION_FAILED,
                    "ui_message": UI_MESSAGES["verification_failed"],
                    "eye_detected": True,
                    "pad_result": pad_res,
                    "temporal_metrics": temporal_eval,
                    "consecutive_frames": 0,
                    "challenge": {
                        "step_id": failed_step["id"] if failed_step else None,
                        "instruction": failed_step["instruction"] if failed_step else "",
                        "step_index": failed_idx,
                        "total_steps": len(self._challenge_steps),
                        "passed": False,
                        "timed_out": True,
                    },
                }

            step_done = self._evaluate_challenge_step(record)
            if step_done:
                self._challenge_index += 1
                self._challenge_start_time = None
                self._challenge_step_frames = []
                if self._challenge_index >= len(self._challenge_steps):
                    self._challenge_passed = True

            current_step = self._current_challenge
            challenge_info = {
                "step_id": current_step["id"] if current_step else "done",
                "instruction": current_step["instruction"] if current_step else "All challenge steps completed.",
                "step_index": self._challenge_index,
                "total_steps": len(self._challenge_steps),
                "passed": self._challenge_passed,
                "timed_out": False,
            }

            if not self._challenge_passed:
                return {
                    "status": "verifying_liveness",
                    "state_machine_state": STATE_VERIFYING_LIVENESS,
                    "is_live": False,
                    "confidence_pct": round(overall_confidence, 1),
                    "liveness_score_pct": liveness_score_pct,
                    "guidance_message": current_step["instruction"] if current_step else UI_MESSAGES["verifying_liveness"],
                    "ui_message": UI_MESSAGES["verifying_liveness"],
                    "eye_detected": True,
                    "pad_result": pad_res,
                    "temporal_metrics": temporal_eval,
                    "consecutive_frames": len(self.frame_history),
                    "challenge": challenge_info,
                }

        if challenge_info is None:
            challenge_info = {
                "step_id": "done",
                "instruction": "All challenge steps completed.",
                "step_index": len(self._challenge_steps),
                "total_steps": len(self._challenge_steps),
                "passed": self._challenge_passed or not self.enable_challenge,
                "timed_out": False,
            }

        # Strict result authorization synthesis:
        # Must satisfy PAD clean, non-zero liveness, temporal dynamics, confidence threshold, AND challenge
        challenge_ok = self._challenge_passed or (not self.enable_challenge)
        if not is_pad_clean or has_any_spoof_frame:
            self._reset_challenge_only()
            return {
                "status": "spoof_detected",
                "state_machine_state": STATE_PAD_DETECTED,
                "is_live": False,
                "confidence_pct": round(pad_res["pad_confidence_pct"], 1),
                "liveness_score_pct": 0.0,
                "guidance_message": MSG_SPOOF_DETECTED,
                "ui_message": UI_MESSAGES["spoof_detected"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": temporal_eval,
                "consecutive_frames": len(self.frame_history),
                "challenge": challenge_info,
            }
        elif is_temporally_live and challenge_ok and overall_confidence >= LIVENESS_CONFIDENCE_THRESHOLD and liveness_score_pct >= 65.0:
            return {
                "status": "live_verified",
                "state_machine_state": STATE_LIVE_VERIFIED,
                "is_live": True,
                "confidence_pct": round(overall_confidence, 1),
                "liveness_score_pct": liveness_score_pct,
                "guidance_message": MSG_LIVE_VERIFIED,
                "ui_message": UI_MESSAGES["live_verified"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": temporal_eval,
                "consecutive_frames": len(self.frame_history),
                "challenge": challenge_info,
            }
        elif self._challenge_passed and (not is_temporally_live or overall_confidence < LIVENESS_CONFIDENCE_THRESHOLD or liveness_score_pct <= 0.0):
            # All challenge steps elapsed yet temporal physiological dynamics failed threshold -> reset & fail cleanly
            self.frame_history.clear()
            self._reset_challenge_only()
            return {
                "status": "verification_failed",
                "state_machine_state": STATE_VERIFICATION_FAILED,
                "is_live": False,
                "confidence_pct": round(overall_confidence, 1),
                "liveness_score_pct": liveness_score_pct,
                "guidance_message": MSG_VERIFICATION_FAILED,
                "ui_message": UI_MESSAGES["verification_failed"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": temporal_eval,
                "consecutive_frames": 0,
                "challenge": challenge_info,
            }
        else:
            blinks = self.blink_count
            guidance = "Eye detected. Please blink naturally directly in front of the camera." if blinks == 0 else MSG_LIVENESS_FAILED
            return {
                "status": "verifying_liveness",
                "state_machine_state": STATE_VERIFYING_LIVENESS,
                "is_live": False,
                "is_static_spoof": bool(temporal_eval.get("is_static_spoof", False)),
                "confidence_pct": round(overall_confidence, 1),
                "liveness_score_pct": liveness_score_pct,
                "guidance_message": guidance,
                "ui_message": UI_MESSAGES["verifying_liveness"],
                "eye_detected": True,
                "pad_result": pad_res,
                "temporal_metrics": temporal_eval,
                "consecutive_frames": len(self.frame_history),
                "challenge": challenge_info,
            }

    def _evaluate_temporal_dynamics(self) -> Dict:
        """
        Evaluate blink dynamics, ocular micro-saccades, and rigid-motion spoofs.
        """
        window = self.frame_history[-self.max_buffer_size :]
        openness_curve = [r["openness"] for r in window]
        min_ear = float(np.min(openness_curve))
        max_ear = float(np.max(openness_curve))
        ear_range = max_ear - min_ear

        # 1. Natural Physiological Blink Detection (buffer OR active session blink excursion)
        blinks = self.blink_count
        blink_detected = (
            (blinks >= 1 or self._session_blink_confirmed)
            and (min_ear <= 0.24 and max_ear >= 0.28 and ear_range >= 0.06)
        )

        # 2. Cached internal optical flow over consecutive pairs
        motion_vals = [r["internal_motion"] for r in window[1:] if "internal_motion" in r]
        mean_internal_motion = float(np.mean(motion_vals)) if motion_vals else 0.0

        # Static print or frozen photo: near-zero micro-motion without blink
        is_static_spoof = (mean_internal_motion < 0.03) and (not blink_detected)

        # Rigid handheld screen/photo shaking rapidly with zero blinking
        is_rigid_spoof = (mean_internal_motion >= 2.2) and (not blink_detected)

        # Genuine ocular micro-motion & micro-saccades (allowing natural head+gaze turns during interactive challenge)
        saccade_detected = (0.04 <= mean_internal_motion <= 4.5) and not is_rigid_spoof

        # Liveness scoring (0.0 when static spoof / 0 motion)
        if is_static_spoof or is_rigid_spoof:
            temporal_score = 0.0
        else:
            temporal_score = 0.0
            if blink_detected:
                temporal_score += 0.65
            if saccade_detected:
                temporal_score += 0.35

        temporal_score = float(np.clip(temporal_score, 0.0, 1.0))
        is_live = (
            blink_detected
            and saccade_detected
            and (not is_static_spoof)
            and (not is_rigid_spoof)
            and temporal_score >= TEMPORAL_LIVENESS_THRESHOLD
        )

        return {
            "is_live": is_live,
            "blink_detected": blink_detected,
            "saccade_detected": saccade_detected,
            "is_static_spoof": is_static_spoof,
            "is_rigid_spoof": is_rigid_spoof,
            "min_ear": round(min_ear, 3),
            "max_ear": round(max_ear, 3),
            "ear_range": round(ear_range, 3),
            "mean_internal_motion": round(mean_internal_motion, 3),
            "temporal_score": round(temporal_score, 3),
        }

    def reset(self):
        """Reset temporal buffer, detector memory, and challenge-response state."""
        self.frame_history.clear()
        self.detector.reset()
        self._missing_eye_streak = 0
        self._reset_challenge_only()


# ---------------------------------------------------------------------------
# 4. Authoritative Server-Side Liveness Session Manager & Single-Use Tokens
# ---------------------------------------------------------------------------
class LivenessSessionManager:
    """
    Maintains live camera liveness sessions and binds successful verifications to
    short-lived, single-use verification tokens that are independently verified
    before eye image analysis is allowed to execute.
    """

    def __init__(self, session_ttl_seconds: float = 120.0, token_ttl_seconds: float = SESSION_TOKEN_TTL_SECONDS):
        self._sessions: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.session_ttl_seconds = session_ttl_seconds
        self.token_ttl_seconds = token_ttl_seconds

    def _cleanup_expired_locked(self, now: float) -> None:
        expired = [sid for sid, d in self._sessions.items() if (now - d.get("last_active", 0.0)) > self.session_ttl_seconds]
        for sid in expired:
            del self._sessions[sid]

    def process_frame(
        self,
        session_id: str,
        frame_bgr: np.ndarray,
        camera_device_id: Optional[str] = None,
        enable_challenge: bool = True,
    ) -> Dict:
        """Process one live webcam frame under an isolated session and emit telemetry."""
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            existing = self._sessions.get(session_id)
            if existing is None:
                existing = {
                    "verifier": LiveHumanEyeVerifier(enable_challenge=enable_challenge),
                    "last_active": now,
                    "camera_device_id": camera_device_id,
                    "is_verified": False,
                    "verification_token": None,
                    "verified_at": None,
                    "verified_frame_bgr": None,
                    "liveness_score_pct": 0.0,
                    "pad_score": 0.0,
                }
                self._sessions[session_id] = existing
            elif camera_device_id and existing.get("camera_device_id") and existing["camera_device_id"] != camera_device_id:
                # Camera device changed -> strictly reset verification
                existing["verifier"].reset()
                existing["camera_device_id"] = camera_device_id
                existing["is_verified"] = False
                existing["verification_token"] = None
                existing["verified_at"] = None
                existing["verified_frame_bgr"] = None
                existing["liveness_score_pct"] = 0.0
                existing["pad_score"] = 0.0

            existing["last_active"] = now
            verifier: LiveHumanEyeVerifier = existing["verifier"]
            liveness_res = verifier.add_frame(frame_bgr)

            pad_data = liveness_res.get("pad_result") or {}
            temporal = liveness_res.get("temporal_metrics") or {}
            liveness_score_pct = float(liveness_res.get("liveness_score_pct", 0.0))
            confidence_pct = float(liveness_res.get("confidence_pct", 0.0))
            pad_score = float(pad_data.get("pad_score", 0.0)) if pad_data else 0.0
            is_pad_spoof = bool(pad_data.get("is_spoof", False))

            if (
                liveness_res.get("status") == "live_verified"
                and liveness_res.get("is_live")
                and not is_pad_spoof
                and pad_score >= PAD_GENUINE_THRESHOLD
                and liveness_score_pct >= 65.0
                and confidence_pct >= LIVENESS_CONFIDENCE_THRESHOLD
                and temporal.get("blink_detected", False)
            ):
                token = secrets.token_urlsafe(24)
                existing["is_verified"] = True
                existing["verification_token"] = token
                existing["verified_at"] = now
                existing["verified_frame_bgr"] = frame_bgr.copy()
                existing["liveness_score_pct"] = liveness_score_pct
                existing["pad_score"] = pad_score
                liveness_res["verificationToken"] = token
                liveness_res["verificationExpiresInSec"] = self.token_ttl_seconds
            elif liveness_res.get("status") in ("spoof_detected", "verification_failed", "waiting_for_eye"):
                existing["is_verified"] = False
                existing["verification_token"] = None
                existing["verified_at"] = None
                existing["verified_frame_bgr"] = None
                existing["liveness_score_pct"] = 0.0
                if liveness_res.get("status") == "spoof_detected":
                    verifier.frame_history.clear()

            liveness_res["sessionId"] = session_id
            liveness_res["blink_count"] = verifier.blink_count
            liveness_res["pad_analysis"] = pad_data
            liveness_res["confidence"] = liveness_score_pct
            liveness_res["liveness_score_pct"] = liveness_score_pct
            return liveness_res

    def reset_session(self, session_id: Optional[str]) -> None:
        """Delete or reset verification session state and revoke active tokens."""
        if not session_id:
            return
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def validate_and_consume_token(
        self,
        session_id: Optional[str],
        verification_token: Optional[str],
        consume: bool = True,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Independently verify that a live camera session passed liveness, PAD, and challenge
        verification, owns a valid non-expired single-use verification token, and meets all thresholds.
        """
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            if not session_id or session_id not in self._sessions:
                return False, "Missing or invalid live camera verification session.", None

            session_data = self._sessions[session_id]
            stored_token = session_data.get("verification_token")
            verified_at = session_data.get("verified_at")
            is_verified = bool(session_data.get("is_verified", False))
            liveness_score_pct = float(session_data.get("liveness_score_pct", 0.0))
            pad_score = float(session_data.get("pad_score", 0.0))

            if not is_verified or not stored_token or verified_at is None:
                return False, "Live human eye verification has not completed.", None

            if not verification_token or not secrets.compare_digest(str(stored_token), str(verification_token)):
                return False, "Invalid or mismatched verification token.", None

            if (now - float(verified_at)) > self.token_ttl_seconds:
                session_data["is_verified"] = False
                session_data["verification_token"] = None
                return False, "Verification session token has expired. Please re-verify.", None

            if liveness_score_pct <= 0.0 or liveness_score_pct < 65.0:
                return False, "Liveness score below required threshold.", None

            if pad_score < PAD_GENUINE_THRESHOLD:
                return False, "PAD presentation attack check did not pass.", None

            snapshot = {
                "session_id": session_id,
                "liveness_score_pct": liveness_score_pct,
                "pad_score": pad_score,
                "verified_at": verified_at,
                "verified_frame_bgr": session_data.get("verified_frame_bgr"),
            }

            if consume:
                session_data["is_verified"] = False
                session_data["verification_token"] = None
                session_data["verified_at"] = None

            return True, "OK", snapshot


SESSION_MANAGER = LivenessSessionManager()


# ---------------------------------------------------------------------------
# 5. Batch Verification for API Request Payload (Stream or Frame Sequence)
# ---------------------------------------------------------------------------
def verify_webcam_frame_sequence(frame_bytes_list: List[bytes]) -> Dict:
    """
    Validate a sequence of consecutive live webcam frames directly.
    Rejects static single images, phone displays, prints, and replayed videos.
    Returns structured PAD and liveness decision.
    """
    if not frame_bytes_list or len(frame_bytes_list) == 0:
        return {
            "status": "waiting_for_eye",
            "state_machine_state": STATE_WAITING_FOR_EYE,
            "is_live": False,
            "confidence_pct": 0.0,
            "liveness_score_pct": 0.0,
            "guidance_message": MSG_WAITING_FOR_EYE,
            "ui_message": UI_MESSAGES["waiting_for_eye"],
            "error": "No frames provided.",
        }

    # Disable interactive gaze challenge in stateless batch sequence mode
    verifier = LiveHumanEyeVerifier(min_frames=min(8, len(frame_bytes_list)), enable_challenge=False)

    last_result = None
    for b in frame_bytes_list:
        np_arr = np.frombuffer(b, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        last_result = verifier.add_frame(frame_bgr)

    if not last_result:
        return {
            "status": "waiting_for_eye",
            "state_machine_state": STATE_WAITING_FOR_EYE,
            "is_live": False,
            "confidence_pct": 0.0,
            "liveness_score_pct": 0.0,
            "guidance_message": MSG_WAITING_FOR_EYE,
            "ui_message": UI_MESSAGES["waiting_for_eye"],
            "error": "Unable to decode webcam frame sequence.",
        }

    return last_result
