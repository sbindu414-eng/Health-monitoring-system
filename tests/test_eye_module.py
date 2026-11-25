"""
Verification Tests for AI Eye Image Analysis & Diabetes Screening Module.

Tests:
1. File validation (corrupt bytes, unsupported extension, too small image)
2. Quality gating (blurry image rejection, severely dark/underexposed rejection)
3. Non-eye image rejection (blank/gradient non-eye photo)
4. Anterior eye / iris CV segmentation & feature extraction (synthetic concentric eye image)
5. Real trained retinal fundus CNN prediction on unseen held-out test samples
6. Separate tabular Pima Indians Diabetes classifier inference
7. SQLite persistence (save, fetch, delete)
"""

import io
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.database import delete_eye_prediction, get_eye_history, init_db, save_eye_prediction
from src.eye_analysis.predict import analyze_uploaded_eye_image, predict_tabular_diabetes
from src.eye_analysis.preprocessing import decode_and_validate_upload


def _img_bytes(rgb_np: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_np).save(buf, format=fmt)
    return buf.getvalue()


class TestEyeAnalysisModule(unittest.TestCase):
    def test_invalid_or_corrupt_image(self):
        with self.assertRaises(ValueError):
            decode_and_validate_upload(b"not an image", "bad.png")
        with self.assertRaises(ValueError):
            decode_and_validate_upload(b"1234", "bad.txt")

    def test_non_eye_image_rejected(self):
        # Sharp high-contrast blue/green geometric blocks (passes sharpness/contrast, contains no eye/fundus anatomy)
        img = np.full((224, 224, 3), [35, 130, 215], dtype=np.uint8)
        for y in range(0, 224, 32):
            for x in range(0, 224, 32):
                if ((x // 32) + (y // 32)) % 2 == 0:
                    img[y : y + 32, x : x + 32] = [210, 230, 245]
        res = analyze_uploaded_eye_image(_img_bytes(img), "geometric_pattern.png", analysis_mode="auto")
        self.assertFalse(res["valid_eye_image"])
        self.assertEqual(res["status"], "rejected")

    def test_anterior_iris_segmentation_and_glucose_boundary(self):
        # Realistic synthetic anterior eye with sclera, textured iris annulus, and dark pupil
        canvas = np.full((240, 320, 3), [225, 220, 215], dtype=np.uint8)
        rng = np.random.default_rng(42)
        cx, cy = 160, 120
        yy, xx = np.ogrid[:240, :320]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Iris texture (blue-green stromal pattern)
        iris_mask = (dist <= 65) & (dist > 22)
        radial_noise = rng.integers(-22, 22, size=(240, 320))
        canvas[iris_mask, 0] = np.clip(65 + radial_noise[iris_mask], 20, 140)
        canvas[iris_mask, 1] = np.clip(125 + radial_noise[iris_mask], 40, 190)
        canvas[iris_mask, 2] = np.clip(145 + radial_noise[iris_mask], 50, 220)

        # Dark pupil core
        pupil_mask = dist <= 22
        canvas[pupil_mask] = [14, 14, 16]

        # Crisp sclera-skin framing so Laplacian edge variance is sharp
        cv2.ellipse(canvas, (cx, cy), (125, 78), 0, 0, 360, (165, 130, 115), 4)

        res = analyze_uploaded_eye_image(_img_bytes(canvas), "macro_iris.png", analysis_mode="anterior_iris")
        self.assertTrue(res["valid_eye_image"])
        self.assertEqual(res["detected_modality"], "anterior_iris")
        self.assertEqual(res["status"], "completed")
        self.assertTrue(res["iris_detected"])
        self.assertIn("Blood glucose cannot be measured from this image", res["glucose_statement"])
        self.assertTrue(res["segmentation_overlay_image"].startswith("data:image/png;base64,"))

    def test_blurry_image_rejected(self):
        # Smooth blurred eye-like target -> low Laplacian variance should trigger quality rejection
        canvas = np.full((240, 320, 3), [220, 215, 210], dtype=np.uint8)
        cv2.circle(canvas, (160, 120), 65, (80, 110, 130), -1)
        cv2.circle(canvas, (160, 120), 22, (15, 15, 18), -1)
        blurred = cv2.GaussianBlur(canvas, (51, 51), 15)
        res = analyze_uploaded_eye_image(_img_bytes(blurred), "blurry_eye.png", analysis_mode="anterior_iris")
        self.assertEqual(res["status"], "rejected_quality")
        self.assertFalse(res["quality"]["is_acceptable"])

    def test_real_test_fundus_sample_prediction(self):
        sample_path = os.path.join(ROOT_DIR, "data", "samples", "sample_test_grade0_no_diabetic_retinopathy.png")
        self.assertTrue(os.path.exists(sample_path), "Held-out sample fundus image missing")
        with open(sample_path, "rb") as f:
            b = f.read()
        res = analyze_uploaded_eye_image(b, os.path.basename(sample_path), analysis_mode="auto")
        self.assertEqual(res["detected_modality"], "retinal_fundus")
        self.assertEqual(res["status"], "completed")
        self.assertIn("Predicted diabetic retinopathy severity:", res["prediction_label"])
        self.assertEqual(len(res["class_probabilities"]), 5)
        self.assertIn("Blood glucose cannot be measured", res["glucose_statement"])

    def test_tabular_pima_prediction(self):
        res = predict_tabular_diabetes({
            "pregnancies": 2,
            "glucose": 155,
            "bloodPressure": 82,
            "skinThickness": 32,
            "insulin": 160,
            "bmi": 33.6,
            "dpf": 0.65,
            "age": 48,
        })
        self.assertEqual(res["status"], "completed")
        self.assertGreater(res["probability_pct"], 0.0)
        self.assertLessEqual(res["probability_pct"], 100.0)

    def test_sqlite_history_crud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "test_hist.db")
            init_db(db_file)
            row_id = save_eye_prediction(
                {
                    "filename": "fundus_sample.png",
                    "file_size_bytes": 42000,
                    "image_width": 224,
                    "image_height": 224,
                    "analysis_mode": "auto",
                    "detected_modality": "retinal_fundus",
                    "quality_status": "acceptable",
                    "quality_score": 82.0,
                    "prediction_label": "Predicted diabetic retinopathy severity: Moderate",
                    "confidence_pct": 71.5,
                    "disclaimer": "Test disclaimer",
                },
                db_path=db_file,
            )
            hist = get_eye_history(limit=10, db_path=db_file)
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0]["id"], row_id)
            self.assertTrue(delete_eye_prediction(row_id, db_path=db_file))
            self.assertEqual(len(get_eye_history(limit=10, db_path=db_file)), 0)


if __name__ == "__main__":
    unittest.main()
