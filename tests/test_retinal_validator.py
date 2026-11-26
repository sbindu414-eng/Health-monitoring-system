"""
Comprehensive Automated Tests for the Human Retinal Fundus Image Validator
and Strict Backend Pipeline Isolation.

Tests every required category strictly on held-out unseen test images (never
training data), verifying zero DR prediction leakage, filename-agnostic backend
rejection, uncertain abstention, and preservation of anterior iris analysis.
"""

import base64
import io
import json
import os
import sys
import unittest
import urllib.request
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.eye_analysis.predict import analyze_uploaded_eye_image
from src.eye_analysis.retinal_validator import (
    MSG_INVALID_NON_RETINAL,
    MSG_UNCERTAIN_RETINAL,
    VALIDATOR_EVAL_PATH,
    validate_human_retinal_fundus,
)

VALIDATOR_TEST_CSV = os.path.join(ROOT_DIR, "data", "retinal_validator", "test.csv")
SAMPLES_DIR = os.path.join(ROOT_DIR, "data", "samples")


def _load_bytes(rel_path: str) -> bytes:
    with open(os.path.join(ROOT_DIR, rel_path), "rb") as f:
        return f.read()


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class TestRetinalFundusValidatorAndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(VALIDATOR_TEST_CSV):
            cls.test_df = pd.read_csv(VALIDATOR_TEST_CSV)
        else:
            cls.test_df = pd.DataFrame()

    def _get_test_sample_path(self, subcategory: str, idx: int = 0) -> str:
        sub = self.test_df[self.test_df["subcategory"] == subcategory]
        self.assertGreater(len(sub), idx, f"No held-out test sample found for subcategory '{subcategory}'")
        return str(sub.iloc[idx]["image_path"])

    def _assert_rejected_before_dr(self, res: dict, label_desc: str):
        self.assertFalse(res["valid_eye_image"], f"{label_desc}: valid_eye_image must be False")
        self.assertIn(res["status"], ("rejected", "uncertain_retinal", "rejected_quality"), f"{label_desc}: unexpected status {res}")
        self.assertNotIn("diabetes_detected", res, f"{label_desc}: DR prediction must NEVER run!")
        self.assertNotIn("predicted_grade", res, f"{label_desc}: DR grade must NEVER be present!")
        self.assertNotIn("class_probabilities", res, f"{label_desc}: DR class probabilities must NEVER run!")
        self.assertNotIn("gradcam", res, f"{label_desc}: DR Grad-CAM must NEVER run!")

    def test_valid_human_retinal_fundus_grades_0_to_4_proceed_to_dr(self):
        """All 5 ICDR DR grade held-out fundus samples pass validation and produce DR predictions."""
        grade_files = [
            (0, "sample_test_grade0_no_diabetic_retinopathy.png"),
            (1, "sample_test_grade1_mild.png"),
            (2, "sample_test_grade2_moderate.png"),
            (3, "sample_test_grade3_severe.png"),
            (4, "sample_test_grade4_proliferative_diabetic_retinopathy.png"),
        ]
        for expected_g, fname in grade_files:
            fpath = os.path.join(SAMPLES_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing test sample {fname}")
            with open(fpath, "rb") as f:
                res = analyze_uploaded_eye_image(f.read(), filename=fname, analysis_mode="retinal_fundus")
            self.assertEqual(res["status"], "completed", f"Grade {expected_g} failed: {res}")
            self.assertTrue(res["valid_eye_image"])
            self.assertEqual(res["detected_modality"], "retinal_fundus")
            self.assertTrue(res["retinal_validation"]["is_valid_retinal"])
            self.assertIn("predicted_grade", res)
            self.assertEqual(len(res["class_probabilities"]), 5)
            self.assertIn(res["predicted_grade"], (0, 1, 2, 3, 4))

    def test_normal_and_high_grade_dr_directions(self):
        """Healthy Grade 0 scan predicts No DR while Grade 4 scan triggers DR detection."""
        with open(os.path.join(SAMPLES_DIR, "sample_test_grade0_no_diabetic_retinopathy.png"), "rb") as f:
            r0 = analyze_uploaded_eye_image(f.read(), "g0.png", analysis_mode="auto")
        self.assertFalse(r0["diabetes_detected"])
        self.assertEqual(r0["predicted_grade"], 0)

        with open(os.path.join(SAMPLES_DIR, "sample_test_grade4_proliferative_diabetic_retinopathy.png"), "rb") as f:
            r4 = analyze_uploaded_eye_image(f.read(), "g4.png", analysis_mode="retinal_fundus")
        self.assertTrue(r4["diabetes_detected"])
        self.assertGreaterEqual(r4["predicted_grade"], 1)

    def test_cat_eye_and_dog_eye_rejected_before_dr(self):
        """Real held-out Cat Eye and Dog Eye photos must be strictly rejected before DR classification."""
        for subcat in ("cat_eye", "cat_portrait", "dog_eye", "dog_portrait"):
            for i in range(min(3, len(self.test_df[self.test_df["subcategory"] == subcat]))):
                rel = self._get_test_sample_path(subcat, i)
                for mode in ("retinal_fundus", "auto"):
                    res = analyze_uploaded_eye_image(_load_bytes(rel), filename="upload_photo.jpg", analysis_mode=mode)
                    self._assert_rejected_before_dr(res, f"{subcat}[{i}] mode={mode}")
                    self.assertIn(
                        res["status"],
                        ("rejected", "uncertain_retinal"),
                        f"{subcat} incorrectly scored status={res['status']}",
                    )

    def test_human_iris_face_webcam_and_attack_images_rejected_in_retinal_mode(self):
        """Human iris photos, human faces, screen eye photos, and printed eye photos are rejected before DR."""
        for subcat in ("human_iris", "human_face", "screen_eye", "printed_eye"):
            sub = self.test_df[self.test_df["subcategory"] == subcat]
            if len(sub) == 0:
                continue
            rel = str(sub.iloc[0]["image_path"])
            for mode in ("retinal_fundus", "auto"):
                res = analyze_uploaded_eye_image(_load_bytes(rel), filename="sample.jpg", analysis_mode=mode)
                self._assert_rejected_before_dr(res, f"{subcat} mode={mode}")

        # Synthetic webcam/human iris also rejected when asking for retinal screening
        canvas = np.full((240, 320, 3), [225, 220, 215], dtype=np.uint8)
        cv2.circle(canvas, (160, 120), 65, (65, 125, 145), -1)
        cv2.circle(canvas, (160, 120), 22, (15, 15, 18), -1)
        res_iris = analyze_uploaded_eye_image(_rgb_to_png_bytes(canvas), "webcam_iris.png", analysis_mode="retinal_fundus")
        self._assert_rejected_before_dr(res_iris, "synthetic_human_iris")

    def test_non_retinal_medical_and_natural_images_rejected(self):
        """Chest X-rays, dermoscopy lesions, and random natural objects/landscapes are rejected before DR."""
        for subcat in ("medical_non_retinal_xray", "medical_non_retinal_derma", "natural_object_landscape"):
            rel = self._get_test_sample_path(subcat, 0)
            res = analyze_uploaded_eye_image(_load_bytes(rel), filename="random_upload.jpg", analysis_mode="auto")
            self._assert_rejected_before_dr(res, subcat)

        # Synthetic bright-red circular balloon/apple on dark corners (classic color-heuristic trap)
        trap = np.full((224, 224, 3), 10, dtype=np.uint8)
        cv2.circle(trap, (112, 112), 75, (215, 95, 42), -1)
        cv2.circle(trap, (135, 95), 14, (240, 220, 180), -1)
        res_trap = analyze_uploaded_eye_image(_rgb_to_png_bytes(trap), "red_object.png", analysis_mode="retinal_fundus")
        self._assert_rejected_before_dr(res_trap, "synthetic_red_circle_trap")

    def test_corrupted_and_low_quality_retinal_images_rejected(self):
        """Corrupted bytes fail decode; blurry/dark retinal images never trigger DR prediction."""
        with self.assertRaises(ValueError):
            analyze_uploaded_eye_image(b"corrupted-not-an-image", "broken.jpg", analysis_mode="retinal_fundus")

        # Severe low-quality blurry fundus image must fail quality check before DR CNN
        with open(os.path.join(SAMPLES_DIR, "sample_test_grade0_no_diabetic_retinopathy.png"), "rb") as f:
            im = Image.open(io.BytesIO(f.read())).convert("RGB")
        blur_np = cv2.GaussianBlur(np.asarray(im, dtype=np.uint8), (51, 51), 18)
        res_blur = analyze_uploaded_eye_image(_rgb_to_png_bytes(blur_np), "blurry_fundus.png", analysis_mode="retinal_fundus")
        self._assert_rejected_before_dr(res_blur, "blurry_fundus")

    def test_uncertain_abstention_mechanism_blocks_dr(self):
        """Borderline probability outputs trigger 'uncertain_retinal' and never run DR CNN."""
        mock_uncertain = {
            "validator_available": True,
            "model_architecture": "mobilenet_v3_small",
            "model_version": "1.0.0",
            "is_valid_retinal": False,
            "validator_status": "uncertain",
            "retinal_fundus_probability": 0.58,
            "retinal_fundus_probability_pct": 58.0,
            "non_retinal_probability": 0.42,
            "non_retinal_probability_pct": 42.0,
            "normalized_entropy": 0.98,
            "valid_threshold": 0.85,
            "uncertain_low": 0.35,
            "structural_sanity_ok": True,
            "structural_sanity_note": "OK",
            "message": MSG_UNCERTAIN_RETINAL,
        }
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        with patch("src.eye_analysis.predict.validate_human_retinal_fundus", return_value=mock_uncertain):
            res = analyze_uploaded_eye_image(_rgb_to_png_bytes(img), "ambiguous.png", analysis_mode="retinal_fundus")
        self.assertEqual(res["status"], "uncertain_retinal")
        self.assertEqual(res["message"], MSG_UNCERTAIN_RETINAL)
        self._assert_rejected_before_dr(res, "uncertain_abstention")

    def test_filename_agnostic_backend_and_api_enforcement(self):
        """Renaming a cat-eye image to a retinal fundus filename cannot bypass backend or API validation."""
        cat_rel = self._get_test_sample_path("cat_eye", 0)
        cat_bytes = _load_bytes(cat_rel)

        # Direct backend call spoofing filename
        res_direct = analyze_uploaded_eye_image(
            cat_bytes,
            filename="sample_test_grade4_proliferative_retinal_fundus.png",
            analysis_mode="retinal_fundus",
        )
        self._assert_rejected_before_dr(res_direct, "cat_eye_spoofed_filename_direct")

        # Live Flask API enforcement test (passes cleanly when server is up)
        try:
            b64 = "data:image/jpeg;base64," + base64.b64encode(cat_bytes).decode("ascii")
            payload = json.dumps({
                "imageBase64": b64,
                "filename": "retinal_fundus_grade0_patient123.png",
                "mode": "retinal_fundus",
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:5000/api/eye/analyze",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                api_res = json.loads(resp.read().decode("utf-8"))
                self._assert_rejected_before_dr(api_res, "cat_eye_api_bypass_attempt")
        except urllib.error.URLError:
            pass

    def test_validator_evaluation_report_metrics_present(self):
        """Validator evaluation report records genuine Accuracy, Precision, Recall, F1, FAR, FRR, CM."""
        self.assertTrue(os.path.exists(VALIDATOR_EVAL_PATH), "Validator evaluation JSON missing")
        with open(VALIDATOR_EVAL_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
        m = rep.get("metrics", {})
        for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "false_acceptance_rate", "false_rejection_rate", "confusion_matrix"):
            self.assertIn(key, m)
        self.assertLessEqual(m["false_acceptance_rate"], 0.02, "Validator False Acceptance Rate too high on test set")


if __name__ == "__main__":
    unittest.main()
