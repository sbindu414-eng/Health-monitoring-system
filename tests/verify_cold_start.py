import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
from src.eye_analysis.predict import load_retinal_bundle, analyze_uploaded_eye_image, get_eye_module_status
from src.eye_analysis.dataset_prep import ROOT_DIR

def verify():
    print("=" * 65)
    print("COLD-START MODEL PERSISTENCE & REPRODUCIBILITY VERIFICATION")
    print("=" * 65)

    # 1. Verify existence of all required models & config files
    required_files = [
        "models/retinal_dr_model.pth",
        "models/class_labels.json",
        "models/preprocessing_config.json",
        "models/evaluation_report.json",
        "models/training_config.json",
        "models/confusion_matrix.png",
        "models/roc_curves.png",
    ]
    for rf in required_files:
        full_p = os.path.join(ROOT_DIR, rf)
        assert os.path.exists(full_p), f"Missing required file: {rf}"
        print(f"  [OK] Found {rf} ({os.path.getsize(full_p):,} bytes)")

    # 2. Check loaded model metadata
    model, ckpt = load_retinal_bundle()
    assert model is not None, "Failed to load model from disk!"
    print("\n--- Model Checkpoint Metadata Loaded From Disk ---")
    print(f"  Architecture:     {ckpt.get('arch')}")
    print(f"  Model Version:    {ckpt.get('model_version')}")
    print(f"  Trained Epoch:    {ckpt.get('best_epoch')}")
    print(f"  Decision Thresh:  {ckpt.get('any_dr_threshold')}")
    print(f"  TTA Enabled:      {ckpt.get('use_tta')}")
    print(f"  CLAHE Preproc:    {ckpt.get('apply_clahe')}")
    print(f"  Input Image Size: {ckpt.get('image_size')}x{ckpt.get('image_size')}")

    # 3. Test reproducibility across 5 clinical test samples
    samples = [
        ("sample_test_grade0_no_diabetic_retinopathy.png", 0, "No diabetic retinopathy"),
        ("sample_test_grade1_mild.png", 1, "Mild"),
        ("sample_test_grade2_moderate.png", 2, "Moderate"),
        ("sample_test_grade3_severe.png", 3, "Severe"),
        ("sample_test_grade4_proliferative_diabetic_retinopathy.png", 4, "Proliferative diabetic retinopathy"),
    ]

    print("\n--- Cold-Start Sample Inferences (2 Passes to Verify Determinism) ---")
    for sname, expected_g, expected_label in samples:
        spath = os.path.join(ROOT_DIR, "data", "samples", sname)
        assert os.path.exists(spath), f"Missing sample file: {spath}"
        with open(spath, "rb") as f:
            b = f.read()

        # Pass 1
        res1 = analyze_uploaded_eye_image(b, filename=sname, analysis_mode="auto")
        # Pass 2
        res2 = analyze_uploaded_eye_image(b, filename=sname, analysis_mode="auto")

        # Check determinism
        assert res1["predicted_grade"] == res2["predicted_grade"], "Inference is not deterministic!"
        assert res1["confidence_pct"] == res2["confidence_pct"], "Confidence differs between runs!"

        p_g = res1["predicted_grade"]
        det = res1["diabetes_detected"]
        conf = res1["confidence_pct"]
        ent = res1["uncertainty"]["normalized_entropy"]
        gcam = bool(res1["gradcam"]["available"])
        print(f"  Scan: {sname[:38]:<38} | Pred Grade: {p_g} ({res1['severity_label']}) | Conf: {conf}% | DR: {det} | Entropy: {ent:.3f} | Grad-CAM: {gcam}")

        # Strict Grade 0 check: healthy eye must not be flagged diabetic
        if expected_g == 0:
            assert not det, "CRITICAL ERROR: Healthy Grade 0 image incorrectly flagged as diabetes_detected!"
            assert p_g == 0, f"Expected Grade 0, got Grade {p_g}"

        # Strict Grade 4 check: proliferative DR must be flagged diabetic
        if expected_g == 4:
            assert det, "CRITICAL ERROR: Proliferative Grade 4 image failed to be flagged as diabetes_detected!"
            assert p_g >= 2, f"Expected referable DR, got Grade {p_g}"

    # 4. Check evaluation metrics consistency
    with open(os.path.join(ROOT_DIR, "models", "evaluation_report.json"), "r") as f:
        rep = json.load(f)
    print("\n--- Preserved Test Evaluation Report (Independent Multi-Center Test Split) ---")
    print(f"  Test Samples:   {rep['test_samples']}")
    print(f"  Exact Accuracy: {rep['overall_metrics']['accuracy'] * 100:.2f}%")
    print(f"  Quadratic QWK:  {rep['overall_metrics']['quadratic_weighted_kappa']:.4f}")
    print(f"  ROC-AUC Macro:  {rep['overall_metrics']['roc_auc_ovr_macro']:.4f}")
    print(f"  Specificity:    {rep['screening_sensitivity_analysis']['any_dr_detection']['specificity'] * 100:.2f}%")
    print(f"  Sensitivity:    {rep['screening_sensitivity_analysis']['any_dr_detection']['sensitivity_recall'] * 100:.2f}%")

    print("\nSUCCESS: All files, weights, and inference pipelines are 100% persistent and deterministic.")

if __name__ == "__main__":
    verify()
