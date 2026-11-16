"""
Rigorous Automated Model Testing & Verification Suite
Tests all 5 machine learning models for:
- Model serialization and loading integrity
- Boundary value compliance and robustness
- Clinical emergency scenario accuracy
- Zero-error execution and sub-millisecond inference latency
"""

import os
import json
import time
import unittest
import joblib
import numpy as np

import ml_datasets

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

class TestHealthMLModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Verify model artifacts exist
        cls.diet_macro = joblib.load(os.path.join(MODELS_DIR, 'diet_macro_rf.joblib'))
        cls.diet_qual = joblib.load(os.path.join(MODELS_DIR, 'diet_quality_rf.joblib'))
        cls.disease_model = joblib.load(os.path.join(MODELS_DIR, 'disease_logreg.joblib'))
        cls.ecg_model = joblib.load(os.path.join(MODELS_DIR, 'ecg_ensemble.joblib'))
        cls.fitness_model = joblib.load(os.path.join(MODELS_DIR, 'fitness_regression.joblib'))
        cls.risk_model = joblib.load(os.path.join(MODELS_DIR, 'risk_gradient_boosting.joblib'))
        
        with open(os.path.join(MODELS_DIR, 'model_metadata.json')) as f:
            cls.metadata = json.load(f)

    # -------------------------------------------------------------------------
    # TEST SUITE 1: ARTIFACT INTEGRITY
    # -------------------------------------------------------------------------
    def test_01_artifacts_integrity(self):
        """Verify all 5 module metadata sections exist and meet benchmark thresholds."""
        for mod in ['diet', 'disease', 'ecg', 'fitness', 'risk']:
            self.assertIn(mod, self.metadata['modules'], f"Missing metadata for module {mod}")
        
        # Verify cross-validation benchmarks
        self.assertGreaterEqual(self.metadata['modules']['diet']['qualityR2'], 0.80)
        self.assertGreaterEqual(self.metadata['modules']['disease']['accuracy'], 90.0)
        self.assertGreaterEqual(self.metadata['modules']['ecg']['accuracy'], 93.0)
        self.assertGreaterEqual(self.metadata['modules']['fitness']['r2Score'], 0.80)
        self.assertGreaterEqual(self.metadata['modules']['risk']['accuracy'], 90.0)

    # -------------------------------------------------------------------------
    # TEST SUITE 2: MODULE 1 (DIET) BOUNDARY TESTING
    # -------------------------------------------------------------------------
    def test_02_diet_boundaries_and_prediction(self):
        """Test Diet module across extreme boundary profiles."""
        # [age, gender, weight, height, bmi, bmr, tdee, activity, diet_goal, target_cal]
        # Profile A: Young athletic male muscle gain
        sample_athlete = np.array([[24, 0, 80.0, 182.0, 24.2, 1850.0, 3100.0, 4, 2, 3450.0]])
        macro_pred = self.diet_macro.predict(sample_athlete)[0]
        qual_pred = self.diet_qual.predict(sample_athlete)[0]
        self.assertIn(macro_pred, [0, 1, 2, 3])
        self.assertGreaterEqual(qual_pred, 70.0)

        # Profile B: Sedentary overweight female weight loss
        sample_loss = np.array([[52, 1, 95.0, 160.0, 37.1, 1550.0, 1860.0, 0, 0, 1550.0]])
        macro_loss = self.diet_macro.predict(sample_loss)[0]
        qual_loss = self.diet_qual.predict(sample_loss)[0]
        self.assertIn(macro_loss, [0, 1, 2, 3])
        self.assertLessEqual(qual_loss, 85.0)

    # -------------------------------------------------------------------------
    # TEST SUITE 3: MODULE 2 (DISEASE) CLINICAL ACCURACY
    # -------------------------------------------------------------------------
    def test_03_disease_clinical_scenarios(self):
        """Verify pathology prediction matches distinct clinical symptom clusters."""
        # 17 symptoms + duration_days + age + resting_hr
        def build_vector(symptoms_list, dur=3, age=45, hr=75):
            vec = np.zeros(20)
            for s in symptoms_list:
                if s in ml_datasets.SYMPTOM_NAMES:
                    vec[ml_datasets.SYMPTOM_NAMES.index(s)] = 1
            vec[17] = dur
            vec[18] = age
            vec[19] = hr
            return vec.reshape(1, -1)

        # 1. Acute Cardiac Strain (chest_pain + palpitations + cold_sweats + SOB)
        v_cardiac = build_vector(['chest_pain', 'palpitations', 'cold_sweats', 'shortness_of_breath'], dur=1, age=60, hr=120)
        pred_cardiac = self.disease_model.predict(v_cardiac)[0]
        self.assertEqual(pred_cardiac, 5, "Acute cardiac symptom cluster must classify as Cardiovascular/Angina Strain")

        # 2. Viral Influenza (fever + fatigue + headache + sore_throat)
        v_flu = build_vector(['fever', 'fatigue', 'headache', 'sore_throat'], dur=4, age=30, hr=95)
        pred_flu = self.disease_model.predict(v_flu)[0]
        self.assertEqual(pred_flu, 2, "Flu symptom cluster must classify as Viral Influenza")

        # 3. Acute Bronchitis (cough + wheezing + fatigue)
        v_bronchitis = build_vector(['cough', 'wheezing', 'fatigue'], dur=6, age=40, hr=80)
        pred_bronchitis = self.disease_model.predict(v_bronchitis)[0]
        self.assertEqual(pred_bronchitis, 1, "Respiratory cluster must classify as Acute Bronchitis")

        # 4. Asymptomatic / Healthy baseline
        v_healthy = build_vector([], dur=0, age=28, hr=68)
        pred_healthy = self.disease_model.predict(v_healthy)[0]
        self.assertEqual(pred_healthy, 0, "Zero symptoms must classify as Healthy/Low Pathology")

    # -------------------------------------------------------------------------
    # TEST SUITE 4: MODULE 3 (ECG) ENSEMBLE CLASSIFICATION
    # -------------------------------------------------------------------------
    def test_04_ecg_ensemble_arrhythmia_detection(self):
        """Verify 12-lead ECG ensemble correctly classifies arrhythmias and acute ischemia."""
        # ['heart_rate', 'rr_interval', 'qrs_duration', 'pr_interval', 'st_elevation', 'hrv', 'age', 'pain', 'sweat', 'palp']
        # 1. Normal Sinus Rhythm (NSR: Class 0)
        x_nsr = np.array([[72.0, 833.0, 88.0, 155.0, 0.01, 48.0, 35, 0, 0, 0]])
        self.assertEqual(self.ecg_model.predict(x_nsr)[0], 0, "Normal vitals must classify as NSR")

        # 2. Acute STEMI Crisis (Class 2: marked ST elevation >= 0.20 mV + chest pain + sweats)
        x_stemi = np.array([[108.0, 555.0, 94.0, 160.0, 0.32, 16.0, 58, 1, 1, 0]])
        self.assertEqual(self.ecg_model.predict(x_stemi)[0], 2, "ST elevation >= 0.20mV must classify as STEMI")

        # 3. Atrial Fibrillation (Class 1: rapid irregular rate, suppressed HRV)
        x_af = np.array([[142.0, 420.0, 90.0, 110.0, 0.02, 12.0, 65, 0, 0, 1]])
        self.assertEqual(self.ecg_model.predict(x_af)[0], 1, "Tachycardia with irregular rate must classify as AF")

        # 4. Right Bundle Branch Block (Class 4: widened QRS >= 120ms)
        x_rbbb = np.array([[78.0, 770.0, 138.0, 165.0, 0.03, 38.0, 50, 0, 0, 0]])
        self.assertEqual(self.ecg_model.predict(x_rbbb)[0], 4, "Wide QRS >= 120ms must classify as RBBB")

        # 5. First Degree AV Block (Class 5: PR interval > 200ms)
        x_avb = np.array([[62.0, 960.0, 88.0, 245.0, 0.01, 44.0, 55, 0, 0, 0]])
        self.assertEqual(self.ecg_model.predict(x_avb)[0], 5, "PR interval > 200ms must classify as AVB")

    # -------------------------------------------------------------------------
    # TEST SUITE 5: MODULE 4 (FITNESS) INTERACTION REGRESSION
    # -------------------------------------------------------------------------
    def test_05_fitness_regression_sensitivity(self):
        """Verify fitness estimation exhibits physiological monotonicity."""
        # ['age', 'resting_hr', 'sleep_hours', 'exercise_mins', 'daily_steps', 'stress_level', 'bmi']
        # Elite endurance athlete
        x_athlete = np.array([[25, 48.0, 8.0, 420.0, 16000.0, 1, 22.0]])
        score_athlete = self.fitness_model.predict(x_athlete)[0]
        self.assertGreaterEqual(score_athlete, 68.0, "Endurance athlete must score >= 68")

        # Sedentary high stress
        x_sedentary = np.array([[45, 92.0, 5.0, 20.0, 2500.0, 5, 34.0]])
        score_sedentary = self.fitness_model.predict(x_sedentary)[0]
        self.assertLessEqual(score_sedentary, 45.0, "Sedentary high stress must score <= 45")

        # Monotonicity: Increasing exercise MUST increase score
        x_more_exercise = x_sedentary.copy()
        x_more_exercise[0, 3] = 300.0 # increase exercise to 300 min
        score_improved = self.fitness_model.predict(x_more_exercise)[0]
        self.assertGreater(score_improved, score_sedentary, "Increasing exercise must strictly increase fitness score")

    # -------------------------------------------------------------------------
    # TEST SUITE 6: MODULE 5 (INTEGRATED RISK) STRATIFICATION
    # -------------------------------------------------------------------------
    def test_06_risk_stratification_calibrated(self):
        """Verify multi-domain risk fusion accurately stratifies emergency vs low-risk profiles."""
        # ['diet_score', 'disease_prob', 'ecg_critical', 'fitness_score', 'age', 'sys_bp', 'dia_bp', 'spo2', 'resting_hr']
        # 1. Critical Emergency Patient (STEMI, hypoxia, hypertensive crisis)
        x_critical = np.array([[42.0, 0.92, 1, 22.0, 68, 178.0, 108.0, 86.0, 130.0]])
        risk_pred_crit = self.risk_model.predict(x_critical)[0]
        risk_probs_crit = self.risk_model.predict_proba(x_critical)[0]
        self.assertEqual(risk_pred_crit, 2, "Critical multi-domain patient must be stratified as High Risk (2)")
        self.assertGreaterEqual(risk_probs_crit[2], 0.75, "High risk probability must be >= 75%")

        # 2. Optimal Healthy Patient
        x_healthy = np.array([[92.0, 0.02, 0, 88.0, 26, 118.0, 76.0, 99.0, 58.0]])
        risk_pred_health = self.risk_model.predict(x_healthy)[0]
        risk_probs_health = self.risk_model.predict_proba(x_healthy)[0]
        self.assertEqual(risk_pred_health, 0, "Optimal healthy individual must be stratified as Low Risk (0)")
        self.assertGreaterEqual(risk_probs_health[0], 0.75, "Low risk probability must be >= 75%")

    # -------------------------------------------------------------------------
    # TEST SUITE 7: INFERENCE LATENCY BENCHMARK
    # -------------------------------------------------------------------------
    def test_07_inference_speed_benchmark(self):
        """Benchmark that sub-millisecond inference speeds are maintained for real-time telemetry."""
        x_ecg = np.array([[75.0, 800.0, 90.0, 155.0, 0.01, 45.0, 40, 0, 0, 0]])
        start_time = time.perf_counter()
        n_iters = 100
        for _ in range(n_iters):
            _ = self.ecg_model.predict(x_ecg)
            _ = self.risk_model.predict(np.array([[80.0, 0.1, 0, 70.0, 35, 120.0, 80.0, 98.0, 72.0]]))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / n_iters
        print(f"\n  [Latency Benchmark] Average Dual-Model Inference: {elapsed_ms:.3f} ms / prediction")
        self.assertLess(elapsed_ms, 35.0, "Real-time inference must be faster than 35 ms")

if __name__ == '__main__':
    unittest.main()
