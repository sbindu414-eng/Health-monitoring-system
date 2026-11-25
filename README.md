# Modular ML Framework for Personalized Health Monitoring & AI Eye Screening

[![React](https://img.shields.io/badge/React-19.3-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646cff.svg)](https://vitejs.dev/)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.14-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-ISC-green.svg)](LICENSE)

> A multi-domain machine learning clinical decision support dashboard built with **React**, **Python**, **Scikit-Learn**, **PyTorch**, **OpenCV**, and **SQLite**, integrating nutritional diet planning, symptom pathology diagnosis, 12-lead ECG classification, physical fitness estimation, composite risk stratification, and **AI Eye Image Analysis & Diabetes Screening**.

---

## Modules Overview

1. **Diet Suggestion Module** — Random Forest Classifier & Regressor for BMI, TDEE, and macronutrient planning.
2. **Disease Diagnosis Module** — Multiclass Logistic Regression across 17 clinical symptom indicators.
3. **12-Lead ECG Classification Module** — Soft-Voting Ensemble (`RF + GradientBoosting + LogisticRegression`) with HTML5 Canvas ECG sweep preview.
4. **Fitness Estimation Module** — Interaction Ridge Regression estimating continuous fitness score (0–100) and VO2 max.
5. **Integrated Risk Stratification** — Gradient Boosted Decision Trees synthesizing cross-domain risk scores.
6. **AI Eye Image Analysis & Diabetes Screening** *(New)*:
   - **Retinal Fundus Diabetic Retinopathy Grading (CNN):** Deep learning transfer learning (`EfficientNet-B0` with 4-view TTA & validation-calibrated binary threshold) trained on patient-disjoint clinical fundus photographs across the 5-grade ICDR scale (`0: No DR`, `1: Mild`, `2: Moderate`, `3: Severe`, `4: Proliferative DR`).
   - **Anterior Eye / Iris Computer Vision Pipeline (Experimental):** OpenCV eye detection (`haarcascade_eye.xml`), pupil localization, concentric Hough iris boundary detection, eyelid/glare masking, sharpness/exposure quality rejection, and feature extraction. Supports both **Image Upload** and **Laptop Camera Snapshot Capture** in the UI. **Explicitly non-diagnostic for diabetes/glucose (never routed through the retinal fundus DR CNN).**
   - **Separate Tabular Clinical Diabetes Screening:** Trained on the **Pima Indians Diabetes Database** (`768` clinical records) using leak-free median imputation and Logistic Regression (`ROC-AUC = 0.8126` on held-out test set).
   - **SQLite Prediction Persistence:** Stores eye image and tabular screening history in `backend/eye_predictions.db`.

---

## Important Medical & Scientific Boundaries

This project strictly separates four distinct medical tasks and never generates fake labels or fabricated blood glucose measurements:

| Task | Supported? | Implementation & Dataset |
| :--- | :--- | :--- |
| **Diabetic Retinopathy Detection (Retinal Fundus)** | **Yes (Research / Screening)** | `EfficientNet-B0` CNN trained on DeepDRiD / RetinaMNIST 224×224 fundus photographs (grades 0–4) |
| **Anterior Iris Segmentation & Quality CV** | **Yes (Experimental CV Only)** | OpenCV pupil/iris segmentation, eyelid/glare masking, focus & brightness checks. **Not used for diabetes diagnosis.** |
| **Clinical / Tabular Diabetes Risk Prediction** | **Yes (Separate Tabular Model)** | Trained strictly on Pima Indians Diabetes tabular features (Glucose, BMI, Age, BP, Insulin, etc.) |
| **Blood Glucose Measurement (mg/dL) from Eye Image** | **No — Unsupported** | Explicitly blocked; UI & API display: *"Blood glucose cannot be measured from this image using the current model. Please use a validated blood glucose testing device."* |

---

## Datasets Used & Data Sources

1. **Multi-Center Retinal Fundus Dataset (Training, Validation & Testing)**:
   - **Primary Clinical Cohorts**:
     - **EyePACS** (Kaggle Diabetic Retinopathy Detection, MIT License)
     - **IDRiD** (Indian Diabetic Retinopathy Image Dataset, CC BY 4.0)
     - **DeepDRiD / RetinaMNIST** (MedMNIST v2, CC BY 4.0)
     - **APTOS 2019 Blindness Detection** (Aravind Eye Hospital, India)
     - **Messidor-2** (French Clinical Reference Cohort, ADCIS / Inserm)
   - **Prepared Multi-Center Splits (after cross-split perceptual dHash deduplication removing 188 near-duplicates)**:
     - Multi-Center Training (`train.csv`): `2,800` images
     - Multi-Center Validation (`val.csv`): `785` images
     - Independent Multi-Center Held-Out Test (`test.csv`): `1,424` images
     - External Zero-Shot Cross-Dataset Benchmark (`external_messidor2.csv`): `800` images
   - **Data Caching**: All 12 parquet shards cached in `data/external_cache/` for instant offline ingestion and reproducibility.
2. **Tabular Diabetes Dataset**:
   - **Pima Indians Diabetes Database** (`768` clinical rows, `8` clinical inputs, binary outcome `0: 500`, `1: 268`)
   - Source reference: `https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database` (downloaded cleanly via public UCI/Plotly CSV mirror into `data/pima_diabetes/diabetes.csv`).

---

## Project Structure

```text
major project/
├── backend/
│   ├── database.py             # SQLite persistence layer (eye_predictions & tabular_diabetes_predictions)
│   ├── eye_predictions.db      # Auto-initialized SQLite database
│   ├── server.py               # Python HTTP API server (port 5000) serving ML & Eye endpoints
│   ├── ml_datasets.py          # Synthetic clinical dataset generators for Modules 1–5
│   ├── train_models.py         # Training script for Modules 1–5
│   ├── test_models.py          # Unit tests for Modules 1–5
│   └── trained_models/         # Saved .joblib models for Modules 1–5
├── data/
│   ├── retinal_fundus/         # Prepared RetinaMNIST images + train.csv / val.csv / test.csv
│   ├── pima_diabetes/          # diabetes.csv (768 records)
│   └── samples/                # Unseen held-out test fundus samples (grades 0–4) for UI testing
├── models/
│   ├── retinal_dr_model.pth    # Best trained EfficientNet-B0 checkpoint (Epoch 7)
│   ├── preprocessing_config.json
│   ├── class_labels.json
│   ├── evaluation_report.json  # Held-out test set metrics & per-class confusion matrix
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── training_history.png
│   ├── pima_diabetes_model.joblib
│   └── pima_diabetes_eval.json
├── notebooks/
│   └── eye_analysis_exploration.ipynb
├── src/
│   └── eye_analysis/
│       ├── dataset_prep.py     # Dataset download, integrity verification, dHash dedup, splits
│       ├── preprocessing.py    # Validation, blur/exposure quality gate, modality check, CLAHE + transforms
│       ├── iris_pipeline.py    # OpenCV eye/pupil/iris segmentation, eyelid/glare masking, CV features
│       ├── train_retinal.py    # Transfer learning CNN training script
│       ├── train_tabular.py    # Separate Pima tabular classifier trainer
│       ├── evaluate.py         # Independent held-out test set evaluator & plot generator
│       └── predict.py          # Production prediction engine + CLI predictor
├── tests/
│   └── test_eye_module.py      # Automated validation, iris CV, CNN, tabular, and SQLite tests
├── frontend/                   # React 19 + Vite 8 + Tailwind CSS + Lucide icons
│   └── src/
│       ├── components/
│       │   ├── EyeAnalysisTab.jsx  # "AI Eye Image Analysis" UI, sample loader, plots, Pima risk, history
│       │   ├── OverviewTab.jsx
│       │   ├── DietTab.jsx / DiseaseTab.jsx / EcgTab.jsx / FitnessTab.jsx / RiskTab.jsx
│       │   └── Navbar.jsx / TabNavigation.jsx / Login.jsx
│       └── services/api.js
├── requirements.txt
└── README.md
```

---

## Setup, Dataset Preparation, Training & Running

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
cd frontend; npm install; cd ..
```

### 2. Download & Prepare Datasets (Retinal Fundus + Pima Diabetes)
```powershell
python -m src.eye_analysis.dataset_prep
```
*(Optional Kaggle APTOS 2019 local folder)*:
```powershell
python -m src.eye_analysis.dataset_prep --kaggle-aptos-dir path/to/aptos2019
```

### 3. Train Models
```powershell
# Train Modules 1–5
python backend/train_models.py

# Train Separate Pima Tabular Diabetes Model
python -m src.eye_analysis.train_tabular

# Train Retinal Fundus DR Classification CNN (EfficientNet-B0, 14 epochs)
python -m src.eye_analysis.train_retinal --arch efficientnet_b0 --epochs 14 --batch-size 32 --lr 0.0003
```

### 4. Evaluate on Held-Out Test Split
```powershell
python -m src.eye_analysis.evaluate
```

### 5. Run Verification Tests
```powershell
python -m unittest tests/test_eye_module.py -v
python backend/test_models.py
```

### 6. Run Backend & Frontend
Terminal 1 (Backend on `http://localhost:5000/`):
```powershell
python backend/server.py
```
Terminal 2 (Frontend on `http://localhost:5173/`):
```powershell
cd frontend
npm run dev
```
Or build frontend (`npm run build`) and serve directly from `http://localhost:5000/`.

---

## Real Held-Out Evaluation Results

### Retinal Fundus CNN (`EfficientNet-B0` + 4-View TTA, Multi-Center Held-Out Test Split `n = 1,424`)
- **Backbone Architecture**: `EfficientNet-B0` (selected over `ResNet18` via validation multi-metric comparison: Val QWK `0.7777` vs `0.7397`)
- **Loss Function**: `WeightedFocalLoss(gamma=1.5)` with protective Class 0 weight `1.20` to prevent healthy false positives
- **5-Class Overall Accuracy**: `52.74%`
- **Quadratic Weighted Kappa (QWK)**: `0.6417` (Substantial agreement on ordinal 5-grade scale)
- **Macro F1-Score**: `38.86%` | **Weighted F1-Score**: `44.89%`
- **One-vs-Rest ROC-AUC (Macro)**: `0.8056` | **Weighted**: `0.8014`
- **Any DR (`>= Grade 1`) Screening**: **`Sensitivity: 50.80%`** | **`Specificity: 95.42%`** | **`ROC-AUC: 0.8355`**
- **Referable DR (`>= Grade 2`) Screening**: **`Sensitivity: 59.97%`** | **`Specificity: 96.15%`** | **`ROC-AUC: 0.8948`**

#### Multi-Center Camera Breakdown (Held-Out Test `n=1,424`):
| Dataset Source | Samples | Accuracy | QWK | Any-DR Sensitivity | Any-DR Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **APTOS 2019** (Aravind Eye Hospital, India) | 68 | **73.53%** | **0.6888** | 66.67% | **100.00%** |
| **IDRiD** (Indian Diabetic Retinopathy Image Dataset) | 102 | **63.73%** | **0.7401** | **80.88%** | 88.24% |
| **DeepDRiD / RetinaMNIST** (Multi-Center Clinical) | 365 | **56.71%** | **0.7011** | 53.08% | **99.35%** |
| **EyePACS** (Kaggle DR Primary Screening) | 889 | **48.26%** | **0.6048** | 45.63% | 93.69% |

#### External Zero-Shot Benchmark (Messidor-2, France `n = 800`):
- **Accuracy**: `47.00%` | **QWK**: `0.4972` | **Macro F1**: `35.46%` | **Any-DR Specificity**: `98.39%` (healthy eyes safely identified zero-shot across French clinical population)

### Separate Tabular Diabetes Screening (`Pima Indians Diabetes`, 20% Test Split `n = 154`)
- **Selected Model**: `LogisticRegression` (with median zero-imputation + scaling inside Pipeline)
- **Test Accuracy**: `73.38%` | **Recall (Sensitivity)**: `70.37%` | **Specificity**: `75.00%` | **ROC-AUC**: `0.8126`

### Explainability & Uncertainty Quantification
- **Real PyTorch Grad-CAM Activation**: Hooks into final Conv2d block (`features[-1]`), masks non-retinal FOV black borders (`gray > 10`), produces base64 PNG overlay + peak focus quadrant description.
- **Uncertainty Gating**: Normalized Shannon Entropy ($H(p) / \log_2(5) > 0.85$) and Top-2 Probability Margin ($< 15\%$) flag ambiguous scans with clinical disclaimers.
- **Clinical Reference Standard**: Automated AAO / ETDRS ICDR severity scale summary and recommendations.

### Known Limitations
- **Class Imbalance & Subtle Lesion Detection**: `Grade 1 (Mild DR)` with isolated microaneurysms has lower sensitivity (`1.1%` exact match) as small lesions (< 15 µm) are smoothed at `224×224` resolution. However, Referable DR (`>= Grade 2`) achieves `59.97%` sensitivity and `96.15%` specificity.
- **Not a Standalone Diagnostic Device**: All predictions are intended as academic clinical decision support demonstrations; patients require formal dilated ophthalmic evaluation by an eye care specialist.

---

## Backend API Endpoints Added

- `GET /api/eye/status` — Returns model training status, evaluation metrics, sample test filenames, and upload limits.
- `POST /api/eye/analyze` — Validates uploaded eye image (`multipart/form-data` or JSON/sample), assesses quality, classifies retinal DR or runs iris CV segmentation, persists record to SQLite, and returns JSON results.
- `GET /api/eye/history` — Retrieves stored eye image & tabular prediction history from SQLite.
- `DELETE /api/eye/history` / `DELETE /api/eye/history/<id>` — Clears or deletes prediction history records.
- `GET /api/eye/samples/<filename>` — Serves held-out test sample images safely.
- `GET /api/eye/plots/<filename>` — Serves generated evaluation charts (`confusion_matrix.png`, `roc_curves.png`, `training_history.png`).
- `POST /api/diabetes/predict` — Runs separate Pima tabular diabetes screening inference and stores history.
