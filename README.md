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
   - **Retinal Fundus Diabetic Retinopathy Grading (CNN):** Deep learning transfer learning (`MobileNetV3-Small`) trained on patient-disjoint clinical fundus photographs across the 5-grade ICDR scale (`0: No DR`, `1: Mild`, `2: Moderate`, `3: Severe`, `4: Proliferative DR`).
   - **Anterior Eye / Iris Computer Vision Pipeline (Experimental):** OpenCV eye detection (`haarcascade_eye.xml`), pupil localization, concentric Hough iris boundary detection, eyelid/glare masking, sharpness/exposure quality rejection, and feature extraction. Supports both **Image Upload** and **Laptop Camera Snapshot Capture** in the UI. **Explicitly non-diagnostic for diabetes/glucose.**
   - **Separate Tabular Clinical Diabetes Screening:** Trained on the **Pima Indians Diabetes Database** (`768` clinical records) using leak-free median imputation and Logistic Regression (`ROC-AUC = 0.8126` on held-out test set).
   - **SQLite Prediction Persistence:** Stores eye image and tabular screening history in `backend/eye_predictions.db`.

---

## Important Medical & Scientific Boundaries

This project strictly separates four distinct medical tasks and never generates fake labels or fabricated blood glucose measurements:

| Task | Supported? | Implementation & Dataset |
| :--- | :--- | :--- |
| **Diabetic Retinopathy Detection (Retinal Fundus)** | **Yes (Research / Screening)** | `MobileNetV3-Small` CNN trained on DeepDRiD / RetinaMNIST 224×224 fundus photographs (grades 0–4) |
| **Anterior Iris Segmentation & Quality CV** | **Yes (Experimental CV Only)** | OpenCV pupil/iris segmentation, eyelid/glare masking, focus & brightness checks. **Not used for diabetes diagnosis.** |
| **Clinical / Tabular Diabetes Risk Prediction** | **Yes (Separate Tabular Model)** | Trained strictly on Pima Indians Diabetes tabular features (Glucose, BMI, Age, BP, Insulin, etc.) |
| **Blood Glucose Measurement (mg/dL) from Eye Image** | **No — Unsupported** | Explicitly blocked; UI & API display: *"Blood glucose cannot be measured from this image using the current model. Please use a validated blood glucose testing device."* |

---

## Datasets Used & Data Sources

1. **Retinal Fundus Dataset (Primary Training & Evaluation)**:
   - **RetinaMNIST (224×224 RGB from DeepDRiD Benchmark, MedMNIST v2, CC BY 4.0)**
   - Public Mirror: `https://huggingface.co/datasets/danjacobellis/retinamnist_224` (Origin: `https://zenodo.org/records/10519652`)
   - **Why used**: Official Kaggle competitions ([APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) and [EyePACS](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)) require personal Kaggle API tokens and competition rule acceptance. DeepDRiD / RetinaMNIST provides real clinical RGB retinal fundus scans under the identical 5-grade ICDR scale (`0–4`) with strict patient-disjoint train/val/test separation.
   - **Prepared Splits (after cross-split perceptual dHash deduplication removing 42 near-duplicates)**:
     - Training: `1,080` images
     - Validation: `113` images
     - Held-out Test: `365` images
   - **Optional Local Kaggle APTOS 2019 Ingestion**: If you download APTOS 2019 locally (`train.csv` + `train_images/`), `src/eye_analysis/dataset_prep.py --kaggle-aptos-dir <path>` supports preprocessing and splitting it directly.
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
│   ├── retinal_dr_model.pth    # Best trained MobileNetV3-Small checkpoint (Epoch 10)
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

# Train Retinal Fundus DR Classification CNN (MobileNetV3-Small, 15 epochs)
python -m src.eye_analysis.train_retinal --arch mobilenet_v3_small --epochs 15 --batch-size 32 --lr 0.0003
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

### Retinal Fundus CNN (`MobileNetV3-Small`, Held-Out Test Split `n = 365`)
- **5-Class Overall Accuracy**: `61.92%`
- **Weighted F1-Score**: `61.54%` | **Macro F1-Score**: `50.07%`
- **One-vs-Rest ROC-AUC (Macro)**: `0.8548` | **Weighted**: `0.8668`
- **Referable DR (`>= Grade 2`) Screening Sensitivity**: **`84.88%`** (`Specificity: 89.12%`, `ROC-AUC: 0.9408`, `False Negatives: 26`)
- **Any DR (`>= Grade 1`) Screening Sensitivity**: **`84.36%`** (`Specificity: 77.27%`, `ROC-AUC: 0.8968`, `False Negatives: 33`)

### Separate Tabular Diabetes Screening (`Pima Indians Diabetes`, 20% Test Split `n = 154`)
- **Selected Model**: `LogisticRegression` (with median zero-imputation + scaling inside Pipeline)
- **Test Accuracy**: `73.38%` | **Recall (Sensitivity)**: `70.37%` | **Specificity**: `75.00%` | **ROC-AUC**: `0.8126`

### Known Limitations
- **Class Imbalance & Rare Grade Sensitivity**: Grade 4 (`Proliferative diabetic retinopathy`, `n=66` train, `n=20` test) has low exact-grade recall (`15.0%`), with `13/20` cases predicted into adjacent `Grade 3 (Severe)` (still flagged as Referable DR). `Grade 1 (Mild)` (`n=39` test) also exhibits boundary overlap with `No DR` (`38.46%` recall).
- **Resolution & External Generalization**: Training at `224×224` on RetinaMNIST/DeepDRiD (`1,080` training scans) cannot match multi-center clinical grading performance trained on 50,000+ high-resolution fundus images.
- **Not a Clinical Device**: All outputs are experimental AI academic screening demonstrations, not clinical diagnoses.

---

## Backend API Endpoints Added

- `GET /api/eye/status` — Returns model training status, evaluation metrics, sample test filenames, and upload limits.
- `POST /api/eye/analyze` — Validates uploaded eye image (`multipart/form-data` or JSON/sample), assesses quality, classifies retinal DR or runs iris CV segmentation, persists record to SQLite, and returns JSON results.
- `GET /api/eye/history` — Retrieves stored eye image & tabular prediction history from SQLite.
- `DELETE /api/eye/history` / `DELETE /api/eye/history/<id>` — Clears or deletes prediction history records.
- `GET /api/eye/samples/<filename>` — Serves held-out test sample images safely.
- `GET /api/eye/plots/<filename>` — Serves generated evaluation charts (`confusion_matrix.png`, `roc_curves.png`, `training_history.png`).
- `POST /api/diabetes/predict` — Runs separate Pima tabular diabetes screening inference and stores history.
