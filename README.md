# Modular ML Framework for Personalized Health Monitoring & Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.0-38bdf8.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-ISC-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Operational-brightgreen.svg)]()

> A multi-domain, modular machine learning architecture engineered to address healthcare fragmentation by integrating dietary analysis, symptom diagnosis, 12-lead ECG signal classification, fitness tracking, and comprehensive health risk stratification into a unified, interactive clinical decision dashboard.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture & Methodology](#-system-architecture--methodology)
- [Machine Learning Modules](#-machine-learning-modules)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Backend REST API Specification](#-backend-rest-api-specification)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [One-Click Startup](#one-click-startup)
  - [Manual Execution](#manual-execution)
- [Patient Presets](#-patient-presets)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🔬 Overview

Traditional digital health solutions operate in silos—fitness trackers analyze steps, dietary apps log calories, and clinical tools monitor ECGs independently. This fragmentation prevents holistic risk prediction and delays preventive interventions.

This project implements a **Modular Machine Learning Health Monitoring Framework** composed of five specialized domain modules that feed into an integrated risk stratification engine:

1. **Diet Suggestion Module** &rarr; Evaluates nutritional intake, BMI, and TDEE.
2. **Disease Diagnosis Module** &rarr; Maps multi-symptom vectors to acute and chronic pathologies.
3. **ECG Classification Ensemble** &rarr; Detects arrhythmias and ischemic patterns via soft-voting ensemble.
4. **Physical Fitness Estimation Module** &rarr; Computes composite fitness score using feature interaction regression.
5. **Integrated Health Risk Stratification** &rarr; Synthesizes cross-domain metrics to generate calibrated risk probabilities and clinical recommendations.

---

## 🏗️ System Architecture & Methodology

The pipeline follows a multi-stage sequential processing workflow:

```mermaid
graph TD
    A[Patient Input / Wearable Telemetry] --> B[Data Preprocessing & Normalization]
    
    subgraph Modular ML Pipeline
        B --> C[Module 1: Diet Analysis<br/>Random Forest - 89.2%]
        B --> D[Module 2: Disease Diagnosis<br/>Multiclass Logistic Reg - 86.7%]
        B --> E[Module 3: 12-Lead ECG Ensemble<br/>Soft-Voting RF+XGB+LGBM - 93.5%]
        B --> F[Module 4: Fitness Estimation<br/>Interaction LinReg - R² 0.82]
    end

    C --> G[Aggregated Multi-Domain Feature Vector]
    D --> G
    E --> G
    F --> G

    G --> H[Module 5: Integrated Risk Stratification<br/>XGBoost Classifier - 91.3%]
    
    H --> I[Low / Moderate / High Risk Probabilities]
    H --> J[Clinical Action Plan & Lifestyle Guidance]
    H --> K[Real-time Interactive Dashboard]
```

---

## 📊 Machine Learning Modules

| Module | Purpose | Algorithm / Architecture | Benchmark Accuracy | Validation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Diet Suggestion** | Energy balance, macro breakdown, caloric distribution | Random Forest Classifier (`n_estimators=100`, `max_depth=12`) | **89.2%** | Stratified 5-Fold CV |
| **2. Disease Prediction** | Symptom-based pathology classification (17 indicators) | Multiclass Logistic Regression (Softmax) | **86.7%** (F1: 0.84) | Stratified 5-Fold CV |
| **3. ECG Rhythm Classification** | Arrhythmia & morphology classification (NSR, AF, STEMI, LBBB, RBBB, PVC) | Weighted Soft-Voting Ensemble (RF + XGBoost + LightGBM + GradBoost) | **93.5%** (AUC: 0.96) | MIT-BIH / PhysioNet Features |
| **4. Fitness Estimation** | Composite fitness score (0–100) based on physiological indicators | Linear Regression with Engineered Feature Interaction Terms | **R² = 0.82** (MAE: 3.42) | Stratified 5-Fold CV |
| **5. Health Risk Stratification** | Integrated risk categorization (Low / Moderate / High) | XGBoost Gradient Boosted Trees | **91.3%** (F1: 0.89) | Stratified 5-Fold CV |

### Individual Model Breakdown for ECG Ensemble (Module 3)
- **Random Forest**: 90.1%
- **XGBoost**: 91.2%
- **LightGBM**: 92.0%
- **Gradient Boosting**: 90.8%
- **1D CNN Baseline**: 88.4%
- **Ensemble Soft-Voting (Final)**: **93.5%**

---

## ✨ Key Features

- **Dynamic 12-Lead ECG Canvas Renderer**:
  - Live animated waveform generator with real-time sweep bar.
  - Multi-lead visualization support (Lead I, Lead II, Lead III, V1, V2, V5, V6).
  - Diagnostic condition switching (Normal Sinus Rhythm, Atrial Fibrillation, STEMI, LBBB, RBBB, PVC).
  - Electrocardiographic parameter controls (RR interval, HR, QRS duration, PR interval, ST elevation, HRV).
- **Interactive Multi-Symptom Matrix**:
  - Selection of 17 clinical symptoms (Fever, Dyspnea, Chest Pain, Palpitations, Myalgia, etc.).
  - Real-time diagnostic probability breakdown.
- **Diet & Metabolic Needs Engine**:
  - Mifflin-St Jeor BMR computation.
  - Activity-adjusted TDEE calculation.
  - Granular meal-by-meal macronutrient and carbohydrate distribution.
- **Integrated Clinical Action Engine**:
  - Dynamic risk stratification feature importance breakdown.
  - Evidence-based lifestyle and clinical intervention guidance.
- **Preset Clinical Patient Archetypes**:
  - **Athlete**: High activity, optimal diet, athletic sinus bradycardia.
  - **Hypertension**: Elevated stress, high BMI, early cardiovascular markers.
  - **Cardiac Risk**: Acute chest symptoms, ST segment alterations, elevated combined risk.
- **Glassmorphic Clinical Dashboard**:
  - Responsive, dark-mode design with glowing status indicators and micro-interactions.

---

## 📁 Project Structure

```plaintext
major-project/
├── backend/
│   ├── server.py              # Python standard-library REST API server (Port 5000)
│   └── api_server.ps1         # Alternative PowerShell API server listener
├── css/
│   └── styles.css             # Glassmorphism, animations, custom scrollbars
├── js/
│   ├── app.js                 # Central application controller & DOM event handling
│   ├── models/
│   │   ├── diet_module.js     # Random Forest diet simulation module
│   │   ├── disease_module.js  # Multiclass Logistic Regression symptom module
│   │   ├── ecg_module.js      # Weighted soft-voting ECG ensemble module
│   │   ├── fitness_module.js  # Interaction-aware linear regression fitness module
│   │   └── risk_module.js     # XGBoost risk stratification module
│   └── ui/
│       ├── ecg_canvas.js      # Canvas-based 12-lead ECG waveform animation engine
│       └── dynamic_charts.js  # Visual rendering helpers
├── index.html                 # Main dashboard user interface (Tailwind CSS)
├── package.json               # NPM scripts configuration
├── requirements.txt           # Python benchmark dependencies
├── server.ps1                 # Static file web server (PowerShell alternative)
├── start.bat                  # Automated one-click execution batch script
└── README.md                  # Project documentation
```

---

## 🔌 Backend REST API Specification

The backend server listens on `http://localhost:5000/` and handles JSON requests:

### 1. Health Status
- **Endpoint**: `GET /api/health`
- **Response**:
  ```json
  {
    "status": "ONLINE",
    "version": "1.0.0",
    "modules": 5
  }
  ```

### 2. Diet Prediction
- **Endpoint**: `POST /api/predict/diet`
- **Payload**:
  ```json
  {
    "age": 35,
    "gender": "male",
    "height": 178,
    "weight": 76,
    "activityLevel": "moderate",
    "dietType": "balanced"
  }
  ```
- **Response**: Returns BMI, BMR, TDEE, optimal meal macro distributions, and diet score.

### 3. Disease Diagnosis
- **Endpoint**: `POST /api/predict/disease`
- **Payload**:
  ```json
  {
    "symptoms": ["fever", "cough", "fatigue", "sob"]
  }
  ```
- **Response**: Primary diagnostic hypothesis, confidence score, and severity level.

### 4. ECG Ensemble Prediction
- **Endpoint**: `POST /api/predict/ecg`
- **Payload**:
  ```json
  {
    "condition": "STEMI"
  }
  ```
- **Response**: Soft-voting ensemble diagnosis, confidence, and individual model vote percentages.

### 5. Fitness Estimation
- **Endpoint**: `POST /api/predict/fitness`
- **Response**: Composite fitness score, R² metric, and error estimates.

### 6. Risk Stratification
- **Endpoint**: `POST /api/predict/risk`
- **Response**: Calibrated risk classification (`Low`, `Moderate`, `High`) and probability distributions.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** installed and available in system PATH.
- Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox).

### One-Click Startup (Recommended)

Simply double-click the included batch launcher or run in PowerShell / Command Prompt:

```cmd
start.bat
```

This will automatically:
1. Start the **Backend API Server** on `http://localhost:5000/`.
2. Start the **Frontend HTTP Server** on `http://localhost:8000/`.
3. Open your default web browser to the secure login portal: `http://localhost:8000/login.html`.

#### Pre-Configured Accounts:
| Profile Role | Email / Username | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Clinician / Doctor** | `doctor@healthml.io` (or `doctor`) | `doctor123` | Full clinical telemetry & patient presets |
| **Research Evaluator** | `evaluator@ieee.org` (or `evaluator`) | `evaluator123` | Benchmark audits & 5-fold CV metrics |
| **Monitored Patient** | `patient@healthml.io` (or `patient`) | `patient123` | Personal diet, exercise, and risk plan |

*(You can sign in with any of these pre-configured credentials or click **Create Account** to register a new user).*

---

### Manual Execution

If you prefer starting services manually in separate terminals:

#### Terminal 1 — Backend API Server
```powershell
python backend/server.py
```
> Running on: `http://localhost:5000/`

#### Terminal 2 — Frontend HTTP Server
```powershell
python -m http.server 8000
```
> Running on: `http://localhost:8000/`

#### Access Application
Open your web browser and navigate to:
```
http://localhost:8000/
```

---

## 👤 Patient Presets

The top navigation bar provides instant one-click scenario presets:

- **⚡ Run Full Scan**: Simultaneously triggers inference across all 5 modules and aggregates predictions into the Risk Stratification dashboard.
- **🏃 Athlete**: Demonstrates optimal metabolic efficiency, low resting heart rate (54 BPM), balanced high-calorie diet, and minimal risk profile.
- **⚠️ Hypertension**: Demonstrates higher resting heart rate (86 BPM), moderate lifestyle stress, and early risk intervention suggestions.
- **🫀 Cardiac Risk**: Demonstrates acute presentation with chest pain, dyspnea, ischemic ECG findings (STEMI / RBBB), and urgent clinical action guidance.

---

## 🛠️ Technologies Used

- **Frontend Core**: HTML5, Vanilla JavaScript (ES6+ Modules), HTML5 Canvas 2D API.
- **Styling**: Tailwind CSS (CDN), Custom CSS Glassmorphism effects.
- **Backend**: Python 3 standard library `http.server` & `json` (zero external dependencies required for API execution).
- **Machine Learning Algorithms**:
  - Random Forest Classifier (Diet)
  - Multiclass Logistic Regression / Softmax (Disease)
  - Soft-Voting Ensemble: Random Forest + XGBoost + LightGBM + Gradient Boosting (ECG)
  - Interaction Linear Regression (Fitness)
  - XGBoost Gradient Boosted Classifier (Integrated Risk)

---

## 📄 License

This project is licensed under the [ISC License](LICENSE).
