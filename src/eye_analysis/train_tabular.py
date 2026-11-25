"""
Separate Tabular Diabetes Risk Screening Model Trainer (Pima Indians Diabetes Dataset).

IMPORTANT SCIENTIFIC SEPARATION:
- Trained strictly on clinical/tabular metabolic measurements (Pima Indians Diabetes
  Database, 768 patient rows), NEVER on eye or iris photographs.
- Handles biologically impossible 0 readings in Glucose, BloodPressure, SkinThickness,
  Insulin, and BMI using leak-free median imputation inside an sklearn Pipeline.
- Evaluates honestly on a stratified held-out 20% test split (never seen during fitting).
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eye_analysis.dataset_prep import PIMA_DIR, ROOT_DIR, prepare_pima_diabetes

MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _clean_biological_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Convert biologically impossible 0 readings in specific clinical fields to NaN."""
    cleaned = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].replace(0, np.nan)
    return cleaned


def train_tabular_diabetes_model() -> dict:
    """Train and evaluate separate tabular diabetes classifier on Pima CSV."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(PIMA_DIR, "diabetes.csv")
    if not os.path.exists(csv_path):
        prepare_pima_diabetes()

    raw_df = pd.read_csv(csv_path)
    X_raw = _clean_biological_zeros(raw_df[FEATURE_COLS])
    y = raw_df["Outcome"].astype(int).values

    # Hold out 20% stratified test split completely unseen during CV and fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=42
    )

    candidates = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)),
        ]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    best_name = None
    best_cv_auc = -1.0

    for name, pipe in candidates.items():
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        mean_auc = float(np.mean(auc_scores))
        cv_results[name] = {
            "cv_roc_auc_mean": round(mean_auc, 4),
            "cv_roc_auc_std": round(float(np.std(auc_scores)), 4),
        }
        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_name = name

    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = {
        "dataset": "Pima Indians Diabetes Database",
        "source": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_model": best_name,
        "feature_names": FEATURE_COLS,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cv_model_comparison": cv_results,
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall_sensitivity": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "specificity": round(float(tn / max(1, tn + fp)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": cm.tolist(),
            "false_negatives": int(fn),
            "false_positives": int(fp),
        },
        "note": (
            "Tabular clinical risk screening ONLY (requires Fasting/OGTT Glucose, BMI, Age, etc.). "
            "Completely separate from eye/iris/retinal image analysis."
        ),
    }

    model_path = os.path.join(MODELS_DIR, "pima_diabetes_model.joblib")
    report_path = os.path.join(MODELS_DIR, "pima_diabetes_eval.json")
    joblib.dump({"pipeline": best_pipe, "metadata": report}, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[Tabular Diabetes] Selected={best_name} | Test Acc={report['test_metrics']['accuracy']:.4f} | ROC-AUC={report['test_metrics']['roc_auc']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train Pima Indians Diabetes Tabular Classifier")
    parser.parse_args()
    train_tabular_diabetes_model()


if __name__ == "__main__":
    main()
