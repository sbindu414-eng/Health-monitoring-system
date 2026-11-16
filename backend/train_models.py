"""
Master Machine Learning Training & Cross-Validation Pipeline
Trains, validates, and serializes all 5 specialized clinical ML models using
5-Fold Stratified Cross-Validation, benchmarking Accuracy, F1-Score, and R^2.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, classification_report

import ml_datasets

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_and_evaluate_all():
    print("=" * 75)
    print("  MODULAR HEALTH ML PIPELINE: 5-MODULE MODEL TRAINING & VALIDATION")
    print("=" * 75)
    
    metadata = {
        "trainedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modules": {},
        "benchmarkSummary": {}
    }

    # -------------------------------------------------------------------------
    # 1. DIET SUGGESTION MODULE (Random Forest Classifier + Regressor)
    # -------------------------------------------------------------------------
    print("\n[1/5] Training Module 1: Diet Suggestion Engine (Random Forest)...")
    X_diet, y_diet_qual, y_diet_macro, feat_diet = ml_datasets.generate_diet_dataset()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_diet_macro = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    scores_macro = cross_val_score(rf_diet_macro, X_diet, y_diet_macro, cv=skf, scoring='accuracy')
    
    rf_diet_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    scores_qual = cross_val_score(rf_diet_reg, X_diet, y_diet_qual, cv=5, scoring='r2')
    
    rf_diet_macro.fit(X_diet, y_diet_macro)
    rf_diet_reg.fit(X_diet, y_diet_qual)
    
    joblib.dump(rf_diet_macro, os.path.join(OUTPUT_DIR, 'diet_macro_rf.joblib'))
    joblib.dump(rf_diet_reg, os.path.join(OUTPUT_DIR, 'diet_quality_rf.joblib'))
    
    diet_acc = float(np.mean(scores_macro)) * 100.0
    diet_r2 = float(np.mean(scores_qual))
    print(f"  ✓ Macro Classification 5-Fold Accuracy: {diet_acc:.2f}% (std: {np.std(scores_macro)*100:.2f}%)")
    print(f"  ✓ Quality Regression 5-Fold R² Score:   {diet_r2:.4f}")
    
    metadata["modules"]["diet"] = {
        "algorithm": "Random Forest (Classifier + Regressor)",
        "features": feat_diet,
        "macroAccuracy": round(diet_acc, 2),
        "qualityR2": round(diet_r2, 4)
    }

    # -------------------------------------------------------------------------
    # 2. DISEASE DIAGNOSIS MODULE (Multiclass Logistic Regression)
    # -------------------------------------------------------------------------
    print("\n[2/5] Training Module 2: Disease Diagnosis Engine (Multiclass Logistic Regression)...")
    X_dis, y_dis, feat_dis = ml_datasets.generate_disease_dataset()
    
    pipe_dis = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.2, random_state=42))
    ])
    
    scores_dis_acc = cross_val_score(pipe_dis, X_dis, y_dis, cv=skf, scoring='accuracy')
    scores_dis_f1 = cross_val_score(pipe_dis, X_dis, y_dis, cv=skf, scoring='f1_macro')
    
    pipe_dis.fit(X_dis, y_dis)
    joblib.dump(pipe_dis, os.path.join(OUTPUT_DIR, 'disease_logreg.joblib'))
    
    dis_acc = float(np.mean(scores_dis_acc)) * 100.0
    dis_f1 = float(np.mean(scores_dis_f1))
    print(f"  ✓ Pathology Diagnosis 5-Fold Accuracy: {dis_acc:.2f}%")
    print(f"  ✓ Macro F1-Score:                      {dis_f1:.4f}")
    
    metadata["modules"]["disease"] = {
        "algorithm": "Multiclass Logistic Regression (Softmax)",
        "features": feat_dis,
        "classes": ml_datasets.DISEASE_CLASSES,
        "accuracy": round(dis_acc, 2),
        "f1Macro": round(dis_f1, 4)
    }

    # -------------------------------------------------------------------------
    # 3. 12-LEAD ECG CLASSIFICATION MODULE (Soft-Voting Ensemble)
    # -------------------------------------------------------------------------
    print("\n[3/5] Training Module 3: 12-Lead ECG Ensemble (RF + GradientBoosting + LogReg)...")
    X_ecg, y_ecg, feat_ecg = ml_datasets.generate_ecg_dataset()
    
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    clf_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    
    ecg_ensemble = VotingClassifier(
        estimators=[('rf', clf_rf), ('gb', clf_gb), ('lr', clf_lr)],
        voting='soft',
        weights=[2.0, 2.0, 1.0]
    )
    
    pipe_ecg = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', ecg_ensemble)
    ])
    
    scores_ecg = cross_val_score(pipe_ecg, X_ecg, y_ecg, cv=skf, scoring='accuracy')
    pipe_ecg.fit(X_ecg, y_ecg)
    joblib.dump(pipe_ecg, os.path.join(OUTPUT_DIR, 'ecg_ensemble.joblib'))
    
    ecg_acc = float(np.mean(scores_ecg)) * 100.0
    print(f"  ✓ ECG Arrhythmia Ensemble 5-Fold Accuracy: {ecg_acc:.2f}% (std: {np.std(scores_ecg)*100:.2f}%)")
    
    metadata["modules"]["ecg"] = {
        "algorithm": "Weighted Soft-Voting Ensemble (RF + GradBoost + LogReg)",
        "features": feat_ecg,
        "classes": ml_datasets.ECG_CLASSES,
        "accuracy": round(ecg_acc, 2)
    }

    # -------------------------------------------------------------------------
    # 4. PHYSICAL FITNESS ESTIMATION MODULE (Feature Interaction Regression)
    # -------------------------------------------------------------------------
    print("\n[4/5] Training Module 4: Physical Fitness Engine (Interaction Ridge Regression)...")
    X_fit, y_fit, feat_fit = ml_datasets.generate_fitness_dataset()
    
    pipe_fit = Pipeline([
        ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))
    ])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_fit_r2 = cross_val_score(pipe_fit, X_fit, y_fit, cv=kf, scoring='r2')
    scores_fit_mae = cross_val_score(pipe_fit, X_fit, y_fit, cv=kf, scoring='neg_mean_absolute_error')
    
    pipe_fit.fit(X_fit, y_fit)
    joblib.dump(pipe_fit, os.path.join(OUTPUT_DIR, 'fitness_regression.joblib'))
    
    fit_r2 = float(np.mean(scores_fit_r2))
    fit_mae = float(-np.mean(scores_fit_mae))
    print(f"  ✓ Continuous Fitness R² Score:         {fit_r2:.4f}")
    print(f"  ✓ Mean Absolute Error (MAE):           {fit_mae:.2f} pts")
    
    metadata["modules"]["fitness"] = {
        "algorithm": "Ridge Regression with Feature Interactions",
        "features": feat_fit,
        "r2Score": round(fit_r2, 4),
        "mae": round(fit_mae, 2)
    }

    # -------------------------------------------------------------------------
    # 5. INTEGRATED HEALTH RISK STRATIFICATION MODULE (Gradient Boosted Trees)
    # -------------------------------------------------------------------------
    print("\n[5/5] Training Module 5: Integrated Risk Engine (Gradient Boosted Trees)...")
    X_risk, y_risk, feat_risk = ml_datasets.generate_risk_dataset()
    
    gb_risk = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42))
    ])
    
    scores_risk_acc = cross_val_score(gb_risk, X_risk, y_risk, cv=skf, scoring='accuracy')
    scores_risk_f1 = cross_val_score(gb_risk, X_risk, y_risk, cv=skf, scoring='f1_macro')
    
    gb_risk.fit(X_risk, y_risk)
    joblib.dump(gb_risk, os.path.join(OUTPUT_DIR, 'risk_gradient_boosting.joblib'))
    
    risk_acc = float(np.mean(scores_risk_acc)) * 100.0
    risk_f1 = float(np.mean(scores_risk_f1))
    print(f"  ✓ Risk Stratification 5-Fold Accuracy:  {risk_acc:.2f}%")
    print(f"  ✓ Macro F1-Score:                       {risk_f1:.4f}")
    
    metadata["modules"]["risk"] = {
        "algorithm": "Gradient Boosted Decision Trees",
        "features": feat_risk,
        "classes": ["Low", "Moderate", "High"],
        "accuracy": round(risk_acc, 2),
        "f1Macro": round(risk_f1, 4)
    }

    # Save summary metadata
    with open(os.path.join(OUTPUT_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 75)
    print("  ✓ ALL 5 MODELS TRAINED, VALIDATED & SAVED SUCCESSFULLY!")
    print(f"  Model Artifacts Directory: {OUTPUT_DIR}")
    print("=" * 75)
    return metadata

if __name__ == '__main__':
    train_and_evaluate_all()
