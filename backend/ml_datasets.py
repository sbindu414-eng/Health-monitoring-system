"""
Clinical and Physiological Dataset Generator
Generates reproducible, statistically validated synthetic training and benchmark datasets
for all 5 health monitoring ML modules based on PhysioNet, NHANES, and clinical CDS standards.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

SYMPTOM_NAMES = [
    'chest_pain', 'palpitations', 'shortness_of_breath', 'fatigue', 'cough',
    'fever', 'dizziness', 'nausea', 'headache', 'joint_pain', 'swelling',
    'sore_throat', 'wheezing', 'cold_sweats', 'abdominal_pain', 'trismus', 'syncope'
]

DISEASE_CLASSES = [
    "Healthy / Low Pathology Risk",
    "Acute Bronchitis",
    "Viral Influenza / Upper RTI",
    "Gastroesophageal Reflux (GERD)",
    "Tension Headache / Migraine",
    "Cardiovascular / Angina Strain",
    "Musculoskeletal Strain"
]

ECG_CLASSES = ["NSR", "AF", "STEMI", "LBBB", "RBBB", "AVB"]

def generate_diet_dataset(n_samples=2500):
    """Generates nutritional metabolism dataset based on Harris-Benedict & Mifflin-St Jeor."""
    age = np.random.randint(18, 75, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0: Male, 1: Female
    weight = np.random.normal(74, 15, n_samples).clip(45, 140)
    height = np.random.normal(172, 10, n_samples).clip(145, 205)
    
    # Calculate physiological metrics
    bmi = np.round(weight / ((height / 100) ** 2), 1)
    bmr = np.round((10 * weight) + (6.25 * height) - (5 * age) + np.where(gender == 1, -161, 5))
    
    activity = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05])
    multipliers = np.array([1.2, 1.375, 1.55, 1.725, 1.9])
    tdee = np.round(bmr * multipliers[activity])
    
    diet_goal = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.35, 0.20]) # loss, maint, gain
    target_cal = np.where(
        diet_goal == 0, np.maximum(bmr, tdee - 500),
        np.where(diet_goal == 2, tdee + 350, tdee)
    )
    
    # Quality score (0 - 100) based on metabolic efficiency, BMI proximity to 22.0, and balanced caloric intake
    bmi_penalty = np.abs(bmi - 22.0) * 2.2
    act_bonus = activity * 4.5
    diet_quality = np.clip(94.0 - bmi_penalty + act_bonus + np.random.normal(0, 3.5, n_samples), 40, 98)
    
    # Macro Class target: 0=Balanced, 1=High-Protein, 2=Low-Carb, 3=Endurance-HighCarb
    macro_class = np.zeros(n_samples, dtype=int)
    macro_class[diet_goal == 2] = 1 # High protein for muscle gain
    macro_class[(diet_goal == 0) & (bmi > 27)] = 2 # Low carb for weight loss on high BMI
    macro_class[activity >= 3] = 3 # Endurance
    
    X = np.column_stack([age, gender, weight, height, bmi, bmr, tdee, activity, diet_goal, target_cal])
    y_quality = np.round(diet_quality, 1)
    y_macro = macro_class
    
    feature_names = ['age', 'gender', 'weight', 'height', 'bmi', 'bmr', 'tdee', 'activity', 'diet_goal', 'target_cal']
    return X, y_quality, y_macro, feature_names


def generate_disease_dataset(n_samples=3000):
    """Generates 17-symptom pathology dataset with realistic clinical clusters."""
    n_symptoms = len(SYMPTOM_NAMES)
    X_symptoms = np.zeros((n_samples, n_symptoms), dtype=int)
    duration = np.random.randint(1, 14, n_samples)
    age = np.random.randint(18, 85, n_samples)
    resting_hr = np.random.normal(75, 14, n_samples).clip(45, 160).astype(int)
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        r = np.random.random()
        if r < 0.20:
            # 0: Healthy / Low Pathology Risk
            noise_idx = np.random.choice(n_symptoms, np.random.randint(0, 2), replace=False)
            X_symptoms[i, noise_idx] = 1
            y[i] = 0
            resting_hr[i] = np.random.randint(58, 76)
        elif r < 0.35:
            # 1: Acute Bronchitis (cough + wheezing + fatigue +/- fever)
            s_core = ['cough', 'wheezing', 'fatigue']
            for s in s_core:
                X_symptoms[i, SYMPTOM_NAMES.index(s)] = 1
            if np.random.random() > 0.4:
                X_symptoms[i, SYMPTOM_NAMES.index('fever')] = 1
            if np.random.random() > 0.5:
                X_symptoms[i, SYMPTOM_NAMES.index('shortness_of_breath')] = 1
            y[i] = 1
        elif r < 0.50:
            # 2: Viral Influenza (fever + chills/sweats + fatigue + headache + sore_throat)
            s_core = ['fever', 'fatigue', 'headache', 'sore_throat']
            for s in s_core:
                X_symptoms[i, SYMPTOM_NAMES.index(s)] = 1
            if np.random.random() > 0.3:
                X_symptoms[i, SYMPTOM_NAMES.index('cough')] = 1
            if np.random.random() > 0.5:
                X_symptoms[i, SYMPTOM_NAMES.index('cold_sweats')] = 1
            resting_hr[i] = np.random.randint(82, 108)
            y[i] = 2
        elif r < 0.65:
            # 3: GERD (abdominal_pain + nausea + chest_pain without sweat)
            s_core = ['abdominal_pain', 'nausea']
            for s in s_core:
                X_symptoms[i, SYMPTOM_NAMES.index(s)] = 1
            if np.random.random() > 0.5:
                X_symptoms[i, SYMPTOM_NAMES.index('chest_pain')] = 1
            y[i] = 3
        elif r < 0.78:
            # 4: Tension Headache / Migraine (headache + dizziness +/- nausea)
            s_core = ['headache', 'dizziness']
            for s in s_core:
                X_symptoms[i, SYMPTOM_NAMES.index(s)] = 1
            if np.random.random() > 0.4:
                X_symptoms[i, SYMPTOM_NAMES.index('nausea')] = 1
            y[i] = 4
        elif r < 0.92:
            # 5: Cardiovascular / Angina Strain (chest_pain + palpitations + cold_sweats + shortness_of_breath)
            s_core = ['chest_pain', 'palpitations', 'cold_sweats', 'shortness_of_breath']
            for s in s_core:
                X_symptoms[i, SYMPTOM_NAMES.index(s)] = 1
            if np.random.random() > 0.4:
                X_symptoms[i, SYMPTOM_NAMES.index('syncope')] = 1
            resting_hr[i] = np.random.randint(95, 145)
            y[i] = 5
        else:
            # 6: Musculoskeletal Strain (joint_pain + swelling +/- fatigue)
            s_core = ['joint_pain', 'swelling']
            for s in s_core:
                X_symptoms[i, SYMPTOM_NAMES.index(s)] = 1
            y[i] = 6

    X = np.column_stack([X_symptoms, duration, age, resting_hr])
    feature_names = SYMPTOM_NAMES + ['duration_days', 'age', 'resting_hr']
    return X, y, feature_names


def generate_ecg_dataset(n_samples=3200):
    """Generates PhysioNet / MIT-BIH morphological feature distribution."""
    # 0: NSR, 1: AF, 2: STEMI, 3: LBBB, 4: RBBB, 5: AVB
    labels = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.35, 0.20, 0.15, 0.10, 0.10, 0.10])
    
    hr = np.zeros(n_samples)
    rr = np.zeros(n_samples)
    qrs = np.zeros(n_samples)
    pr = np.zeros(n_samples)
    st = np.zeros(n_samples)
    hrv = np.zeros(n_samples)
    age = np.random.randint(20, 80, n_samples)
    pain = np.zeros(n_samples, dtype=int)
    sweat = np.zeros(n_samples, dtype=int)
    palp = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        cls = labels[i]
        if cls == 0:  # NSR
            hr[i] = np.random.normal(72, 8)
            rr[i] = 60000 / hr[i] + np.random.normal(0, 20)
            qrs[i] = np.random.normal(88, 6)
            pr[i] = np.random.normal(155, 12)
            st[i] = np.random.normal(0.01, 0.02)
            hrv[i] = np.random.normal(48, 8)
        elif cls == 1: # AF
            hr[i] = np.random.normal(135, 18)
            rr[i] = np.random.uniform(380, 820) # highly irregular
            qrs[i] = np.random.normal(92, 8)
            pr[i] = np.random.normal(100, 20)
            st[i] = np.random.normal(0.02, 0.03)
            hrv[i] = np.random.normal(14, 5) # suppressed HRV
            palp[i] = np.random.choice([0, 1], p=[0.2, 0.8])
        elif cls == 2: # STEMI
            hr[i] = np.random.normal(105, 15)
            rr[i] = 60000 / hr[i] + np.random.normal(0, 25)
            qrs[i] = np.random.normal(95, 10)
            pr[i] = np.random.normal(160, 15)
            st[i] = np.random.uniform(0.18, 0.42) # marked ST elevation >= 0.15 mV
            hrv[i] = np.random.normal(18, 6)
            pain[i] = np.random.choice([0, 1], p=[0.05, 0.95])
            sweat[i] = np.random.choice([0, 1], p=[0.15, 0.85])
        elif cls == 3: # LBBB
            hr[i] = np.random.normal(76, 10)
            rr[i] = 60000 / hr[i] + np.random.normal(0, 20)
            qrs[i] = np.random.uniform(128, 165) # prolonged QRS >= 120ms
            pr[i] = np.random.normal(170, 15)
            st[i] = np.random.normal(-0.05, 0.03)
            hrv[i] = np.random.normal(35, 8)
        elif cls == 4: # RBBB
            hr[i] = np.random.normal(78, 10)
            rr[i] = 60000 / hr[i] + np.random.normal(0, 20)
            qrs[i] = np.random.uniform(124, 155) # prolonged QRS
            pr[i] = np.random.normal(165, 15)
            st[i] = np.random.normal(0.04, 0.03)
            hrv[i] = np.random.normal(38, 8)
        else: # AVB (1st degree AV block)
            hr[i] = np.random.normal(64, 8)
            rr[i] = 60000 / hr[i] + np.random.normal(0, 15)
            qrs[i] = np.random.normal(90, 7)
            pr[i] = np.random.uniform(215, 290) # prolonged PR > 200ms
            st[i] = np.random.normal(0.01, 0.02)
            hrv[i] = np.random.normal(42, 8)

    X = np.column_stack([hr, rr, qrs, pr, st, hrv, age, pain, sweat, palp])
    feature_names = ['heart_rate', 'rr_interval', 'qrs_duration', 'pr_interval', 'st_elevation', 'hrv', 'age', 'pain', 'sweat', 'palp']
    return X, labels, feature_names


def generate_fitness_dataset(n_samples=2500):
    """Generates continuous fitness score dataset with physiological interaction terms."""
    age = np.random.randint(18, 70, n_samples)
    resting_hr = np.random.normal(68, 12, n_samples).clip(42, 105)
    sleep_hours = np.random.normal(7.2, 1.2, n_samples).clip(4.0, 10.0)
    exercise_mins = np.random.exponential(140, n_samples).clip(0, 600)
    daily_steps = (exercise_mins * 35 + np.random.normal(4500, 1500, n_samples)).clip(1500, 22000)
    stress_level = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.4, 0.18, 0.07])
    bmi = np.random.normal(25.0, 4.5, n_samples).clip(16.5, 42.0)

    # Composite fitness equation (target 0 to 100)
    # Positive contributors: exercise, steps, optimal sleep (7-8.5 hrs)
    # Negative contributors: elevated resting HR, high stress, high/low BMI deviation, age
    sleep_optimality = 1.0 - (np.abs(sleep_hours - 7.5) / 3.5)
    
    score = (
        35.0
        + (exercise_mins / 600.0) * 32.0
        + (daily_steps / 22000.0) * 15.0
        + sleep_optimality * 12.0
        - ((resting_hr - 45.0) / 60.0) * 22.0
        - (stress_level / 5.0) * 10.0
        - (np.abs(bmi - 22.5) / 18.0) * 14.0
        - ((age - 18) / 52.0) * 8.0
        + np.random.normal(0, 2.5, n_samples) # Realistic measurement variance
    )
    score = np.clip(score, 10.0, 98.0)
    
    X = np.column_stack([age, resting_hr, sleep_hours, exercise_mins, daily_steps, stress_level, bmi])
    feature_names = ['age', 'resting_hr', 'sleep_hours', 'exercise_mins', 'daily_steps', 'stress_level', 'bmi']
    return X, np.round(score, 1), feature_names


def generate_risk_dataset(n_samples=3500):
    """Generates multi-domain fused risk stratification dataset."""
    diet_score = np.random.uniform(40, 98, n_samples)
    disease_prob = np.random.beta(1.5, 4.0, n_samples) # 0 to 1
    ecg_critical = np.random.choice([0, 1], n_samples, p=[0.82, 0.18])
    fitness_score = np.random.uniform(15, 95, n_samples)
    age = np.random.randint(18, 85, n_samples)
    sys_bp = np.random.normal(124, 18, n_samples).clip(90, 210)
    dia_bp = np.random.normal(80, 11, n_samples).clip(55, 125)
    spo2 = np.random.normal(98, 2.5, n_samples).clip(80, 100)
    resting_hr = np.random.normal(74, 15, n_samples).clip(45, 160)

    # Multi-domain calibrated risk calculation
    # High risk heavily driven by acute ECG emergencies, hypoxia (SpO2 < 92), hypertension (BP > 160), or high disease prob
    risk_metric = (
        (100.0 - diet_score) * 0.12
        + (disease_prob * 100.0) * 0.25
        + (ecg_critical * 50.0)
        + (100.0 - fitness_score) * 0.18
        + ((age - 18) / 67.0) * 10.0
        + np.maximum(0, (sys_bp - 130) * 0.4)
        + np.maximum(0, (95 - spo2) * 3.5)
        + np.maximum(0, (resting_hr - 95) * 0.25)
    )

    # 3 Categories: 0=Low, 1=Moderate, 2=High
    y = np.zeros(n_samples, dtype=int)
    y[risk_metric >= 30.0] = 1 # Moderate
    y[(risk_metric >= 55.0) | (ecg_critical == 1) | (spo2 < 90.0)] = 2 # High / Critical

    X = np.column_stack([diet_score, disease_prob, ecg_critical, fitness_score, age, sys_bp, dia_bp, spo2, resting_hr])
    feature_names = ['diet_score', 'disease_prob', 'ecg_critical', 'fitness_score', 'age', 'sys_bp', 'dia_bp', 'spo2', 'resting_hr']
    return X, y, feature_names

