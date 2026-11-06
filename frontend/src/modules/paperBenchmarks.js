/**
 * IEEE 2025 Research Paper Benchmark Data & Analytics Engine
 * Figures 2 through 9 from the research publication.
 */

export const PaperBenchmarks = {
    fig2_macronutrientComposition: {
        title: "Fig. 2: Macronutrient Composition across Meals",
        description: "Macronutrient intake distribution across breakfast, lunch, dinner, and snacks.",
        labels: ["Breakfast", "Lunch", "Dinner", "Snacks"],
        datasets: [
            { label: "Carbohydrates (g)", data: [55, 65, 85, 20], color: "#38bdf8" },
            { label: "Proteins (g)", data: [25, 40, 35, 10], color: "#34d399" },
            { label: "Fats (g)", data: [20, 25, 30, 15], color: "#fbbf24" }
        ]
    },

    fig3_diseasePrevalence: {
        title: "Fig. 3: Condition Prevalence in Disease Dataset",
        description: "Frequency of top clinical symptom manifestations across the disease prediction cohort.",
        items: [
            { name: "Fever", count: 1420 },
            { name: "Cough", count: 1380 },
            { name: "Fatigue", count: 1290 },
            { name: "Shortness of Breath", count: 1150 },
            { name: "Headache", count: 1080 },
            { name: "Chest Pain", count: 950 },
            { name: "Nausea", count: 890 },
            { name: "Dizziness", count: 840 },
            { name: "Joint Pain", count: 780 },
            { name: "Skin Rash", count: 720 },
            { name: "Abdominal Pain", count: 690 },
            { name: "Muscle Aches", count: 650 },
            { name: "Sore Throat", count: 610 },
            { name: "Loss of Taste/Smell", count: 580 },
            { name: "Chills", count: 520 },
            { name: "Body Weakness", count: 460 },
            { name: "Palpitations", count: 410 }
        ]
    },

    fig4_ecgPrevalence: {
        title: "Fig. 4: Prevalence of ECG Conditions in Dataset",
        description: "Diagnostic group distribution across MIT-BIH & PhysioNet annotated telemetry.",
        items: [
            { code: "RBBB", name: "Right Bundle Branch Block", percentage: 28.4, color: "#f43f5e" },
            { code: "ST_ELEV", name: "ST-Segment Elevation / Ischemia", percentage: 22.1, color: "#ef4444" },
            { code: "AV_BLOCK_1", name: "1st Degree AV Block", percentage: 18.6, color: "#06b6d4" },
            { code: "NSR", name: "Normal Sinus Rhythm", percentage: 14.5, color: "#10b981" },
            { code: "AF", name: "Atrial Fibrillation", percentage: 10.8, color: "#e11d48" },
            { code: "LBBB", name: "Left Bundle Branch Block", percentage: 5.6, color: "#f97316" }
        ]
    },

    fig6_normalizedRiskIndicators: {
        title: "Fig. 6: Normalized Health Indicators Across Risk Tiers",
        description: "Standardized biomarker averages across stratified risk cohorts.",
        indicators: ["Physical Activity", "Sleep Quality", "Cardio Recovery", "Stress Resilience", "Metabolic Balance"],
        tiers: [
            { label: "Low Risk Cohort", color: "#10b981", values: [88, 85, 92, 80, 89] },
            { label: "Moderate Risk Cohort", color: "#f59e0b", values: [62, 64, 68, 58, 65] },
            { label: "High Risk Cohort", color: "#ef4444", values: [32, 41, 45, 30, 38] }
        ]
    },

    fig7_modelComparison: {
        title: "Fig. 7: Comparative Model Performance Across All 5 Modules",
        description: "Standardized benchmark evaluations validating the chosen production models.",
        modules: [
            {
                name: "1. Diet Suggestion",
                metric: "Accuracy (%)",
                models: [
                    { name: "Logistic Regression", score: "82.3%" },
                    { name: "Support Vector Machine", score: "84.1%" },
                    { name: "Decision Tree", score: "85.6%" },
                    { name: "Random Forest (Selected)", score: "89.2%", isSelected: true }
                ]
            },
            {
                name: "2. Disease Prediction",
                metric: "Accuracy (%)",
                models: [
                    { name: "Naive Bayes", score: "78.5%" },
                    { name: "K-Nearest Neighbors", score: "81.0%" },
                    { name: "Decision Tree", score: "82.4%" },
                    { name: "Multiclass Logistic Reg (Selected)", score: "86.7%", isSelected: true }
                ]
            },
            {
                name: "3. ECG Classification",
                metric: "Accuracy / AUC",
                models: [
                    { name: "1D CNN Baseline", score: "88.4%" },
                    { name: "Random Forest", score: "90.1%" },
                    { name: "Gradient Boosting", score: "90.8%" },
                    { name: "XGBoost", score: "91.2%" },
                    { name: "LightGBM", score: "92.0%" },
                    { name: "Weighted Soft-Voting Ensemble (Selected)", score: "93.5% (AUC 0.96)", isSelected: true }
                ]
            },
            {
                name: "4. Fitness Estimation",
                metric: "Regression R² Score",
                models: [
                    { name: "Decision Tree Regressor", score: "R² 0.74" },
                    { name: "Ridge Regression", score: "R² 0.77" },
                    { name: "Linear Regression + Interaction Terms (Selected)", score: "R² 0.82 (MAE 3.42)", isSelected: true }
                ]
            },
            {
                name: "5. Risk Stratification",
                metric: "Accuracy / F1",
                models: [
                    { name: "Decision Tree", score: "84.0%" },
                    { name: "Random Forest", score: "89.5%" },
                    { name: "XGBoost Classifier (Selected)", score: "91.3% (F1 0.89)", isSelected: true }
                ]
            }
        ]
    },

    fig8_ecgConfusionMatrix: {
        title: "Fig. 8: ECG Classification Confusion Matrix (Weighted Ensemble)",
        classes: ["NSR", "AF", "LBBB", "RBBB", "ST_ELEV", "AV_BLOCK_1"],
        matrix: [
            [142, 4, 0, 0, 0, 0],
            [1, 118, 2, 0, 0, 0],
            [0, 1, 89, 1, 0, 0],
            [0, 0, 1, 215, 2, 0],
            [0, 0, 0, 1, 178, 0],
            [0, 0, 2, 0, 0, 148]
        ]
    },

    fig9_riskConfusionMatrix: {
        title: "Fig. 9: Risk Prediction Confusion Matrix (XGBoost)",
        classes: ["Low Risk", "Moderate Risk", "High Risk"],
        matrix: [
            [185, 12, 1],
            [9, 164, 11],
            [0, 14, 142]
        ]
    }
};

