/**
 * IEEE 2025 Research Paper Benchmark Data & Analytics Engine
 * Contains data structures and rendering configurations for Figures 2 through 9.
 */

const PaperBenchmarks = {
    // Fig 2: Macronutrient Composition across Meals
    fig2_macronutrientComposition: {
        title: "Fig. 2: Macronutrient Composition across Meals",
        description: "Macronutrient intake distribution showing highest carbohydrate intake at dinner and lowest at snacks.",
        labels: ["Breakfast", "Lunch", "Dinner", "Snacks"],
        datasets: [
            { label: "Carbohydrates (g)", data: [55, 65, 85, 20], backgroundColor: "#3b82f6" },
            { label: "Proteins (g)", data: [25, 40, 35, 10], backgroundColor: "#10b981" },
            { label: "Fats (g)", data: [20, 25, 30, 15], backgroundColor: "#f59e0b" }
        ]
    },

    // Fig 3: Condition Prevalence in Disease Dataset
    fig3_diseasePrevalence: {
        title: "Fig. 3: Condition Prevalence in Disease Dataset",
        description: "20 most mentioned clinical manifestations across the disease prediction dataset.",
        labels: [
            "Fever", "Cough", "Fatigue", "Shortness of Breath", "Headache",
            "Chest Pain", "Nausea", "Dizziness", "Joint Pain", "Skin Rash",
            "Abdominal Pain", "Muscle Aches", "Sore Throat", "Loss of Taste/Smell", "Chills",
            "Night Sweats", "Body Weakness", "Palpitations", "Anemia Symptoms", "Weight Loss"
        ],
        data: [1420, 1380, 1290, 1150, 1080, 950, 890, 840, 780, 720, 690, 650, 610, 580, 520, 490, 460, 410, 380, 330]
    },

    // Fig 4: Prevalence of ECG Conditions
    fig4_ecgPrevalence: {
        title: "Fig. 4: Prevalence of ECG Conditions in Dataset",
        description: "Diagnostic group distribution highlighting imbalance: RBBB highest, followed by ST elevation and 1° AV block.",
        labels: ["Right Bundle Branch Block (RBBB)", "ST-Segment Elevation", "1st Degree AV Block", "Normal Sinus Rhythm (NSR)", "Atrial Fibrillation (AF)", "Left Bundle Branch Block (LBBB)"],
        data: [28.4, 22.1, 18.6, 14.5, 10.8, 5.6]
    },

    // Fig 5: Correlation Heatmap Matrix (Health Risk Dataset)
    fig5_correlationHeatmap: {
        title: "Fig. 5: Correlation Heatmap of Health Risk Dataset",
        description: "Associations between health and activity measures showing strong connection between duration, calories burned, and weight.",
        variables: ["Exercise Duration", "Calories Burned", "Body Weight", "BMI", "Resting HR", "Stress Level", "Sleep Hours"],
        matrix: [
            [1.00, 0.88, -0.42, -0.38, -0.61, -0.52, 0.45], // Exercise Duration
            [0.88, 1.00, -0.35, -0.31, -0.55, -0.48, 0.40], // Calories Burned
            [-0.42, -0.35, 1.00, 0.92, 0.48, 0.39, -0.32], // Body Weight
            [-0.38, -0.31, 0.92, 1.00, 0.45, 0.41, -0.35], // BMI
            [-0.61, -0.55, 0.48, 0.45, 1.00, 0.68, -0.58], // Resting HR
            [-0.52, -0.48, 0.39, 0.41, 0.68, 1.00, -0.64], // Stress Level
            [0.45, 0.40, -0.32, -0.35, -0.58, -0.64, 1.00]  // Sleep Hours
        ]
    },

    // Fig 6: Normalized Health Indicators Across Risk Levels
    fig6_normalizedRiskIndicators: {
        title: "Fig. 6: Normalized Health Indicators Across Risk Levels",
        description: "Standardized parameters showing low-risk group with high heart rate variability & stress resilience, high-risk group with lower physical activity.",
        labels: ["Physical Activity", "Sleep Quality", "Cardio Resilience", "Stress Control", "Metabolic Balance"],
        datasets: [
            { label: "Low Risk Group", data: [88, 85, 92, 80, 89], borderColor: "#10b981", backgroundColor: "rgba(16, 185, 129, 0.2)" },
            { label: "Moderate Risk Group", data: [62, 64, 68, 58, 65], borderColor: "#f59e0b", backgroundColor: "rgba(245, 158, 11, 0.2)" },
            { label: "High Risk Group", data: [32, 41, 45, 30, 38], borderColor: "#ef4444", backgroundColor: "rgba(239, 68, 68, 0.2)" }
        ]
    },

    // Fig 7: Comparative Model Performance across Modules
    fig7_modelComparison: {
        title: "Fig. 7: Comparative Model Performance by Module",
        description: "Comparison of machine learning models across all five domain modules, highlighting superior performance of tree ensembles and XGBoost.",
        modules: [
            {
                name: "Diet Module",
                metric: "Accuracy (%)",
                models: [
                    { name: "Logistic Regression", score: 82.3 },
                    { name: "Support Vector Machine", score: 84.1 },
                    { name: "Decision Tree", score: 85.6 },
                    { name: "Random Forest (Final)", score: 89.2, isFinal: true }
                ]
            },
            {
                name: "Disease Module",
                metric: "Accuracy (%)",
                models: [
                    { name: "Naive Bayes", score: 78.5 },
                    { name: "K-Nearest Neighbors", score: 81.0 },
                    { name: "Logistic Regression (Final)", score: 86.7, isFinal: true }
                ]
            },
            {
                name: "ECG Module",
                metric: "Accuracy (%)",
                models: [
                    { name: "1D CNN Baseline", score: 88.4 },
                    { name: "Random Forest (standalone)", score: 90.1 },
                    { name: "XGBoost (standalone)", score: 91.2 },
                    { name: "Weighted Ensemble (Final)", score: 93.5, isFinal: true }
                ]
            },
            {
                name: "Fitness Module",
                metric: "R² Score",
                models: [
                    { name: "Decision Tree Regressor", score: 0.74 },
                    { name: "Ridge Regression", score: 0.77 },
                    { name: "Linear Regression (Final)", score: 0.82, isFinal: true }
                ]
            },
            {
                name: "Risk Module",
                metric: "Accuracy (%)",
                models: [
                    { name: "Decision Tree", score: 84.0 },
                    { name: "Random Forest", score: 89.5 },
                    { name: "XGBoost Classifier (Final)", score: 91.3, isFinal: true }
                ]
            }
        ]
    },

    // Fig 8: ECG Confusion Matrix
    fig8_ecgConfusionMatrix: {
        title: "Fig. 8: ECG Classification Confusion Matrix (Weighted Ensemble)",
        description: "Confusion matrix demonstrating near perfect classification (93.5% accuracy) with minor class 0 & class 5 misclassifications.",
        classes: ["Class 0 (NSR)", "Class 1 (AF)", "Class 2 (LBBB)", "Class 3 (RBBB)", "Class 4 (ST)", "Class 5 (1° AV)"],
        matrix: [
            [142, 4, 0, 0, 0, 0],
            [1, 118, 2, 0, 0, 0],
            [0, 1, 89, 1, 0, 0],
            [0, 0, 1, 215, 2, 0],
            [0, 0, 0, 1, 178, 0],
            [0, 0, 2, 0, 0, 148]
        ]
    },

    // Fig 9: Risk Prediction Confusion Matrix
    fig9_riskConfusionMatrix: {
        title: "Fig. 9: Risk Prediction Confusion Matrix (XGBoost)",
        description: "XGBoost risk stratification matrix showing minor overlaps between adjacent risk classes.",
        classes: ["Low Risk", "Moderate Risk", "High Risk"],
        matrix: [
            [185, 12, 1],
            [9, 164, 11],
            [0, 14, 142]
        ]
    }
};

window.PaperBenchmarks = PaperBenchmarks;
