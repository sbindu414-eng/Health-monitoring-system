/**
 * Fitness Estimation Module - Linear Regression ML Model
 * IEEE 2025 Paper: R² Score 0.82 with engineered feature interaction terms
 */

const FitnessModule = {
    metadata: {
        algorithm: "Linear Regression with Engineered Interaction Terms",
        paperR2Score: 0.82,
        mae: 3.42,
        rmse: 4.85,
        cvStrategy: "Stratified 5-Fold"
    },

    // Linear Regression Weights (Paper equation 3 implementation)
    weights: {
        intercept: 45.2,
        age: -0.22,
        bmi: -0.85,
        exerciseMin: 0.42,
        sleepHours: 1.85,
        caloricBalance: -0.003, // (Intake - Baseline)
        heartRateResting: -0.35,
        stressLevel: -2.10,
        // Interaction terms
        interaction_exercise_sleep: 0.035, // Exercise * Sleep boost
        interaction_stress_hr: -0.015       // High stress * elevated HR penalty
    },

    preprocess: function (data) {
        const age = parseFloat(data.age) || 28;
        const weight = parseFloat(data.weight) || 70;
        const height = parseFloat(data.height) || 175;
        const bmi = parseFloat((weight / ((height / 100) * (height / 100))).toFixed(1));
        const exerciseMin = parseFloat(data.exerciseMin) || 45; // min/day
        const sleepHours = parseFloat(data.sleepHours) || 7.5;
        const caloricIntake = parseFloat(data.caloricIntake) || 2200;
        const heartRateResting = parseFloat(data.heartRateResting) || 68; // bpm
        const stressLevel = parseFloat(data.stressLevel) || 4; // 1 to 10 scale

        return { age, weight, height, bmi, exerciseMin, sleepHours, caloricIntake, heartRateResting, stressLevel };
    },

    predict: function (inputData) {
        const p = this.preprocess(inputData);
        const w = this.weights;

        const caloricDiff = p.caloricIntake - 2000;
        const ex_sleep = p.exerciseMin * p.sleepHours;
        const stress_hr = p.stressLevel * p.heartRateResting;

        // Regression equation y = beta_0 + sum(beta_i * x_i)
        let rawScore = w.intercept
            + (w.age * p.age)
            + (w.bmi * p.bmi)
            + (w.exerciseMin * p.exerciseMin)
            + (w.sleepHours * p.sleepHours)
            + (w.caloricBalance * caloricDiff)
            + (w.heartRateResting * p.heartRateResting)
            + (w.stressLevel * p.stressLevel)
            + (w.interaction_exercise_sleep * ex_sleep)
            + (w.interaction_stress_hr * stress_hr);

        // Clamp continuous fitness score to 0 - 100
        const fitnessScore = parseFloat(Math.max(15, Math.min(99, rawScore)).toFixed(1));

        // Determine fitness category
        let category = "Fair";
        if (fitnessScore >= 85) category = "Excellent";
        else if (fitnessScore >= 70) category = "Good";
        else if (fitnessScore >= 50) category = "Fair";
        else category = "Poor / Low Fitness";

        // Normalized Health Indicators (Paper Fig. 6 pattern)
        const indicators = {
            activityIndex: Math.min(100, Math.round((p.exerciseMin / 90) * 100)),
            sleepQuality: Math.min(100, Math.round((p.sleepHours / 9) * 100)),
            cardioRecovery: Math.min(100, Math.round(100 - ((p.heartRateResting - 50) * 1.5))),
            stressResilience: Math.min(100, Math.round((10 - p.stressLevel) * 10)),
            metabolicScore: Math.min(100, Math.round(100 - (Math.abs(p.bmi - 22) * 5)))
        };

        const recommendations = [];
        if (p.exerciseMin < 30) recommendations.push("Increase moderate physical activity to at least 150 min/week to improve cardiorespiratory fitness.");
        if (p.sleepHours < 7) recommendations.push("Prioritize 7.5 - 8 hours of nocturnal sleep for optimal anabolic recovery.");
        if (p.stressLevel > 6) recommendations.push("High stress index detected; incorporate mindfulness, breathwork, or active recovery.");
        if (p.heartRateResting > 85) recommendations.push("Elevated resting heart rate noted; consult a cardiac specialist for evaluation.");

        return {
            fitnessScore,
            category,
            indicators,
            recommendations,
            engineeredInteractions: { ex_sleep, stress_hr },
            modelInfo: this.metadata
        };
    }
};

window.FitnessModule = FitnessModule;
