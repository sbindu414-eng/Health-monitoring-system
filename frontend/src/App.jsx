import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import TabNavigation from './components/TabNavigation';
import OverviewTab from './components/OverviewTab';
import DietTab from './components/DietTab';
import DiseaseTab from './components/DiseaseTab';
import EcgTab from './components/EcgTab';
import FitnessTab from './components/FitnessTab';
import RiskTab from './components/RiskTab';
import BenchmarksTab from './components/BenchmarksTab';
import Login from './components/Login';

import { DietModule } from './modules/dietModule';
import { DiseaseModule } from './modules/diseaseModule';
import { ECGModule } from './modules/ecgModule';
import { FitnessModule } from './modules/fitnessModule';
import { RiskModule } from './modules/riskModule';
import { api } from './services/api';

export default function App() {
    // 1. Auth State
    const [session, setSession] = useState(() => {
        try {
            const saved = localStorage.getItem('healthml_session');
            return saved ? JSON.parse(saved) : null;
        } catch {
            return null;
        }
    });

    // 2. Active Tab Navigation
    const [activeTab, setActiveTab] = useState('overview');

    // 3. Centralized Patient Data State
    const [patientData, setPatientData] = useState({
        age: 45,
        gender: 'male',
        height: 175,
        weight: 82,
        activityLevel: 'moderate',
        dietType: 'balanced',
        dietGoal: 'maintenance',
        sleepHours: 6.5,
        exerciseMin: 35,
        heartRateResting: 74,
        stressLevel: 6,
        symptoms: ['cough', 'fatigue', 'headache', 'chest_pain'],
        ecgSymptoms: ['chest_pain', 'palpitations', 'shortness_breath'],
        ecgHasFile: false
    });

    // 4. Clinical Context for Disease Module
    const [diseaseContext, setDiseaseContext] = useState({
        duration: '1_3_days',
        reportedSeverity: 'moderate'
    });

    // 5. Prediction Results Cache
    const [results, setResults] = useState({
        diet: null,
        disease: null,
        ecg: null,
        fitness: null,
        risk: null
    });

    const [isScanning, setIsScanning] = useState(false);

    // Initial computation on mount
    useEffect(() => {
        executeFullPipeline();
    }, []);

    const handleChangePatientData = (field, val) => {
        setPatientData((prev) => ({ ...prev, [field]: val }));
    };

    const handleChangeDiseaseContext = (field, val) => {
        setDiseaseContext((prev) => ({ ...prev, [field]: val }));
    };

    // Disease symptom toggle
    const handleToggleDiseaseSymptom = (id) => {
        setPatientData((prev) => {
            const nextSymptoms = prev.symptoms.includes(id)
                ? prev.symptoms.filter((s) => s !== id)
                : [...prev.symptoms, id];
            const nextState = { ...prev, symptoms: nextSymptoms };

            // Real-time re-run of disease module
            const dRes = DiseaseModule.predict(nextSymptoms, {
                ...diseaseContext,
                age: prev.age,
                restingHeartRate: prev.heartRateResting
            });
            setResults((r) => ({ ...r, disease: dRes }));

            return nextState;
        });
    };

    const handleClearDiseaseSymptoms = () => {
        setPatientData((prev) => ({ ...prev, symptoms: [] }));
        const dRes = DiseaseModule.predict([], diseaseContext);
        setResults((r) => ({ ...r, disease: dRes }));
    };

    // Module execution runners
    const handleRunDiet = () => {
        const res = DietModule.predict(patientData);
        setResults((prev) => ({ ...prev, diet: res }));
    };

    const handleRunDisease = () => {
        const res = DiseaseModule.predict(patientData.symptoms, {
            ...diseaseContext,
            age: patientData.age,
            restingHeartRate: patientData.heartRateResting
        });
        setResults((prev) => ({ ...prev, disease: res }));
    };

    const handleRunEcg = () => {
        const res = ECGModule.predict({
            age: patientData.age,
            gender: patientData.gender,
            symptoms: patientData.ecgSymptoms,
            hasFile: patientData.ecgHasFile
        });
        setResults((prev) => ({ ...prev, ecg: res }));
    };

    const handleRunFitness = () => {
        const res = FitnessModule.predict(patientData);
        setResults((prev) => ({ ...prev, fitness: res }));
    };

    const handleRunRisk = () => {
        // Gather latest sub-module results
        const dRes = results.diet || DietModule.predict(patientData);
        const disRes = results.disease || DiseaseModule.predict(patientData.symptoms, diseaseContext);
        const ecgRes = results.ecg || ECGModule.predict({
            age: patientData.age,
            gender: patientData.gender,
            symptoms: patientData.ecgSymptoms,
            hasFile: patientData.ecgHasFile
        });
        const fitRes = results.fitness || FitnessModule.predict(patientData);

        const rRes = RiskModule.predict({
            bmi: dRes.bmi,
            diseaseLabel: disRes.primaryDiagnosis,
            diseaseSeverity: disRes.severity,
            ecgLabel: ecgRes.conditionName,
            ecgSeverity: ecgRes.severity,
            fitnessScore: fitRes.fitnessScore,
            dietRating: dRes.dietScore
        });

        setResults({
            diet: dRes,
            disease: disRes,
            ecg: ecgRes,
            fitness: fitRes,
            risk: rRes
        });
    };

    // Sequential 5-Module Execution
    const executeFullPipeline = () => {
        setIsScanning(true);

        const diet = DietModule.predict(patientData);
        const disease = DiseaseModule.predict(patientData.symptoms, {
            ...diseaseContext,
            age: patientData.age,
            restingHeartRate: patientData.heartRateResting
        });
        const ecg = ECGModule.predict({
            age: patientData.age,
            gender: patientData.gender,
            symptoms: patientData.ecgSymptoms,
            hasFile: patientData.ecgHasFile
        });
        const fitness = FitnessModule.predict(patientData);
        const risk = RiskModule.predict({
            bmi: diet.bmi,
            diseaseLabel: disease.primaryDiagnosis,
            diseaseSeverity: disease.severity,
            ecgLabel: ecg.conditionName,
            ecgSeverity: ecg.severity,
            fitnessScore: fitness.fitnessScore,
            dietRating: diet.dietScore
        });

        setResults({ diet, disease, ecg, fitness, risk });
        setIsScanning(false);
    };

    const handleRunFullScan = () => {
        executeFullPipeline();
        setActiveTab('risk');
    };

    // Patient Profile Presets
    const handleLoadPreset = (presetType) => {
        let newPatient = { ...patientData };

        if (presetType === 'athlete') {
            newPatient = {
                age: 26,
                gender: 'male',
                height: 180,
                weight: 74,
                activityLevel: 'active',
                dietType: 'balanced',
                dietGoal: 'muscle_gain',
                sleepHours: 8.5,
                exerciseMin: 75,
                heartRateResting: 52,
                stressLevel: 2,
                symptoms: [],
                ecgSymptoms: [],
                ecgHasFile: false
            };
        } else if (presetType === 'hypertension') {
            newPatient = {
                age: 58,
                gender: 'female',
                height: 162,
                weight: 88,
                activityLevel: 'light',
                dietType: 'balanced',
                dietGoal: 'weight_loss',
                sleepHours: 6.0,
                exerciseMin: 20,
                heartRateResting: 82,
                stressLevel: 7,
                symptoms: ['fatigue', 'dizziness', 'chest_pain', 'sob'],
                ecgSymptoms: ['chest_pain', 'dizziness', 'shortness_breath', 'fatigue', 'slow_hr'],
                ecgHasFile: false
            };
        } else if (presetType === 'cardiac') {
            newPatient = {
                age: 64,
                gender: 'male',
                height: 172,
                weight: 92,
                activityLevel: 'sedentary',
                dietType: 'low_carb',
                dietGoal: 'weight_loss',
                sleepHours: 5.5,
                exerciseMin: 10,
                heartRateResting: 94,
                stressLevel: 9,
                symptoms: ['fever', 'cough', 'chest_pain', 'sob', 'weakness'],
                ecgSymptoms: ['chest_pain', 'jaw_arm_pain', 'cold_sweat', 'shortness_breath', 'syncope', 'nausea'],
                ecgHasFile: false
            };
        }

        setPatientData(newPatient);

        // Run all models for new preset
        const diet = DietModule.predict(newPatient);
        const disease = DiseaseModule.predict(newPatient.symptoms, diseaseContext);
        const ecg = ECGModule.predict({
            age: newPatient.age,
            gender: newPatient.gender,
            symptoms: newPatient.ecgSymptoms,
            hasFile: newPatient.ecgHasFile
        });
        const fitness = FitnessModule.predict(newPatient);
        const risk = RiskModule.predict({
            bmi: diet.bmi,
            diseaseLabel: disease.primaryDiagnosis,
            diseaseSeverity: disease.severity,
            ecgLabel: ecg.conditionName,
            ecgSeverity: ecg.severity,
            fitnessScore: fitness.fitnessScore,
            dietRating: diet.dietScore
        });

        setResults({ diet, disease, ecg, fitness, risk });
        setActiveTab('risk');
    };

    const handleSignOut = () => {
        localStorage.removeItem('healthml_session');
        setSession(null);
    };

    if (!session) {
        return <Login onLoginSuccess={(sess) => setSession(sess)} />;
    }

    return (
        <div className="min-h-screen bg-[#070a12] text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30">
            {/* Navigation Header */}
            <Navbar
                session={session}
                onSignOut={handleSignOut}
                onLoadPreset={handleLoadPreset}
                onRunFullScan={handleRunFullScan}
                isScanning={isScanning}
            />

            {/* Sub-Header Tabs */}
            <TabNavigation activeTab={activeTab} onSelectTab={setActiveTab} />

            {/* Main Application Content Area */}
            <main className="max-w-7xl mx-auto px-4 py-6 flex-1 w-full space-y-6">
                {activeTab === 'overview' && (
                    <OverviewTab
                        onSelectTab={setActiveTab}
                        onRunFullScan={handleRunFullScan}
                        results={results}
                    />
                )}

                {activeTab === 'diet' && (
                    <DietTab
                        patientData={patientData}
                        onChangePatientData={handleChangePatientData}
                        onRunDiet={handleRunDiet}
                        result={results.diet}
                    />
                )}

                {activeTab === 'disease' && (
                    <DiseaseTab
                        selectedSymptoms={patientData.symptoms}
                        onToggleSymptom={handleToggleDiseaseSymptom}
                        onClearSymptoms={handleClearDiseaseSymptoms}
                        diseaseContext={diseaseContext}
                        onChangeContext={handleChangeDiseaseContext}
                        onRunDisease={handleRunDisease}
                        result={results.disease}
                    />
                )}

                {activeTab === 'ecg' && (
                    <EcgTab
                        patientData={patientData}
                        onChangePatientData={handleChangePatientData}
                        onRunEcg={handleRunEcg}
                        result={results.ecg}
                    />
                )}

                {activeTab === 'fitness' && (
                    <FitnessTab
                        patientData={patientData}
                        onChangePatientData={handleChangePatientData}
                        onRunFitness={handleRunFitness}
                        result={results.fitness}
                    />
                )}

                {activeTab === 'risk' && (
                    <RiskTab
                        result={results.risk}
                        onRunRisk={handleRunRisk}
                        isRunning={isScanning}
                    />
                )}

                {activeTab === 'benchmarks' && <BenchmarksTab />}
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500 mt-auto">
                <p>
                    A Modular Machine Learning Framework for Personalized Health Monitoring and Risk Prediction &copy; 2025-2026
                </p>
            </footer>
        </div>
    );
}
