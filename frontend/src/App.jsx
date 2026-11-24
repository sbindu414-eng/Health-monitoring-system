import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import TabNavigation from './components/TabNavigation';
import OverviewTab from './components/OverviewTab';
import DietTab from './components/DietTab';
import DiseaseTab from './components/DiseaseTab';
import EcgTab from './components/EcgTab';
import FitnessTab from './components/FitnessTab';
import RiskTab from './components/RiskTab';
import Login from './components/Login';

import { DietModule } from './modules/dietModule';
import { DiseaseModule } from './modules/diseaseModule';
import { ECGModule } from './modules/ecgModule';
import { FitnessModule } from './modules/fitnessModule';
import { RiskModule } from './modules/riskModule';

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

    // 2. Active Tab Navigation - Default to 'overview'
    const [activeTab, setActiveTab] = useState('overview');

    // 3. Centralized Patient Data State
    const [patientData, setPatientData] = useState({
        age: 45,
        gender: 'male',
        height: 175,
        weight: 80,
        activityLevel: 'moderate',
        dietType: 'balanced',
        dietGoal: 'maintenance',
        sleepHours: 7.0,
        exerciseMin: 35,
        heartRateResting: 72,
        stressLevel: 4,
        symptoms: [],
        ecgSymptoms: [],
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

    // Initial inference run on mount
    useEffect(() => {
        executeFullPipeline(patientData, diseaseContext);
    }, []);

    const handleChangePatientData = (field, val) => {
        setPatientData((prev) => {
            const updated = { ...prev, [field]: val };
            executeFullPipeline(updated, diseaseContext);
            return updated;
        });
    };

    const handleChangeDiseaseContext = (field, val) => {
        setDiseaseContext((prev) => {
            const updated = { ...prev, [field]: val };
            executeFullPipeline(patientData, updated);
            return updated;
        });
    };

    // Sequential 5-Module ML Inference Engine
    const executeFullPipeline = (currentPatient = patientData, currentContext = diseaseContext) => {
        setIsScanning(true);

        const diet = DietModule.predict(currentPatient);
        const disease = DiseaseModule.predict(currentPatient.symptoms, {
            ...currentContext,
            age: currentPatient.age,
            restingHeartRate: currentPatient.heartRateResting
        });
        const ecg = ECGModule.predict({
            age: currentPatient.age,
            gender: currentPatient.gender,
            symptoms: currentPatient.ecgSymptoms,
            hasFile: currentPatient.ecgHasFile
        });
        const fitness = FitnessModule.predict({
            ...currentPatient,
            heartRateResting: currentPatient.heartRateResting
        });
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
        executeFullPipeline(patientData, diseaseContext);
        setActiveTab('risk');
    };

    // Clinical Patient Profile Preset Loader
    const handleLoadPreset = (presetType) => {
        let presetPatient;
        let presetContext = { duration: '1_3_days', reportedSeverity: 'moderate' };

        if (presetType === 'athlete') {
            presetPatient = {
                age: 26,
                gender: 'male',
                height: 182,
                weight: 76,
                activityLevel: 'very_active',
                dietType: 'high_protein',
                dietGoal: 'muscle_gain',
                sleepHours: 8.5,
                exerciseMin: 90,
                heartRateResting: 54,
                stressLevel: 2,
                symptoms: [],
                ecgSymptoms: [],
                ecgHasFile: false
            };
        } else if (presetType === 'cardiac') {
            presetPatient = {
                age: 58,
                gender: 'male',
                height: 172,
                weight: 88,
                activityLevel: 'sedentary',
                dietType: 'high_carb',
                dietGoal: 'maintenance',
                sleepHours: 5.5,
                exerciseMin: 10,
                heartRateResting: 98,
                stressLevel: 8,
                symptoms: ['chest_pain', 'shortness_of_breath', 'fatigue'],
                ecgSymptoms: ['chest_pain', 'palpitations'],
                ecgHasFile: false
            };
            presetContext = { duration: '1_3_days', reportedSeverity: 'severe' };
        } else {
            presetPatient = {
                age: 45,
                gender: 'male',
                height: 175,
                weight: 80,
                activityLevel: 'moderate',
                dietType: 'balanced',
                dietGoal: 'maintenance',
                sleepHours: 7.0,
                exerciseMin: 35,
                heartRateResting: 72,
                stressLevel: 4,
                symptoms: [],
                ecgSymptoms: [],
                ecgHasFile: false
            };
        }

        setPatientData(presetPatient);
        setDiseaseContext(presetContext);
        executeFullPipeline(presetPatient, presetContext);
        setActiveTab('risk');
    };

    // Disease symptom toggle
    const handleToggleDiseaseSymptom = (id) => {
        setPatientData((prev) => {
            const nextSymptoms = prev.symptoms.includes(id)
                ? prev.symptoms.filter((s) => s !== id)
                : [...prev.symptoms, id];
            const nextState = { ...prev, symptoms: nextSymptoms };

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
        setPatientData((prev) => {
            const nextState = { ...prev, symptoms: [] };
            const dRes = DiseaseModule.predict([], diseaseContext);
            setResults((r) => ({ ...r, disease: dRes }));
            return nextState;
        });
    };

    // Manual Module Execution runners
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
        executeFullPipeline();
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
            {/* Top Navigation Bar */}
            <Navbar
                session={session}
                onSignOut={handleSignOut}
                onLoadPreset={handleLoadPreset}
                onRunFullScan={handleRunFullScan}
                isScanning={isScanning}
            />

            {/* Multi-Domain Module Navigation Tabs */}
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
            </main>
        </div>
    );
}
