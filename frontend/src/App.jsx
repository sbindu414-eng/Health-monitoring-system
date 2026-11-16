import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import TabNavigation from './components/TabNavigation';
import LiveTelemetryTicker from './components/LiveTelemetryTicker';
import LiveTelemetryTab from './components/LiveTelemetryTab';
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
import { liveTelemetry } from './services/liveTelemetryService';

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

    // 2. Active Tab Navigation - Default to 'live' for real-time biosensor demonstration
    const [activeTab, setActiveTab] = useState('live');

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

    // 6. Real-Time Telemetry & Audit Stream State
    const [telemetry, setTelemetry] = useState(null);
    const [isLiveStreaming, setIsLiveStreaming] = useState(true);
    const [auditLog, setAuditLog] = useState([
        { time: new Date().toLocaleTimeString(), level: 'INFO', message: 'Autonomous IoT Biosensor Telemetry Engine initiated.' },
        { time: new Date().toLocaleTimeString(), level: 'INFO', message: 'Real-time sliding-window ML inference active.' }
    ]);

    const [isScanning, setIsScanning] = useState(false);
    const lastInferenceTimeRef = useRef(0);

    // Subscribe to Real-Time Telemetry Stream on mount
    useEffect(() => {
        liveTelemetry.start();

        const unsubscribe = liveTelemetry.subscribe((packet) => {
            setTelemetry(packet);

            // Automatically feed live telemetry into patient state without user input!
            setPatientData((prev) => {
                const updated = {
                    ...prev,
                    heartRateResting: packet.heartRate,
                    symptoms: packet.activeSymptoms || [],
                    ecgSymptoms: packet.activeSymptoms || []
                };

                // Throttled Sliding-Window ML Inference Loop (runs every 1.2s on live stream)
                const now = performance.now();
                if (now - lastInferenceTimeRef.current > 1200) {
                    lastInferenceTimeRef.current = now;
                    executeSlidingWindowInference(updated, packet);
                }

                return updated;
            });

            // Log critical/high events to audit trail
            if (packet.severity === 'Critical' || packet.severity === 'High') {
                setAuditLog((prev) => [
                    {
                        time: packet.timestamp,
                        level: packet.severity === 'Critical' ? 'CRITICAL' : 'HIGH',
                        message: `Telemetry Anomaly detected: ${packet.ecgCondition} rhythm with HR ${packet.heartRate} BPM, SpO2 ${packet.spO2}%.`
                    },
                    ...prev.slice(0, 30)
                ]);
            }
        });

        // Initial run
        executeFullPipeline();

        return () => {
            unsubscribe();
            liveTelemetry.stop();
        };
    }, []);

    const handleChangePatientData = (field, val) => {
        setPatientData((prev) => ({ ...prev, [field]: val }));
    };

    const handleChangeDiseaseContext = (field, val) => {
        setDiseaseContext((prev) => ({ ...prev, [field]: val }));
    };

    // Automated Sliding-Window Real-Time ML Inference
    const executeSlidingWindowInference = (currentPatient, packet) => {
        const diet = DietModule.predict(currentPatient);
        const disease = DiseaseModule.predict(currentPatient.symptoms, {
            ...diseaseContext,
            age: currentPatient.age,
            restingHeartRate: packet ? packet.heartRate : currentPatient.heartRateResting
        });
        const ecg = ECGModule.predict({
            age: currentPatient.age,
            gender: currentPatient.gender,
            symptoms: currentPatient.ecgSymptoms,
            hasFile: currentPatient.ecgHasFile
        });
        const fitness = FitnessModule.predict({
            ...currentPatient,
            heartRateResting: packet ? packet.heartRate : currentPatient.heartRateResting
        });
        const risk = RiskModule.predict({
            bmi: diet.bmi,
            diseaseLabel: disease.primaryDiagnosis,
            diseaseSeverity: disease.severity,
            ecgLabel: packet ? packet.ecgCondition : ecg.conditionName,
            ecgSeverity: packet ? packet.severity : ecg.severity,
            fitnessScore: fitness.fitnessScore,
            dietRating: diet.dietScore
        });

        setResults({ diet, disease, ecg, fitness, risk });
    };

    // Full Pipeline Execution
    const executeFullPipeline = () => {
        setIsScanning(true);
        executeSlidingWindowInference(patientData, telemetry);
        setIsScanning(false);
    };

    const handleRunFullScan = () => {
        executeFullPipeline();
        setActiveTab('risk');
    };

    // Toggle Live Telemetry Stream
    const handleToggleLiveStream = () => {
        if (isLiveStreaming) {
            liveTelemetry.stop();
            setIsLiveStreaming(false);
            setAuditLog((prev) => [
                { time: new Date().toLocaleTimeString(), level: 'INFO', message: 'Live sensor telemetry stream paused by clinician.' },
                ...prev
            ]);
        } else {
            liveTelemetry.start();
            setIsLiveStreaming(true);
            setAuditLog((prev) => [
                { time: new Date().toLocaleTimeString(), level: 'INFO', message: 'Live sensor telemetry stream resumed.' },
                ...prev
            ]);
        }
    };

    // Scenario Change Handler (Wearable Stream)
    const handleScenarioChange = (scenarioId) => {
        liveTelemetry.setScenario(scenarioId);
        setAuditLog((prev) => [
            {
                time: new Date().toLocaleTimeString(),
                level: scenarioId === 'stemi' ? 'CRITICAL' : scenarioId === 'afib' || scenarioId === 'hypoxia' ? 'HIGH' : 'INFO',
                message: `IoT Biosensor stream switched to scenario: ${scenarioId.toUpperCase()}.`
            },
            ...prev
        ]);
    };

    // Live Webcam Optical PPG Detection
    const handleApplyLiveBpm = (bpm) => {
        setPatientData((prev) => ({ ...prev, heartRateResting: bpm }));
        setAuditLog((prev) => [
            { time: new Date().toLocaleTimeString(), level: 'INFO', message: `Webcam Optical PPG captured live heart rate: ${bpm} BPM.` },
            ...prev
        ]);
        executeSlidingWindowInference({ ...patientData, heartRateResting: bpm }, telemetry);
    };

    // Live Audio Cough / Stridor Detection
    const handleApplyAudioSymptom = (symId) => {
        setPatientData((prev) => {
            if (prev.symptoms.includes(symId)) return prev;
            const updatedSymptoms = [...prev.symptoms, symId];
            setAuditLog((l) => [
                { time: new Date().toLocaleTimeString(), level: 'WARN', message: `Live Audio Auscultation detected acoustic event: ${symId.toUpperCase()}.` },
                ...l
            ]);
            executeSlidingWindowInference({ ...prev, symptoms: updatedSymptoms }, telemetry);
            return { ...prev, symptoms: updatedSymptoms };
        });
    };

    // Disease symptom toggle (Manual override)
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
        setPatientData((prev) => ({ ...prev, symptoms: [] }));
        const dRes = DiseaseModule.predict([], diseaseContext);
        setResults((r) => ({ ...r, disease: dRes }));
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

    const handleLoadPreset = (presetType) => {
        handleScenarioChange(presetType);
        setActiveTab('live');
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
                isLiveStreaming={isLiveStreaming}
                onToggleLiveStream={handleToggleLiveStream}
                onOpenLiveTab={() => setActiveTab('live')}
            />

            {/* Continuous Real-Time Telemetry Vitals Ticker HUD */}
            <LiveTelemetryTicker
                telemetry={telemetry}
                riskResult={results.risk}
                isLive={isLiveStreaming}
                onOpenLiveTab={() => setActiveTab('live')}
            />

            {/* Multi-Domain Module Navigation Tabs */}
            <TabNavigation activeTab={activeTab} onSelectTab={setActiveTab} />

            {/* Main Application Content Area */}
            <main className="max-w-7xl mx-auto px-4 py-6 flex-1 w-full space-y-6">
                {/* Real-Time Live Biosensors & Telemetry Center */}
                {activeTab === 'live' && (
                    <LiveTelemetryTab
                        telemetry={telemetry}
                        onScenarioChange={handleScenarioChange}
                        onApplyLiveBpm={handleApplyLiveBpm}
                        onApplyAudioSymptom={handleApplyAudioSymptom}
                        auditLog={auditLog}
                        riskResult={results.risk}
                    />
                )}

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
