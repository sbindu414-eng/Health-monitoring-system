import React, { useState } from 'react';
import {
    Radio, Activity, Heart, Droplets, Thermometer, Wind,
    Zap, CheckCircle2, ShieldAlert
} from 'lucide-react';
import WebcamPpgSensor from './WebcamPpgSensor';
import LiveAudioAuscultation from './LiveAudioAuscultation';
import EcgCanvas from './EcgCanvas';

export default function LiveTelemetryTab({
    telemetry,
    onScenarioChange,
    onApplyLiveBpm,
    onApplyAudioSymptom,
    auditLog = [],
    riskResult
}) {
    const [streamSource, setStreamSource] = useState('webcam'); // 'webcam' | 'audio'

    return (
        <div className="space-y-6">
            {/* Real-Time Control Center Banner */}
            <div className="rounded-2xl p-6 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl relative overflow-hidden space-y-3">
                <div className="flex flex-wrap justify-between items-center gap-2">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                        <Radio className="w-3.5 h-3.5 text-rose-500 animate-pulse" />
                        <span>Autonomous Real-Time Biosensor Ingestion Engine</span>
                    </div>
                    <span className="text-[10px] text-emerald-400 font-mono bg-emerald-950/60 px-2.5 py-1 rounded-full border border-emerald-500/40">
                        ⚡ Continuous Zero-Input Live Inference Active
                    </span>
                </div>
                <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                    Real-Time Telemetry &amp; Autonomous ML Pipeline
                </h2>
                <p className="text-xs sm:text-sm text-slate-300 max-w-3xl leading-relaxed">
                    This framework continuously ingests live physiological parameters (heart rate, SpO2, blood pressure, skin temp, HRV, audio respiration, and 12-lead ECG signals) without asking the user to manually type inputs. All 5 ML modules infer continuously in sliding windows.
                </p>
            </div>

            {/* Ingestion Mode Selector Tabs */}
            <div className="flex flex-wrap gap-2 p-1.5 bg-slate-950/80 rounded-2xl border border-slate-800 text-xs">
                <button
                    type="button"
                    onClick={() => setStreamSource('webcam')}
                    className={`px-4 py-2 rounded-xl font-semibold transition-all flex items-center gap-2 ${
                        streamSource === 'webcam'
                            ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30'
                            : 'text-slate-400 hover:text-slate-200'
                    }`}
                >
                    <Activity className="w-4 h-4" />
                    <span>Live Optical PPG (Webcam Pulse Sensor)</span>
                </button>
                <button
                    type="button"
                    onClick={() => setStreamSource('audio')}
                    className={`px-4 py-2 rounded-xl font-semibold transition-all flex items-center gap-2 ${
                        streamSource === 'audio'
                            ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                            : 'text-slate-400 hover:text-slate-200'
                    }`}
                >
                    <Wind className="w-4 h-4" />
                    <span>Live Audio Respiration &amp; Cough Auscultation</span>
                </button>
            </div>

            {/* Webcam Mode */}
            {streamSource === 'webcam' && (
                <WebcamPpgSensor onBpmDetected={onApplyLiveBpm} isLiveAutoSync={true} />
            )}

            {/* Audio Mode */}
            {streamSource === 'audio' && (
                <LiveAudioAuscultation
                    onSymptomDetected={onApplyAudioSymptom}
                    respirationRate={telemetry?.respirationRate ?? 16}
                    telemetry={telemetry}
                />
            )}

            {/* Real-Time Live Vitals Dashboard Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                {/* Heart Rate */}
                <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-1">
                    <div className="flex items-center justify-between text-slate-400 text-xs">
                        <span>Heart Rate</span>
                        <Heart className="w-4 h-4 text-rose-500 animate-bounce" />
                    </div>
                    <div className="text-2xl font-bold font-mono text-slate-100">
                        {telemetry?.heartRate ?? 72} <span className="text-xs font-normal text-slate-500">BPM</span>
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">RR: {telemetry?.rrInterval ?? 800}ms</span>
                </div>

                {/* SpO2 */}
                <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-1">
                    <div className="flex items-center justify-between text-slate-400 text-xs">
                        <span>Blood Oxygen</span>
                        <Droplets className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div className="text-2xl font-bold font-mono text-cyan-300">
                        {telemetry?.spO2 ?? 98}%
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                        {telemetry?.spO2 < 92 ? '⚠️ Hypoxemia' : 'Normal Saturation'}
                    </span>
                </div>

                {/* Blood Pressure */}
                <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-1">
                    <div className="flex items-center justify-between text-slate-400 text-xs">
                        <span>Blood Pressure</span>
                        <Activity className="w-4 h-4 text-amber-400" />
                    </div>
                    <div className="text-xl font-bold font-mono text-slate-100">
                        {telemetry?.bloodPressure?.systolic ?? 120}/{telemetry?.bloodPressure?.diastolic ?? 78}
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">mmHg</span>
                </div>

                {/* Skin Temp */}
                <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-1">
                    <div className="flex items-center justify-between text-slate-400 text-xs">
                        <span>Skin Temp</span>
                        <Thermometer className="w-4 h-4 text-emerald-400" />
                    </div>
                    <div className="text-2xl font-bold font-mono text-emerald-300">
                        {telemetry?.skinTemp ?? 36.8}°C
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                        {telemetry?.skinTemp > 38.0 ? '🌡️ Febrile' : 'Afebrile'}
                    </span>
                </div>

                {/* Respiration */}
                <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-2xl space-y-1">
                    <div className="flex items-center justify-between text-slate-400 text-xs">
                        <span>Respiration</span>
                        <Wind className="w-4 h-4 text-indigo-400" />
                    </div>
                    <div className="text-2xl font-bold font-mono text-indigo-300">
                        {telemetry?.respirationRate ?? 16}
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">breaths/min</span>
                </div>

                {/* Live Composite Risk */}
                <div className="p-4 bg-purple-950/20 border border-purple-500/30 rounded-2xl space-y-1">
                    <div className="flex items-center justify-between text-purple-300 text-xs">
                        <span>ML Risk Index</span>
                        <ShieldAlert className="w-4 h-4 text-purple-400" />
                    </div>
                    <div className="text-2xl font-bold font-mono text-purple-300">
                        {riskResult?.riskScore ?? 24}%
                    </div>
                    <span className="text-[10px] text-purple-400 font-mono font-bold">
                        {riskResult?.riskLevel ?? 'Low'} Tier
                    </span>
                </div>
            </div>

            {/* Live Lead II Telemetry Oscilloscope Canvas */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
                <div className="flex justify-between items-center border-b border-slate-800 pb-2.5">
                    <div className="flex items-center gap-2">
                        <Zap className="w-4 h-4 text-rose-500" />
                        <h3 className="text-sm font-bold text-slate-200">
                            Real-Time Lead II Continuous Waveform Telemetry
                        </h3>
                    </div>
                    <span className="text-xs font-mono font-bold text-rose-400 bg-rose-950/40 px-2.5 py-0.5 rounded border border-rose-500/30">
                        Pattern: {telemetry?.ecgCondition || 'NSR'} ({telemetry?.severity || 'Normal'})
                    </span>
                </div>
                <EcgCanvas condition={telemetry?.ecgCondition || 'NSR'} lead="Lead II" speed={2.2} />
            </div>
        </div>
    );
}

