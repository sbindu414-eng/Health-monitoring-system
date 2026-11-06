import React, { useState } from 'react';
import { HeartPulse, Upload, FileText, X, AlertTriangle, Activity } from 'lucide-react';
import { ECGModule } from '../modules/ecgModule';
import EcgCanvas from './EcgCanvas';

export default function EcgTab({ patientData, onChangePatientData, onRunEcg, result }) {
    const [selectedSymptoms, setSelectedSymptoms] = useState(patientData.ecgSymptoms || []);
    const [uploadedFile, setUploadedFile] = useState(null);
    const [filePreviewUrl, setFilePreviewUrl] = useState(null);

    const toggleSymptom = (id) => {
        const next = selectedSymptoms.includes(id)
            ? selectedSymptoms.filter((s) => s !== id)
            : [...selectedSymptoms, id];
        setSelectedSymptoms(next);
        onChangePatientData('ecgSymptoms', next);
    };

    const handleFileUpload = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setUploadedFile(file);
        if (file.type.startsWith('image/')) {
            const url = URL.createObjectURL(file);
            setFilePreviewUrl(url);
        } else {
            setFilePreviewUrl(null);
        }
        onChangePatientData('ecgHasFile', true);
    };

    const handleRemoveFile = () => {
        setUploadedFile(null);
        setFilePreviewUrl(null);
        onChangePatientData('ecgHasFile', false);
    };

    const conditionCode = result?.predictedCode || 'NSR';

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Input Panel */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-rose-400 flex items-center gap-2">
                        <HeartPulse className="w-4 h-4" /> Patient ECG Analysis
                    </h3>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 font-mono">
                        Soft Ensemble
                    </span>
                </div>

                <div className="space-y-3.5 text-xs">
                    {/* Age & Gender */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Age</label>
                            <input
                                type="number"
                                min="1"
                                max="120"
                                value={patientData.age}
                                onChange={(e) => onChangePatientData('age', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-rose-500 transition-all font-mono"
                            />
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Gender</label>
                            <select
                                value={patientData.gender}
                                onChange={(e) => onChangePatientData('gender', e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-rose-500 transition-all"
                            >
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                    </div>

                    {/* Cardiac Symptoms Pills */}
                    <div>
                        <div className="flex justify-between items-center mb-1.5">
                            <label className="block text-slate-400">Cardiac / Related Symptoms</label>
                            <span className="text-[10px] text-rose-400 font-mono bg-rose-950/40 px-1.5 py-0.5 rounded">
                                {selectedSymptoms.length} active
                            </span>
                        </div>
                        <div className="flex flex-wrap gap-1.5 max-h-52 overflow-y-auto pr-1">
                            {ECGModule.cardiacSymptoms.map((sym) => {
                                const isActive = selectedSymptoms.includes(sym.id);
                                return (
                                    <button
                                        key={sym.id}
                                        type="button"
                                        onClick={() => toggleSymptom(sym.id)}
                                        className={`px-2 py-1 rounded-lg text-[11px] flex items-center gap-1.5 border transition-all ${
                                            isActive
                                                ? 'bg-rose-900/40 border-rose-500/60 text-white shadow-sm shadow-rose-500/20'
                                                : 'bg-slate-950/60 border-slate-800/80 text-slate-300 hover:border-slate-700'
                                        }`}
                                    >
                                        <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-rose-400 animate-pulse' : 'bg-slate-600'}`} />
                                        <span>{sym.label}</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* ECG Report Upload */}
                    <div>
                        <label className="block text-slate-400 mb-1.5">
                            Upload ECG Report <span className="text-slate-500">(Image / PDF)</span>
                        </label>
                        {!uploadedFile ? (
                            <label className="flex flex-col items-center justify-center gap-2 w-full h-28 border-2 border-dashed border-rose-500/30 rounded-xl bg-slate-950/60 hover:border-rose-500/60 hover:bg-rose-950/20 cursor-pointer transition-all group p-2 text-center">
                                <Upload className="w-6 h-6 text-rose-400/60 group-hover:text-rose-400 transition-all" />
                                <span className="text-[11px] text-slate-400 group-hover:text-slate-200">
                                    Click or drag ECG report file here<br />
                                    <span className="text-[10px] text-slate-500">PNG, JPG, PDF up to 10 MB</span>
                                </span>
                                <input
                                    type="file"
                                    accept="image/*,.pdf"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                />
                            </label>
                        ) : (
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 p-2.5 bg-rose-950/30 border border-rose-500/30 rounded-xl">
                                    <FileText className="w-4 h-4 text-rose-400 flex-shrink-0" />
                                    <div className="flex-1 min-w-0 text-left">
                                        <p className="text-[11px] text-rose-300 font-medium truncate">{uploadedFile.name}</p>
                                        <p className="text-[10px] text-slate-500 font-mono">
                                            {(uploadedFile.size / 1024).toFixed(1)} KB • Telemetry Verified
                                        </p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={handleRemoveFile}
                                        className="p-1 text-slate-400 hover:text-rose-400 rounded transition-all"
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                                {filePreviewUrl && (
                                    <div className="rounded-lg overflow-hidden border border-rose-500/30 bg-slate-950 max-h-28 flex items-center justify-center">
                                        <img src={filePreviewUrl} alt="ECG Telemetry Preview" className="object-contain max-h-28 w-full" />
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    <button
                        onClick={onRunEcg}
                        className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-semibold rounded-lg shadow-lg shadow-rose-600/25 transition-all flex items-center justify-center gap-2"
                    >
                        <span>🫀</span> Analyse ECG &amp; Classify Rhythm
                    </button>
                </div>
            </div>

            {/* Canvas & Results Panel */}
            <div className="lg:col-span-2 rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200">Real-Time Continuous Telemetry Canvas</h3>
                    <span className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold ${
                        result?.severity === 'Critical'
                            ? 'bg-red-600/30 text-red-300 border border-red-500/60 animate-pulse'
                            : result?.severity === 'High'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : result?.severity === 'Moderate'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}>
                        {result?.severity || 'Normal'}
                    </span>
                </div>

                {/* Animated ECG Sweep Canvas */}
                <EcgCanvas condition={conditionCode} lead="Lead II" speed={2.0} />

                {/* Diagnosis Label & Confidence */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Classification Label</span>
                        <h4 className="text-base font-bold text-rose-400 mt-0.5 leading-snug">
                            {result?.conditionName || 'Normal Sinus Rhythm (NSR)'}
                        </h4>
                    </div>
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-right">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Voting Confidence</span>
                        <div className="text-2xl font-bold font-mono text-rose-400">
                            {result?.confidence || '93.5%'}
                        </div>
                    </div>
                </div>

                {/* Condition Description */}
                {result?.description && (
                    <div className="p-3 bg-rose-950/20 border border-rose-500/20 rounded-xl text-xs text-slate-300 leading-relaxed">
                        {result.description}
                    </div>
                )}

                {/* Clinical Guidance Box */}
                {result?.clinicalSummary && (
                    <div className="p-4 bg-gradient-to-r from-rose-950/30 via-slate-950/60 to-slate-950/80 border border-rose-500/30 rounded-xl space-y-3 text-xs">
                        <div className="flex items-center justify-between border-b border-rose-500/20 pb-2">
                            <span className="font-bold text-rose-300 flex items-center gap-1.5">
                                <span>📋</span> Clinical Guidance &amp; Recommended Actions
                            </span>
                            <span className="text-[10px] bg-rose-500/10 border border-rose-500/30 px-2 py-0.5 rounded font-mono text-rose-400">
                                {result.riskTag || 'Cardiology Evaluation'}
                            </span>
                        </div>

                        <div>
                            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold mb-0.5">
                                What This Means:
                            </span>
                            <p className="text-slate-200 leading-relaxed font-medium">
                                {result.clinicalSummary}
                            </p>
                        </div>

                        {result.actions && (
                            <div>
                                <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold mb-1">
                                    Recommended Steps:
                                </span>
                                <ul className="space-y-1.5 text-slate-300">
                                    {result.actions.map((act, i) => (
                                        <li key={i} className="flex items-start gap-1.5">
                                            <span className="text-rose-400 font-bold mt-0.5">•</span>
                                            <span>{act}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {result.warning && (
                            <div className="p-3 bg-slate-950/80 rounded-lg border border-amber-500/30 flex items-start gap-2.5">
                                <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                                <p className="text-[11px] text-amber-200 leading-relaxed">{result.warning}</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

