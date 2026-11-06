import React from 'react';
import { Stethoscope, AlertTriangle, CheckCircle, Clock, Activity, RotateCcw } from 'lucide-react';
import { DiseaseModule } from '../modules/diseaseModule';

export default function DiseaseTab({
    selectedSymptoms,
    onToggleSymptom,
    onClearSymptoms,
    diseaseContext,
    onChangeContext,
    onRunDisease,
    result
}) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Symptom Selector Panel */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-cyan-400 flex items-center gap-2">
                        <Stethoscope className="w-4 h-4" /> Binary Symptom Vector (17 Features)
                    </h3>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono">
                        Logistic Reg
                    </span>
                </div>

                <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Select active patient symptoms:</span>
                    <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950/60 border border-cyan-500/30 px-2 py-0.5 rounded">
                        {selectedSymptoms.length} Selected
                    </span>
                </div>

                {/* 17 Symptom Pills */}
                <div className="flex flex-wrap gap-2 max-h-72 overflow-y-auto pr-1">
                    {DiseaseModule.symptomsList.map((sym) => {
                        const isActive = selectedSymptoms.includes(sym.id);
                        return (
                            <button
                                key={sym.id}
                                type="button"
                                onClick={() => onToggleSymptom(sym.id)}
                                className={`px-2.5 py-1.5 rounded-lg text-xs flex items-center gap-1.5 border transition-all ${
                                    isActive
                                        ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-md shadow-cyan-500/20'
                                        : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700'
                                }`}
                            >
                                <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-cyan-400 animate-pulse' : 'bg-slate-600'}`} />
                                <span>{sym.label}</span>
                            </button>
                        );
                    })}
                </div>

                {/* Clinical Context Parameters */}
                <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-800/80">
                    <div>
                        <label className="block text-slate-400 mb-1">Symptom Duration</label>
                        <select
                            value={diseaseContext.duration}
                            onChange={(e) => onChangeContext('duration', e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-cyan-500 transition-all"
                        >
                            <option value="acute_24h">Acute (&lt; 24 Hours)</option>
                            <option value="1_3_days">1 – 3 Days</option>
                            <option value="4_7_days">4 – 7 Days</option>
                            <option value="persistent_week">&gt; 1 Week (Persistent)</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-slate-400 mb-1">Reported Severity</label>
                        <select
                            value={diseaseContext.reportedSeverity}
                            onChange={(e) => onChangeContext('reportedSeverity', e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-cyan-500 transition-all"
                        >
                            <option value="mild">Mild (Low impact)</option>
                            <option value="moderate">Moderate (Disruptive)</option>
                            <option value="severe">Severe (Acute discomfort)</option>
                        </select>
                    </div>
                </div>

                <div className="flex gap-2 pt-1">
                    <button
                        onClick={onRunDisease}
                        className="flex-1 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold rounded-lg shadow-lg shadow-cyan-600/25 transition-all text-xs"
                    >
                        Run Disease Predictor
                    </button>
                    <button
                        onClick={onClearSymptoms}
                        title="Clear all selected symptoms"
                        className="px-3 py-2.5 bg-slate-950 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-700 text-xs transition-all flex items-center gap-1"
                    >
                        <RotateCcw className="w-3.5 h-3.5" /> Clear
                    </button>
                </div>
            </div>

            {/* Diagnosis Output Panel */}
            <div className="lg:col-span-2 rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200">Multiclass Softmax Diagnosis Output</h3>
                    <span className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold ${
                        result?.severity === 'High'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : result?.severity === 'Moderate'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}>
                        {result?.severity || 'Low'} Severity
                    </span>
                </div>

                {/* Primary Result Banner */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Primary Predicted Pathology</span>
                        <h4 className="text-xl font-bold text-cyan-300 mt-0.5">
                            {result?.primaryDiagnosis || 'Healthy / Low Pathology Risk'}
                        </h4>
                    </div>
                    <div className="text-right">
                        <span className="text-[10px] text-slate-400 uppercase">Confidence Score</span>
                        <div className="text-2xl font-bold font-mono text-cyan-400">
                            {result?.confidence || '94.2%'}
                        </div>
                    </div>
                </div>

                {/* Probability Distribution Vector */}
                {result?.allPredictions && (
                    <div className="space-y-2">
                        <h4 className="text-xs font-semibold text-slate-300">Disease Category Probability Vector (Softmax)</h4>
                        <div className="space-y-2 bg-slate-950/50 p-3 rounded-xl border border-slate-800">
                            {result.allPredictions.map((p, i) => (
                                <div key={i} className="space-y-1">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-slate-300">{p.disease}</span>
                                        <span className="font-mono text-cyan-400 font-semibold">{p.percentage}</span>
                                    </div>
                                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-cyan-500 rounded-full transition-all duration-500"
                                            style={{ width: p.percentage }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Dynamic Clinical Guidance */}
                <div className="p-4 bg-gradient-to-r from-cyan-950/30 via-slate-950/60 to-slate-950/80 border border-cyan-500/30 rounded-xl space-y-3 text-xs">
                    <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
                        <span className="font-bold text-cyan-300 flex items-center gap-1.5">
                            <span>📋</span> Recommended Action Plan & Next Steps
                        </span>
                        <span className="text-[10px] text-cyan-400 bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 rounded font-mono">
                            {result?.paramBadge || 'Dynamic Synthesis'}
                        </span>
                    </div>

                    {/* What this means */}
                    <div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                            Clinical Meaning:
                        </span>
                        <p className="text-slate-200 mt-0.5 leading-relaxed font-medium">
                            {result?.summary || 'Evaluating symptoms...'}
                        </p>
                    </div>

                    {/* Actionable Steps */}
                    {result?.actionSteps && (
                        <div className="space-y-1">
                            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-semibold">
                                Immediate Steps You Should Take:
                            </span>
                            <ul className="space-y-1.5 mt-1 text-slate-300">
                                {result.actionSteps.map((step, idx) => (
                                    <li key={idx} className="flex items-start gap-2">
                                        <span className="text-cyan-400 font-bold mt-0.5">•</span>
                                        <span>{step}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Red flag warnings */}
                    {result?.whenToSeeDoctor && (
                        <div className="p-3 bg-slate-950/80 rounded-lg border border-amber-500/30 text-[11px] text-amber-300 flex items-start gap-2.5">
                            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                            <div>
                                <span className="font-bold text-amber-300">When to Seek Immediate Medical Care:</span>
                                <p className="text-slate-300 mt-0.5 leading-relaxed">{result.whenToSeeDoctor}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

