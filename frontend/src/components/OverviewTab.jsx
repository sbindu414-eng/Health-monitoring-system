import React from 'react';
import { ArrowRight, Cpu, Zap } from 'lucide-react';

export default function OverviewTab({ onSelectTab, onRunFullScan, results }) {
    const modules = [
        {
            id: 'diet',
            num: '01',
            name: 'Diet Suggestion Module',
            algo: 'Random Forest Classifier',
            acc: '89.2% Accuracy',
            cv: 'Stratified 5-Fold CV',
            desc: 'Analyzes age, gender, BMI, TDEE, and circadian meal split ratios to formulate macro-calibrated nutritional plans.',
            color: 'from-emerald-500/20 to-emerald-500/5',
            border: 'border-emerald-500/30',
            text: 'text-emerald-400',
            status: results.diet ? 'Analyzed' : 'Ready'
        },
        {
            id: 'disease',
            num: '02',
            name: 'Disease Diagnosis Module',
            algo: 'Multiclass Logistic Regression',
            acc: '86.7% Accuracy (F1: 0.84)',
            cv: 'Stratified 5-Fold CV',
            desc: 'Transforms 17 binary symptom vectors and duration metrics into calibrated disease probabilities with red-flag alerts.',
            color: 'from-cyan-500/20 to-cyan-500/5',
            border: 'border-cyan-500/30',
            text: 'text-cyan-400',
            status: results.disease ? 'Diagnosed' : 'Ready'
        },
        {
            id: 'ecg',
            num: '03',
            name: '12-Lead ECG Classification',
            algo: 'Weighted Soft-Voting Ensemble',
            acc: '93.5% Accuracy (AUC: 0.96)',
            cv: 'PhysioNet & MIT-BIH',
            desc: 'Classifies NSR, AF, LBBB, RBBB, ST-Elevation, and 1° AV Block from symptoms, age, and uploaded ECG reports with real-time waveform animation.',
            color: 'from-rose-500/20 to-rose-500/5',
            border: 'border-rose-500/30',
            text: 'text-rose-400',
            status: results.ecg ? 'Classified' : 'Ready'
        },
        {
            id: 'fitness',
            num: '04',
            name: 'Physical Fitness Estimation',
            algo: 'Linear Regression + Interactions',
            acc: 'R² = 0.82 (MAE: 3.42)',
            cv: 'Engineered Interactions',
            desc: 'Estimates continuous physiological fitness score (0-100) using sleep, exercise duration, resting HR, and stress interaction terms.',
            color: 'from-amber-500/20 to-amber-500/5',
            border: 'border-amber-500/30',
            text: 'text-amber-400',
            status: results.fitness ? 'Estimated' : 'Ready'
        },
        {
            id: 'risk',
            num: '05',
            name: 'Integrated Risk Stratification',
            algo: 'XGBoost Gradient Boosted Trees',
            acc: '91.3% Accuracy (F1: 0.89)',
            cv: 'Stratified 5-Fold CV',
            desc: 'Synthesizes cross-domain outputs from Modules 1-4 into calibrated risk categories (Low / Moderate / High) and CDSS action plans.',
            color: 'from-purple-500/20 to-purple-500/5',
            border: 'border-purple-500/30',
            text: 'text-purple-400',
            status: results.risk ? 'Stratified' : 'Ready'
        }
    ];

    return (
        <div className="space-y-6">
            {/* Hero Banner */}
            <div className="rounded-2xl p-6 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-2xl relative overflow-hidden">
                <div className="relative z-10 max-w-3xl space-y-3">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                        <Cpu className="w-3.5 h-3.5" /> Autonomous Multi-Domain ML Architecture
                    </div>
                    <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                        Modular Machine Learning for Integrated Health Risk Surveillance
                    </h2>
                    <p className="text-sm text-slate-300 leading-relaxed">
                        A unified clinical decision support pipeline addressing healthcare fragmentation by fusing dietary modeling, multi-symptom pathology classification, 12-lead ECG telemetry, and fitness regression into holistic risk prediction.
                    </p>
                    <div className="pt-2 flex flex-wrap gap-3">
                        <button
                            onClick={onRunFullScan}
                            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-rose-600 hover:from-indigo-500 hover:to-rose-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/25 flex items-center gap-2 transition-all"
                        >
                            <Zap className="w-4 h-4" /> Run 5-Module Sequential Scan
                        </button>
                        <button
                            onClick={() => onSelectTab('risk')}
                            className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-200 font-medium text-xs border border-slate-700 transition-all flex items-center gap-1.5"
                        >
                            <span>🛡️</span> View Risk Stratification <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Architecture Pipeline Flow Diagram */}
            <div className="rounded-2xl p-6 bg-slate-900/60 border border-slate-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                        <span>🔄</span> Sequential Pipeline Workflow
                    </h3>
                    <span className="text-[11px] text-indigo-400 font-mono">End-to-End Processing</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative">
                    {modules.map((m, idx) => (
                        <div
                            key={m.id}
                            onClick={() => onSelectTab(m.id)}
                            className={`p-4 rounded-xl border bg-gradient-to-b ${m.color} ${m.border} cursor-pointer hover:scale-[1.02] transition-all space-y-2`}
                        >
                            <div className="flex justify-between items-center">
                                <span className="text-xs font-mono font-bold text-slate-500">{m.num}</span>
                                <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                                    m.status !== 'Ready' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-slate-800 text-slate-400'
                                }`}>
                                    {m.status}
                                </span>
                            </div>
                            <h4 className={`text-xs font-bold ${m.text}`}>{m.name}</h4>
                            <p className="text-[11px] font-mono text-slate-300">{m.algo}</p>
                            <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/80 flex items-center justify-between">
                                <span>{m.acc}</span>
                                <ArrowRight className="w-3 h-3 text-slate-500" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

