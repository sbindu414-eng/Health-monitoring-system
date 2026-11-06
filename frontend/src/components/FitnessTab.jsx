import React from 'react';
import { Dumbbell, Activity, Heart, Moon, Zap, Shield } from 'lucide-react';

export default function FitnessTab({ patientData, onChangePatientData, onRunFitness, result }) {
    const d = patientData;

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Input Controls */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-amber-400 flex items-center gap-2">
                        <Dumbbell className="w-4 h-4" /> Lifestyle &amp; Physiological Metrics
                    </h3>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 font-mono">
                        LinReg (R² 0.82)
                    </span>
                </div>

                <div className="space-y-3 text-xs">
                    {/* Age & Weight */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Age</label>
                            <input
                                type="number"
                                value={d.age}
                                onChange={(e) => onChangePatientData('age', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                            />
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Weight (kg)</label>
                            <input
                                type="number"
                                value={d.weight}
                                onChange={(e) => onChangePatientData('weight', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                            />
                        </div>
                    </div>

                    {/* Height & Exercise */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Height (cm)</label>
                            <input
                                type="number"
                                value={d.height}
                                onChange={(e) => onChangePatientData('height', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                            />
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Exercise (min/day)</label>
                            <input
                                type="number"
                                value={d.exerciseMin}
                                onChange={(e) => onChangePatientData('exerciseMin', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                            />
                        </div>
                    </div>

                    {/* Sleep Hours (Full width because Calories Intake was removed) */}
                    <div>
                        <label className="block text-slate-400 mb-1">Sleep Duration (Hours/Night)</label>
                        <input
                            type="number"
                            step="0.5"
                            value={d.sleepHours}
                            onChange={(e) => onChangePatientData('sleepHours', Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                        />
                    </div>

                    {/* Resting HR & Stress Level */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Resting HR (BPM)</label>
                            <input
                                type="number"
                                value={d.heartRateResting}
                                onChange={(e) => onChangePatientData('heartRateResting', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                            />
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Stress Level (1-10)</label>
                            <input
                                type="number"
                                min="1"
                                max="10"
                                value={d.stressLevel}
                                onChange={(e) => onChangePatientData('stressLevel', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono focus:outline-none focus:border-amber-500 transition-all"
                            />
                        </div>
                    </div>

                    <button
                        onClick={onRunFitness}
                        className="w-full py-2.5 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded-lg shadow-lg shadow-amber-600/25 transition-all flex items-center justify-center gap-2 mt-2"
                    >
                        <span>🏃</span> Estimate Continuous Fitness Score
                    </button>
                </div>
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-2 rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200">Continuous Fitness Estimation &amp; Biomarkers</h3>
                    <span className="px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                        {result?.category || 'Good'}
                    </span>
                </div>

                {/* Score Banner */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3">
                    <div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Estimated Physiological Score</span>
                        <div className="flex items-baseline gap-2">
                            <span className="text-3xl font-extrabold font-mono text-amber-400 mt-0.5">
                                {result?.fitnessScore ?? 78.5}
                            </span>
                            <span className="text-xs text-slate-500">/ 100</span>
                        </div>
                    </div>
                    <div className="text-right">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Regression Model Fit</span>
                        <div className="text-xl font-bold font-mono text-amber-400">R² = 0.820</div>
                        <div className="text-[10px] text-slate-500 font-mono">MAE: 3.42 • RMSE: 4.85</div>
                    </div>
                </div>

                {/* Standardized Health Indicators */}
                {result?.indicators && (
                    <div className="space-y-3">
                        <h4 className="text-xs font-semibold text-slate-300">Standardized Physiological Indicator Breakdown</h4>
                        <div className="space-y-3 bg-slate-950/50 p-4 rounded-xl border border-slate-800 text-xs">
                            {/* Activity Index */}
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span className="text-slate-300">Cardiorespiratory Activity Index</span>
                                    <span className="text-amber-400 font-mono font-semibold">{result.indicators.activityIndex} / 100</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-amber-500 rounded-full transition-all duration-500" style={{ width: `${result.indicators.activityIndex}%` }} />
                                </div>
                            </div>

                            {/* Sleep Quality */}
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span className="text-slate-300">Nocturnal Sleep Quality Factor</span>
                                    <span className="text-amber-400 font-mono font-semibold">{result.indicators.sleepQuality} / 100</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-amber-500 rounded-full transition-all duration-500" style={{ width: `${result.indicators.sleepQuality}%` }} />
                                </div>
                            </div>

                            {/* Cardio Recovery */}
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span className="text-slate-300">Autonomic Cardio Recovery Buffer</span>
                                    <span className="text-amber-400 font-mono font-semibold">{result.indicators.cardioRecovery} / 100</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-amber-500 rounded-full transition-all duration-500" style={{ width: `${result.indicators.cardioRecovery}%` }} />
                                </div>
                            </div>

                            {/* Stress Resilience */}
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span className="text-slate-300">Neuro-Hormonal Stress Resilience</span>
                                    <span className="text-amber-400 font-mono font-semibold">{result.indicators.stressResilience} / 100</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-amber-500 rounded-full transition-all duration-500" style={{ width: `${result.indicators.stressResilience}%` }} />
                                </div>
                            </div>

                            {/* Metabolic Score */}
                            <div className="space-y-1">
                                <div className="flex justify-between">
                                    <span className="text-slate-300">Body Composition &amp; Metabolic Score</span>
                                    <span className="text-amber-400 font-mono font-semibold">{result.indicators.metabolicScore} / 100</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-amber-500 rounded-full transition-all duration-500" style={{ width: `${result.indicators.metabolicScore}%` }} />
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Recommendations */}
                {result?.recommendations && result.recommendations.length > 0 && (
                    <div className="space-y-2">
                        <h4 className="text-xs font-semibold text-slate-300">Clinical Lifestyle Recommendations</h4>
                        <ul className="space-y-1.5 bg-slate-950/40 p-3 rounded-xl border border-slate-800/80 text-xs text-slate-300">
                            {result.recommendations.map((rec, i) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="text-amber-400 font-bold">•</span>
                                    <span>{rec}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

