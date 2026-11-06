import React from 'react';
import { ShieldAlert, AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';

export default function RiskTab({ result, onRunRisk, isRunning }) {
    const riskLevel = result?.riskLevel || 'Moderate';
    const isHigh = riskLevel === 'High';
    const isMod = riskLevel === 'Moderate';

    const levelColor = isHigh
        ? 'text-rose-400'
        : isMod
        ? 'text-amber-400'
        : 'text-emerald-400';

    const badgeBg = isHigh
        ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
        : isMod
        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
        : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';

    return (
        <div className="rounded-2xl p-6 bg-slate-900/80 border border-slate-800 shadow-2xl relative overflow-hidden space-y-6">
            {/* Header Banner */}
            <div className="flex flex-wrap justify-between items-center border-b border-slate-800 pb-4 gap-4">
                <div>
                    <span className="px-2.5 py-0.5 text-xs rounded font-mono font-bold bg-purple-500/10 text-purple-300 border border-purple-500/30">
                        Aggregate XGBoost Classifier (Paper 91.3% Acc)
                    </span>
                    <h3 className={`text-2xl sm:text-3xl font-extrabold ${levelColor} mt-1.5 tracking-tight`}>
                        {riskLevel.toUpperCase()} RISK STRATIFICATION
                    </h3>
                </div>
                <div className="flex items-center gap-4">
                    <span className={`px-4 py-1.5 rounded-xl text-sm font-bold font-mono border ${badgeBg}`}>
                        {riskLevel}
                    </span>
                    <div className="text-right">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Composite Score</span>
                        <div className={`text-3xl font-extrabold font-mono ${levelColor}`}>
                            {result?.riskScore ?? 42}%
                        </div>
                    </div>
                </div>
            </div>

            {/* Calibrated Probability Stratification Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-slate-950/80 border border-emerald-500/30 rounded-xl p-4 text-center">
                    <span className="text-[10px] text-emerald-400 uppercase font-bold tracking-wider">
                        Low Risk Probability
                    </span>
                    <div className="text-2xl font-bold font-mono text-emerald-300 mt-1">
                        {result?.probabilities?.low || '22.0%'}
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono">P(Y = 0 | X)</span>
                </div>
                <div className="bg-slate-950/80 border border-amber-500/30 rounded-xl p-4 text-center">
                    <span className="text-[10px] text-amber-400 uppercase font-bold tracking-wider">
                        Moderate Risk Probability
                    </span>
                    <div className="text-2xl font-bold font-mono text-amber-300 mt-1">
                        {result?.probabilities?.moderate || '65.0%'}
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono">P(Y = 1 | X)</span>
                </div>
                <div className="bg-slate-950/80 border border-rose-500/30 rounded-xl p-4 text-center">
                    <span className="text-[10px] text-rose-400 uppercase font-bold tracking-wider">
                        High Risk Probability
                    </span>
                    <div className="text-2xl font-bold font-mono text-rose-300 mt-1">
                        {result?.probabilities?.high || '13.0%'}
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono">P(Y = 2 | X)</span>
                </div>
            </div>

            {/* Clinical Action & Lifestyle Recommendations (Full Width) */}
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                        <span>📋</span> Clinical Decision Support (CDSS) Action Plan
                    </h4>
                    <span className="text-[10px] text-purple-400 font-mono bg-purple-950/40 px-2 py-0.5 rounded border border-purple-500/30">
                        Synthesized Cross-Domain Directives
                    </span>
                </div>

                <ul className="space-y-2.5">
                    {result?.clinicalActionPlan ? (
                        result.clinicalActionPlan.map((action, idx) => (
                            <li
                                key={idx}
                                className="p-3.5 bg-slate-950/70 rounded-xl text-xs text-slate-200 border-l-4 border-purple-500 shadow-md flex items-start gap-2.5"
                            >
                                <span className="font-bold text-purple-400 mt-0.5">{idx + 1}.</span>
                                <span className="leading-relaxed font-medium">{action}</span>
                            </li>
                        ))
                    ) : (
                        <li className="p-3 bg-slate-950/60 rounded-xl text-xs text-slate-400">
                            Run the full pipeline to generate clinical recommendations.
                        </li>
                    )}
                </ul>
            </div>

            {/* Recalculate CTA */}
            <div className="pt-2 flex justify-end">
                <button
                    onClick={onRunRisk}
                    disabled={isRunning}
                    className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-semibold rounded-xl text-xs shadow-lg shadow-purple-600/25 transition-all flex items-center gap-2"
                >
                    <Zap className="w-4 h-4" /> Recalculate Risk Stratification
                </button>
            </div>
        </div>
    );
}

