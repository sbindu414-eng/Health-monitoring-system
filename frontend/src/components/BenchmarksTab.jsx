import React from 'react';
import { BarChart3, Award, FileSpreadsheet, Grid, ShieldCheck } from 'lucide-react';
import { PaperBenchmarks } from '../modules/paperBenchmarks';

export default function BenchmarksTab() {
    const {
        fig2_macronutrientComposition,
        fig3_diseasePrevalence,
        fig4_ecgPrevalence,
        fig6_normalizedRiskIndicators,
        fig7_modelComparison,
        fig8_ecgConfusionMatrix,
        fig9_riskConfusionMatrix
    } = PaperBenchmarks;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="rounded-2xl p-6 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 shadow-xl space-y-2">
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-xs font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                    <BarChart3 className="w-3.5 h-3.5" /> IEEE Transactions on Healthcare Engineering (2025)
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-white">
                    Validated Experimental Benchmarks &amp; Figures 2–9
                </h2>
                <p className="text-xs text-slate-300">
                    Statistical evaluation, comparative baseline models, and confusion matrices validating the 5 production ML modules.
                </p>
            </div>

            {/* Model Comparison Table (Fig 7) */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                        <Award className="w-4 h-4 text-amber-400" /> {fig7_modelComparison.title}
                    </h3>
                    <span className="text-[10px] text-slate-400 font-mono">5-Fold Stratified Cross-Validation</span>
                </div>
                <p className="text-xs text-slate-400">{fig7_modelComparison.description}</p>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {fig7_modelComparison.modules.map((mod, i) => (
                        <div key={i} className="p-4 bg-slate-950/70 rounded-xl border border-slate-800 space-y-2.5">
                            <div className="flex justify-between items-center text-xs font-bold text-slate-200">
                                <span>{mod.name}</span>
                                <span className="text-[10px] text-slate-500 font-mono">{mod.metric}</span>
                            </div>
                            <div className="space-y-1.5">
                                {mod.models.map((m, j) => (
                                    <div
                                        key={j}
                                        className={`flex justify-between items-center text-xs p-2 rounded ${
                                            m.isSelected
                                                ? 'bg-indigo-500/15 border border-indigo-500/40 text-indigo-200 font-semibold'
                                                : 'text-slate-400'
                                        }`}
                                    >
                                        <span>{m.name}</span>
                                        <span className={`font-mono ${m.isSelected ? 'text-indigo-400 font-bold' : ''}`}>
                                            {m.score}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Grid 2-cols: ECG & Risk Confusion Matrices (Fig 8 & 9) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Fig 8: ECG Confusion Matrix */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <h4 className="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                            <Grid className="w-4 h-4" /> {fig8_ecgConfusionMatrix.title}
                        </h4>
                        <span className="text-[10px] font-mono text-emerald-400">93.5% Accuracy</span>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-center text-xs font-mono">
                            <thead>
                                <tr className="text-slate-500 border-b border-slate-800">
                                    <th className="p-2 text-left">Actual \ Pred</th>
                                    {fig8_ecgConfusionMatrix.classes.map((c, idx) => (
                                        <th key={idx} className="p-2">{c}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {fig8_ecgConfusionMatrix.matrix.map((row, rIdx) => (
                                    <tr key={rIdx} className="border-b border-slate-800/40">
                                        <td className="p-2 text-left font-bold text-slate-400">{fig8_ecgConfusionMatrix.classes[rIdx]}</td>
                                        {row.map((val, cIdx) => {
                                            const isDiag = rIdx === cIdx;
                                            return (
                                                <td
                                                    key={cIdx}
                                                    className={`p-2 rounded ${
                                                        isDiag
                                                            ? 'bg-rose-500/20 text-rose-300 font-bold'
                                                            : val > 0
                                                            ? 'bg-slate-800/60 text-slate-400'
                                                            : 'text-slate-600'
                                                    }`}
                                                >
                                                    {val}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Fig 9: Risk Prediction Confusion Matrix */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <h4 className="text-xs font-bold text-purple-400 flex items-center gap-1.5">
                            <Grid className="w-4 h-4" /> {fig9_riskConfusionMatrix.title}
                        </h4>
                        <span className="text-[10px] font-mono text-emerald-400">91.3% Accuracy</span>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-center text-xs font-mono">
                            <thead>
                                <tr className="text-slate-500 border-b border-slate-800">
                                    <th className="p-2 text-left">Actual \ Pred</th>
                                    {fig9_riskConfusionMatrix.classes.map((c, idx) => (
                                        <th key={idx} className="p-2">{c}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {fig9_riskConfusionMatrix.matrix.map((row, rIdx) => (
                                    <tr key={rIdx} className="border-b border-slate-800/40">
                                        <td className="p-2 text-left font-bold text-slate-400">{fig9_riskConfusionMatrix.classes[rIdx]}</td>
                                        {row.map((val, cIdx) => {
                                            const isDiag = rIdx === cIdx;
                                            return (
                                                <td
                                                    key={cIdx}
                                                    className={`p-2 rounded ${
                                                        isDiag
                                                            ? 'bg-purple-500/20 text-purple-300 font-bold'
                                                            : val > 0
                                                            ? 'bg-slate-800/60 text-slate-400'
                                                            : 'text-slate-600'
                                                    }`}
                                                >
                                                    {val}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* Prevalence Figures (Fig 3 & 4) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Fig 3: Symptoms Prevalence */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
                    <h4 className="text-xs font-bold text-cyan-400">{fig3_diseasePrevalence.title}</h4>
                    <p className="text-[11px] text-slate-400">{fig3_diseasePrevalence.description}</p>
                    <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                        {fig3_diseasePrevalence.items.map((it, idx) => {
                            const pct = (it.count / 1420) * 100;
                            return (
                                <div key={idx} className="space-y-0.5 text-xs">
                                    <div className="flex justify-between text-slate-300">
                                        <span>{it.name}</span>
                                        <span className="font-mono text-cyan-400">{it.count} cases</span>
                                    </div>
                                    <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-cyan-500 rounded-full" style={{ width: `${pct}%` }} />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Fig 4: ECG Prevalence */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
                    <h4 className="text-xs font-bold text-rose-400">{fig4_ecgPrevalence.title}</h4>
                    <p className="text-[11px] text-slate-400">{fig4_ecgPrevalence.description}</p>
                    <div className="space-y-2">
                        {fig4_ecgPrevalence.items.map((it, idx) => (
                            <div key={idx} className="space-y-1 text-xs">
                                <div className="flex justify-between text-slate-300">
                                    <span>{it.name}</span>
                                    <span className="font-mono font-bold" style={{ color: it.color }}>{it.percentage}%</span>
                                </div>
                                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full rounded-full" style={{ width: `${it.percentage}%`, backgroundColor: it.color }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

