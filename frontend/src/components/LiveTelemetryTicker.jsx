import React from 'react';
import { Heart, Activity, Thermometer, Droplets, Zap, ShieldAlert, Radio, AlertOctagon } from 'lucide-react';

export default function LiveTelemetryTicker({ telemetry, riskResult, isLive = true, onOpenLiveTab }) {
    if (!telemetry) return null;

    const isCritical = telemetry.severity === 'Critical' || (riskResult && riskResult.riskLevel === 'High');
    const isModerate = telemetry.severity === 'Moderate' || (riskResult && riskResult.riskLevel === 'Moderate');

    const statusBadgeClass = isCritical
        ? 'bg-red-500/20 text-red-300 border-red-500/50 animate-pulse'
        : isModerate
        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
        : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';

    return (
        <div className="border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md px-4 py-2 text-xs">
            {/* Critical Emergency Banner if in crisis */}
            {isCritical && (
                <div className="mb-2 p-2 bg-red-950/60 border border-red-500/60 rounded-xl text-red-200 flex flex-wrap items-center justify-between gap-2 animate-pulse">
                    <div className="flex items-center gap-2">
                        <AlertOctagon className="w-4 h-4 text-red-400 flex-shrink-0 animate-bounce" />
                        <span className="font-bold text-xs uppercase tracking-wider">
                            EMERGENCY ALERT: Acute Physiological Anomaly Detected ({telemetry.ecgCondition} • {telemetry.severity})
                        </span>
                    </div>
                    <span className="text-[10px] font-mono bg-red-600/30 px-2 py-0.5 rounded border border-red-400/50">
                        AUTOMATED ICU TELEMETRY DISPATCH ACTIVE
                    </span>
                </div>
            )}

            <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
                {/* Live Stream Status Pill */}
                <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-[11px] bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                        <Radio className="w-3.5 h-3.5 text-rose-500 animate-pulse" />
                        <span className="font-semibold">LIVE SENSOR STREAM</span>
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                        Pkt #{telemetry.packetId} • {telemetry.latencyMs}ms
                    </span>
                </div>

                {/* Live Vitals Ticker Grid */}
                <div className="flex flex-wrap items-center gap-3 sm:gap-5 text-slate-300 text-[11px] font-mono">
                    {/* Heart Rate */}
                    <div className="flex items-center gap-1.5">
                        <Heart className="w-3.5 h-3.5 text-rose-400 animate-bounce" />
                        <span className="text-slate-400">HR:</span>
                        <span className="font-bold text-slate-100 text-xs">{telemetry.heartRate}</span>
                        <span className="text-[10px] text-slate-500">BPM</span>
                    </div>

                    {/* SpO2 */}
                    <div className="flex items-center gap-1.5">
                        <Droplets className="w-3.5 h-3.5 text-cyan-400" />
                        <span className="text-slate-400">SpO2:</span>
                        <span className={`font-bold text-xs ${telemetry.spO2 < 92 ? 'text-rose-400' : 'text-slate-100'}`}>
                            {telemetry.spO2}%
                        </span>
                    </div>

                    {/* Blood Pressure */}
                    <div className="flex items-center gap-1.5 hidden md:flex">
                        <Activity className="w-3.5 h-3.5 text-amber-400" />
                        <span className="text-slate-400">BP:</span>
                        <span className="font-bold text-slate-100 text-xs">
                            {telemetry.bloodPressure?.systolic}/{telemetry.bloodPressure?.diastolic}
                        </span>
                        <span className="text-[10px] text-slate-500">mmHg</span>
                    </div>

                    {/* Skin Temp */}
                    <div className="flex items-center gap-1.5 hidden lg:flex">
                        <Thermometer className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-slate-400">Temp:</span>
                        <span className="font-bold text-slate-100 text-xs">{telemetry.skinTemp}°C</span>
                    </div>

                    {/* ECG Code */}
                    <div className="flex items-center gap-1.5">
                        <Zap className="w-3.5 h-3.5 text-rose-400" />
                        <span className="text-slate-400">ECG:</span>
                        <span className="font-bold text-rose-300 text-xs">{telemetry.ecgCondition}</span>
                    </div>

                    {/* Real-Time ML Risk Score */}
                    {riskResult && (
                        <div className="flex items-center gap-1.5 pl-2 border-l border-slate-800">
                            <ShieldAlert className="w-3.5 h-3.5 text-purple-400" />
                            <span className="text-slate-400">Live Risk:</span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${statusBadgeClass}`}>
                                {riskResult.riskScore}% ({riskResult.riskLevel})
                            </span>
                        </div>
                    )}
                </div>

                {/* Live Stream Inspector Button */}
                <button
                    onClick={onOpenLiveTab}
                    className="px-2.5 py-1 text-[11px] font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-indigo-500/30 transition-all flex items-center gap-1"
                >
                    <span>Inspect Stream</span> &rarr;
                </button>
            </div>
        </div>
    );
}

