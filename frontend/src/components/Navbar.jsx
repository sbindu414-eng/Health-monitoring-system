import React from 'react';
import { Activity, Zap, LogOut, Radio, ShieldCheck } from 'lucide-react';

export default function Navbar({
    session,
    onSignOut,
    onLoadPreset,
    onRunFullScan,
    isScanning,
    isLiveStreaming,
    onToggleLiveStream,
    onOpenLiveTab
}) {
    return (
        <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
                {/* Brand */}
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                        <Activity className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h1 className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-300 to-cyan-400">
                                Modular ML Health Monitoring Framework
                            </h1>
                            <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/30 animate-pulse">
                                <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-ping" /> REAL-TIME IoT ENGINE
                            </span>
                        </div>
                    </div>
                </div>

                {/* Actions & Profiles */}
                <div className="flex items-center gap-3">
                    {/* Live Stream Switch */}
                    <button
                        onClick={onToggleLiveStream}
                        title={isLiveStreaming ? "Pause Live Biosensor Stream" : "Resume Live Biosensor Stream"}
                        className={`px-3 py-1.5 text-xs font-semibold rounded-lg flex items-center gap-1.5 border transition-all ${
                            isLiveStreaming
                                ? 'bg-rose-950/40 border-rose-500/50 text-rose-300 shadow-sm shadow-rose-500/20'
                                : 'bg-slate-900 border-slate-700 text-slate-400 hover:text-slate-200'
                        }`}
                    >
                        <Radio className={`w-3.5 h-3.5 ${isLiveStreaming ? 'text-rose-400 animate-pulse' : 'text-slate-500'}`} />
                        <span>{isLiveStreaming ? 'Live Stream: ACTIVE' : 'Live Stream: PAUSED'}</span>
                    </button>

                    {/* Patient Presets */}
                    <div className="hidden lg:flex items-center gap-1.5">
                        <button
                            onClick={() => onLoadPreset('athlete')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-emerald-500/30 transition-all"
                            title="Load Athlete Telemetry"
                        >
                            🏃 Athlete
                        </button>
                        <button
                            onClick={() => onLoadPreset('exercise')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-amber-300 border border-amber-500/30 transition-all"
                            title="Load Exercise Tachycardia Stream"
                        >
                            ⚡ Exertion
                        </button>
                        <button
                            onClick={() => onLoadPreset('stemi')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-rose-300 border border-rose-500/30 transition-all"
                            title="Load STEMI Ischemia Crisis Stream"
                        >
                            🚨 STEMI Alert
                        </button>
                    </div>

                    <button
                        onClick={onRunFullScan}
                        disabled={isScanning}
                        className="px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-md shadow-purple-500/25 flex items-center gap-1.5 transition-all disabled:opacity-50"
                    >
                        <Zap className="w-3.5 h-3.5" />
                        <span>{isScanning ? 'Inferring...' : 'Infer Pipeline'}</span>
                    </button>

                    {/* Active User Badge & Sign Out */}
                    {session && (
                        <div className="flex items-center gap-2 pl-3 border-l border-slate-800">
                            <div className="w-8 h-8 rounded-full bg-slate-800 border border-indigo-500/40 flex items-center justify-center text-sm shadow">
                                {session.avatar || '👨‍⚕️'}
                            </div>
                            <div className="hidden sm:block text-left leading-tight">
                                <div className="text-xs font-bold text-slate-200">{session.name}</div>
                                <div className="text-[10px] text-indigo-400 font-mono">{session.role || 'Clinician'}</div>
                            </div>
                            <button
                                onClick={onSignOut}
                                title="Sign Out"
                                className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-all ml-1"
                            >
                                <LogOut className="w-4 h-4" />
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
}
