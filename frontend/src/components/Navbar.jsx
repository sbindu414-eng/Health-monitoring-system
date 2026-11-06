import React from 'react';
import { Activity, Zap, LogOut, User, ShieldCheck } from 'lucide-react';

export default function Navbar({ session, onSignOut, onLoadPreset, onRunFullScan, isScanning }) {
    return (
        <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
                {/* Brand */}
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                        <Activity className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-300 to-cyan-400">
                            Modular ML Health Monitoring Framework
                        </h1>
                        <p className="text-[11px] text-slate-400">IEEE 2025 • Clinical Decision & Risk Stratification</p>
                    </div>
                </div>

                {/* Actions & Profiles */}
                <div className="flex items-center gap-3">
                    {/* Patient Presets */}
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400 hidden lg:inline">Presets:</span>
                        <button
                            onClick={() => onLoadPreset('athlete')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-emerald-500/30 transition-all"
                            title="Load Healthy Young Athlete Preset"
                        >
                            🏃 Athlete
                        </button>
                        <button
                            onClick={() => onLoadPreset('hypertension')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-amber-300 border border-amber-500/30 transition-all"
                            title="Load Hypertensive Conduction Block Preset"
                        >
                            🩺 Hypertension
                        </button>
                        <button
                            onClick={() => onLoadPreset('cardiac')}
                            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 text-rose-300 border border-rose-500/30 transition-all"
                            title="Load Acute Cardiac Ischemia Preset"
                        >
                            🫀 Cardiac Risk
                        </button>

                        <button
                            onClick={onRunFullScan}
                            disabled={isScanning}
                            className="ml-1 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white shadow-md shadow-purple-500/25 flex items-center gap-1.5 transition-all disabled:opacity-50"
                        >
                            <Zap className="w-3.5 h-3.5" />
                            <span>{isScanning ? 'Scanning...' : 'Run Full Scan'}</span>
                        </button>
                    </div>

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

