import React from 'react';
import { Home, Salad, Stethoscope, HeartPulse, Dumbbell, ShieldAlert, BarChart3 } from 'lucide-react';

const tabs = [
    { id: 'overview', label: 'Overview & Flow', icon: Home, badge: 'Pipeline' },
    { id: 'diet', label: 'Diet Suggestion', icon: Salad, badge: 'RF 89.2%' },
    { id: 'disease', label: 'Disease Diagnosis', icon: Stethoscope, badge: 'LogReg 86.7%' },
    { id: 'ecg', label: 'ECG Classification', icon: HeartPulse, badge: 'Ensemble 93.5%' },
    { id: 'fitness', label: 'Fitness Estimation', icon: Dumbbell, badge: 'LinReg R² 0.82' },
    { id: 'risk', label: 'Integrated Risk', icon: ShieldAlert, badge: 'XGBoost 91.3%' },
    { id: 'benchmarks', label: 'Paper Benchmarks', icon: BarChart3, badge: 'Figures 2-9' },
];

export default function TabNavigation({ activeTab, onSelectTab }) {
    return (
        <nav className="max-w-7xl mx-auto px-4 flex overflow-x-auto text-xs font-medium border-b border-slate-800/80 bg-slate-950/40">
            {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                    <button
                        key={tab.id}
                        onClick={() => onSelectTab(tab.id)}
                        className={`px-4 py-3 flex items-center gap-2 whitespace-nowrap border-b-2 transition-all ${
                            isActive
                                ? 'bg-slate-900/90 text-white border-indigo-500 font-semibold shadow-inner'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40 border-transparent'
                        }`}
                    >
                        <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-slate-500'}`} />
                        <span>{tab.label}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                            isActive ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40' : 'bg-slate-800 text-slate-400'
                        }`}>
                            {tab.badge}
                        </span>
                    </button>
                );
            })}
        </nav>
    );
}

