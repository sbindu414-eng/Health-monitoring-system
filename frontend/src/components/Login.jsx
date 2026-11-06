import React, { useState } from 'react';
import { Activity, ShieldCheck, UserCheck, Stethoscope, Lock, Mail, ArrowRight, User } from 'lucide-react';

const DEMO_ACCOUNTS = [
    {
        name: "Dr. Bindu S",
        email: "doctor@healthml.io",
        username: "doctor",
        password: "doctor123",
        role: "Clinician / Doctor",
        avatar: "👨‍⚕️",
        badge: "Full Clinical CDS Access"
    },
    {
        name: "Prof. Reviewer (IEEE)",
        email: "evaluator@ieee.org",
        username: "evaluator",
        password: "evaluator123",
        role: "Research Evaluator",
        avatar: "🔬",
        badge: "Experimental Benchmarks"
    },
    {
        name: "Alex Morgan",
        email: "patient@healthml.io",
        username: "patient",
        password: "patient123",
        role: "Monitored Patient",
        avatar: "👤",
        badge: "Wearable & Diet Telemetry"
    },
    {
        name: "System Administrator",
        email: "admin@healthml.io",
        username: "admin",
        password: "admin123",
        role: "Administrator",
        avatar: "🛡️",
        badge: "Root Infrastructure"
    }
];

export default function Login({ onLoginSuccess }) {
    const [mode, setMode] = useState('login'); // 'login' | 'register'
    const [username, setUsername] = useState('doctor@healthml.io');
    const [password, setPassword] = useState('doctor123');
    const [name, setName] = useState('');
    const [role, setRole] = useState('doctor');
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');

    const handleLogin = (e) => {
        e?.preventDefault();
        setErrorMsg('');
        setLoading(true);

        setTimeout(() => {
            const matched = DEMO_ACCOUNTS.find(
                (u) =>
                    (u.email.toLowerCase() === username.trim().toLowerCase() ||
                        u.username.toLowerCase() === username.trim().toLowerCase()) &&
                    u.password === password
            );

            if (matched) {
                const session = {
                    name: matched.name,
                    email: matched.email,
                    role: matched.role,
                    avatar: matched.avatar,
                    loginTime: new Date().toISOString()
                };
                localStorage.setItem('healthml_session', JSON.stringify(session));
                onLoginSuccess(session);
            } else if (username && password.length >= 4) {
                const session = {
                    name: username.split('@')[0],
                    email: username,
                    role: role === 'doctor' ? 'Clinician' : role === 'evaluator' ? 'Evaluator' : 'Patient',
                    avatar: role === 'doctor' ? '👨‍⚕️' : '👤',
                    loginTime: new Date().toISOString()
                };
                localStorage.setItem('healthml_session', JSON.stringify(session));
                onLoginSuccess(session);
            } else {
                setErrorMsg('Invalid email or password. You can also click any Quick Demo Profile below.');
            }
            setLoading(false);
        }, 400);
    };

    const handleQuickLogin = (account) => {
        setUsername(account.email);
        setPassword(account.password);
        setLoading(true);
        setTimeout(() => {
            const session = {
                name: account.name,
                email: account.email,
                role: account.role,
                avatar: account.avatar,
                loginTime: new Date().toISOString()
            };
            localStorage.setItem('healthml_session', JSON.stringify(session));
            onLoginSuccess(session);
            setLoading(false);
        }, 300);
    };

    const handleGuestAccess = () => {
        const guestSession = {
            name: "Guest Evaluator",
            email: "guest@healthml.io",
            role: "Guest Access",
            avatar: "👤",
            loginTime: new Date().toISOString()
        };
        localStorage.setItem('healthml_session', JSON.stringify(guestSession));
        onLoginSuccess(guestSession);
    };

    return (
        <div className="min-h-screen bg-[#070a12] text-slate-100 flex flex-col justify-between relative overflow-hidden">
            {/* Glowing Orbs */}
            <div className="fixed top-0 left-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />
            <div className="fixed bottom-0 right-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl pointer-events-none" />
            <div className="fixed top-1/3 right-10 w-72 h-72 bg-emerald-600/10 rounded-full blur-3xl pointer-events-none" />

            {/* Header */}
            <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md px-6 py-4 flex items-center justify-between z-10">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                        <Activity className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-300 to-cyan-400">
                            Modular ML Health Monitoring Framework
                        </h1>
                        <p className="text-[11px] text-slate-400">Secure Clinical Decision &amp; Telemetry Access</p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" /> System Online
                    </span>
                </div>
            </header>

            {/* Main Container */}
            <main className="flex-1 flex items-center justify-center p-4 z-10 my-8">
                <div className="w-full max-w-md space-y-4">
                    {/* Auth Card */}
                    <div className="rounded-2xl p-6 sm:p-8 space-y-5 bg-slate-900/85 backdrop-blur-xl border border-slate-800 shadow-2xl">
                        <div className="text-center space-y-1.5">
                            <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 mb-1 shadow-inner">
                                <Lock className="w-6 h-6" />
                            </div>
                            <h2 className="text-xl font-bold text-slate-100">
                                {mode === 'login' ? 'Sign In to Dashboard' : 'Create New Account'}
                            </h2>
                            <p className="text-xs text-slate-400">
                                {mode === 'login'
                                    ? 'Enter credentials to access personalized patient & telemetry modules'
                                    : 'Register to experience the 5-module predictive pipeline'}
                            </p>
                        </div>

                        {/* Mode Switcher */}
                        <div className="grid grid-cols-2 p-1 bg-slate-950/90 rounded-xl border border-slate-800 text-xs font-semibold">
                            <button
                                type="button"
                                onClick={() => setMode('login')}
                                className={`py-2 rounded-lg transition-all ${
                                    mode === 'login' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                Sign In
                            </button>
                            <button
                                type="button"
                                onClick={() => setMode('register')}
                                className={`py-2 rounded-lg transition-all ${
                                    mode === 'register' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                Create Account
                            </button>
                        </div>

                        {errorMsg && (
                            <div className="p-3 rounded-xl text-xs flex items-center gap-2 border bg-rose-950/40 border-rose-500/40 text-rose-300">
                                <span>⚠️</span>
                                <span className="flex-1">{errorMsg}</span>
                            </div>
                        )}

                        {/* Form */}
                        <form onSubmit={handleLogin} className="space-y-4 text-xs">
                            {mode === 'register' && (
                                <div>
                                    <label className="block text-slate-300 mb-1 font-medium">Full Name</label>
                                    <input
                                        type="text"
                                        required
                                        placeholder="Dr. Alex Morgan"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-all"
                                    />
                                </div>
                            )}

                            <div>
                                <label className="block text-slate-300 mb-1 font-medium">Email or Username</label>
                                <input
                                    type="text"
                                    required
                                    placeholder="doctor@healthml.io"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-all font-mono"
                                />
                            </div>

                            <div>
                                <label className="block text-slate-300 mb-1 font-medium">Password</label>
                                <input
                                    type="password"
                                    required
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-all font-mono"
                                />
                            </div>

                            <div>
                                <label className="block text-slate-300 mb-1 font-medium">Clinical Role</label>
                                <select
                                    value={role}
                                    onChange={(e) => setRole(e.target.value)}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-all"
                                >
                                    <option value="doctor">Clinician / Doctor (Full CDS Access)</option>
                                    <option value="evaluator">Research Evaluator (Paper Benchmarks)</option>
                                    <option value="patient">Monitored Patient</option>
                                </select>
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                            >
                                {loading ? (
                                    <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                                ) : (
                                    <>
                                        <span>{mode === 'login' ? 'Sign In to Dashboard' : 'Register & Enter'}</span>
                                        <ArrowRight className="w-3.5 h-3.5" />
                                    </>
                                )}
                            </button>
                        </form>

                        {/* Guest Access Link */}
                        <div className="text-center pt-1">
                            <button
                                type="button"
                                onClick={handleGuestAccess}
                                className="text-xs text-indigo-400 hover:text-indigo-300 transition-all"
                            >
                                Continue as Guest Evaluator &rarr;
                            </button>
                        </div>
                    </div>

                    {/* Quick Demo Accounts */}
                    <div className="rounded-2xl p-4 bg-slate-900/60 border border-slate-800 space-y-2.5">
                        <span className="text-[11px] font-semibold text-slate-400 block text-center">
                            ⚡ Quick 1-Click Demo Profiles (Pre-configured)
                        </span>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                            {DEMO_ACCOUNTS.map((acc, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    onClick={() => handleQuickLogin(acc)}
                                    className="p-2 rounded-xl bg-slate-950/80 hover:bg-slate-800/80 border border-slate-800/80 text-left transition-all hover:scale-[1.02]"
                                >
                                    <div className="flex items-center gap-1.5 font-medium text-slate-200 text-[11px]">
                                        <span>{acc.avatar}</span>
                                        <span className="truncate">{acc.name}</span>
                                    </div>
                                    <span className="text-[10px] text-indigo-400 block truncate">{acc.role}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-900 bg-slate-950/80 py-3 text-center text-xs text-slate-500 z-10">
                A Modular Machine Learning Framework for Personalized Health Monitoring and Risk Prediction &copy; 2025-2026
            </footer>
        </div>
    );
}

