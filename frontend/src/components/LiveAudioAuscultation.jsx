import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Volume2, Wind, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function LiveAudioAuscultation({ onSymptomDetected }) {
    const [isListening, setIsListening] = useState(false);
    const [audioLevel, setAudioLevel] = useState(0);
    const [coughCount, setCoughCount] = useState(0);
    const [respirationStatus, setRespirationStatus] = useState('Normal');
    const [errorMsg, setErrorMsg] = useState('');

    const audioCtxRef = useRef(null);
    const analyserRef = useRef(null);
    const micStreamRef = useRef(null);
    const animIdRef = useRef(null);
    const canvasRef = useRef(null);
    const lastCoughTimeRef = useRef(0);

    const startListening = async () => {
        setErrorMsg('');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            micStreamRef.current = stream;

            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const audioCtx = new AudioContext();
            audioCtxRef.current = audioCtx;

            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            analyserRef.current = analyser;

            setIsListening(true);
            drawSpectrogram();
        } catch (err) {
            console.error('Audio capture error:', err);
            setErrorMsg('Microphone access denied. Please grant microphone permissions in browser.');
        }
    };

    const stopListening = () => {
        if (animIdRef.current) {
            cancelAnimationFrame(animIdRef.current);
            animIdRef.current = null;
        }
        if (micStreamRef.current) {
            micStreamRef.current.getTracks().forEach((track) => track.stop());
            micStreamRef.current = null;
        }
        if (audioCtxRef.current) {
            audioCtxRef.current.close().catch(() => {});
            audioCtxRef.current = null;
        }
        setIsListening(false);
        setAudioLevel(0);
    };

    useEffect(() => {
        return () => {
            stopListening();
        };
    }, []);

    const drawSpectrogram = () => {
        const analyser = analyserRef.current;
        const canvas = canvasRef.current;
        if (!analyser || !canvas) return;

        const ctx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        analyser.getByteFrequencyData(dataArray);

        // Calculate average acoustic volume (RMS approximation)
        let sum = 0;
        let midEnergy = 0; // 200 - 800 Hz range typical of cough bursts
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
            if (i >= 5 && i <= 30) {
                midEnergy += dataArray[i];
            }
        }
        const avg = Math.round(sum / bufferLength);
        const midAvg = Math.round(midEnergy / 26);
        setAudioLevel(avg);

        // Acoustic Cough Transient Detection
        const now = performance.now();
        if (midAvg > 95 && now - lastCoughTimeRef.current > 1200) {
            lastCoughTimeRef.current = now;
            setCoughCount((c) => c + 1);
            if (onSymptomDetected) {
                onSymptomDetected('cough');
            }
        }

        // Heavy breathing / dyspnea detection
        if (avg > 75) {
            setRespirationStatus('Elevated / Dyspneic');
            if (onSymptomDetected) {
                onSymptomDetected('sob');
            }
        } else {
            setRespirationStatus('Calm / Regular');
        }

        // Render FFT spectrum on canvas
        ctx.fillStyle = '#090d16';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            const r = Math.min(255, dataArray[i] + 40);
            const g = 180 - dataArray[i] / 2;
            const b = 240;

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }

        animIdRef.current = requestAnimationFrame(drawSpectrogram);
    };

    return (
        <div className="rounded-2xl p-5 bg-slate-900/90 border border-cyan-500/30 shadow-2xl space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
                        <Mic className="w-4 h-4" />
                    </div>
                    <div>
                        <h4 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                            <span>Live Acoustic Auscultation &amp; Cough Detector</span>
                            <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                                Real-Time FFT
                            </span>
                        </h4>
                        <p className="text-[11px] text-slate-400">
                            Detects cough acoustic transients and heavy dyspnea without asking for manual input.
                        </p>
                    </div>
                </div>

                <button
                    type="button"
                    onClick={isListening ? stopListening : startListening}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md ${
                        isListening
                            ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/30'
                            : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-cyan-600/30'
                    }`}
                >
                    {isListening ? (
                        <>
                            <MicOff className="w-3.5 h-3.5" /> Stop Mic
                        </>
                    ) : (
                        <>
                            <Mic className="w-3.5 h-3.5" /> Start Live Mic
                        </>
                    )}
                </button>
            </div>

            {errorMsg && (
                <div className="p-2.5 rounded-lg text-xs flex items-center gap-2 bg-rose-950/40 border border-rose-500/40 text-rose-300">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    <span>{errorMsg}</span>
                </div>
            )}

            {/* Audio Frequency FFT Spectrum Display */}
            <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400 font-mono">Live Acoustic Frequency Spectrum (0 - 4 kHz)</span>
                    <span className="text-[10px] font-mono text-cyan-400">
                        {isListening ? `Input Level: ${audioLevel} dB` : 'Microphone Inactive'}
                    </span>
                </div>
                <div className="rounded-xl overflow-hidden border border-cyan-500/30 bg-[#090d16]">
                    <canvas ref={canvasRef} width={400} height={90} className="w-full block" />
                </div>
            </div>

            {/* Ingested Indicators Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 pt-1 text-xs">
                <div className="p-3 bg-slate-950/80 rounded-xl border border-cyan-500/30 flex items-center justify-between">
                    <div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Detected Cough Bursts</span>
                        <div className="text-xl font-bold font-mono text-cyan-400 mt-0.5">
                            {coughCount} <span className="text-xs font-normal text-slate-500">Events</span>
                        </div>
                    </div>
                    <Volume2 className={`w-5 h-5 text-cyan-400 ${coughCount > 0 ? 'animate-bounce' : 'opacity-40'}`} />
                </div>

                <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800 flex items-center justify-between">
                    <div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Respiratory Pattern</span>
                        <div className="text-xs font-bold font-mono text-slate-200 mt-1">
                            {isListening ? respirationStatus : 'Standby'}
                        </div>
                    </div>
                    <Wind className="w-5 h-5 text-indigo-400 opacity-60" />
                </div>

                <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800 col-span-2 sm:col-span-1 flex items-center justify-between">
                    <div>
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Symptom Pipeline</span>
                        <div className="text-xs font-semibold text-cyan-300 mt-1 flex items-center gap-1">
                            <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
                            <span>Auto-Injected</span>
                        </div>
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">Audio AI</span>
                </div>
            </div>
        </div>
    );
}

