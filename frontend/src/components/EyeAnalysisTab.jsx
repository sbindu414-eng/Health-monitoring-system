import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Eye,
    Upload,
    Camera,
    CameraOff,
    Aperture,
    Sparkles,
    AlertTriangle,
    CheckCircle2,
    XCircle,
    Loader2,
    BarChart3,
    Activity,
    ShieldAlert,
    ShieldCheck,
    Info,
    RotateCcw,
    Image as ImageIcon,
} from 'lucide-react';
import { api } from '../services/api';

const UI_STATUS_MESSAGES = {
    waiting_for_camera:   'Start your camera to begin.',
    waiting_for_eye:      'Waiting for eye. Please look directly into the camera.',
    eye_detected:         'Eye detected. Verifying that it is live.',
    verifying_liveness:   'Please follow the eye movement instructions.',
    spoof_detected:       'Possible screen or photograph detected. Analysis blocked.',
    verification_failed:  'Live human eye could not be verified. Please try again.',
    live_verified:        'Live eye verified. Starting analysis.',
    analysis_in_progress: 'Analysis in progress...',
    completed:            'Eye analysis completed successfully.',
};

const LIVENESS_STATUS_LABELS = {
    waiting_for_camera:   'WAITING_FOR_CAMERA',
    waiting_for_eye:      'WAITING_FOR_EYE',
    eye_detected:         'EYE_DETECTED',
    verifying_liveness:   'VERIFYING_LIVENESS',
    spoof_detected:       'PRESENTATION_ATTACK_DETECTED',
    live_verified:        'LIVE_EYE_VERIFIED',
    analysis_in_progress: 'ANALYSIS_RUNNING',
    completed:            'ANALYSIS_COMPLETED',
    verification_failed:  'VERIFICATION_FAILED',
};

const MSG_LIVENESS_FAILED = UI_STATUS_MESSAGES.verification_failed;
const MSG_SPOOF_DETECTED  = UI_STATUS_MESSAGES.spoof_detected;

export default function EyeAnalysisTab({ session }) {
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const captureCanvasRef = useRef(null);
    const streamRef = useRef(null);
    const sessionIdRef = useRef('');
    const verificationTokenRef = useRef('');
    const streamIntervalRef = useRef(null);
    const isCheckingRef = useRef(false);

    const [inputSource, setInputSource] = useState('upload'); // 'upload' | 'camera'
    const [cameraActive, setCameraActive] = useState(false);
    const [cameraReady, setCameraReady] = useState(false);
    const [cameraDevices, setCameraDevices] = useState([]);
    const [selectedDeviceId, setSelectedDeviceId] = useState('');

    // Live Eye Verification & Presentation Attack Detection (PAD) state
    const [livenessState, setLivenessState] = useState('waiting_for_camera');
    const [livenessFeedback, setLivenessFeedback] = useState(UI_STATUS_MESSAGES.waiting_for_camera);
    const [livenessTelemetry, setLivenessTelemetry] = useState(null);
    const [isLiveVerified, setIsLiveVerified] = useState(false);

    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedSample, setSelectedSample] = useState('');
    const [previewUrl, setPreviewUrl] = useState('');
    const [analysisMode, setAnalysisMode] = useState('auto');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');

    const [moduleStatus, setModuleStatus] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [showEvalPanel, setShowEvalPanel] = useState(false);

    const refreshCameraDevices = useCallback(async () => {
        if (!navigator.mediaDevices?.enumerateDevices) return;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cams = devices.filter((d) => d.kind === 'videoinput');
            setCameraDevices(cams);
            if (!selectedDeviceId && cams.length > 0 && cams[0].deviceId) {
                setSelectedDeviceId(cams[0].deviceId);
            }
        } catch {
            // Ignore enumerate permission errors before camera starts
        }
    }, [selectedDeviceId]);

    const loadStatus = async () => {
        try {
            const st = await api.getEyeStatus();
            setModuleStatus(st);
        } catch (err) {
            console.error('Eye module status error:', err);
        }
    };

    useEffect(() => {
        loadStatus();
    }, []);

    useEffect(() => {
        return () => {
            if (previewUrl && previewUrl.startsWith('blob:')) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

    const stopCamera = useCallback(() => {
        if (streamIntervalRef.current) {
            clearInterval(streamIntervalRef.current);
            streamIntervalRef.current = null;
        }
        if (sessionIdRef.current) {
            api.resetLivenessSession(sessionIdRef.current).catch(() => {});
            sessionIdRef.current = '';
        }
        verificationTokenRef.current = '';
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setCameraActive(false);
        setCameraReady(false);
        setLivenessState('waiting_for_camera');
        setLivenessFeedback(UI_STATUS_MESSAGES.waiting_for_camera);
        setLivenessTelemetry(null);
        setIsLiveVerified(false);
        isCheckingRef.current = false;
    }, []);

    useEffect(() => {
        return () => {
            stopCamera();
        };
    }, [stopCamera]);

    const runVerifiedLiveAnalysis = async (activeSessionId, activeToken) => {
        if (!activeSessionId || !activeToken) {
            setLivenessState('verification_failed');
            setLivenessFeedback(UI_STATUS_MESSAGES.verification_failed);
            setIsLiveVerified(false);
            return;
        }
        setLivenessState('analysis_in_progress');
        setLivenessFeedback(UI_STATUS_MESSAGES.analysis_in_progress);
        setIsAnalyzing(true);
        setErrorMsg('');
        try {
            const video = videoRef.current;
            const canvas = captureCanvasRef.current || document.createElement('canvas');
            const width = video?.videoWidth || 640;
            const height = video?.videoHeight || 480;
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            if (video) {
                ctx.drawImage(video, 0, 0, width, height);
            }

            const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
            if (!blob) throw new Error('Failed to capture verified frame from webcam');

            const file = new File([blob], `live_webcam_eye_${Date.now()}.png`, { type: 'image/png' });
            setSelectedFile(file);
            if (previewUrl && previewUrl.startsWith('blob:')) {
                URL.revokeObjectURL(previewUrl);
            }
            setPreviewUrl(URL.createObjectURL(file));

            // Consume single-use verification token immediately on client side too
            verificationTokenRef.current = '';

            const res = await api.analyzeEyeImage({
                file,
                mode: 'live_camera',
                sessionId: activeSessionId,
                verificationToken: activeToken,
                source: 'camera',
                userRole: session?.role || 'Clinician',
            });

            setAnalysisResult(res);
            if (res?.status === 'completed') {
                setLivenessState('completed');
                setLivenessFeedback(UI_STATUS_MESSAGES.completed);
                await loadStatus();
            } else if (res?.status === 'spoof_detected') {
                setLivenessState('spoof_detected');
                setLivenessFeedback(UI_STATUS_MESSAGES.spoof_detected);
            } else {
                setLivenessState('verification_failed');
                setLivenessFeedback(UI_STATUS_MESSAGES.verification_failed);
            }
        } catch (err) {
            setErrorMsg(err.message || 'Live eye analysis failed.');
            setLivenessState('verification_failed');
            setLivenessFeedback(UI_STATUS_MESSAGES.verification_failed);
        } finally {
            setIsAnalyzing(false);
            setIsLiveVerified(false);
            verificationTokenRef.current = '';
        }
    };

    const startLivenessStreamLoop = (activeSessionId, deviceIdUsed = selectedDeviceId) => {
        if (streamIntervalRef.current) {
            clearInterval(streamIntervalRef.current);
            streamIntervalRef.current = null;
        }

        const offscreenCanvas = document.createElement('canvas');
        const vW = videoRef.current?.videoWidth || 640;
        const vH = videoRef.current?.videoHeight || 480;
        offscreenCanvas.width = vW;
        offscreenCanvas.height = vH;
        const ctx = offscreenCanvas.getContext('2d');

        streamIntervalRef.current = setInterval(async () => {
            if (isCheckingRef.current || !videoRef.current || !sessionIdRef.current) return;
            if (sessionIdRef.current !== activeSessionId) return;
            if (videoRef.current.readyState < 2) return;

            try {
                isCheckingRef.current = true;
                ctx.drawImage(videoRef.current, 0, 0, vW, vH);
                const frameDataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.85);

                const res = await api.verifyLivenessStream({
                    sessionId: activeSessionId,
                    frameBase64: frameDataUrl,
                    cameraDeviceId: deviceIdUsed || null,
                });

                if (!res || sessionIdRef.current !== activeSessionId) return;

                setLivenessTelemetry(res);

                const padData = res.pad_analysis || res.pad_result;
                const livenessScore = Number(res.liveness_score_pct ?? res.confidence ?? 0);

                if (res.status === 'spoof_detected' || padData?.is_spoof) {
                    verificationTokenRef.current = '';
                    setLivenessState('spoof_detected');
                    setLivenessFeedback(UI_STATUS_MESSAGES.spoof_detected);
                    setIsLiveVerified(false);
                    setAnalysisResult(null);
                } else if (res.status === 'verification_failed') {
                    verificationTokenRef.current = '';
                    setLivenessState('verification_failed');
                    setLivenessFeedback(UI_STATUS_MESSAGES.verification_failed);
                    setIsLiveVerified(false);
                    setAnalysisResult(null);
                } else if (res.status === 'live_verified' && res.is_live && res.verificationToken && livenessScore >= 65) {
                    verificationTokenRef.current = res.verificationToken;
                    setLivenessState('live_verified');
                    setIsLiveVerified(true);
                    setLivenessFeedback(UI_STATUS_MESSAGES.live_verified);
                    if (streamIntervalRef.current) {
                        clearInterval(streamIntervalRef.current);
                        streamIntervalRef.current = null;
                    }
                    await runVerifiedLiveAnalysis(activeSessionId, res.verificationToken);
                } else if (res.status === 'eye_detected') {
                    verificationTokenRef.current = '';
                    setIsLiveVerified(false);
                    setLivenessState('eye_detected');
                    setLivenessFeedback(UI_STATUS_MESSAGES.eye_detected);
                } else if (res.status === 'verifying_liveness') {
                    verificationTokenRef.current = '';
                    setIsLiveVerified(false);
                    setLivenessState('verifying_liveness');
                    setLivenessFeedback(UI_STATUS_MESSAGES.verifying_liveness);
                } else if (res.status === 'waiting_for_eye') {
                    verificationTokenRef.current = '';
                    setIsLiveVerified(false);
                    setLivenessState('waiting_for_eye');
                    setLivenessFeedback(UI_STATUS_MESSAGES.waiting_for_eye);
                } else {
                    verificationTokenRef.current = '';
                    setIsLiveVerified(false);
                    setLivenessState('verification_failed');
                    setLivenessFeedback(UI_STATUS_MESSAGES.verification_failed);
                }
            } catch (err) {
                console.warn('Liveness verification stream frame error:', err);
            } finally {
                isCheckingRef.current = false;
            }
        }, 140);
    };

    const startCamera = async (overrideDeviceId = null) => {
        setErrorMsg('');
        setCameraReady(false);
        setAnalysisResult(null);
        verificationTokenRef.current = '';
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setErrorMsg('Laptop webcam API is unavailable in this browser.');
            return;
        }
        // Always stop any previous stream & reset previous session on camera restart or device switch
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
        }
        if (sessionIdRef.current) {
            api.resetLivenessSession(sessionIdRef.current).catch(() => {});
        }
        const devId = overrideDeviceId !== null ? overrideDeviceId : selectedDeviceId;
        try {
            const videoConstraint = devId
                ? { width: { ideal: 640 }, height: { ideal: 480 }, deviceId: { exact: devId } }
                : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' };
            const stream = await navigator.mediaDevices.getUserMedia({
                video: videoConstraint,
                audio: false,
            });
            streamRef.current = stream;
            await refreshCameraDevices();
            const newSessionId = `cam_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
            sessionIdRef.current = newSessionId;
            setLivenessState('waiting_for_eye');
            setLivenessFeedback(UI_STATUS_MESSAGES.waiting_for_eye);
            setLivenessTelemetry(null);
            setIsLiveVerified(false);
            setCameraActive(true);

            setTimeout(async () => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    await videoRef.current.play();
                    setCameraReady(true);
                    startLivenessStreamLoop(newSessionId, devId);
                }
            }, 80);
        } catch (err) {
            setErrorMsg(`Unable to access laptop camera: ${err.message || 'Permission denied'}`);
            stopCamera();
        }
    };

    const handleCameraDeviceChange = async (nextDeviceId) => {
        setSelectedDeviceId(nextDeviceId);
        if (cameraActive) {
            await startCamera(nextDeviceId);
        }
    };

    const handleResetVerification = async () => {
        if (streamIntervalRef.current) {
            clearInterval(streamIntervalRef.current);
            streamIntervalRef.current = null;
        }
        if (sessionIdRef.current) {
            await api.resetLivenessSession(sessionIdRef.current);
        }
        verificationTokenRef.current = '';
        const newSessionId = `cam_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
        sessionIdRef.current = newSessionId;
        setLivenessState(cameraActive ? 'waiting_for_eye' : 'waiting_for_camera');
        setLivenessFeedback(cameraActive ? UI_STATUS_MESSAGES.waiting_for_eye : UI_STATUS_MESSAGES.waiting_for_camera);
        setLivenessTelemetry(null);
        setIsLiveVerified(false);
        setAnalysisResult(null);
        setErrorMsg('');
        if (cameraActive && videoRef.current) {
            startLivenessStreamLoop(newSessionId, selectedDeviceId);
        }
    };

    const handleCaptureSnapshot = async () => {
        if (!isLiveVerified || !verificationTokenRef.current || livenessState !== 'live_verified') {
            setErrorMsg(UI_STATUS_MESSAGES.verification_failed);
            return;
        }
        if (sessionIdRef.current && verificationTokenRef.current) {
            await runVerifiedLiveAnalysis(sessionIdRef.current, verificationTokenRef.current);
        }
    };

    const handleSourceSwitch = (nextSource) => {
        setInputSource(nextSource);
        setErrorMsg('');
        if (nextSource === 'upload') {
            stopCamera();
        } else {
            refreshCameraDevices();
            setSelectedFile(null);
            setSelectedSample('');
            setPreviewUrl('');
            setAnalysisResult(null);
            setLivenessState('waiting_for_camera');
            setLivenessFeedback(UI_STATUS_MESSAGES.waiting_for_camera);
        }
    };

    const handleFileSelect = (file) => {
        if (!file) return;
        setErrorMsg('');
        setSelectedSample('');
        setSelectedFile(file);
        setAnalysisResult(null);
        if (previewUrl && previewUrl.startsWith('blob:')) {
            URL.revokeObjectURL(previewUrl);
        }
        setPreviewUrl(URL.createObjectURL(file));
    };

    const handleSelectSample = (sampleName) => {
        setErrorMsg('');
        setSelectedFile(null);
        setSelectedSample(sampleName);
        setAnalysisResult(null);
        setPreviewUrl(`/api/eye/samples/${encodeURIComponent(sampleName)}`);
    };

    const handleClearSelection = () => {
        setSelectedFile(null);
        setSelectedSample('');
        if (previewUrl && previewUrl.startsWith('blob:')) {
            URL.revokeObjectURL(previewUrl);
        }
        setPreviewUrl('');
        setAnalysisResult(null);
        setErrorMsg('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleAnalyzeImage = async () => {
        if (inputSource === 'camera') {
            setErrorMsg(UI_STATUS_MESSAGES.verification_failed);
            return;
        }
        if (!selectedFile && !selectedSample) {
            setErrorMsg('Please upload an eye image or choose a held-out test sample first.');
            return;
        }
        setErrorMsg('');
        setIsAnalyzing(true);
        try {
            const res = await api.analyzeEyeImage({
                file: selectedFile,
                sampleName: selectedSample,
                mode: analysisMode,
                source: 'upload',
                userRole: session?.role || 'Clinician',
            });
            setAnalysisResult(res);
            await loadStatus();
        } catch (err) {
            setErrorMsg(err.message || 'Analysis failed.');
        } finally {
            setIsAnalyzing(false);
            setIsLiveVerified(false);
            verificationTokenRef.current = '';
        }
    };

    const evalSummary = moduleStatus?.retinal_evaluation || {};
    const overallMetrics = evalSummary?.overall_metrics || {};
    const sensitivityMetrics = evalSummary?.screening_sensitivity_analysis || {};
    const externalEval = evalSummary?.external_cross_dataset_validation || null;
    const perSourceMetrics = evalSummary?.per_source_metrics || {};
    const modelComparison = moduleStatus?.retinal_comparison || null;
    const validatorEval = moduleStatus?.retinal_validator_evaluation || {};
    const validatorMetrics = validatorEval?.metrics || {};
    const samples = moduleStatus?.sample_images || [];

    const probEntries = analysisResult?.class_probabilities
        ? Object.entries(analysisResult.class_probabilities)
        : [];

    return (
        <div className="space-y-6">
            {/* Medical & Scientific Scope Banner */}
            <div className="rounded-2xl p-5 bg-gradient-to-r from-slate-900 via-sky-950/35 to-slate-900 border border-sky-500/30 shadow-xl space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono bg-sky-500/15 text-sky-300 border border-sky-500/30">
                            <Eye className="w-3.5 h-3.5" /> Module 06 • AI Eye Image Analysis
                        </span>
                        <span className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-slate-950/70 border border-slate-700 text-slate-300">
                            {moduleStatus?.retinal_model_trained
                                ? `CNN Ready (${moduleStatus.retinal_architecture}${moduleStatus.retinal_model_version ? ` v${moduleStatus.retinal_model_version}` : ''})`
                                : 'Checking Model Status...'}
                        </span>
                    </div>
                    <button
                        type="button"
                        onClick={() => setShowEvalPanel((s) => !s)}
                        className="px-3 py-1 rounded-lg bg-slate-950/80 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-sky-300 flex items-center gap-1.5 transition-all"
                    >
                        <BarChart3 className="w-3.5 h-3.5" />
                        <span>{showEvalPanel ? 'Hide Model Metrics & Plots' : 'View Real Test Evaluation Metrics'}</span>
                    </button>
                </div>

                <h2 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
                    AI Eye Image Analysis &amp; Diabetic Retinopathy Screening
                </h2>
                <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
                    Scientifically separates <strong>Retinal Fundus Screening</strong> (5-grade ICDR CNN trained across multi-center EyePACS, IDRiD, DeepDRiD/RetinaMNIST &amp; APTOS 2019 scans with Grad-CAM explainability, uncertainty flagging &amp; zero-shot Messidor-2 external benchmark), <strong>Anterior Eye / Iris CV Segmentation</strong> (experimental OpenCV pupil/iris segmentation &amp; quality gating), and <strong>Clinical Tabular Diabetes Risk</strong> (separate Pima Indians Diabetes model).
                </p>

                <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-500/40 text-amber-200 text-xs flex items-start gap-2.5">
                    <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                        <strong className="text-amber-300">Scientific &amp; Clinical Boundary:</strong>{' '}
                        An ordinary external iris photograph <strong>cannot</strong> measure blood glucose (mg/dL) or diagnose diabetes. Retinal fundus scans screen for diabetic retinopathy lesions, while tabular diabetes risk requires clinical lab inputs. For blood sugar measurement, always use a validated blood glucose testing device.
                    </div>
                </div>
            </div>

            {/* Optional Expandable Held-Out Test Evaluation Panel */}
            {showEvalPanel && evalSummary?.overall_metrics && (
                <div className="rounded-2xl p-5 bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                        <div>
                            <h3 className="text-sm font-bold text-sky-300 flex items-center gap-2">
                                <BarChart3 className="w-4 h-4" /> Held-Out Multi-Center Test Evaluation (Unseen Test Set, n={evalSummary.test_samples})
                            </h3>
                            <p className="text-[11px] text-slate-400 mt-0.5">
                                Backbone: {evalSummary.backbone_architecture} (Best Epoch {evalSummary.best_epoch}) • Deduplicated EyePACS + IDRiD + DeepDRiD + APTOS 2019 test split
                            </p>
                        </div>
                        <span className="text-[10px] font-mono px-2.5 py-1 rounded bg-slate-950 border border-slate-700 text-slate-300">
                            Academic Screening Benchmark (Not Clinical Validation)
                        </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-center">
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">QWK (Kappa)</div>
                            <div className="text-lg font-bold font-mono text-cyan-300 mt-0.5">
                                {overallMetrics.quadratic_weighted_kappa !== undefined ? overallMetrics.quadratic_weighted_kappa.toFixed(3) : 'N/A'}
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Accuracy</div>
                            <div className="text-lg font-bold font-mono text-sky-300 mt-0.5">
                                {(overallMetrics.accuracy * 100).toFixed(2)}%
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Macro F1</div>
                            <div className="text-lg font-bold font-mono text-emerald-300 mt-0.5">
                                {(overallMetrics.f1_macro * 100).toFixed(2)}%
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">OvR ROC-AUC</div>
                            <div className="text-lg font-bold font-mono text-purple-300 mt-0.5">
                                {overallMetrics.roc_auc_ovr_macro?.toFixed(3) ?? 'N/A'}
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Any-DR Sens.</div>
                            <div className="text-lg font-bold font-mono text-amber-300 mt-0.5">
                                {((sensitivityMetrics.any_dr_detection?.sensitivity_recall || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Any-DR Spec.</div>
                            <div className="text-lg font-bold font-mono text-teal-300 mt-0.5">
                                {((sensitivityMetrics.any_dr_detection?.specificity || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Referable Sens.</div>
                            <div className="text-lg font-bold font-mono text-rose-300 mt-0.5">
                                {((sensitivityMetrics.referable_dr_detection_grade_2plus?.sensitivity_recall || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>

                    {/* Per-Dataset Breakdown & External Messidor-2 Benchmark */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {Object.keys(perSourceMetrics).length > 0 && (
                            <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-3.5 space-y-2">
                                <div className="text-xs font-bold text-slate-200">Test Breakdown by Camera / Dataset Source</div>
                                <div className="space-y-1.5 text-[11px] font-mono">
                                    {Object.entries(perSourceMetrics).map(([srcName, m]) => (
                                        <div key={srcName} className="flex flex-wrap justify-between items-center gap-2 py-1 border-b border-slate-800/70 last:border-b-0">
                                            <span className="text-sky-300">{srcName} (n={m.samples})</span>
                                            <span className="text-slate-300">
                                                QWK: {m.qwk?.toFixed(3)} · Acc: {(m.accuracy * 100).toFixed(1)}% · Sens/Spec: {(m.any_dr_sensitivity * 100).toFixed(0)}%/{(m.any_dr_specificity * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {externalEval?.overall_metrics && (
                            <div className="bg-slate-950/90 border border-cyan-500/30 rounded-xl p-3.5 space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-cyan-300">
                                        External Cross-Dataset Validation: Messidor-2 (n={externalEval.test_samples})
                                    </span>
                                    <span className="text-[10px] font-mono text-cyan-400">Zero-Shot Unseen Cohort</span>
                                </div>
                                <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 text-center text-[11px] font-mono pt-1">
                                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                        <div className="text-[9px] text-slate-400">QWK</div>
                                        <div className="text-cyan-300 font-bold">{externalEval.overall_metrics.quadratic_weighted_kappa?.toFixed(3)}</div>
                                    </div>
                                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                        <div className="text-[9px] text-slate-400">Acc</div>
                                        <div className="text-sky-300 font-bold">{(externalEval.overall_metrics.accuracy * 100).toFixed(1)}%</div>
                                    </div>
                                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                        <div className="text-[9px] text-slate-400">Macro F1</div>
                                        <div className="text-emerald-300 font-bold">{(externalEval.overall_metrics.f1_macro * 100).toFixed(1)}%</div>
                                    </div>
                                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                        <div className="text-[9px] text-slate-400">Any-DR Sens</div>
                                        <div className="text-amber-300 font-bold">{((externalEval.screening_sensitivity_analysis?.any_dr_detection?.sensitivity_recall || 0) * 100).toFixed(1)}%</div>
                                    </div>
                                    <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                        <div className="text-[9px] text-slate-400">Any-DR Spec</div>
                                        <div className="text-teal-300 font-bold">{((externalEval.screening_sensitivity_analysis?.any_dr_detection?.specificity || 0) * 100).toFixed(1)}%</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {modelComparison?.candidates?.length > 0 && (
                        <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-3.5 space-y-2">
                            <div className="text-xs font-bold text-slate-200">
                                Validation-Driven Architecture Selection (Winner: <span className="text-emerald-300 font-mono">{modelComparison.selected_winner}</span>)
                            </div>
                            <div className="flex flex-wrap gap-3 text-[11px] font-mono">
                                {modelComparison.candidates.map((cand) => (
                                    <div
                                        key={cand.arch}
                                        className={`px-3 py-1.5 rounded-lg border ${
                                            cand.arch === modelComparison.selected_winner
                                                ? 'border-emerald-500/50 bg-emerald-950/30 text-emerald-200'
                                                : 'border-slate-800 bg-slate-900 text-slate-300'
                                        }`}
                                    >
                                        <span className="font-bold">{cand.arch}</span> (ep {cand.best_epoch}): Val QWK={cand.val_qwk?.toFixed(3)} · F1={(cand.val_macro_f1 * 100).toFixed(1)}% · Acc={(cand.val_accuracy * 100).toFixed(1)}%
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-2 text-center space-y-1">
                            <img src="/api/eye/plots/confusion_matrix.png" alt="Confusion Matrix" className="w-full rounded-lg" />
                            <div className="text-[11px] text-slate-400 font-mono">Confusion Matrix (Test Split)</div>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-2 text-center space-y-1">
                            <img src="/api/eye/plots/roc_curves.png" alt="ROC Curves" className="w-full rounded-lg" />
                            <div className="text-[11px] text-slate-400 font-mono">One-vs-Rest ROC Curves</div>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-2 text-center space-y-1">
                            <img src="/api/eye/plots/training_history.png" alt="Training Curves" className="w-full rounded-lg" />
                            <div className="text-[11px] text-slate-400 font-mono">Training &amp; Validation Loss/F1/QWK</div>
                        </div>
                    </div>

                    {/* Retinal Fundus Input Validator Evaluation Summary */}
                    {validatorEval?.metrics && (
                        <div className="bg-slate-950/90 border border-emerald-500/30 rounded-xl p-3.5 space-y-2.5">
                            <div className="flex flex-wrap justify-between items-center gap-2">
                                <span className="text-xs font-bold text-emerald-300">
                                    Human Retinal Fundus Image Validator ({validatorEval.architecture} • Unseen Test Split, n={validatorEval.test_samples})
                                </span>
                                <span className="text-[10px] font-mono text-slate-400">
                                    Accept Cutoff ≥ {((validatorEval.valid_threshold || 0.8) * 100).toFixed(1)}% • Abstain Band {((validatorEval.uncertain_low || 0.35) * 100).toFixed(0)}–{((validatorEval.valid_threshold || 0.8) * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-center text-[11px] font-mono">
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">Accuracy</div>
                                    <div className="text-emerald-300 font-bold">{(validatorMetrics.accuracy * 100).toFixed(2)}%</div>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">Precision</div>
                                    <div className="text-sky-300 font-bold">{(validatorMetrics.precision * 100).toFixed(2)}%</div>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">Recall</div>
                                    <div className="text-cyan-300 font-bold">{(validatorMetrics.recall * 100).toFixed(2)}%</div>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">F1 Score</div>
                                    <div className="text-teal-300 font-bold">{(validatorMetrics.f1_score * 100).toFixed(2)}%</div>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">ROC-AUC</div>
                                    <div className="text-purple-300 font-bold">{validatorMetrics.roc_auc?.toFixed(4)}</div>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">False Accept (FAR)</div>
                                    <div className="text-rose-300 font-bold">{(validatorMetrics.false_acceptance_rate * 100).toFixed(2)}%</div>
                                </div>
                                <div className="p-2 rounded bg-slate-900 border border-slate-800">
                                    <div className="text-[9px] text-slate-400">False Reject (FRR)</div>
                                    <div className="text-amber-300 font-bold">{(validatorMetrics.false_rejection_rate * 100).toFixed(2)}%</div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Main Eye Upload & Prediction Workspace */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Upload / Camera Capture, Modality Mode, Preview & Sample Loader */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                    <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                        <h3 className="text-sm font-bold text-sky-400 flex items-center gap-2">
                            {inputSource === 'camera' ? <Camera className="w-4 h-4" /> : <Upload className="w-4 h-4" />}
                            <span>{inputSource === 'camera' ? 'Laptop Camera Capture' : 'Eye Image Upload'}</span>
                        </h3>
                        <span className="text-[10px] px-2 py-0.5 rounded bg-sky-500/10 text-sky-300 font-mono">
                            Max {moduleStatus?.max_upload_mb || 10} MB
                        </span>
                    </div>

                    {/* Source Toggle: Keep BOTH Upload Image & Laptop Camera Capture */}
                    <div className="grid grid-cols-2 gap-1.5 p-1 bg-slate-950 rounded-xl border border-slate-800 text-xs">
                        <button
                            type="button"
                            onClick={() => handleSourceSwitch('upload')}
                            className={`py-1.5 px-2 rounded-lg font-medium flex items-center justify-center gap-1.5 transition-all ${
                                inputSource === 'upload'
                                    ? 'bg-sky-600 text-white shadow-sm'
                                    : 'text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            <Upload className="w-3.5 h-3.5" />
                            <span>Upload Image</span>
                        </button>
                        <button
                            type="button"
                            onClick={() => handleSourceSwitch('camera')}
                            className={`py-1.5 px-2 rounded-lg font-medium flex items-center justify-center gap-1.5 transition-all ${
                                inputSource === 'camera'
                                    ? 'bg-sky-600 text-white shadow-sm'
                                    : 'text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            <Camera className="w-3.5 h-3.5" />
                            <span>Laptop Camera</span>
                        </button>
                    </div>

                    {/* Analysis Mode Selector */}
                    <div className="space-y-1.5 text-xs">
                        <label className="block text-slate-400">Analysis Pipeline Mode</label>
                        <select
                            value={analysisMode}
                            onChange={(e) => setAnalysisMode(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-2 text-slate-200 text-xs focus:outline-none focus:border-sky-500 transition-all"
                        >
                            <option value="auto">Auto-Detect Modality (Recommended)</option>
                            <option value="retinal_fundus">Retinal Fundus Photograph (DR Grading CNN)</option>
                            <option value="anterior_iris">Anterior Eye / Iris Photo (OpenCV CV Pipeline)</option>
                        </select>
                    </div>

                    {inputSource === 'upload' ? (
                        <>
                            {/* Upload Dropzone */}
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="border-2 border-dashed border-slate-700 hover:border-sky-500/70 bg-slate-950/70 rounded-xl p-4 text-center cursor-pointer transition-all space-y-2"
                            >
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept=".png,.jpg,.jpeg,.bmp,.webp,.tif,.tiff"
                                    className="hidden"
                                    onChange={(e) => handleFileSelect(e.target.files?.[0])}
                                />
                                <ImageIcon className="w-7 h-7 text-sky-400 mx-auto opacity-80" />
                                <div className="text-xs font-medium text-slate-200">
                                    {selectedFile
                                        ? selectedFile.name
                                        : selectedSample
                                        ? `Test Sample: ${selectedSample}`
                                        : 'Click to upload retinal fundus or anterior eye image'}
                                </div>
                                <div className="text-[10px] text-slate-500 font-mono">
                                    Supports PNG, JPG, JPEG, WEBP, BMP, TIFF
                                </div>
                            </div>

                            {/* Quick Held-Out Test Samples */}
                            {samples.length > 0 && (
                                <div className="space-y-1.5">
                                    <div className="text-[11px] text-slate-400 flex items-center justify-between">
                                        <span>Or load unseen held-out test fundus sample:</span>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                        {samples.map((sName) => {
                                            const labelHint = sName.replace('sample_test_', '').replace('.png', '').replace(/_/g, ' ');
                                            const active = selectedSample === sName;
                                            return (
                                                <button
                                                    key={sName}
                                                    type="button"
                                                    onClick={() => handleSelectSample(sName)}
                                                    className={`px-2 py-1 rounded-md text-[10px] font-mono border transition-all ${
                                                        active
                                                            ? 'bg-sky-500/20 border-sky-400 text-sky-200'
                                                            : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                                                    }`}
                                                >
                                                    {labelHint}
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </>
                    ) : (
                        /* Laptop Camera Live Verification & Analysis Box */
                        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 space-y-3">
                            {cameraDevices.length > 1 && (
                                <div className="flex items-center justify-between gap-2 text-[11px]">
                                    <span className="text-slate-400">Webcam Device</span>
                                    <select
                                        value={selectedDeviceId}
                                        onChange={(e) => handleCameraDeviceChange(e.target.value)}
                                        className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-slate-200 text-[11px] max-w-[190px]"
                                    >
                                        {cameraDevices.map((cam, idx) => (
                                            <option key={cam.deviceId || idx} value={cam.deviceId}>
                                                {cam.label || `Camera ${idx + 1}`}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            )}

                            <div className="relative rounded-lg overflow-hidden bg-black aspect-video flex items-center justify-center border border-slate-800">
                                <video
                                    ref={videoRef}
                                    playsInline
                                    muted
                                    className={`w-full h-full object-cover ${cameraActive ? 'block' : 'hidden'}`}
                                />
                                {!cameraActive && (
                                    <div className="text-center p-4 space-y-1.5 text-slate-400">
                                        <Camera className="w-7 h-7 mx-auto text-sky-400 opacity-70" />
                                        <div className="text-xs font-medium text-slate-200">Start your camera to begin.</div>
                                        <p className="text-[10px] text-slate-500">
                                            Captures a live human eye directly in front of the webcam with multi-frame liveness, challenge-response &amp; PAD verification.
                                        </p>
                                    </div>
                                )}
                                {cameraActive && (
                                    <>
                                        {/* Dynamic Target HUD Box */}
                                        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                                            <div
                                                className={`w-36 h-24 rounded-full border-2 border-dashed flex items-center justify-center transition-all ${
                                                    livenessState === 'live_verified' || livenessState === 'completed'
                                                        ? 'border-emerald-400/90 shadow-lg shadow-emerald-500/20'
                                                        : livenessState === 'spoof_detected' || livenessState === 'verification_failed'
                                                        ? 'border-rose-400/90 shadow-lg shadow-rose-500/20'
                                                        : livenessState === 'eye_detected'
                                                        ? 'border-sky-400/90 shadow-lg shadow-sky-500/20'
                                                        : 'border-slate-500/60'
                                                }`}
                                            >
                                                <span
                                                    className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                                                        livenessState === 'live_verified' || livenessState === 'completed'
                                                            ? 'text-emerald-300 bg-slate-950/90'
                                                            : livenessState === 'spoof_detected' || livenessState === 'verification_failed'
                                                            ? 'text-rose-300 bg-slate-950/90'
                                                            : 'text-sky-200 bg-slate-950/80'
                                                    }`}
                                                >
                                                    {livenessState === 'live_verified'
                                                        ? '✓ Live Verified'
                                                        : livenessState === 'completed'
                                                        ? '✓ Analysis Completed'
                                                        : livenessState === 'spoof_detected'
                                                        ? '⚠ Spoof Blocked'
                                                        : livenessState === 'verification_failed'
                                                        ? '✕ Verification Failed'
                                                        : 'Center Real Eye Here'}
                                                </span>
                                            </div>
                                        </div>

                                        {/* Status Pill in Video Corner */}
                                        <div className="absolute top-2 left-2 z-10">
                                            <span
                                                className={`px-2 py-0.5 rounded-md text-[10px] font-mono font-bold flex items-center gap-1 shadow-md border ${
                                                    livenessState === 'live_verified' || livenessState === 'completed'
                                                        ? 'bg-emerald-950/90 text-emerald-300 border-emerald-500/50'
                                                        : livenessState === 'spoof_detected' || livenessState === 'verification_failed'
                                                        ? 'bg-rose-950/90 text-rose-300 border-rose-500/60 animate-pulse'
                                                        : livenessState === 'eye_detected'
                                                        ? 'bg-sky-950/90 text-sky-300 border-sky-500/50'
                                                        : livenessState === 'verifying_liveness' || livenessState === 'analysis_in_progress'
                                                        ? 'bg-indigo-950/90 text-indigo-300 border-indigo-500/50'
                                                        : 'bg-amber-950/90 text-amber-300 border-amber-500/50'
                                                }`}
                                            >
                                                {livenessState === 'verifying_liveness' || livenessState === 'analysis_in_progress' ? (
                                                    <Loader2 className="w-2.5 h-2.5 animate-spin" />
                                                ) : livenessState === 'live_verified' || livenessState === 'completed' ? (
                                                    <CheckCircle2 className="w-2.5 h-2.5" />
                                                ) : livenessState === 'spoof_detected' || livenessState === 'verification_failed' ? (
                                                    <ShieldAlert className="w-2.5 h-2.5" />
                                                ) : (
                                                    <Eye className="w-2.5 h-2.5" />
                                                )}
                                                <span>{LIVENESS_STATUS_LABELS[livenessState] || livenessState}</span>
                                            </span>
                                        </div>
                                    </>
                                )}
                                <canvas ref={captureCanvasRef} className="hidden" />
                            </div>

                            {/* Camera Action Buttons */}
                            <div className="flex gap-2">
                                {!cameraActive ? (
                                    <button
                                        type="button"
                                        onClick={() => startCamera()}
                                        className="flex-1 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-all shadow-md shadow-emerald-700/20"
                                    >
                                        <Camera className="w-3.5 h-3.5" /> Start Camera
                                    </button>
                                ) : (
                                    <>
                                        <button
                                            type="button"
                                            onClick={handleCaptureSnapshot}
                                            disabled={!isLiveVerified || livenessState !== 'live_verified'}
                                            title={isLiveVerified ? 'Capture verified live eye snapshot' : 'Verification required before capture'}
                                            className="flex-1 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                        >
                                            <Aperture className="w-3.5 h-3.5" /> Capture Snapshot
                                        </button>
                                        <button
                                            type="button"
                                            onClick={handleResetVerification}
                                            title="Reset verification state and start fresh scan"
                                            className="px-2.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center justify-center gap-1 transition-all border border-slate-700"
                                        >
                                            <RotateCcw className="w-3.5 h-3.5" /> Reset
                                        </button>
                                        <button
                                            type="button"
                                            onClick={stopCamera}
                                            className="px-3 py-2 rounded-lg bg-rose-600/80 hover:bg-rose-500 text-white text-xs font-semibold flex items-center justify-center gap-1 transition-all"
                                        >
                                            <CameraOff className="w-3.5 h-3.5" /> Stop
                                        </button>
                                    </>
                                )}
                            </div>

                            {/* Live Liveness & Presentation Attack Detection HUD Details */}
                            <div className="space-y-2 pt-1">
                                {/* Real-time Status Card */}
                                <div className="flex items-center justify-between text-[11px] font-mono px-1">
                                    <span className="text-slate-400">Live Verification:</span>
                                    <span className="font-semibold text-slate-200">
                                        {LIVENESS_STATUS_LABELS[livenessState] || livenessState}
                                    </span>
                                </div>

                                {/* Specific Required State Messages & Guidance */}
                                {!cameraActive && (
                                    <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300 text-xs flex items-start gap-2">
                                        <Info className="w-4 h-4 text-sky-400 flex-shrink-0 mt-0.5" />
                                        <span>{UI_STATUS_MESSAGES.waiting_for_camera}</span>
                                    </div>
                                )}

                                {cameraActive && livenessState === 'spoof_detected' && (
                                    <div className="p-2.5 rounded-lg bg-rose-950/60 border border-rose-500 text-rose-200 text-xs flex items-start gap-2 shadow-lg">
                                        <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                                        <div>
                                            <strong className="text-rose-300 block">Presentation Attack Detected:</strong>
                                            <span>{UI_STATUS_MESSAGES.spoof_detected}</span>
                                        </div>
                                    </div>
                                )}

                                {cameraActive && livenessState === 'verification_failed' && (
                                    <div className="p-2.5 rounded-lg bg-amber-950/60 border border-amber-500 text-amber-200 text-xs flex items-start gap-2 shadow-lg">
                                        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                                        <div>
                                            <strong className="text-amber-300 block">Verification Failed:</strong>
                                            <span>{UI_STATUS_MESSAGES.verification_failed}</span>
                                        </div>
                                    </div>
                                )}

                                {cameraActive && livenessState === 'live_verified' && (
                                    <div className="p-2.5 rounded-lg bg-emerald-950/50 border border-emerald-500/60 text-emerald-200 text-xs flex items-start gap-2 shadow-lg">
                                        <ShieldCheck className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                                        <div>
                                            <strong className="text-emerald-300 block">Verified:</strong>
                                            <span>{UI_STATUS_MESSAGES.live_verified}</span>
                                        </div>
                                    </div>
                                )}

                                {cameraActive && livenessState === 'completed' && (
                                    <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/50 text-emerald-200 text-xs flex items-start gap-2">
                                        <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                                        <span>{UI_STATUS_MESSAGES.completed}</span>
                                    </div>
                                )}

                                {cameraActive && (livenessState === 'verifying_liveness' || livenessState === 'eye_detected' || livenessState === 'waiting_for_eye' || livenessState === 'analysis_in_progress') && (
                                    <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300 text-xs flex items-start gap-2">
                                        <Info className="w-4 h-4 text-sky-400 flex-shrink-0 mt-0.5" />
                                        <div className="space-y-1">
                                            <span className="block">
                                                {livenessFeedback || UI_STATUS_MESSAGES[livenessState]}
                                            </span>
                                            {livenessState === 'verifying_liveness' && livenessTelemetry?.challenge && !livenessTelemetry.challenge.passed && (
                                                <div className="p-2 rounded bg-sky-950/60 border border-sky-500/40 text-sky-200 text-[11px]">
                                                    <strong className="text-sky-300 block font-mono">
                                                        Challenge Step {Math.min((livenessTelemetry.challenge.step_index ?? 0) + 1, livenessTelemetry.challenge.total_steps || 4)} of {livenessTelemetry.challenge.total_steps || 4}:
                                                    </strong>
                                                    <span className="font-semibold text-white">{livenessTelemetry.challenge.instruction}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* Telemetry Metric Badges (No hardcoded 100% fallback!) */}
                                {cameraActive && livenessTelemetry && (
                                    <div className="grid grid-cols-3 gap-1.5 p-2 bg-slate-950 rounded-lg border border-slate-800/80 text-[10px] font-mono text-center">
                                        <div>
                                            <span className="text-slate-500 block">Blinks</span>
                                            <span className="text-sky-300 font-bold">{livenessTelemetry.blink_count ?? 0}</span>
                                        </div>
                                        <div>
                                            <span className="text-slate-500 block">PAD Genuine</span>
                                            <span
                                                className={`font-bold ${
                                                    livenessTelemetry.pad_analysis?.is_spoof
                                                        ? 'text-rose-400'
                                                        : livenessTelemetry.pad_analysis?.pad_confidence_pct !== undefined
                                                        ? 'text-emerald-300'
                                                        : 'text-slate-400'
                                                }`}
                                            >
                                                {livenessTelemetry.pad_analysis?.pad_confidence_pct !== undefined
                                                    ? `${livenessTelemetry.pad_analysis.pad_confidence_pct}%${livenessTelemetry.pad_analysis.is_spoof ? ' (Spoof)' : ''}`
                                                    : '--'}
                                            </span>
                                        </div>
                                        <div>
                                            <span className="text-slate-500 block">Liveness</span>
                                            <span
                                                className={`font-bold ${
                                                    livenessTelemetry.is_live
                                                        ? 'text-emerald-300'
                                                        : Number(livenessTelemetry.liveness_score_pct ?? livenessTelemetry.confidence ?? 0) <= 0
                                                        ? 'text-rose-400'
                                                        : 'text-amber-300'
                                                }`}
                                            >
                                                {livenessTelemetry.liveness_score_pct !== undefined
                                                    ? `${livenessTelemetry.liveness_score_pct}%`
                                                    : livenessTelemetry.confidence !== undefined
                                                    ? `${livenessTelemetry.confidence}%`
                                                    : '0%'}
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Preview Box */}
                    {previewUrl && (
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-2 space-y-2">
                            <div className="flex justify-between items-center text-[10px] text-slate-400 px-1">
                                <span>Selected Image Preview</span>
                                <button
                                    type="button"
                                    onClick={handleClearSelection}
                                    className="text-rose-400 hover:text-rose-300 font-mono"
                                >
                                    Clear
                                </button>
                            </div>
                            <img
                                src={previewUrl}
                                alt="Eye Preview"
                                className="w-full max-h-52 object-contain rounded-lg bg-black"
                            />
                        </div>
                    )}

                    {errorMsg && (
                        <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/40 text-rose-300 text-xs flex items-start gap-2">
                            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                            <span>{errorMsg}</span>
                        </div>
                    )}

                    {inputSource === 'upload' ? (
                        <button
                            type="button"
                            onClick={handleAnalyzeImage}
                            disabled={isAnalyzing || (!selectedFile && !selectedSample)}
                            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-semibold text-xs shadow-lg shadow-sky-600/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                        >
                            {isAnalyzing ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Analysing Eye Image...</span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-4 h-4" />
                                    <span>Analyse Image</span>
                                </>
                            )}
                        </button>
                    ) : (
                        <div className="text-[11px] font-mono text-center text-slate-400 bg-slate-950/70 border border-slate-800 rounded-xl py-2 px-3">
                            Live camera analysis unlocks automatically after PAD &amp; challenge-response pass.
                        </div>
                    )}
                </div>

                {/* Right 2 Columns: Analysis Results, Quality Status, Class Probabilities, Iris CV */}
                <div className="lg:col-span-2 rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                    <div className="flex flex-wrap justify-between items-center gap-2 border-b border-slate-800 pb-3">
                        <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                            <Activity className="w-4 h-4 text-sky-400" /> Diagnostic &amp; Image Quality Output
                        </h3>
                        {analysisResult?.quality && (
                            <span
                                className={`px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold flex items-center gap-1 border ${
                                    analysisResult.quality.is_acceptable
                                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                                        : 'bg-rose-500/15 text-rose-300 border-rose-500/40'
                                }`}
                            >
                                {analysisResult.quality.is_acceptable ? (
                                    <CheckCircle2 className="w-3.5 h-3.5" />
                                ) : (
                                    <XCircle className="w-3.5 h-3.5" />
                                )}
                                <span>
                                    Quality: {analysisResult.quality.status.toUpperCase()} ({analysisResult.quality.quality_score}/100)
                                </span>
                            </span>
                        )}
                    </div>

                    {isAnalyzing && (
                        <div className="py-16 text-center space-y-3">
                            <Loader2 className="w-10 h-10 text-sky-400 animate-spin mx-auto" />
                            <div className="text-sm font-semibold text-slate-200">
                                Running image validation, quality checks &amp; ML inference...
                            </div>
                            <p className="text-xs text-slate-400">
                                Checking anatomy modality, blur/exposure thresholds, and model predictions.
                            </p>
                        </div>
                    )}

                    {!isAnalyzing && !analysisResult && (
                        <div className="py-16 text-center space-y-2 text-slate-400">
                            <Eye className="w-10 h-10 mx-auto text-slate-600" />
                            <div className="text-sm font-medium text-slate-300">
                                Upload a retinal fundus or anterior eye image and click &ldquo;Analyse Image&rdquo;
                            </div>
                            <p className="text-xs text-slate-500 max-w-md mx-auto">
                                Retinal fundus scans are classified into 5 ICDR diabetic retinopathy grades; anterior eye photos undergo OpenCV iris segmentation &amp; quality inspection.
                            </p>
                        </div>
                    )}

                    {!isAnalyzing && analysisResult && (
                        <div className="space-y-4">

                            {/* Binary / Uncertainty Result Card — Strictly for Retinal Fundus Scans ONLY */}
                            {analysisResult.status === 'completed' && analysisResult.detected_modality === 'retinal_fundus' && analysisResult.binary_label && (
                                <div
                                    className={`rounded-2xl p-5 border-2 text-center shadow-lg ${
                                        analysisResult.is_uncertain
                                            ? 'bg-yellow-950/35 border-yellow-400/70 shadow-yellow-900/25'
                                            : analysisResult.diabetes_detected
                                                ? 'bg-amber-950/40 border-amber-400/70 shadow-amber-900/30'
                                                : 'bg-emerald-950/40 border-emerald-400/70 shadow-emerald-900/30'
                                    }`}
                                >
                                    <div className={`text-2xl sm:text-3xl font-black tracking-tight mb-1 ${
                                        analysisResult.is_uncertain
                                            ? 'text-yellow-300'
                                            : analysisResult.diabetes_detected
                                                ? 'text-amber-300'
                                                : 'text-emerald-300'
                                    }`}>
                                        {analysisResult.is_uncertain
                                            ? '⚠ PREDICTION UNCERTAIN — REVIEW RECOMMENDED'
                                            : analysisResult.diabetes_detected
                                                ? '⚠ DIABETIC RETINOPATHY DETECTED'
                                                : '✓ NO DIABETIC RETINOPATHY SIGNS'}
                                    </div>
                                    <div className={`text-sm font-semibold mb-2 ${
                                        analysisResult.is_uncertain
                                            ? 'text-yellow-200'
                                            : analysisResult.diabetes_detected
                                                ? 'text-amber-200'
                                                : 'text-emerald-200'
                                    }`}>
                                        {analysisResult.is_uncertain ? 'Provisional Grade: ' : 'Severity: '}
                                        {analysisResult.severity_tier || '—'}
                                        {analysisResult.confidence_pct > 0 && ` · Model Confidence: ${analysisResult.confidence_pct}%`}
                                    </div>
                                    {analysisResult.is_uncertain && analysisResult.uncertainty_warning && (
                                        <p className="text-xs text-yellow-200/95 bg-yellow-950/50 border border-yellow-500/40 rounded-xl px-3.5 py-2 max-w-2xl mx-auto mb-2">
                                            {analysisResult.uncertainty_warning}
                                            {analysisResult.uncertainty?.uncertainty_reasons?.length > 0 && (
                                                <span className="block text-[10px] font-mono text-yellow-300/80 mt-1">
                                                    Triggers: {analysisResult.uncertainty.uncertainty_reasons.join(' • ')} (Entropy: {analysisResult.uncertainty.normalized_entropy}, Top-2 Margin: {analysisResult.uncertainty.top2_margin_pct}%)
                                                </span>
                                            )}
                                        </p>
                                    )}
                                    {analysisResult.severity_description && (
                                        <p className="text-xs text-slate-300 max-w-lg mx-auto">
                                            {analysisResult.severity_description}
                                        </p>
                                    )}
                                    {analysisResult.any_dr_probability_pct !== undefined && (
                                        <div className="mt-3 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-950/60 border border-slate-700 text-xs font-mono text-slate-300">
                                            Any DR Probability: <span className={`font-bold ${analysisResult.diabetes_detected ? 'text-amber-300' : 'text-emerald-300'}`}>{analysisResult.any_dr_probability_pct}%</span>
                                            {analysisResult.any_dr_threshold_pct && (
                                                <span className="text-slate-500">(Threshold: {analysisResult.any_dr_threshold_pct}%)</span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Presentation Attack Detection (PAD) Rejection Banner */}
                            {analysisResult.status === 'spoof_detected' && (
                                <div className="rounded-2xl p-5 border-2 bg-rose-950/40 border-rose-500 text-center space-y-2.5 shadow-xl">
                                    <div className="text-xl sm:text-2xl font-black tracking-tight text-rose-300 flex items-center justify-center gap-2">
                                        <ShieldAlert className="w-6 h-6 text-rose-400" />
                                        <span>POSSIBLE SCREEN OR PHOTOGRAPH DETECTED</span>
                                    </div>
                                    <p className="text-sm font-bold text-rose-200 max-w-xl mx-auto">
                                        {UI_STATUS_MESSAGES.spoof_detected}
                                    </p>
                                    <p className="text-xs text-slate-300 max-w-lg mx-auto leading-relaxed">
                                        Presentation Attack Detection (PAD) detected high-frequency 2D spectral Moiré screen lines, low LBP texture entropy, diffuse reflection, or device bezels.
                                    </p>
                                    <div className="text-[11px] font-mono text-rose-300/80 pt-1">
                                        Diagnostic results suppressed: Only a directly present live human eye is accepted.
                                    </div>
                                </div>
                            )}

                            {/* Liveness Verification Failure Banner */}
                            {(analysisResult.status === 'liveness_failed' || analysisResult.status === 'verification_failed') && (
                                <div className="rounded-2xl p-5 border-2 bg-amber-950/40 border-amber-500 text-center space-y-2.5 shadow-xl">
                                    <div className="text-xl sm:text-2xl font-black tracking-tight text-amber-300 flex items-center justify-center gap-2">
                                        <AlertTriangle className="w-6 h-6 text-amber-400" />
                                        <span>LIVE HUMAN EYE NOT VERIFIED</span>
                                    </div>
                                    <p className="text-sm font-bold text-amber-200 max-w-xl mx-auto">
                                        {UI_STATUS_MESSAGES.verification_failed}
                                    </p>
                                    <p className="text-xs text-slate-300 max-w-lg mx-auto leading-relaxed">
                                        The system requires multi-frame landmark tracking, natural eyelid blink dynamics, interactive eye-movement challenge completion, and non-spoof verification.
                                    </p>
                                    <div className="text-[11px] font-mono text-amber-300/80 pt-1">
                                        Diagnostic results suppressed: Static photographs, screen captures, or replayed videos are rejected.
                                    </div>
                                </div>
                            )}

                            {/* Strict Retinal Fundus Validation Rejection / Uncertain Abstention Banners */}
                            {analysisResult.status === 'uncertain_retinal' && (
                                <div className="rounded-2xl p-5 border-2 bg-amber-950/40 border-amber-500 text-center space-y-2 shadow-xl">
                                    <div className="text-xl sm:text-2xl font-black tracking-tight text-amber-300 flex items-center justify-center gap-2">
                                        <AlertTriangle className="w-6 h-6 text-amber-400" />
                                        <span>UNABLE TO VERIFY RETINAL FUNDUS IMAGE</span>
                                    </div>
                                    <p className="text-sm font-bold text-amber-200 max-w-xl mx-auto">
                                        {analysisResult.message}
                                    </p>
                                    {analysisResult.retinal_validation && (
                                        <div className="text-[11px] font-mono text-amber-300/90">
                                            Retinal Fundus Probability: {analysisResult.retinal_validation.retinal_fundus_probability_pct}% (Required ≥ {(analysisResult.retinal_validation.valid_threshold * 100).toFixed(1)}%) • DR classification intentionally blocked.
                                        </div>
                                    )}
                                </div>
                            )}

                            {analysisResult.status === 'rejected' && (
                                <div className="rounded-2xl p-5 border-2 bg-rose-950/40 border-rose-500 text-center space-y-2 shadow-xl">
                                    <div className="text-xl sm:text-2xl font-black tracking-tight text-rose-300 flex items-center justify-center gap-2">
                                        <XCircle className="w-6 h-6 text-rose-400" />
                                        <span>INVALID / NON-RETINAL IMAGE REJECTED</span>
                                    </div>
                                    <p className="text-sm font-bold text-rose-200 max-w-xl mx-auto">
                                        {analysisResult.message}
                                    </p>
                                    {analysisResult.retinal_validation && (
                                        <div className="text-[11px] font-mono text-rose-300/90">
                                            Retinal Fundus Score: {analysisResult.retinal_validation.retinal_fundus_probability_pct}% • Non-Retinal Score: {analysisResult.retinal_validation.non_retinal_probability_pct}% • DR CNN skipped.
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Neutral Computer Vision Card — Strictly for External Anterior Eye / Iris Photos */}
                            {analysisResult.status === 'completed' && analysisResult.detected_modality === 'anterior_iris' && (
                                <div className="rounded-2xl p-5 border-2 bg-sky-950/30 border-sky-400/50 text-center space-y-2 shadow-lg">
                                    <div className="text-xl sm:text-2xl font-black tracking-tight text-sky-300">
                                        EXTERNAL ANTERIOR IRIS SEGMENTED — NOT A RETINAL SCAN
                                    </div>
                                    <p className="text-xs text-slate-200 max-w-xl mx-auto leading-relaxed">
                                        Diabetes / diabetic retinopathy <strong>cannot</strong> be diagnosed from front-of-eye iris photos or webcam captures, so the retinal CNN is intentionally not run on this photo.
                                    </p>
                                    <p className="text-[11px] text-sky-300/90">
                                        Use a <strong>Retinal Fundus Photograph</strong> (e.g., Grade 0–4 samples on the left) for DR screening, or enter clinical labs in the <strong>Pima Screening Form</strong> below.
                                    </p>
                                </div>
                            )}

                            {/* Prediction Header Card */}
                            <div
                                className={`rounded-xl p-4 border ${
                                    analysisResult.status === 'completed'
                                        ? analysisResult.detected_modality === 'retinal_fundus'
                                            ? analysisResult.diabetes_detected
                                                ? 'bg-amber-950/25 border-amber-500/40'
                                                : 'bg-emerald-950/20 border-emerald-500/40'
                                            : 'bg-sky-950/25 border-sky-500/40'
                                        : 'bg-rose-950/25 border-rose-500/40'
                                }`}
                            >
                                <div className="flex flex-wrap justify-between items-start gap-3">
                                    <div className="space-y-1">
                                        <div className="flex flex-wrap items-center gap-2">
                                            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-slate-950/80 border border-slate-700 text-sky-300">
                                                Modality: {analysisResult.detected_modality?.replace('_', ' ')}
                                            </span>
                                            {analysisResult.model_name && (
                                                <span className="text-[10px] font-mono text-slate-400">
                                                    {analysisResult.model_name}
                                                </span>
                                            )}
                                        </div>
                                        <h4 className="text-base font-bold text-white">
                                            {analysisResult.prediction_label}
                                        </h4>
                                        {analysisResult.modality_note && (
                                            <p className="text-xs text-slate-300">{analysisResult.modality_note}</p>
                                        )}
                                        {analysisResult.message && (
                                            <p className="text-xs text-rose-300 font-medium">{analysisResult.message}</p>
                                        )}
                                    </div>

                                    {analysisResult.status === 'completed' && (
                                        <div className="text-right bg-slate-950/80 px-3.5 py-2 rounded-xl border border-slate-800">
                                            <div className="text-[10px] text-slate-400 font-mono uppercase">
                                                {analysisResult.detected_modality === 'retinal_fundus' ? 'Model Confidence' : 'Quality Score'}
                                            </div>
                                            <div className="text-2xl font-extrabold font-mono text-sky-300">
                                                {analysisResult.confidence_pct}%
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Quality Metrics Bar */}
                            {analysisResult.quality && (
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-center">
                                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                                        <div className="text-[10px] text-slate-400 font-mono">Sharpness (Laplacian)</div>
                                        <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                                            {analysisResult.quality.sharpness_laplacian_var}
                                        </div>
                                    </div>
                                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                                        <div className="text-[10px] text-slate-400 font-mono">Mean Brightness</div>
                                        <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                                            {analysisResult.quality.mean_brightness}/255
                                        </div>
                                    </div>
                                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                                        <div className="text-[10px] text-slate-400 font-mono">Contrast Std</div>
                                        <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                                            {analysisResult.quality.contrast_std}
                                        </div>
                                    </div>
                                    <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                                        <div className="text-[10px] text-slate-400 font-mono">Glare Ratio</div>
                                        <div className="text-sm font-bold font-mono text-slate-200 mt-0.5">
                                            {(analysisResult.quality.glare_ratio * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Retinal Fundus DR Class Probability Distribution */}
                            {analysisResult.status === 'completed' && analysisResult.detected_modality === 'retinal_fundus' && probEntries.length > 0 && (
                                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-3">
                                    <div className="flex flex-wrap justify-between items-center gap-2">
                                        <h5 className="text-xs font-bold text-slate-200">
                                            Diabetic Retinopathy Severity Probabilities
                                        </h5>
                                        {analysisResult.referable_dr_probability_pct !== undefined && (
                                            <span className="text-[11px] font-mono text-amber-300">
                                                Referable DR (Moderate+): {analysisResult.referable_dr_probability_pct}%
                                            </span>
                                        )}
                                    </div>
                                    <div className="space-y-2">
                                        {probEntries.map(([clsName, pct]) => {
                                            const isHighest = clsName === analysisResult.severity_label;
                                            const isHealthyClass = clsName.toLowerCase().includes('no diabetic');
                                            const barClass = isHighest
                                                ? isHealthyClass
                                                    ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                                                    : 'bg-gradient-to-r from-amber-500 to-rose-500'
                                                : 'bg-gradient-to-r from-sky-500 to-indigo-500';
                                            return (
                                                <div key={clsName} className="space-y-1">
                                                    <div className="flex justify-between text-[11px] font-mono">
                                                        <span className={`${isHighest ? 'text-white font-bold' : 'text-slate-300'}`}>{clsName}</span>
                                                        <span className={`font-semibold ${isHighest ? (isHealthyClass ? 'text-emerald-300' : 'text-amber-300') : 'text-sky-300'}`}>{pct}%</span>
                                                    </div>
                                                    <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                                                        <div
                                                            className={`h-2 rounded-full transition-all duration-500 ${barClass}`}
                                                            style={{ width: `${Math.min(100, Math.max(1, pct))}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                    {analysisResult.clinical_guidance && (
                                        <div className="text-xs text-slate-300 pt-2 border-t border-slate-800">
                                            <strong className="text-sky-300">Screening Guidance:</strong> {analysisResult.clinical_guidance}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Real PyTorch Grad-CAM Explainability & Clinical Screening Report */}
                            {analysisResult.status === 'completed' && analysisResult.detected_modality === 'retinal_fundus' && (analysisResult.gradcam?.available || analysisResult.clinical_report) && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {analysisResult.gradcam?.available && (
                                        <div className="bg-slate-950/75 border border-slate-800 rounded-xl p-4 space-y-2.5">
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="text-xs font-bold text-cyan-300">
                                                    CNN Grad-CAM Spatial Activation Map
                                                </div>
                                                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-300">
                                                    {analysisResult.gradcam.target_layer}
                                                </span>
                                            </div>
                                            {analysisResult.gradcam_overlay_image && (
                                                <img
                                                    src={analysisResult.gradcam_overlay_image}
                                                    alt="Retinal Grad-CAM Overlay"
                                                    className="w-full max-h-52 object-contain rounded-lg bg-black border border-slate-800"
                                                />
                                            )}
                                            <div className="text-[11px] font-mono text-slate-300 space-y-1">
                                                <div>
                                                    Peak Region: <span className="text-cyan-300">{analysisResult.gradcam.focus_region}</span>
                                                </div>
                                                <div>
                                                    High-Activation Coverage: <span className="text-slate-200">{analysisResult.gradcam.peak_activation_pct}%</span> of retinal FOV
                                                </div>
                                            </div>
                                            {analysisResult.gradcam.disclaimer && (
                                                <p className="text-[10px] text-slate-400 leading-relaxed border-t border-slate-800/80 pt-2">
                                                    {analysisResult.gradcam.disclaimer}
                                                </p>
                                            )}
                                        </div>
                                    )}

                                    {analysisResult.clinical_report && (
                                        <div className="bg-slate-950/75 border border-slate-800 rounded-xl p-4 space-y-2.5 text-xs">
                                            <div className="flex flex-wrap items-center justify-between gap-1">
                                                <div className="font-bold text-sky-300">ICDR Structured Screening Summary</div>
                                                <span className="text-[10px] font-mono text-slate-400">AAO / ETDRS</span>
                                            </div>
                                            <div className="font-semibold text-white">{analysisResult.clinical_report.headline}</div>
                                            <p className="text-slate-300 leading-relaxed">{analysisResult.clinical_report.summary}</p>
                                            <div className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 space-y-1">
                                                <div className="text-[10px] font-mono text-slate-400 uppercase">ICDR Clinical Criteria</div>
                                                <div className="text-[11px] text-slate-300">{analysisResult.clinical_report.clinical_criteria}</div>
                                            </div>
                                            <div className="p-2.5 rounded-lg bg-sky-950/30 border border-sky-500/30 space-y-1">
                                                <div className="text-[10px] font-mono text-sky-300 uppercase">Recommended Next Step</div>
                                                <div className="text-[11px] text-slate-200">{analysisResult.clinical_report.recommended_action}</div>
                                            </div>
                                            {analysisResult.clinical_report.gradcam_focus_note && (
                                                <div className="text-[11px] text-slate-400 font-mono">
                                                    {analysisResult.clinical_report.gradcam_focus_note}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Anterior Iris Segmentation Visualization & CV Features */}
                            {analysisResult.detected_modality === 'anterior_iris' && analysisResult.iris_analysis && (
                                <div className="space-y-4">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-950/70 border border-slate-800 rounded-xl p-4">
                                    <div className="space-y-2">
                                        <div className="text-xs font-bold text-emerald-300">
                                            OpenCV Iris Segmentation Overlay
                                        </div>
                                        {analysisResult.segmentation_overlay_image && (
                                            <img
                                                src={analysisResult.segmentation_overlay_image}
                                                alt="Iris Segmentation Overlay"
                                                className="w-full max-h-48 object-contain rounded-lg bg-black border border-slate-800"
                                            />
                                        )}
                                        <div className="text-[10px] text-slate-400 font-mono">
                                            Cyan = Pupil • Green = Iris Annulus • Amber = Eyelid Mask Bounds
                                        </div>
                                    </div>

                                    <div className="space-y-2 text-xs">
                                        <div className="font-bold text-slate-200">Extracted Iris CV Features</div>
                                        <ul className="space-y-1.5 text-[11px] font-mono text-slate-300">
                                            <li className="flex justify-between">
                                                <span>Iris Detected / Segmented:</span>
                                                <span className="text-emerald-300">
                                                    {analysisResult.iris_detected ? 'Yes' : 'No'} / {analysisResult.segmentation_completed ? 'Completed' : 'Partial'}
                                                </span>
                                            </li>
                                            <li className="flex justify-between">
                                                <span>Pupil / Iris Radius Ratio:</span>
                                                <span>{analysisResult.extracted_features?.pupil_to_iris_ratio}</span>
                                            </li>
                                            <li className="flex justify-between">
                                                <span>Usable Iris Area:</span>
                                                <span>{analysisResult.extracted_features?.usable_iris_area_pct}%</span>
                                            </li>
                                            <li className="flex justify-between">
                                                <span>Glare Masked:</span>
                                                <span>{analysisResult.extracted_features?.glare_masked_pct}%</span>
                                            </li>
                                            <li className="flex justify-between">
                                                <span>Texture Gradient Energy:</span>
                                                <span>{analysisResult.extracted_features?.texture_gradient_energy}</span>
                                            </li>
                                            <li className="flex justify-between">
                                                <span>Mean Iris HSV:</span>
                                                <span>
                                                    H:{analysisResult.extracted_features?.mean_hsv?.hue_deg}° S:{analysisResult.extracted_features?.mean_hsv?.saturation_pct}% V:{analysisResult.extracted_features?.mean_hsv?.value_pct}%
                                                </span>
                                            </li>
                                        </ul>
                                    </div>
                                    </div>
                                </div>
                            )}

                            {/* Required Blood Glucose Scientific Notice + Disclaimer */}
                            <div className="space-y-2">
                                <div className="p-3 rounded-xl bg-indigo-950/30 border border-indigo-500/30 text-xs text-indigo-200 flex items-start gap-2">
                                    <Info className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
                                    <span>
                                        <strong>Blood Glucose Notice:</strong> {analysisResult.glucose_statement}
                                    </span>
                                </div>
                                <p className="text-[11px] text-slate-400 leading-relaxed bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                                    {analysisResult.disclaimer}
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
