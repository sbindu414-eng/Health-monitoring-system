import React, { useState, useEffect, useRef } from 'react';
import {
    Eye,
    Upload,
    Sparkles,
    AlertTriangle,
    CheckCircle2,
    XCircle,
    Loader2,
    History,
    Trash2,
    BarChart3,
    Activity,
    ShieldAlert,
    Info,
    Image as ImageIcon,
    Calculator,
} from 'lucide-react';
import { api } from '../services/api';

const DEFAULT_PIMA_INPUTS = {
    pregnancies: 1,
    glucose: 118,
    bloodPressure: 74,
    skinThickness: 24,
    insulin: 88,
    bmi: 26.8,
    dpf: 0.42,
    age: 38,
};

export default function EyeAnalysisTab({ session }) {
    const fileInputRef = useRef(null);

    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedSample, setSelectedSample] = useState('');
    const [previewUrl, setPreviewUrl] = useState('');
    const [analysisMode, setAnalysisMode] = useState('auto');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');

    const [moduleStatus, setModuleStatus] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [historyItems, setHistoryItems] = useState([]);
    const [showEvalPanel, setShowEvalPanel] = useState(false);

    // Separate Tabular Diabetes Risk State (Pima Indians Diabetes Database)
    const [pimaInputs, setPimaInputs] = useState(DEFAULT_PIMA_INPUTS);
    const [pimaResult, setPimaResult] = useState(null);
    const [isRunningPima, setIsRunningPima] = useState(false);

    const loadStatusAndHistory = async () => {
        try {
            const [st, hist] = await Promise.all([api.getEyeStatus(), api.getEyeHistory(25)]);
            setModuleStatus(st);
            setHistoryItems(hist.eye_history || []);
        } catch (err) {
            console.error('Eye module status/history error:', err);
        }
    };

    useEffect(() => {
        loadStatusAndHistory();
    }, []);

    useEffect(() => {
        return () => {
            if (previewUrl && previewUrl.startsWith('blob:')) {
                URL.revokeObjectURL(previewUrl);
            }
        };
    }, [previewUrl]);

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
                userRole: session?.role || 'Clinician',
            });
            setAnalysisResult(res);
            await loadStatusAndHistory();
        } catch (err) {
            setErrorMsg(err.message || 'Analysis failed.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleDeleteHistory = async (id) => {
        try {
            await api.deleteEyeHistoryItem(id);
            setHistoryItems((prev) => prev.filter((item) => item.id !== id));
        } catch (err) {
            setErrorMsg(err.message);
        }
    };

    const handleClearHistory = async () => {
        try {
            await api.clearEyeHistory();
            setHistoryItems([]);
        } catch (err) {
            setErrorMsg(err.message);
        }
    };

    const handleRunPimaRisk = async (e) => {
        e?.preventDefault();
        setIsRunningPima(true);
        try {
            const res = await api.predictTabularDiabetes(pimaInputs);
            setPimaResult(res);
        } catch (err) {
            setErrorMsg(err.message);
        } finally {
            setIsRunningPima(false);
        }
    };

    const evalSummary = moduleStatus?.retinal_evaluation || {};
    const overallMetrics = evalSummary?.overall_metrics || {};
    const sensitivityMetrics = evalSummary?.screening_sensitivity_analysis || {};
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
                                ? `CNN Ready (${moduleStatus.retinal_architecture})`
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
                    Scientifically distinguishes between <strong>Retinal Fundus Screening</strong> (5-grade ICDR diabetic retinopathy grading via transfer learning CNN trained on DeepDRiD / RetinaMNIST), <strong>Anterior Eye / Iris CV Segmentation</strong> (experimental OpenCV pupil/iris segmentation &amp; quality gating), and <strong>Clinical Tabular Diabetes Risk</strong> (separate Pima Indians Diabetes model).
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
                                <BarChart3 className="w-4 h-4" /> Held-Out Test Split Evaluation (Unseen Test Set, n={evalSummary.test_samples})
                            </h3>
                            <p className="text-[11px] text-slate-400 mt-0.5">
                                Backbone: {evalSummary.backbone_architecture} (Best Epoch {evalSummary.best_epoch}) • Patient-disjoint RetinaMNIST / DeepDRiD split
                            </p>
                        </div>
                        <span className="text-[10px] font-mono px-2.5 py-1 rounded bg-slate-950 border border-slate-700 text-slate-300">
                            Academic Internal Benchmark (Not Clinical Validation)
                        </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-center">
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
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Weighted F1</div>
                            <div className="text-lg font-bold font-mono text-indigo-300 mt-0.5">
                                {(overallMetrics.f1_weighted * 100).toFixed(2)}%
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">OvR ROC-AUC</div>
                            <div className="text-lg font-bold font-mono text-purple-300 mt-0.5">
                                {overallMetrics.roc_auc_ovr_macro?.toFixed(3) ?? 'N/A'}
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Any-DR Sensitivity</div>
                            <div className="text-lg font-bold font-mono text-amber-300 mt-0.5">
                                {((sensitivityMetrics.any_dr_detection?.sensitivity_recall || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                            <div className="text-[10px] text-slate-400 uppercase font-mono">Referable DR Sens.</div>
                            <div className="text-lg font-bold font-mono text-rose-300 mt-0.5">
                                {((sensitivityMetrics.referable_dr_detection_grade_2plus?.sensitivity_recall || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>

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
                            <div className="text-[11px] text-slate-400 font-mono">Training &amp; Validation Loss/F1</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Main Eye Upload & Prediction Workspace */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Upload, Modality Mode, Preview & Sample Loader */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                    <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                        <h3 className="text-sm font-bold text-sky-400 flex items-center gap-2">
                            <Upload className="w-4 h-4" /> Eye Image Upload
                        </h3>
                        <span className="text-[10px] px-2 py-0.5 rounded bg-sky-500/10 text-sky-300 font-mono">
                            Max {moduleStatus?.max_upload_mb || 10} MB
                        </span>
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
                            {/* Prediction Header Card */}
                            <div
                                className={`rounded-xl p-4 border ${
                                    analysisResult.status === 'completed'
                                        ? analysisResult.detected_modality === 'retinal_fundus' && analysisResult.predicted_grade >= 2
                                            ? 'bg-amber-950/25 border-amber-500/40'
                                            : 'bg-emerald-950/20 border-emerald-500/40'
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
                                        <h4 className="text-lg sm:text-xl font-extrabold text-white">
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

                            {/* Retinal Fundus 5-Class Probability Distribution */}
                            {analysisResult.detected_modality === 'retinal_fundus' && probEntries.length > 0 && (
                                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-3">
                                    <div className="flex flex-wrap justify-between items-center gap-2">
                                        <h5 className="text-xs font-bold text-slate-200">
                                            Diabetic Retinopathy Class Probability Distribution
                                        </h5>
                                        <span className="text-[11px] font-mono text-amber-300">
                                            Referable DR (Moderate+) Prob: {analysisResult.referable_dr_probability_pct}%
                                        </span>
                                    </div>
                                    <div className="space-y-2">
                                        {probEntries.map(([clsName, pct]) => (
                                            <div key={clsName} className="space-y-1">
                                                <div className="flex justify-between text-[11px] font-mono">
                                                    <span className="text-slate-300">{clsName}</span>
                                                    <span className="text-sky-300 font-semibold">{pct}%</span>
                                                </div>
                                                <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                                                    <div
                                                        className="bg-gradient-to-r from-sky-500 to-indigo-500 h-2 rounded-full transition-all duration-500"
                                                        style={{ width: `${Math.min(100, Math.max(1, pct))}%` }}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                    {analysisResult.clinical_guidance && (
                                        <div className="text-xs text-slate-300 pt-2 border-t border-slate-800">
                                            <strong className="text-sky-300">Screening Guidance:</strong> {analysisResult.clinical_guidance}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Experimental Anterior Iris Segmentation Visualization & CV Features */}
                            {analysisResult.detected_modality === 'anterior_iris' && analysisResult.iris_analysis && (
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

            {/* Separate Tabular Diabetes Screening Section (Pima Indians Diabetes Database) */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex flex-wrap justify-between items-center gap-2 border-b border-slate-800 pb-3">
                    <div>
                        <h3 className="text-sm font-bold text-purple-300 flex items-center gap-2">
                            <Calculator className="w-4 h-4" /> Separate Tabular Clinical Diabetes Risk Screening (Pima Indians Diabetes Model)
                        </h3>
                        <p className="text-[11px] text-slate-400 mt-0.5">
                            Trained strictly on 768 clinical tabular records (ROC-AUC: {moduleStatus?.tabular_diabetes_evaluation?.test_metrics?.roc_auc ?? '0.8126'}), completely separate from image analysis.
                        </p>
                    </div>
                    <span className="text-[10px] font-mono px-2.5 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30">
                        Tabular Clinical Data Only
                    </span>
                </div>

                <form onSubmit={handleRunPimaRisk} className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 text-xs">
                    {[
                        ['pregnancies', 'Pregnancies'],
                        ['glucose', 'Glucose (mg/dL)'],
                        ['bloodPressure', 'Diastolic BP'],
                        ['skinThickness', 'Skin Fold (mm)'],
                        ['insulin', 'Insulin (mu U/ml)'],
                        ['bmi', 'BMI'],
                        ['dpf', 'Pedigree Func'],
                        ['age', 'Age (yrs)'],
                    ].map(([key, label]) => (
                        <div key={key}>
                            <label className="block text-[10px] text-slate-400 mb-1">{label}</label>
                            <input
                                type="number"
                                step={key === 'bmi' || key === 'dpf' ? '0.01' : '1'}
                                value={pimaInputs[key]}
                                onChange={(e) => setPimaInputs((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-slate-200 font-mono text-xs focus:outline-none focus:border-purple-500"
                            />
                        </div>
                    ))}
                    <div className="col-span-2 sm:col-span-4 lg:col-span-8 flex flex-wrap items-center justify-between gap-3 pt-1">
                        <button
                            type="submit"
                            disabled={isRunningPima}
                            className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs flex items-center gap-2 transition-all disabled:opacity-50"
                        >
                            {isRunningPima ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Calculator className="w-3.5 h-3.5" />}
                            <span>Predict Tabular Diabetes Risk</span>
                        </button>

                        {pimaResult && (
                            <div className="flex flex-wrap items-center gap-3 bg-slate-950/90 border border-purple-500/30 px-4 py-2 rounded-xl text-xs">
                                <span className="font-bold text-purple-200">{pimaResult.prediction_label}</span>
                                <span className="font-mono text-purple-300">
                                    Probability: {pimaResult.probability_pct}% ({pimaResult.risk_tier} Tier)
                                </span>
                            </div>
                        )}
                    </div>
                </form>
            </div>

            {/* Prediction History Section (SQLite Backed) */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex flex-wrap justify-between items-center gap-2 border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                        <History className="w-4 h-4 text-sky-400" /> Eye Analysis Prediction History (SQLite Database)
                    </h3>
                    {historyItems.length > 0 && (
                        <button
                            type="button"
                            onClick={handleClearHistory}
                            className="px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-rose-950/50 border border-slate-700 hover:border-rose-500/40 text-[11px] text-rose-300 flex items-center gap-1 transition-all"
                        >
                            <Trash2 className="w-3 h-3" /> Clear History
                        </button>
                    )}
                </div>

                {historyItems.length === 0 ? (
                    <div className="text-xs text-slate-500 py-4 text-center">
                        No eye image analyses recorded yet. Upload and analyse an image to store its prediction in SQLite.
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                            <thead className="text-[10px] uppercase font-mono text-slate-400 border-b border-slate-800">
                                <tr>
                                    <th className="py-2 pr-3">Time</th>
                                    <th className="py-2 pr-3">File</th>
                                    <th className="py-2 pr-3">Modality</th>
                                    <th className="py-2 pr-3">Quality</th>
                                    <th className="py-2 pr-3">Prediction</th>
                                    <th className="py-2 pr-3">Confidence</th>
                                    <th className="py-2 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/70">
                                {historyItems.map((item) => (
                                    <tr key={item.id} className="hover:bg-slate-950/50">
                                        <td className="py-2 pr-3 font-mono text-[11px] text-slate-400 whitespace-nowrap">
                                            {new Date(item.created_at).toLocaleString()}
                                        </td>
                                        <td className="py-2 pr-3 font-mono text-slate-200 max-w-[180px] truncate">
                                            {item.filename}
                                        </td>
                                        <td className="py-2 pr-3 font-mono text-[11px] text-sky-300">
                                            {item.detected_modality}
                                        </td>
                                        <td className="py-2 pr-3">
                                            <span
                                                className={`text-[10px] px-2 py-0.5 rounded font-mono ${
                                                    item.quality_status === 'acceptable'
                                                        ? 'bg-emerald-500/15 text-emerald-300'
                                                        : 'bg-rose-500/15 text-rose-300'
                                                }`}
                                            >
                                                {item.quality_status} ({item.quality_score})
                                            </span>
                                        </td>
                                        <td className="py-2 pr-3 text-slate-200 font-medium">
                                            {item.prediction_label}
                                        </td>
                                        <td className="py-2 pr-3 font-mono text-sky-300">
                                            {item.confidence_pct}%
                                        </td>
                                        <td className="py-2 text-right">
                                            <button
                                                type="button"
                                                onClick={() => handleDeleteHistory(item.id)}
                                                className="text-slate-500 hover:text-rose-400 p-1 rounded"
                                                title="Delete entry"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}
