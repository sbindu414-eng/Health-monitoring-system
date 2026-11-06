/**
 * Main Application Controller & Guided Pipeline Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // Auth Session Validation (Redirect to login.html if not signed in)
    function checkAuthSession() {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get("guest") === "true") {
            const guestSession = { name: "Guest Reviewer", role: "Guest Access", avatar: "👤" };
            renderUserSession(guestSession);
            return guestSession;
        }

        const sessionStr = localStorage.getItem("healthml_session") || sessionStorage.getItem("healthml_session");
        if (!sessionStr) {
            window.location.href = "login.html";
            return null;
        }

        try {
            const session = JSON.parse(sessionStr);
            if (!session || !session.name) {
                window.location.href = "login.html";
                return null;
            }
            renderUserSession(session);
            return session;
        } catch (e) {
            window.location.href = "login.html";
            return null;
        }
    }

    function renderUserSession(session) {
        if (document.getElementById("userName")) document.getElementById("userName").innerText = session.name;
        if (document.getElementById("userRole")) document.getElementById("userRole").innerText = session.role || "Clinician";
        if (document.getElementById("userAvatar")) document.getElementById("userAvatar").innerText = session.avatar || "👨‍⚕️";
    }

    window.handleSignOut = function() {
        localStorage.removeItem("healthml_session");
        sessionStorage.removeItem("healthml_session");
        window.location.href = "login.html?logout=true";
    };

    const currentSession = checkAuthSession();
    if (!currentSession) return; // Halt pipeline initialization if redirecting to login.html

    // State management
    const appState = {
        currentTab: "overview",
        ecgRenderer: null,
        patientData: {
            age: 45,
            gender: "male",
            height: 175,
            weight: 82,
            activityLevel: "moderate",
            dietType: "balanced",
            dietGoal: "maintenance",
            sleepHours: 6.5,
            caloricIntake: 2400,
            exerciseMin: 35,
            heartRateResting: 74,
            stressLevel: 6,
            symptoms: ["cough", "fatigue", "headache", "chest_pain"],
            ecgCondition: "RBBB"
        },
        moduleResults: {
            diet: null,
            disease: null,
            ecg: null,
            fitness: null,
            risk: null
        },
        ecgState: {
            symptoms: [],
            hasFile: false
        }
    };

    // Initialize ECG Renderer
    appState.ecgRenderer = new ECGCanvasRenderer("ecgCanvas");

    // Tab Navigation Listener
    const navItems = document.querySelectorAll("[data-tab]");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tabName = item.getAttribute("data-tab");
            switchTab(tabName);
        });
    });

    function switchTab(tabName) {
        appState.currentTab = tabName;

        // Update active nav button styling
        navItems.forEach(nav => {
            if (nav.getAttribute("data-tab") === tabName) {
                nav.classList.add("bg-slate-800", "text-white", "border-b-2", "border-indigo-500");
                nav.classList.remove("text-slate-400");
            } else {
                nav.classList.remove("bg-slate-800", "text-white", "border-b-2", "border-indigo-500");
                nav.classList.add("text-slate-400");
            }
        });

        // Toggle Tab Panels
        const tabPanels = document.querySelectorAll(".tab-panel");
        tabPanels.forEach(panel => {
            if (panel.id === `tab-${tabName}`) {
                panel.classList.remove("hidden");
            } else {
                panel.classList.add("hidden");
            }
        });

        // Trigger ECG canvas when ECG tab becomes visible
        if (tabName === "ecg") {
            setTimeout(() => {
                appState.ecgRenderer.resize();
                const selectedCond = document.getElementById("ecgConditionSelect")?.value || "NSR";
                appState.ecgRenderer.start(selectedCond, "Lead II");
            }, 100);
        } else {
            appState.ecgRenderer.stop();
        }
    }

    // Symptom Matrix Renderer (Module 2)
    function renderSymptomSelector() {
        const container = document.getElementById("symptomPillContainer");
        if (!container) return;

        let html = "";
        DiseaseModule.symptomsList.forEach(s => {
            const isActive = appState.patientData.symptoms.includes(s.id);
            html += `
                <div class="symptom-pill glass-panel px-3 py-2 rounded-lg text-xs flex items-center gap-2 ${isActive ? 'active' : 'text-slate-300'}"
                     data-symptom-id="${s.id}">
                    <span class="w-2 h-2 rounded-full ${isActive ? 'bg-cyan-400 animate-pulse' : 'bg-slate-600'}"></span>
                    <span>${s.label}</span>
                </div>
            `;
        });

        container.innerHTML = html;

        // Toggle click handler
        container.querySelectorAll(".symptom-pill").forEach(pill => {
            pill.addEventListener("click", () => {
                const symId = pill.getAttribute("data-symptom-id");
                const idx = appState.patientData.symptoms.indexOf(symId);
                if (idx > -1) {
                    appState.patientData.symptoms.splice(idx, 1);
                } else {
                    appState.patientData.symptoms.push(symId);
                }
                renderSymptomSelector();
                runDiseaseModule();
            });
        });
    }

    // Execute Module 1: Diet Suggestion
    function runDietModule() {
        const data = {
            age: document.getElementById("dietAge")?.value || appState.patientData.age,
            gender: document.getElementById("dietGender")?.value || appState.patientData.gender,
            height: document.getElementById("dietHeight")?.value || appState.patientData.height,
            weight: document.getElementById("dietWeight")?.value || appState.patientData.weight,
            activityLevel: document.getElementById("dietActivity")?.value || appState.patientData.activityLevel,
            dietType: document.getElementById("dietType")?.value || appState.patientData.dietType,
            dietGoal: document.getElementById("dietGoal")?.value || appState.patientData.dietGoal || "maintenance",
            sleepHours: document.getElementById("dietSleep")?.value || appState.patientData.sleepHours
        };

        const res = DietModule.predict(data);
        appState.moduleResults.diet = res;
        appState.patientData.caloricIntake = res.targetCalories; // Auto-sync calculated target calories
        if (document.getElementById("fitCalories")) {
            document.getElementById("fitCalories").value = res.targetCalories;
        }

        // Render UI
        document.getElementById("dietScoreVal").innerText = res.dietScore;
        document.getElementById("dietRatingBadge").innerText = res.ratingCategory;
        document.getElementById("dietBmiVal").innerText = res.bmi;
        document.getElementById("dietTdeeVal").innerText = res.tdee + " kcal";
        if (document.getElementById("dietTargetCalVal")) {
            document.getElementById("dietTargetCalVal").innerText = res.targetCalories + " kcal";
        }

        // Meals Breakdown
        document.getElementById("mealBreakfast").innerText = `${res.meals.breakfast.cal} kcal (${res.meals.breakfast.carbs}g C, ${res.meals.breakfast.protein}g P)`;
        document.getElementById("mealLunch").innerText = `${res.meals.lunch.cal} kcal (${res.meals.lunch.carbs}g C, ${res.meals.lunch.protein}g P)`;
        document.getElementById("mealDinner").innerText = `${res.meals.dinner.cal} kcal (${res.meals.dinner.carbs}g C, ${res.meals.dinner.protein}g P)`;
        document.getElementById("mealSnacks").innerText = `${res.meals.snacks.cal} kcal (${res.meals.snacks.carbs}g C, ${res.meals.snacks.protein}g P)`;

        // Dynamic Meal Split Titles (Calculated from Goal, Activity, and Sleep)
        if (document.getElementById("mealTitleBreakfast")) document.getElementById("mealTitleBreakfast").innerText = `🍳 Breakfast (${res.meals.breakfast.pct}%)`;
        if (document.getElementById("mealTitleLunch")) document.getElementById("mealTitleLunch").innerText = `🥗 Lunch (${res.meals.lunch.pct}%)`;
        if (document.getElementById("mealTitleDinner")) document.getElementById("mealTitleDinner").innerText = `🍲 Dinner (${res.meals.dinner.pct}%)`;
        if (document.getElementById("mealTitleSnacks")) document.getElementById("mealTitleSnacks").innerText = `🥜 Snacks (${res.meals.snacks.pct}%)`;
        if (document.getElementById("mealSplitPatternTag")) {
            document.getElementById("mealSplitPatternTag").innerText = `${res.dietGoalLabel} • Dynamic Split`;
        }

        // Sample Food Plates (Dynamically synthesized from computed macros and diet parameters)
        if (document.getElementById("foodBreakfast")) document.getElementById("foodBreakfast").innerText = res.meals.breakfast.sampleFood;
        if (document.getElementById("foodLunch")) document.getElementById("foodLunch").innerText = res.meals.lunch.sampleFood;
        if (document.getElementById("foodDinner")) document.getElementById("foodDinner").innerText = res.meals.dinner.sampleFood;
        if (document.getElementById("foodSnacks")) document.getElementById("foodSnacks").innerText = res.meals.snacks.sampleFood;

        // Recommendations List
        const recList = document.getElementById("dietRecList");
        if (recList) {
            recList.innerHTML = res.recommendations.map(r => `<li class="flex items-start gap-2 text-xs text-emerald-300"><span class="text-emerald-500 font-bold">✓</span> ${r}</li>`).join("");
        }
    }

    // Execute Module 2: Disease Prediction
    function runDiseaseModule() {
        const res = DiseaseModule.predict(appState.patientData.symptoms);
        appState.moduleResults.disease = res;

        // Render UI
        document.getElementById("diseasePrimary").innerText = res.primaryDiagnosis;
        document.getElementById("diseaseConfidence").innerText = res.confidence;
        document.getElementById("diseaseSeverity").innerText = res.severity;

        // Render plain-English summary, actionable steps, and red flag warnings
        if (document.getElementById("diseaseRecSummary")) {
            document.getElementById("diseaseRecSummary").innerText = res.summary || res.recommendation;
        }
        if (document.getElementById("diseaseActionStepsList")) {
            document.getElementById("diseaseActionStepsList").innerHTML = (res.actionSteps || []).map(step => `
                <li class="flex items-start gap-2 text-xs">
                    <span class="text-cyan-400 font-bold mt-0.5">•</span>
                    <span class="text-slate-200">${step}</span>
                </li>
            `).join("");
        }
        if (document.getElementById("diseaseWarningText")) {
            document.getElementById("diseaseWarningText").innerText = res.whenToSeeDoctor || "Consult a doctor if symptoms worsen or do not subside within 48 hours.";
        }
        if (document.getElementById("diseaseRecText")) {
            document.getElementById("diseaseRecText").innerText = res.recommendation;
        }

        // Render Disease Probability Breakdown
        const probContainer = document.getElementById("diseaseProbBreakdown");
        if (probContainer) {
            probContainer.innerHTML = res.allPredictions.map(p => `
                <div class="space-y-1">
                    <div class="flex justify-between text-xs">
                        <span class="text-slate-300">${p.disease}</span>
                        <span class="font-mono text-cyan-400">${p.percentage}</span>
                    </div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full bg-cyan-500 rounded-full" style="width: ${p.percentage}"></div>
                    </div>
                </div>
            `).join("");
        }
    }

    // ── ECG Symptom Pill Renderer ────────────────────────────────────────────
    function renderEcgSymptomPills() {
        const container = document.getElementById("ecgSymptomPills");
        if (!container) return;
        let html = "";
        ECGModule.cardiacSymptoms.forEach(s => {
            const isActive = appState.ecgState.symptoms.includes(s.id);
            html += `
                <div class="ecg-sym-pill glass-panel px-2.5 py-1.5 rounded-lg text-[11px] flex items-center gap-1.5 cursor-pointer select-none transition-all
                    ${isActive ? 'bg-rose-900/40 border-rose-500/60 text-white' : 'text-slate-300 hover:border-slate-600'}"
                     data-ecg-sym="${s.id}">
                    <span class="w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-rose-400 animate-pulse' : 'bg-slate-600'}"></span>
                    <span>${s.label}</span>
                </div>`;
        });
        container.innerHTML = html;

        container.querySelectorAll(".ecg-sym-pill").forEach(pill => {
            pill.addEventListener("click", () => {
                const symId = pill.getAttribute("data-ecg-sym");
                const idx   = appState.ecgState.symptoms.indexOf(symId);
                if (idx > -1) appState.ecgState.symptoms.splice(idx, 1);
                else          appState.ecgState.symptoms.push(symId);
                renderEcgSymptomPills();
            });
        });
    }

    // Execute Module 3: ECG Pattern Classification
    function runEcgModule() {
        const age     = parseInt(document.getElementById("ecgAge")?.value) || 45;
        const gender  = document.getElementById("ecgGender")?.value        || "male";
        const symptoms = appState.ecgState.symptoms;
        const hasFile  = appState.ecgState.hasFile;

        const res = ECGModule.predict({ age, gender, symptoms, hasFile });
        appState.moduleResults.ecg = res;

        // Severity badge
        const badge = document.getElementById("ecgSeverityBadge");
        if (badge) {
            badge.innerText = res.severity;
            const colourMap = {
                Normal:   "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40",
                Mild:     "bg-sky-500/20 text-sky-300 border border-sky-500/40",
                Moderate: "bg-amber-500/20 text-amber-300 border border-amber-500/40",
                High:     "bg-rose-500/20 text-rose-300 border border-rose-500/40",
                Critical: "bg-red-600/30 text-red-300 border border-red-500/60 animate-pulse"
            };
            badge.className = "px-2.5 py-0.5 rounded text-xs font-mono font-bold " + (colourMap[res.severity] || "badge-ecg");
        }

        // Classification labels
        const diagEl = document.getElementById("ecgDiagnosisName");
        if (diagEl) diagEl.innerText = res.conditionName;
        const confEl = document.getElementById("ecgConfidenceVal");
        if (confEl) confEl.innerText = res.confidence;

        // Description box
        const descBox  = document.getElementById("ecgDescBox");
        const descText = document.getElementById("ecgDescText");
        if (descBox && descText) {
            descText.innerText = res.description;
            descBox.classList.remove("hidden");
        }

        // Clinical guidance box
        const clinBox = document.getElementById("ecgClinicalBox");
        if (clinBox) {
            const summEl = document.getElementById("ecgClinicalSummary");
            const actEl  = document.getElementById("ecgActionList");
            const warnEl = document.getElementById("ecgWarningText");
            const tagEl  = document.getElementById("ecgRiskTag");
            if (summEl) summEl.innerText = res.clinicalSummary || res.description;
            if (tagEl)  tagEl.innerText  = res.riskTag || "";
            if (actEl)  actEl.innerHTML  = (res.actions || []).map(a =>
                `<li class="flex items-start gap-1.5 mt-1"><span class="text-rose-400 font-bold mt-0.5">•</span><span>${a}</span></li>`
            ).join("");
            if (warnEl) warnEl.innerText = res.warning || "";
            clinBox.classList.remove("hidden");
        }

        // Canvas animation
        if (appState.ecgRenderer) {
            appState.ecgRenderer.start(res.predictedCode || "NSR", "Lead II");
        }
    }

    // Execute Module 4: Fitness Estimation
    function runFitnessModule() {
        const data = {
            age: document.getElementById("fitAge")?.value || appState.patientData.age,
            weight: document.getElementById("fitWeight")?.value || appState.patientData.weight,
            height: document.getElementById("fitHeight")?.value || appState.patientData.height,
            exerciseMin: document.getElementById("fitExercise")?.value || appState.patientData.exerciseMin,
            sleepHours: document.getElementById("fitSleep")?.value || appState.patientData.sleepHours,
            caloricIntake: document.getElementById("fitCalories")?.value || appState.patientData.caloricIntake,
            heartRateResting: document.getElementById("fitHr")?.value || appState.patientData.heartRateResting,
            stressLevel: document.getElementById("fitStress")?.value || appState.patientData.stressLevel
        };

        const res = FitnessModule.predict(data);
        appState.moduleResults.fitness = res;

        // Render UI
        document.getElementById("fitnessScoreVal").innerText = res.fitnessScore;
        document.getElementById("fitnessCategoryBadge").innerText = res.category;

        // Render Fitness Indicator Bars
        const indContainer = document.getElementById("fitnessIndicatorsList");
        if (indContainer) {
            indContainer.innerHTML = `
                <div class="space-y-1">
                    <div class="flex justify-between text-xs"><span class="text-slate-300">Activity Index</span><span class="text-amber-400 font-mono">${res.indicators.activityIndex}/100</span></div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full"><div class="h-full bg-amber-500 rounded-full" style="width: ${res.indicators.activityIndex}%"></div></div>
                </div>
                <div class="space-y-1">
                    <div class="flex justify-between text-xs"><span class="text-slate-300">Sleep Quality</span><span class="text-amber-400 font-mono">${res.indicators.sleepQuality}/100</span></div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full"><div class="h-full bg-amber-500 rounded-full" style="width: ${res.indicators.sleepQuality}%"></div></div>
                </div>
                <div class="space-y-1">
                    <div class="flex justify-between text-xs"><span class="text-slate-300">Cardio Recovery</span><span class="text-amber-400 font-mono">${res.indicators.cardioRecovery}/100</span></div>
                    <div class="w-full h-1.5 bg-slate-800 rounded-full"><div class="h-full bg-amber-500 rounded-full" style="width: ${res.indicators.cardioRecovery}%"></div></div>
                </div>
            `;
        }
    }

    // Execute Module 5: Integrated Health Risk Stratification
    function runRiskModule() {
        // Collect latest outputs from Modules 1-4
        if (!appState.moduleResults.diet) runDietModule();
        if (!appState.moduleResults.disease) runDiseaseModule();
        if (!appState.moduleResults.ecg) runEcgModule();
        if (!appState.moduleResults.fitness) runFitnessModule();

        const aggData = {
            bmi: appState.moduleResults.diet.bmi,
            diseaseLabel: appState.moduleResults.disease.primaryDiagnosis,
            diseaseSeverity: appState.moduleResults.disease.severity,
            ecgLabel: appState.moduleResults.ecg.conditionName,
            ecgSeverity: appState.moduleResults.ecg.severity,
            fitnessScore: appState.moduleResults.fitness.fitnessScore,
            dietRating: appState.moduleResults.diet.dietScore
        };

        const res = RiskModule.predict(aggData);
        appState.moduleResults.risk = res;

        // Render UI
        document.getElementById("riskLevelTitle").innerText = res.riskLevel + " RISK";
        document.getElementById("riskScoreVal").innerText = res.riskScore + "%";
        document.getElementById("riskBadge").innerText = res.riskLevel;

        // Probabilities
        document.getElementById("probLow").innerText = res.probabilities.low;
        document.getElementById("probMod").innerText = res.probabilities.moderate;
        document.getElementById("probHigh").innerText = res.probabilities.high;

        // Clinical Recommendations
        const planContainer = document.getElementById("riskActionPlanList");
        if (planContainer) {
            planContainer.innerHTML = res.clinicalActionPlan.map(p => `
                <li class="p-2 bg-slate-900/60 rounded text-xs text-slate-200 border-l-2 border-purple-500">${p}</li>
            `).join("");
        }
    }

    // Run Complete Integrated Pipeline (Figure 1 Methodology)
    function runCompleteHealthPipeline() {
        runDietModule();
        runDiseaseModule();
        runEcgModule();
        runFitnessModule();
        runRiskModule();

        switchTab("risk");
    }

    // Preset Patient Profiles Loader
    function loadPresetProfile(presetType) {
        if (presetType === "athlete") {
            appState.patientData = {
                age: 26, gender: "male", height: 180, weight: 74, activityLevel: "active", dietType: "balanced", dietGoal: "muscle_gain",
                sleepHours: 8.5, caloricIntake: 2800, exerciseMin: 75, heartRateResting: 52, stressLevel: 2,
                symptoms: []
            };
            appState.ecgState = { symptoms: [], hasFile: false };
        } else if (presetType === "hypertension") {
            appState.patientData = {
                age: 58, gender: "female", height: 162, weight: 88, activityLevel: "light", dietType: "balanced", dietGoal: "weight_loss",
                sleepHours: 6.0, caloricIntake: 2300, exerciseMin: 20, heartRateResting: 82, stressLevel: 7,
                symptoms: ["fatigue", "dizziness", "chest_pain", "sob"]
            };
            appState.ecgState = { symptoms: ["chest_pain", "dizziness", "shortness_breath", "fatigue", "slow_hr"], hasFile: false };
        } else if (presetType === "cardiac") {
            appState.patientData = {
                age: 64, gender: "male", height: 172, weight: 92, activityLevel: "sedentary", dietType: "low_carb", dietGoal: "weight_loss",
                sleepHours: 5.5, caloricIntake: 2600, exerciseMin: 10, heartRateResting: 94, stressLevel: 9,
                symptoms: ["fever", "cough", "chest_pain", "sob", "weakness"]
            };
            appState.ecgState = { symptoms: ["chest_pain", "jaw_arm_pain", "cold_sweat", "shortness_breath", "syncope", "nausea"], hasFile: false };
        }

        // Sync inputs in UI
        syncInputsWithState();
        renderSymptomSelector();
        renderEcgSymptomPills();
        runCompleteHealthPipeline();
    }

    function syncInputsWithState() {
        const d = appState.patientData;
        if (document.getElementById("dietAge")) document.getElementById("dietAge").value = d.age;
        if (document.getElementById("dietGender")) document.getElementById("dietGender").value = d.gender;
        if (document.getElementById("dietHeight")) document.getElementById("dietHeight").value = d.height;
        if (document.getElementById("dietWeight")) document.getElementById("dietWeight").value = d.weight;
        if (document.getElementById("dietActivity")) document.getElementById("dietActivity").value = d.activityLevel;
        if (document.getElementById("dietType")) document.getElementById("dietType").value = d.dietType;
        if (document.getElementById("dietGoal")) document.getElementById("dietGoal").value = d.dietGoal || "maintenance";
        if (document.getElementById("dietSleep")) document.getElementById("dietSleep").value = d.sleepHours;

        // ECG age & gender sync
        if (document.getElementById("ecgAge"))    document.getElementById("ecgAge").value    = d.age;
        if (document.getElementById("ecgGender")) document.getElementById("ecgGender").value = d.gender;

        if (document.getElementById("fitAge")) document.getElementById("fitAge").value = d.age;
        if (document.getElementById("fitWeight")) document.getElementById("fitWeight").value = d.weight;
        if (document.getElementById("fitHeight")) document.getElementById("fitHeight").value = d.height;
        if (document.getElementById("fitExercise")) document.getElementById("fitExercise").value = d.exerciseMin;
        if (document.getElementById("fitSleep")) document.getElementById("fitSleep").value = d.sleepHours;
        if (document.getElementById("fitCalories")) document.getElementById("fitCalories").value = d.caloricIntake;
        if (document.getElementById("fitHr")) document.getElementById("fitHr").value = d.heartRateResting;
        if (document.getElementById("fitStress")) document.getElementById("fitStress").value = d.stressLevel;
    }

    // Attach Event Listeners
    document.getElementById("btnRunFullPipeline")?.addEventListener("click", runCompleteHealthPipeline);
    document.getElementById("btnRunDiet")?.addEventListener("click", runDietModule);
    document.getElementById("btnRunDisease")?.addEventListener("click", runDiseaseModule);
    document.getElementById("btnRunEcg")?.addEventListener("click", runEcgModule);
    document.getElementById("btnRunFitness")?.addEventListener("click", runFitnessModule);
    document.getElementById("btnRunRisk")?.addEventListener("click", runRiskModule);

    // Presets listeners
    document.getElementById("presetAthlete")?.addEventListener("click", () => loadPresetProfile("athlete"));
    document.getElementById("presetHypertension")?.addEventListener("click", () => loadPresetProfile("hypertension"));
    document.getElementById("presetCardiac")?.addEventListener("click", () => loadPresetProfile("cardiac"));

    // ECG file upload handler
    const ecgFileInput = document.getElementById("ecgFileInput");
    if (ecgFileInput) {
        ecgFileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (!file) return;
            appState.ecgState.hasFile = true;
            // Show status bar
            const statusEl = document.getElementById("ecgFileStatus");
            const nameEl   = document.getElementById("ecgFileName");
            const sizeEl   = document.getElementById("ecgFileSize");
            if (statusEl) statusEl.classList.remove("hidden");
            if (nameEl)   nameEl.innerText = file.name;
            if (sizeEl)   sizeEl.innerText = (file.size / 1024).toFixed(1) + " KB";
            // Show image thumbnail if it's an image
            if (file.type.startsWith("image/")) {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    const wrap = document.getElementById("ecgImagePreviewWrap");
                    const img  = document.getElementById("ecgImagePreview");
                    if (img)  img.src = ev.target.result;
                    if (wrap) wrap.classList.remove("hidden");
                };
                reader.readAsDataURL(file);
            }
        });
    }
    // File remove button
    const ecgFileRemove = document.getElementById("ecgFileRemove");
    if (ecgFileRemove) {
        ecgFileRemove.addEventListener("click", () => {
            appState.ecgState.hasFile = false;
            const fileInput = document.getElementById("ecgFileInput");
            if (fileInput) fileInput.value = "";
            const statusEl = document.getElementById("ecgFileStatus");
            if (statusEl) statusEl.classList.add("hidden");
            const wrap = document.getElementById("ecgImagePreviewWrap");
            if (wrap) wrap.classList.add("hidden");
        });
    }

    // Initial render
    renderSymptomSelector();
    renderEcgSymptomPills();
    runDietModule();
    runDiseaseModule();
    runEcgModule();
    runFitnessModule();
    runRiskModule();
});
