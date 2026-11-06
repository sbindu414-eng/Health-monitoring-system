import React from 'react';
import { Salad, Flame, Scale, Moon, Target, ShieldCheck } from 'lucide-react';

export default function DietTab({ patientData, onChangePatientData, onRunDiet, result }) {
    const d = patientData;

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Input Controls */}
            <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2">
                        <Salad className="w-4 h-4" /> Nutritional & Biometric Inputs
                    </h3>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono">
                        Random Forest
                    </span>
                </div>

                <div className="space-y-3 text-xs">
                    {/* Age & Gender */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Age (Years)</label>
                            <input
                                type="number"
                                value={d.age}
                                onChange={(e) => onChangePatientData('age', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all font-mono"
                            />
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Gender</label>
                            <select
                                value={d.gender}
                                onChange={(e) => onChangePatientData('gender', e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all"
                            >
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>
                    </div>

                    {/* Height & Weight */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Height (cm)</label>
                            <input
                                type="number"
                                value={d.height}
                                onChange={(e) => onChangePatientData('height', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all font-mono"
                            />
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Weight (kg)</label>
                            <input
                                type="number"
                                value={d.weight}
                                onChange={(e) => onChangePatientData('weight', Number(e.target.value))}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all font-mono"
                            />
                        </div>
                    </div>

                    {/* Activity Level */}
                    <div>
                        <label className="block text-slate-400 mb-1">Physical Activity Level</label>
                        <select
                            value={d.activityLevel}
                            onChange={(e) => onChangePatientData('activityLevel', e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all"
                        >
                            <option value="sedentary">Sedentary (Little or no exercise)</option>
                            <option value="light">Lightly Active (1-3 days/week)</option>
                            <option value="moderate">Moderately Active (3-5 days/week)</option>
                            <option value="active">Active (6-7 days/week)</option>
                            <option value="very_active">Very Active (Heavy training / manual work)</option>
                        </select>
                    </div>

                    {/* Diet Preference & Goal */}
                    <div className="grid grid-cols-2 gap-2">
                        <div>
                            <label className="block text-slate-400 mb-1">Dietary Paradigm</label>
                            <select
                                value={d.dietType}
                                onChange={(e) => onChangePatientData('dietType', e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all"
                            >
                                <option value="balanced">Balanced Whole Food</option>
                                <option value="vegetarian">Vegetarian (Lacto-Ovo)</option>
                                <option value="vegan">Plant-Based (Vegan)</option>
                                <option value="low_carb">Low-Carbohydrate</option>
                                <option value="keto">Ketogenic (High Fat)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-slate-400 mb-1">Dietary Goal</label>
                            <select
                                value={d.dietGoal}
                                onChange={(e) => onChangePatientData('dietGoal', e.target.value)}
                                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all font-semibold text-emerald-300"
                            >
                                <option value="weight_loss">Weight Loss (-500 kcal)</option>
                                <option value="maintenance">Maintenance (TDEE)</option>
                                <option value="muscle_gain">Muscle Hypertrophy (+350 kcal)</option>
                            </select>
                        </div>
                    </div>

                    {/* Sleep Hours */}
                    <div>
                        <label className="block text-slate-400 mb-1">Average Nocturnal Sleep (Hours)</label>
                        <input
                            type="number"
                            step="0.5"
                            value={d.sleepHours}
                            onChange={(e) => onChangePatientData('sleepHours', Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500 transition-all font-mono"
                        />
                    </div>

                    <button
                        onClick={onRunDiet}
                        className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow-lg shadow-emerald-600/25 transition-all flex items-center justify-center gap-2 mt-2"
                    >
                        <span>🥗</span> Generate Macro & Caloric Prescription
                    </button>
                </div>
            </div>

            {/* Results Output */}
            <div className="lg:col-span-2 rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                    <h3 className="text-sm font-bold text-slate-200">Random Forest Caloric & Nutritional Profile</h3>
                    <span className="px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                        {result?.ratingCategory || 'Optimal'}
                    </span>
                </div>

                {/* Metrics Highlights */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-center">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Diet Quality</span>
                        <div className="text-2xl font-bold font-mono text-emerald-400 mt-0.5">
                            {result?.dietScore ?? 89} <span className="text-xs text-slate-500">/ 100</span>
                        </div>
                    </div>
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-center">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Body Mass Index</span>
                        <div className="text-2xl font-bold font-mono text-slate-200 mt-0.5">
                            {result?.bmi ?? 24.2}
                        </div>
                    </div>
                    <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-center">
                        <span className="text-[10px] text-slate-400 uppercase tracking-wider">Estimated TDEE</span>
                        <div className="text-2xl font-bold font-mono text-slate-200 mt-0.5">
                            {result?.tdee ?? 2350} <span className="text-xs text-slate-500">kcal</span>
                        </div>
                    </div>
                    <div className="bg-emerald-950/30 border border-emerald-500/40 rounded-xl p-3 text-center">
                        <span className="text-[10px] text-emerald-400 uppercase tracking-wider font-bold">Target Calories</span>
                        <div className="text-2xl font-bold font-mono text-emerald-300 mt-0.5">
                            {result?.targetCalories ?? 1850} <span className="text-xs text-emerald-400">kcal</span>
                        </div>
                    </div>
                </div>

                {/* Meals Breakdown Cards */}
                {result?.meals && (
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h4 className="text-xs font-semibold text-slate-300">Circadian Meal Distribution & Sample Food Plates</h4>
                            <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 font-mono">
                                {result.dietGoalLabel}
                            </span>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                            {/* Breakfast */}
                            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 space-y-1.5">
                                <div className="flex justify-between items-center text-slate-300 font-semibold">
                                    <span>🍳 Breakfast ({result.meals.breakfast.pct}%)</span>
                                    <span className="font-mono text-emerald-400">{result.meals.breakfast.cal} kcal</span>
                                </div>
                                <div className="text-[11px] text-slate-400 font-mono">
                                    Carbs: {result.meals.breakfast.carbs}g • Protein: {result.meals.breakfast.protein}g • Fat: {result.meals.breakfast.fat}g
                                </div>
                                <p className="text-[11px] text-slate-300 pt-1 border-t border-slate-800/60 leading-relaxed">
                                    {result.meals.breakfast.sampleFood}
                                </p>
                            </div>

                            {/* Lunch */}
                            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 space-y-1.5">
                                <div className="flex justify-between items-center text-slate-300 font-semibold">
                                    <span>🥗 Lunch ({result.meals.lunch.pct}%)</span>
                                    <span className="font-mono text-emerald-400">{result.meals.lunch.cal} kcal</span>
                                </div>
                                <div className="text-[11px] text-slate-400 font-mono">
                                    Carbs: {result.meals.lunch.carbs}g • Protein: {result.meals.lunch.protein}g • Fat: {result.meals.lunch.fat}g
                                </div>
                                <p className="text-[11px] text-slate-300 pt-1 border-t border-slate-800/60 leading-relaxed">
                                    {result.meals.lunch.sampleFood}
                                </p>
                            </div>

                            {/* Dinner */}
                            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 space-y-1.5">
                                <div className="flex justify-between items-center text-slate-300 font-semibold">
                                    <span>🍲 Dinner ({result.meals.dinner.pct}%)</span>
                                    <span className="font-mono text-emerald-400">{result.meals.dinner.cal} kcal</span>
                                </div>
                                <div className="text-[11px] text-slate-400 font-mono">
                                    Carbs: {result.meals.dinner.carbs}g • Protein: {result.meals.dinner.protein}g • Fat: {result.meals.dinner.fat}g
                                </div>
                                <p className="text-[11px] text-slate-300 pt-1 border-t border-slate-800/60 leading-relaxed">
                                    {result.meals.dinner.sampleFood}
                                </p>
                            </div>

                            {/* Snacks */}
                            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 space-y-1.5">
                                <div className="flex justify-between items-center text-slate-300 font-semibold">
                                    <span>🥜 Snacks ({result.meals.snacks.pct}%)</span>
                                    <span className="font-mono text-emerald-400">{result.meals.snacks.cal} kcal</span>
                                </div>
                                <div className="text-[11px] text-slate-400 font-mono">
                                    Carbs: {result.meals.snacks.carbs}g • Protein: {result.meals.snacks.protein}g • Fat: {result.meals.snacks.fat}g
                                </div>
                                <p className="text-[11px] text-slate-300 pt-1 border-t border-slate-800/60 leading-relaxed">
                                    {result.meals.snacks.sampleFood}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Hand Portion Rule Guide */}
                <div className="p-3.5 bg-gradient-to-r from-emerald-950/30 to-slate-900 border border-emerald-500/20 rounded-xl space-y-2">
                    <div className="flex items-center justify-between">
                        <h4 className="text-xs font-bold text-emerald-400 flex items-center gap-1.5">
                            <span>✋</span> Practical Portion Guide (Hand-Size Estimation)
                        </h4>
                        <span className="text-[10px] text-slate-400 font-mono">No Scale Required</span>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                        <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80">
                            <span className="text-emerald-400 font-semibold block">✋ Palm Size</span>
                            <span className="text-slate-300 text-[10px]">Protein (~20-30g): Eggs, chicken, fish, paneer, tofu</span>
                        </div>
                        <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80">
                            <span className="text-emerald-400 font-semibold block">✊ Closed Fist</span>
                            <span className="text-slate-300 text-[10px]">Veggies / Greens: Broccoli, spinach, leafy salads</span>
                        </div>
                        <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80">
                            <span className="text-emerald-400 font-semibold block">🤲 Cupped Hand</span>
                            <span className="text-slate-300 text-[10px]">Carbs: Rice, oats, quinoa, sweet potatoes, rotis</span>
                        </div>
                        <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80">
                            <span className="text-emerald-400 font-semibold block">👍 Thumb Size</span>
                            <span className="text-slate-300 text-[10px]">Fats: Olive oil, ghee, butter, avocado, nuts</span>
                        </div>
                    </div>
                </div>

                {/* Dietary Directives */}
                {result?.recommendations && (
                    <div className="space-y-2">
                        <h4 className="text-xs font-semibold text-slate-300">Evidence-Based Clinical Directives</h4>
                        <ul className="space-y-1.5 bg-slate-950/40 p-3 rounded-xl border border-slate-800/80 text-xs text-slate-300">
                            {result.recommendations.map((rec, i) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="text-emerald-400 font-bold">•</span>
                                    <span>{rec}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
}

