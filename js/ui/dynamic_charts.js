/**
 * Dynamic Chart Renderer & Visualizer (Clean Light Theme)
 * Utilities to render Chart.js graphs, confusion matrix heatmaps, and radar charts.
 */

const DynamicCharts = {
    instances: {},

    // Helper to safely destroy existing chart instance before re-creating
    destroyChart: function (id) {
        if (this.instances[id]) {
            this.instances[id].destroy();
            delete this.instances[id];
        }
    },

    // Render Fig 2: Macronutrient Composition Bar Chart
    renderFig2: function (canvasId) {
        this.destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = PaperBenchmarks.fig2_macronutrientComposition;
        this.instances[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: data.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { color: '#334155', font: { weight: '500' } } },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(226, 232, 240, 0.8)' } },
                    y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(226, 232, 240, 0.8)' }, title: { display: true, text: 'Grams (g)', color: '#475569' } }
                }
            }
        });
    },

    // Render Fig 3: Disease Symptom Prevalence Horizontal Bar Chart
    renderFig3: function (canvasId) {
        this.destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = PaperBenchmarks.fig3_diseasePrevalence;
        this.instances[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Mention Frequency / Count',
                    data: data.data,
                    backgroundColor: 'rgba(8, 145, 178, 0.85)',
                    borderColor: '#0891b2',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(226, 232, 240, 0.8)' } },
                    y: { ticks: { color: '#334155', font: { size: 11, weight: '500' } }, grid: { display: false } }
                }
            }
        });
    },

    // Render Fig 4: ECG Condition Prevalence Doughnut Chart
    renderFig4: function (canvasId) {
        this.destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = PaperBenchmarks.fig4_ecgPrevalence;
        this.instances[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.data,
                    backgroundColor: [
                        '#e11d48', '#f97316', '#eab308', '#059669', '#0891b2', '#6366f1'
                    ],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#334155', boxWidth: 12, font: { size: 11 } } }
                }
            }
        });
    },

    // Render Fig 6: Normalized Health Indicators Radar Chart
    renderFig6: function (canvasId) {
        this.destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        const data = PaperBenchmarks.fig6_normalizedRiskIndicators;
        this.instances[canvasId] = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: data.labels,
                datasets: data.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(203, 213, 225, 0.7)' },
                        grid: { color: 'rgba(203, 213, 225, 0.7)' },
                        pointLabels: { color: '#334155', font: { size: 12, weight: 'bold' } },
                        ticks: { color: '#64748b', backdropColor: 'transparent', stepSize: 20 }
                    }
                },
                plugins: {
                    legend: { position: 'top', labels: { color: '#334155', font: { weight: '500' } } }
                }
            }
        });
    },

    // Render Fig 7: Comparative Model Performance Grouped Bar Chart
    renderFig7: function (canvasId) {
        this.destroyChart(canvasId);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        // Flatten data into grouped format
        const labels = [
            "Diet: LogReg (82.3%)", "Diet: SVM (84.1%)", "Diet: DecTree (85.6%)", "Diet: RandForest (89.2%)",
            "Disease: NaiveBayes (78.5%)", "Disease: KNN (81.0%)", "Disease: LogReg (86.7%)",
            "ECG: 1D-CNN (88.4%)", "ECG: RandForest (90.1%)", "ECG: XGBoost (91.2%)", "ECG: Ensemble (93.5%)",
            "Fitness: DecTree R²(0.74)", "Fitness: Ridge R²(0.77)", "Fitness: LinReg R²(0.82)",
            "Risk: DecTree (84.0%)", "Risk: RandForest (89.5%)", "Risk: XGBoost (91.3%)"
        ];

        const scores = [82.3, 84.1, 85.6, 89.2, 78.5, 81.0, 86.7, 88.4, 90.1, 91.2, 93.5, 74.0, 77.0, 82.0, 84.0, 89.5, 91.3];
        const colors = scores.map((s, idx) => [3, 6, 10, 13, 16].includes(idx) ? '#059669' : '#3b82f6'); // Highlight final paper models in Emerald

        this.instances[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Performance Metric (Accuracy % / R² Score x100)',
                    data: scores,
                    backgroundColor: colors,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (context) { return `Score: ${context.parsed.y}%`; }
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: '#475569', font: { size: 10 } }, grid: { display: false } },
                    y: { min: 60, max: 100, ticks: { color: '#64748b' }, grid: { color: 'rgba(226, 232, 240, 0.8)' } }
                }
            }
        });
    },

    // Render HTML Heatmap Table for Correlation Heatmap (Fig 5)
    renderFig5CorrelationTable: function (containerId) {
        const el = document.getElementById(containerId);
        if (!el) return;

        const data = PaperBenchmarks.fig5_correlationHeatmap;
        let html = `<div class="overflow-x-auto"><table class="w-full text-xs text-center border-collapse"><thead><tr><th class="p-2 text-left text-slate-600 bg-slate-100 font-semibold border border-slate-200">Feature</th>`;

        data.variables.forEach(v => {
            html += `<th class="p-2 text-slate-700 bg-slate-100 font-semibold border border-slate-200">${v}</th>`;
        });
        html += `</tr></thead><tbody>`;

        data.matrix.forEach((row, i) => {
            html += `<tr><td class="p-2 text-left font-semibold text-slate-800 border border-slate-200 bg-slate-50">${data.variables[i]}</td>`;
            row.forEach((val, j) => {
                let bgColor = "#ffffff";
                let textColor = "#334155";

                if (val === 1.0) {
                    bgColor = "rgba(59, 130, 246, 0.25)"; textColor = "#1d4ed8";
                } else if (val > 0) {
                    bgColor = `rgba(59, 130, 246, ${val * 0.22})`;
                    textColor = "#1e40af";
                } else {
                    bgColor = `rgba(239, 68, 68, ${Math.abs(val) * 0.22})`;
                    textColor = "#b91c1c";
                }

                html += `<td class="p-2 border border-slate-200 font-mono font-bold transition-all hover:scale-105" style="background-color: ${bgColor}; color: ${textColor}">${val > 0 ? '+' : ''}${val.toFixed(2)}</td>`;
            });
            html += `</tr>`;
        });

        html += `</tbody></table></div>`;
        el.innerHTML = html;
    },

    // Render Confusion Matrix Grid (Fig 8 & Fig 9)
    renderConfusionMatrix: function (containerId, matrixData, classLabels, title) {
        const container = document.getElementById(containerId);
        if (!container) return;

        let maxVal = 0;
        matrixData.forEach(row => row.forEach(val => { if (val > maxVal) maxVal = val; }));

        let html = `
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                <h4 class="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> ${title}
                </h4>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-center border-separate border-spacing-1">
                        <thead>
                            <tr>
                                <th class="p-1.5 text-left text-slate-500 font-normal">Actual / Pred</th>
                                ${classLabels.map(l => `<th class="p-1.5 text-slate-700 font-medium">${l}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
        `;

        matrixData.forEach((row, i) => {
            html += `<tr><td class="p-1.5 text-left text-slate-700 font-medium">${classLabels[i]}</td>`;
            row.forEach((val, j) => {
                const ratio = val / maxVal;
                const isDiagonal = (i === j);
                let bgColor = isDiagonal
                    ? `rgba(16, 185, 129, ${Math.max(0.12, ratio * 0.35)})`
                    : `rgba(239, 68, 68, ${Math.max(0.04, ratio * 0.25)})`;

                let textColor = isDiagonal ? '#047857' : (val > 0 ? '#b91c1c' : '#94a3b8');

                html += `
                    <td class="p-3 rounded font-mono font-bold text-sm transition-all hover:brightness-95" 
                        style="background-color: ${bgColor}; color: ${textColor}"
                        title="Actual: ${classLabels[i]} | Predicted: ${classLabels[j]} | Count: ${val}">
                        ${val}
                    </td>
                `;
            });
            html += `</tr>`;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        container.innerHTML = html;
    }
};

window.DynamicCharts = DynamicCharts;
