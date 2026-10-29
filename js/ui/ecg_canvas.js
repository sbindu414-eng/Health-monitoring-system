/**
 * Interactive 12-Lead ECG Waveform Canvas Engine
 * Real-time HTML5 canvas renderer simulating cardiac lead waveforms and paper grid.
 */

class ECGCanvasRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext("2d");
        this.isRunning = false;
        this.speed = 2.0; // mm/s equivalent sweep
        this.gain = 1.0;
        this.currentCondition = "NSR";
        this.sweepX = 0;
        this.lead = "Lead II";
        this.animationId = null;

        this.buffer = [];
        this.resize();
        window.addEventListener("resize", () => this.resize());
    }

    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width || 800;
        this.canvas.height = rect.height || 240;
        this.clear();
        this.drawGrid();
    }

    clear() {
        if (!this.ctx) return;
        this.ctx.fillStyle = "#090d16"; // Dark ECG display background
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    drawGrid() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const ctx = this.ctx;

        ctx.lineWidth = 0.5;

        // Small 5px grid (1mm equivalent)
        ctx.strokeStyle = "rgba(239, 68, 68, 0.12)"; // Subtle red ECG grid
        ctx.beginPath();
        for (let x = 0; x < w; x += 10) {
            ctx.moveTo(x, 0); ctx.lineTo(x, h);
        }
        for (let y = 0; y < h; y += 10) {
            ctx.moveTo(0, y); ctx.lineTo(w, y);
        }
        ctx.stroke();

        // Large 25px grid (5mm equivalent)
        ctx.strokeStyle = "rgba(239, 68, 68, 0.30)";
        ctx.beginPath();
        for (let x = 0; x < w; x += 50) {
            ctx.moveTo(x, 0); ctx.lineTo(x, h);
        }
        for (let y = 0; y < h; y += 50) {
            ctx.moveTo(0, y); ctx.lineTo(w, y);
        }
        ctx.stroke();

        // Lead label text
        ctx.font = "bold 13px system-ui, sans-serif";
        ctx.fillStyle = "#ef4444";
        ctx.fillText((this.lead || "Lead II").toString().toUpperCase() + " | 25mm/s | 10mm/mV | 0.05-150Hz", 15, 25);
    }

    // Mathematical P-Q-R-S-T wave synthesizer for cardiac cycles
    getWaveY(time, condition) {
        const midY = this.canvas.height / 2;
        let yOffset = 0;

        // Normalize cycle time t in [0, 1]
        let cyclePeriod = 0.8; // seconds
        if (condition === "AF") cyclePeriod = 0.52 + Math.sin(time * 7) * 0.15; // Irregular RR
        else if (condition === "AV_BLOCK_1") cyclePeriod = 0.95;
        else if (condition === "LBBB" || condition === "RBBB") cyclePeriod = 0.82;

        const phase = (time % cyclePeriod) / cyclePeriod;

        // P Wave
        if (condition === "AF") {
            // Fibrillatory waves (no true P wave)
            yOffset += Math.sin(time * 35) * 3 + Math.cos(time * 50) * 2;
        } else if (condition === "AV_BLOCK_1") {
            // Extended PR segment (P wave happens early in phase: 0.05 - 0.15)
            if (phase >= 0.05 && phase < 0.15) {
                const pPhase = (phase - 0.05) / 0.10;
                yOffset -= Math.sin(pPhase * Math.PI) * 12;
            }
        } else {
            // Standard P wave (0.10 - 0.20)
            if (phase >= 0.10 && phase < 0.20) {
                const pPhase = (phase - 0.10) / 0.10;
                yOffset -= Math.sin(pPhase * Math.PI) * 12;
            }
        }

        // Q-R-S Complex
        let qrsStart = (condition === "AV_BLOCK_1") ? 0.35 : 0.25;
        let qrsDuration = (condition === "LBBB" || condition === "RBBB") ? 0.18 : 0.10;

        if (phase >= qrsStart && phase < (qrsStart + qrsDuration)) {
            const qrsPhase = (phase - qrsStart) / qrsDuration;

            if (condition === "RBBB") {
                // Notch rSR' pattern (bunny ears)
                if (qrsPhase < 0.3) yOffset -= Math.sin(qrsPhase / 0.3 * Math.PI) * 45;
                else if (qrsPhase < 0.5) yOffset += 15;
                else yOffset -= Math.sin((qrsPhase - 0.5) / 0.5 * Math.PI) * 65;
            } else if (condition === "LBBB") {
                // Wide notched R wave
                if (qrsPhase < 0.4) yOffset -= Math.sin(qrsPhase / 0.4 * Math.PI) * 55;
                else if (qrsPhase < 0.6) yOffset -= 35; // notch plateau
                else yOffset -= Math.sin((qrsPhase - 0.6) / 0.4 * Math.PI) * 50;
            } else {
                // Normal QRS spike
                if (qrsPhase < 0.2) yOffset += 12; // Q dip
                else if (qrsPhase < 0.7) yOffset -= 75; // R peak
                else yOffset += 22; // S dip
            }
        }

        // ST Segment & T Wave
        let stStart = qrsStart + qrsDuration;
        let tStart = stStart + 0.10;

        if (condition === "ST_ELEV") {
            // High ST Elevation above baseline
            if (phase >= stStart && phase < tStart) {
                yOffset -= 25; // ST Elevation
            } else if (phase >= tStart && phase < (tStart + 0.20)) {
                const tPhase = (phase - tStart) / 0.20;
                yOffset -= (25 + Math.sin(tPhase * Math.PI) * 20); // Elevated T wave
            }
        } else if (condition === "LBBB") {
            // ST Depression
            if (phase >= stStart && phase < (tStart + 0.20)) {
                yOffset += 15;
            }
        } else {
            // Normal T wave (0.45 - 0.65)
            if (phase >= tStart && phase < (tStart + 0.20)) {
                const tPhase = (phase - tStart) / 0.20;
                yOffset -= Math.sin(tPhase * Math.PI) * 18;
            }
        }

        return midY + (yOffset * this.gain);
    }

    start(condition = "NSR", lead = "Lead II") {
        this.currentCondition = condition;
        this.lead = lead;
        this.isRunning = true;
        this.sweepX = 0;
        this.clear();
        this.drawGrid();

        let startTime = performance.now();

        const animate = (now) => {
            if (!this.isRunning) return;

            const timeSec = (now - startTime) / 1000;
            const ctx = this.ctx;
            const w = this.canvas.width;

            // Clear sweep cursor head
            const sweepWidth = 15;
            ctx.fillStyle = "#090d16";
            ctx.fillRect(this.sweepX, 0, sweepWidth, this.canvas.height);

            // Redraw grid slice
            ctx.save();
            ctx.beginPath();
            ctx.rect(this.sweepX, 0, sweepWidth, this.canvas.height);
            ctx.clip();
            this.drawGrid();
            ctx.restore();

            // Compute current point
            const y = this.getWaveY(timeSec, this.currentCondition);

            // Draw signal line
            ctx.strokeStyle = (this.currentCondition === "ST_ELEV" || this.currentCondition === "AF") ? "#ef4444" : "#10b981"; // Red for critical, Green for normal/moderate
            ctx.lineWidth = 2.2;
            ctx.lineJoin = "round";

            if (this.lastY !== undefined && this.sweepX > 0) {
                ctx.beginPath();
                ctx.moveTo(this.sweepX - this.speed, this.lastY);
                ctx.lineTo(this.sweepX, y);
                ctx.stroke();
            }

            this.lastY = y;
            this.sweepX += this.speed;

            if (this.sweepX >= w) {
                this.sweepX = 0;
                this.lastY = y;
            }

            this.animationId = requestAnimationFrame(animate);
        };

        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.animationId = requestAnimationFrame(animate);
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) cancelAnimationFrame(this.animationId);
    }

    setCondition(conditionCode) {
        this.currentCondition = conditionCode;
    }

    setLead(leadName) {
        this.lead = leadName;
    }
}

window.ECGCanvasRenderer = ECGCanvasRenderer;
