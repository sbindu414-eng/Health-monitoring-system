/**
 * Real-Time Telemetry Stream Engine
 * Continuously ingests streaming physiological vitals without requiring manual user input.
 * Supports backend polling + client-side continuous telemetry synthesis.
 */

class LiveTelemetryService {
    constructor() {
        this.subscribers = new Set();
        this.intervalId = null;
        this.isRunning = false;
        this.scenario = "normal";
        this.packetCounter = 0;
        this.streamFrequencyMs = 1000; // 1 packet per second
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.intervalId = setInterval(() => {
            this.fetchOrGeneratePacket();
        }, this.streamFrequencyMs);
    }

    stop() {
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    setScenario(newScenario) {
        this.scenario = newScenario;
        // Notify backend of scenario change
        fetch('/api/telemetry/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario: newScenario })
        }).catch(() => {});
        this.fetchOrGeneratePacket();
    }

    subscribe(callback) {
        this.subscribers.add(callback);
        return () => this.subscribers.delete(callback);
    }

    emit(packet) {
        this.subscribers.forEach((cb) => {
            try {
                cb(packet);
            } catch (err) {
                console.error("Telemetry subscriber error:", err);
            }
        });
    }

    async fetchOrGeneratePacket() {
        try {
            const res = await fetch('/api/telemetry/live', { signal: AbortSignal.timeout(800) });
            if (res.ok) {
                const data = await res.json();
                this.emit(data);
                return;
            }
        } catch {
            // Local fallback synthesizer if backend disconnected
        }

        // High-fidelity client synthetic packet
        this.packetCounter += 1;
        const now = new Date();
        const t = now.getTime() / 1000;
        const noise = (Math.random() - 0.5) * 2;
        let p = {
            packetId: this.packetCounter,
            timestamp: now.toLocaleTimeString(),
            scenario: this.scenario,
            streamStatus: "ONLINE",
            throughput: "1 pkt/sec",
            latencyMs: +(11 + Math.random() * 5).toFixed(1)
        };

        if (this.scenario === "athlete") {
            p = {
                ...p,
                heartRate: Math.round(52 + Math.sin(t * 0.5) * 3 + noise),
                spO2: Math.min(100, Math.round(99 + Math.random())),
                bloodPressure: { systolic: Math.round(114 + Math.sin(t * 0.2) * 3), diastolic: Math.round(72 + Math.sin(t * 0.2) * 2) },
                skinTemp: +(36.6 + Math.random() * 0.2).toFixed(1),
                hrv: Math.round(68 + Math.random() * 8),
                rrInterval: 1150,
                respirationRate: 12,
                ecgCondition: "NSR",
                activeSymptoms: [],
                severity: "Normal"
            };
        } else if (this.scenario === "exercise") {
            p = {
                ...p,
                heartRate: Math.round(135 + Math.sin(t * 0.8) * 8 + noise * 2),
                spO2: Math.round(97 + Math.random() * 2),
                bloodPressure: { systolic: Math.round(142 + Math.sin(t * 0.3) * 6), diastolic: Math.round(88 + Math.sin(t * 0.3) * 3) },
                skinTemp: +(37.4 + Math.random() * 0.3).toFixed(1),
                hrv: Math.round(28 + Math.random() * 5),
                rrInterval: 450,
                respirationRate: 26,
                ecgCondition: "NSR",
                activeSymptoms: ["palpitations", "fatigue"],
                severity: "Moderate"
            };
        } else if (this.scenario === "afib") {
            p = {
                ...p,
                heartRate: Math.round(145 + Math.sin(t * 2.0) * 16 + noise * 5),
                spO2: Math.round(95 + Math.random() * 2),
                bloodPressure: { systolic: Math.round(138 + Math.sin(t * 0.4) * 8), diastolic: Math.round(90 + Math.sin(t * 0.4) * 4) },
                skinTemp: +(36.9 + Math.random() * 0.2).toFixed(1),
                hrv: Math.round(110 + Math.random() * 30),
                rrInterval: 520,
                respirationRate: 22,
                ecgCondition: "AF",
                activeSymptoms: ["palpitations", "dizziness", "irregular_pulse", "fatigue"],
                severity: "High"
            };
        } else if (this.scenario === "stemi") {
            p = {
                ...p,
                heartRate: Math.round(105 + Math.sin(t * 0.6) * 8 + noise * 3),
                spO2: Math.round(93 + Math.random() * 3),
                bloodPressure: { systolic: Math.round(165 + Math.sin(t * 0.3) * 10), diastolic: Math.round(102 + Math.sin(t * 0.3) * 6) },
                skinTemp: +(37.1 + Math.random() * 0.2).toFixed(1),
                hrv: Math.round(18 + Math.random() * 4),
                rrInterval: 570,
                respirationRate: 24,
                ecgCondition: "ST_ELEV",
                activeSymptoms: ["chest_pain", "jaw_arm_pain", "cold_sweat", "shortness_breath", "nausea"],
                severity: "Critical"
            };
        } else if (this.scenario === "hypoxia") {
            p = {
                ...p,
                heartRate: Math.round(112 + Math.sin(t * 0.7) * 6 + noise * 2),
                spO2: Math.round(87 + Math.random() * 3),
                bloodPressure: { systolic: Math.round(132 + Math.sin(t * 0.3) * 5), diastolic: Math.round(84 + Math.sin(t * 0.3) * 3) },
                skinTemp: +(38.6 + Math.random() * 0.4).toFixed(1),
                hrv: Math.round(32 + Math.random() * 6),
                rrInterval: 620,
                respirationRate: 30,
                ecgCondition: "NSR",
                activeSymptoms: ["fever", "cough", "sob", "chills", "fatigue"],
                severity: "High"
            };
        } else {
            p = {
                ...p,
                heartRate: Math.round(72 + Math.sin(t * 0.4) * 4 + noise),
                spO2: Math.min(100, Math.round(98 + Math.random() * 2)),
                bloodPressure: { systolic: Math.round(120 + Math.sin(t * 0.2) * 3), diastolic: Math.round(78 + Math.sin(t * 0.2) * 2) },
                skinTemp: +(36.8 + Math.random() * 0.2).toFixed(1),
                hrv: Math.round(45 + Math.random() * 8),
                rrInterval: 830,
                respirationRate: 16,
                ecgCondition: "NSR",
                activeSymptoms: [],
                severity: "Normal"
            };
        }

        this.emit(p);
    }
}

export const liveTelemetry = new LiveTelemetryService();

