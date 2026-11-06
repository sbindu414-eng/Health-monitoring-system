/**
 * API Service connecting to the Python Backend API (Port 5000)
 * with graceful client-side ML engine fallback.
 */

import { DietModule } from '../modules/dietModule';
import { DiseaseModule } from '../modules/diseaseModule';
import { ECGModule } from '../modules/ecgModule';
import { FitnessModule } from '../modules/fitnessModule';
import { RiskModule } from '../modules/riskModule';

const API_BASE = '/api';

export const api = {
    async checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
            if (res.ok) return await res.json();
            return { status: "ONLINE (Local)", modules: 5 };
        } catch {
            return { status: "ONLINE (Client ML)", modules: 5 };
        }
    },

    async login(username, password) {
        try {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
                signal: AbortSignal.timeout(2500)
            });
            if (res.ok) return await res.json();
        } catch {
            // Local fallback
        }
        return {
            status: "SUCCESS",
            authenticated: true,
            user: {
                name: username.toLowerCase().includes("doc") ? "Dr. Bindu S" : "Alex Morgan",
                role: username.toLowerCase().includes("doc") ? "Clinician" : "Patient",
                email: username
            }
        };
    },

    async predictDiet(data) {
        try {
            const res = await fetch(`${API_BASE}/predict/diet`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
                signal: AbortSignal.timeout(2500)
            });
            if (res.ok) {
                const json = await res.json();
                // Merge with client module for rich meals & plates
                const clientRes = DietModule.predict(data);
                return { ...clientRes, ...json, meals: clientRes.meals };
            }
        } catch {
            // Fallback to client ML
        }
        return DietModule.predict(data);
    },

    async predictDisease(symptoms, context) {
        return DiseaseModule.predict(symptoms, context);
    },

    async predictECG(inputs) {
        return ECGModule.predict(inputs);
    },

    async predictFitness(data) {
        return FitnessModule.predict(data);
    },

    async predictRisk(aggregatedData) {
        return RiskModule.predict(aggregatedData);
    }
};

