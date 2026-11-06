import React, { useEffect, useRef } from 'react';

export default function EcgCanvas({ condition = "NSR", lead = "Lead II", speed = 2.0, gain = 1.0 }) {
    const canvasRef = useRef(null);
    const animRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let isRunning = true;
        let sweepX = 0;
        let lastY = undefined;
        let startTime = performance.now();

        const handleResize = () => {
            const rect = canvas.parentElement?.getBoundingClientRect();
            if (rect) {
                canvas.width = rect.width || 800;
                canvas.height = 200;
            }
            drawGrid();
        };

        const drawGrid = () => {
            const w = canvas.width;
            const h = canvas.height;

            ctx.fillStyle = '#090d16';
            ctx.fillRect(0, 0, w, h);

            ctx.lineWidth = 0.5;

            // Small 10px grid
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.12)';
            ctx.beginPath();
            for (let x = 0; x < w; x += 10) {
                ctx.moveTo(x, 0); ctx.lineTo(x, h);
            }
            for (let y = 0; y < h; y += 10) {
                ctx.moveTo(0, y); ctx.lineTo(w, y);
            }
            ctx.stroke();

            // Large 50px grid
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.30)';
            ctx.beginPath();
            for (let x = 0; x < w; x += 50) {
                ctx.moveTo(x, 0); ctx.lineTo(x, h);
            }
            for (let y = 0; y < h; y += 50) {
                ctx.moveTo(0, y); ctx.lineTo(w, y);
            }
            ctx.stroke();

            // Lead info watermark
            ctx.font = 'bold 12px monospace';
            ctx.fillStyle = '#ef4444';
            ctx.fillText(`${lead.toUpperCase()} | 25mm/s | 10mm/mV | 0.05-150Hz | Real-Time Telemetry`, 14, 22);
        };

        const getWaveY = (time, cond) => {
            const midY = canvas.height / 2;
            let yOffset = 0;

            let cyclePeriod = 0.8;
            if (cond === 'AF') cyclePeriod = 0.52 + Math.sin(time * 7) * 0.15;
            else if (cond === 'AV_BLOCK_1') cyclePeriod = 0.95;
            else if (cond === 'LBBB' || cond === 'RBBB') cyclePeriod = 0.82;

            const phase = (time % cyclePeriod) / cyclePeriod;

            // P Wave
            if (cond === 'AF') {
                yOffset += Math.sin(time * 35) * 3 + Math.cos(time * 50) * 2;
            } else if (cond === 'AV_BLOCK_1') {
                if (phase >= 0.05 && phase < 0.15) {
                    const pPhase = (phase - 0.05) / 0.10;
                    yOffset -= Math.sin(pPhase * Math.PI) * 12;
                }
            } else {
                if (phase >= 0.10 && phase < 0.20) {
                    const pPhase = (phase - 0.10) / 0.10;
                    yOffset -= Math.sin(pPhase * Math.PI) * 12;
                }
            }

            // QRS Complex
            const qrsStart = (cond === 'AV_BLOCK_1') ? 0.35 : 0.25;
            const qrsDuration = (cond === 'LBBB' || cond === 'RBBB') ? 0.18 : 0.10;

            if (phase >= qrsStart && phase < (qrsStart + qrsDuration)) {
                const qrsPhase = (phase - qrsStart) / qrsDuration;

                if (cond === 'RBBB') {
                    if (qrsPhase < 0.3) yOffset -= Math.sin(qrsPhase / 0.3 * Math.PI) * 45;
                    else if (qrsPhase < 0.5) yOffset += 15;
                    else yOffset -= Math.sin((qrsPhase - 0.5) / 0.5 * Math.PI) * 65;
                } else if (cond === 'LBBB') {
                    if (qrsPhase < 0.4) yOffset -= Math.sin(qrsPhase / 0.4 * Math.PI) * 55;
                    else if (qrsPhase < 0.6) yOffset -= 35;
                    else yOffset -= Math.sin((qrsPhase - 0.6) / 0.4 * Math.PI) * 50;
                } else {
                    if (qrsPhase < 0.2) yOffset += 12;
                    else if (qrsPhase < 0.7) yOffset -= 75;
                    else yOffset += 22;
                }
            }

            // ST Segment & T Wave
            const stStart = qrsStart + qrsDuration;
            const tStart = stStart + 0.10;

            if (cond === 'ST_ELEV') {
                if (phase >= stStart && phase < tStart) {
                    yOffset -= 25;
                } else if (phase >= tStart && phase < (tStart + 0.20)) {
                    const tPhase = (phase - tStart) / 0.20;
                    yOffset -= (25 + Math.sin(tPhase * Math.PI) * 20);
                }
            } else if (cond === 'LBBB') {
                if (phase >= stStart && phase < (tStart + 0.20)) {
                    yOffset += 15;
                }
            } else {
                if (phase >= tStart && phase < (tStart + 0.20)) {
                    const tPhase = (phase - tStart) / 0.20;
                    yOffset -= Math.sin(tPhase * Math.PI) * 18;
                }
            }

            return midY + (yOffset * gain);
        };

        handleResize();
        window.addEventListener('resize', handleResize);

        const animate = (now) => {
            if (!isRunning) return;

            const timeSec = (now - startTime) / 1000;
            const w = canvas.width;
            const h = canvas.height;

            const sweepWidth = 15;
            ctx.fillStyle = '#090d16';
            ctx.fillRect(sweepX, 0, sweepWidth, h);

            // Redraw grid slice
            ctx.save();
            ctx.beginPath();
            ctx.rect(sweepX, 0, sweepWidth, h);
            ctx.clip();
            drawGrid();
            ctx.restore();

            const y = getWaveY(timeSec, condition);

            ctx.strokeStyle = (condition === 'ST_ELEV' || condition === 'AF') ? '#ef4444' : '#10b981';
            ctx.lineWidth = 2.2;
            ctx.lineJoin = 'round';

            if (lastY !== undefined && sweepX > 0) {
                ctx.beginPath();
                ctx.moveTo(sweepX - speed, lastY);
                ctx.lineTo(sweepX, y);
                ctx.stroke();
            }

            lastY = y;
            sweepX += speed;

            if (sweepX >= w) {
                sweepX = 0;
                lastY = y;
            }

            animRef.current = requestAnimationFrame(animate);
        };

        animRef.current = requestAnimationFrame(animate);

        return () => {
            isRunning = false;
            window.removeEventListener('resize', handleResize);
            if (animRef.current) cancelAnimationFrame(animRef.current);
        };
    }, [condition, lead, speed, gain]);

    return (
        <div className="w-full rounded-xl overflow-hidden border border-rose-500/20 shadow-2xl relative bg-[#090d16]">
            <canvas ref={canvasRef} className="w-full block" style={{ height: '200px' }} />
        </div>
    );
}

