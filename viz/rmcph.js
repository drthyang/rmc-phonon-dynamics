// rmcph.js  —  rendering and DOM wiring for rmcph.html
// S(Q,E) and DOS compute live off the main thread in sqeworker.js.
'use strict';

// ℏ²/(2m_n) — recoil parabola E_recoil(Q) = HBAR2_2MN * Q² (used by renderHeatmap)
// Q in Å⁻¹ (2π convention), E in meV.  ℏ=1.0546e-34 J·s, m_n=1.6749e-27 kg
const HBAR2_2MN = 2.0723;       // meV · Å²
// phonopy band.yaml stores frequencies in THz; all our physics uses meV
const THZ_TO_MEV = 4.135667696; // h·10¹²/e  (h=6.62607e-34 J·s, e=1.60218e-19 C)

// Hook Highcharts (loaded by phononwebsite) to display band frequencies in meV
(function hookHighcharts() {
    if (typeof Highcharts === 'undefined') { setTimeout(hookHighcharts, 50); return; }

    // Scale y-data THz → meV on every setData call
    const origSetData = Highcharts.Series.prototype.setData;
    Highcharts.Series.prototype.setData = function(data, ...args) {
        const conv = Array.isArray(data)
            ? data.map(p => Array.isArray(p) ? [p[0], p[1] * THZ_TO_MEV] : p)
            : data;
        const r = origSetData.call(this, conv, ...args);
        try {
            const bEmin = parseFloat(document.getElementById('band-emin')?.value) ?? 0;
            const bEmax = parseFloat(document.getElementById('band-emax')?.value) ?? 100;
            this.chart?.yAxis?.[0]?.update({ title: { text: 'Energy (meV)' }, min: bEmin, max: bEmax }, false);
            this.chart?.update({
                tooltip: {
                    formatter: function() {
                        return `Energy: <b>${this.y.toFixed(2)} meV</b>`;
                    }
                }
            }, false);
        }
        catch(e) {}
        return r;
    };

    // Override tooltip at render time — runs on every hover, bypasses phononwebsite's formatter
    const origRefresh = Highcharts.Tooltip.prototype.refresh;
    Highcharts.Tooltip.prototype.refresh = function(point, mouseEvent) {
        const opts = this.chart?.options?.tooltip;
        if (opts) {
            const saved = opts.formatter;
            opts.formatter = function() {
                return `Energy: <b>${this.y.toFixed(2)} meV</b>`;
            };
            origRefresh.call(this, point, mouseEvent);
            opts.formatter = saved;
        } else {
            origRefresh.call(this, point, mouseEvent);
        }
    };
})();

// ── Colormaps ────────────────────────────────────────────────────────────────

function interpRGB(t, stops) {
    const n = stops.length - 1;
    const i = Math.min(Math.floor(t * n), n - 1);
    const f = t * n - i;
    const a = stops[i], b = stops[i+1];
    return [
        Math.round(a[0] + f*(b[0]-a[0])),
        Math.round(a[1] + f*(b[1]-a[1])),
        Math.round(a[2] + f*(b[2]-a[2]))
    ];
}

const CMAPS = {
    inferno:  [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[252,188,52],[252,255,164]],
    viridis:  [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
    hot:      [[0,0,0],[255,0,0],[255,255,0],[255,255,255]],
    plasma:   [[13,8,135],[126,3,167],[204,71,120],[248,149,64],[240,249,33]],
    coolwarm: [[59,76,192],[99,130,223],[161,189,225],[221,221,221],[241,182,148],[210,96,72],[180,4,38]],
};

function applyColormap(t, name) {
    return interpRGB(Math.max(0, Math.min(1, t)), CMAPS[name] || CMAPS.inferno);
}

// ── Generic heatmap renderer ─────────────────────────────────────────────────
const ML = 52, MR = 10, MT = 8, MB = 30;

function renderHeatmap(canvas, result, colormap, logScale) {
    const {S, nX, nE, Eaxis, Smax, xMin, xMax, xLabel, labels} = result;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (Smax <= 0 || nX < 2) return;

    const plotW = W - ML - MR;
    const plotH = H - MT - MB;
    if (plotW < 10 || plotH < 10) return;

    const Emin = Eaxis[0], Emax = Eaxis[nE-1];
    const recoilA  = result.recoilA || 0;
    const Ei_fixed = result.Ei || 0;        // incident energy; 0 = unconstrained
    const ki_fixed = Ei_fixed > 0 ? Math.sqrt(Ei_fixed / HBAR2_2MN) : 0;

    // ── Heatmap pixels ───────────────────────────────────────────────────
    const img = ctx.createImageData(plotW, plotH);
    const pix = img.data;

    for (let px = 0; px < plotW; px++) {
        const xi = Math.min(Math.floor(px * nX / plotW), nX - 1);

        // |Q| at this pixel column (powder only; ignored for path)
        const Qpx     = xMin + (px + 0.5) / plotW * (xMax - xMin);

        for (let py = 0; py < plotH; py++) {
            const off = (py * plotW + px) * 4;

            // Energy at this pixel row
            const E = Emin + (1 - (py + 0.5) / plotH) * (Emax - Emin);

            // Kinematically forbidden — white
            let forbidden = false;
            if (Ei_fixed > 0) {
                if (E > Ei_fixed) {
                    forbidden = true;
                } else {
                    const Ef = Ei_fixed - E;
                    const kf = Math.sqrt(Ef / HBAR2_2MN);
                    forbidden = Qpx > ki_fixed + kf || Qpx < Math.abs(ki_fixed - kf);
                }
            } else if (recoilA > 0) {
                forbidden = E > recoilA * Qpx * Qpx;
            }
            if (forbidden) {
                pix[off]=255; pix[off+1]=255; pix[off+2]=255; pix[off+3]=255;
                continue;
            }

            const iE = Math.min(Math.floor((plotH-1-py) * nE / plotH), nE-1);
            let v = S[xi * nE + iE] / Smax;
            if (logScale) v = v > 0 ? Math.log10(1 + 9*v) : 0;
            const [r,g,b] = applyColormap(v, colormap);
            pix[off]=r; pix[off+1]=g; pix[off+2]=b; pix[off+3]=255;
        }
    }
    ctx.putImageData(img, ML, MT);

    // ── Kinematic boundary curves ────────────────────────────────────────
    if (Ei_fixed > 0) {
        ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        const Eplot_max = Math.min(Emax, Ei_fixed);

        // Upper boundary: Q_max(E) = ki + kf
        ctx.strokeStyle = 'rgba(40,40,40,0.75)';
        ctx.beginPath();
        let started = false;
        for (let py = 0; py < plotH; py++) {
            const E = Emin + (1 - (py + 0.5) / plotH) * (Emax - Emin);
            if (E > Ei_fixed || E < Emin) { started = false; continue; }
            const Qb = ki_fixed + Math.sqrt(Math.max(0, Ei_fixed - E) / HBAR2_2MN);
            if (Qb < xMin || Qb > xMax) { started = false; continue; }
            const px = ML + plotW * (Qb - xMin) / (xMax - xMin);
            const py_c = MT + py;
            if (!started) { ctx.moveTo(px, py_c); started = true; }
            else          ctx.lineTo(px, py_c);
        }
        ctx.stroke();

        // Lower boundary: Q_min(E) = |ki − kf|
        ctx.beginPath();
        started = false;
        for (let py = 0; py < plotH; py++) {
            const E = Emin + (1 - (py + 0.5) / plotH) * (Emax - Emin);
            if (E > Ei_fixed || E < Emin) { started = false; continue; }
            const kf = Math.sqrt(Math.max(0, Ei_fixed - E) / HBAR2_2MN);
            const Qb = Math.abs(ki_fixed - kf);
            if (Qb < xMin || Qb > xMax) { started = false; continue; }
            const px = ML + plotW * (Qb - xMin) / (xMax - xMin);
            const py_c = MT + py;
            if (!started) { ctx.moveTo(px, py_c); started = true; }
            else          ctx.lineTo(px, py_c);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    } else if (recoilA > 0) {
        ctx.strokeStyle = 'rgba(60,60,60,0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        let started = false;
        for (let px = 0; px < plotW; px++) {
            const Q  = xMin + (px + 0.5) / plotW * (xMax - xMin);
            const Er = recoilA * Q * Q;
            if (Er < Emin || Er > Emax) { started = false; continue; }
            const py = MT + plotH * (1 - (Er - Emin) / (Emax - Emin));
            if (!started) { ctx.moveTo(ML + px, py); started = true; }
            else          ctx.lineTo(ML + px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // ── E-axis ticks (left) ──────────────────────────────────────────────
    ctx.font = '10px Inter, system-ui, sans-serif'; ctx.fillStyle = '#222';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const nEticks = Math.min(8, Math.floor(plotH / 28));
    for (let ti = 0; ti <= nEticks; ti++) {
        const E  = Emin + (Emax-Emin) * ti / nEticks;
        const py = MT + plotH - plotH * ti / nEticks;
        ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(ML, py); ctx.lineTo(ML+plotW, py); ctx.stroke();
        ctx.fillStyle = '#222';
        ctx.fillText(E.toFixed(1), ML-3, py);
    }
    ctx.save();
    ctx.translate(11, MT + plotH/2);
    ctx.rotate(-Math.PI/2);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('Energy (meV)', 0, 0);
    ctx.restore();

    // ── E=0 dashed line ──────────────────────────────────────────────────
    if (Emin < 0 && Emax > 0) {
        const y0 = MT + plotH * (1 - (0-Emin)/(Emax-Emin));
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 1; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(ML, y0); ctx.lineTo(ML+plotW, y0); ctx.stroke();
        ctx.setLineDash([]);
    }

    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = '#222';

    if (labels.length > 0) {
        // Path: vertical lines + high-symmetry labels
        ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 0.8;
        for (const {xIdx, text} of labels) {
            const px = ML + Math.round(xIdx / (nX-1) * plotW);
            ctx.beginPath(); ctx.moveTo(px, MT); ctx.lineTo(px, MT+plotH); ctx.stroke();
            ctx.fillText(text, px, MT+plotH+4);
        }
    } else {
        // Powder: numeric |Q| ticks
        const nXticks = Math.min(6, Math.floor(plotW / 55));
        ctx.font = '10px Inter, system-ui, sans-serif';
        for (let ti = 0; ti <= nXticks; ti++) {
            const Q  = xMin + (xMax-xMin) * ti / nXticks;
            const px = ML + plotW * ti / nXticks;
            ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(px, MT); ctx.lineTo(px, MT+plotH); ctx.stroke();
            ctx.fillStyle = '#222';
            ctx.fillText(Q.toFixed(2), px, MT+plotH+4);
        }
    }

    // x-axis label + border
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.font = '10px Inter, system-ui, sans-serif'; ctx.fillStyle = '#444';
    ctx.fillText(xLabel, ML + plotW/2, H);
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1;
    ctx.strokeRect(ML, MT, plotW, plotH);
}

// ── DOS renderer ─────────────────────────────────────────────────────────────
// ML_D / MR_D: narrow margins; MT and MB reused from heatmap so y-axes align.
const ML_D = 4, MR_D = 6;

function renderDOS(canvas, dosResult) {
    const { dos, nE, Emin, Emax, dosMax } = dosResult;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (dosMax <= 0) return;

    const plotW = W - ML_D - MR_D;
    const plotH = H - MT - MB;
    if (plotW < 4 || plotH < 10) return;

    // y(Ei) = MT + plotH*(1 - Ei/(nE-1)) — same mapping as renderHeatmap rows
    // Filled area + outline
    ctx.beginPath();
    ctx.moveTo(ML_D, MT + plotH);
    for (let Ei = 0; Ei < nE; Ei++) {
        const x = ML_D + (dos[Ei] / dosMax) * plotW;
        const y = MT + plotH * (1 - Ei / (nE - 1));
        ctx.lineTo(x, y);
    }
    ctx.lineTo(ML_D, MT);
    ctx.closePath();
    ctx.fillStyle = 'rgba(37,99,235,0.15)';
    ctx.fill();

    ctx.beginPath();
    for (let Ei = 0; Ei < nE; Ei++) {
        const x = ML_D + (dos[Ei] / dosMax) * plotW;
        const y = MT + plotH * (1 - Ei / (nE - 1));
        Ei === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(37,99,235,0.75)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Horizontal gridlines aligned to heatmap E-ticks
    const nEticks = Math.min(8, Math.floor(plotH / 28));
    ctx.strokeStyle = 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 0.5;
    for (let ti = 0; ti <= nEticks; ti++) {
        const py = MT + plotH - plotH * ti / nEticks;
        ctx.beginPath(); ctx.moveTo(ML_D, py); ctx.lineTo(ML_D + plotW, py); ctx.stroke();
    }

    // E=0 dashed line
    if (Emin < 0 && Emax > 0) {
        const y0 = MT + plotH * (1 - (0 - Emin) / (Emax - Emin));
        ctx.strokeStyle = 'rgba(100,100,100,0.4)';
        ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(ML_D, y0); ctx.lineTo(ML_D + plotW, y0); ctx.stroke();
        ctx.setLineDash([]);
    }

    // x-axis label
    ctx.font = '10px Inter, system-ui, sans-serif'; ctx.fillStyle = '#444';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('PhDOS', ML_D + plotW / 2, H);

    // Border
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.strokeRect(ML_D, MT, plotW, plotH);
}

// ── Colorbar ─────────────────────────────────────────────────────────────────

function drawColorbar(cbCanvas, colormap) {
    const ctx = cbCanvas.getContext('2d');
    const W = cbCanvas.width, H = cbCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const plotH = H - MT - MB;
    if (plotH < 10) return;

    const bw = 14;
    for (let y = 0; y < plotH; y++) {
        const [r,g,b] = applyColormap(1 - y / plotH, colormap);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(0, MT + y, bw, 1);
    }
    ctx.strokeStyle = '#666'; ctx.lineWidth = 0.5;
    ctx.strokeRect(0, MT, bw, plotH);

    // Tick marks + labels at 0%, 25%, 50%, 75%, 100%
    const nTicks = 4;
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#333'; ctx.textAlign = 'left';
    for (let ti = 0; ti <= nTicks; ti++) {
        const t  = ti / nTicks;
        const y  = MT + t * plotH;
        ctx.strokeStyle = '#666'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(bw, y); ctx.lineTo(bw + 3, y); ctx.stroke();
        const label = ti === 0 ? '1.0' : ti === nTicks ? '0.0' : (1 - t).toFixed(2);
        ctx.textBaseline = ti === 0 ? 'top' : ti === nTicks ? 'bottom' : 'middle';
        ctx.fillText(label, bw + 5, y);
    }
}

// ── Main init ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Band structure y-axis range inputs
    ['band-emin', 'band-emax'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => {
            const min = parseFloat(document.getElementById('band-emin').value);
            const max = parseFloat(document.getElementById('band-emax').value);
            if (isNaN(min) || isNaN(max) || min >= max) return;
            Highcharts?.charts?.forEach(c => c?.yAxis?.[0]?.setExtremes(min, max));
        });
    });

    const fileInput  = document.getElementById('file-input');
    const computeBtn = document.getElementById('sqe-compute');
    const toggleBtn  = document.getElementById('sqe-toggle');
    const panelBody  = document.getElementById('sqe-body');
    const statusEl   = document.getElementById('sqe-status');

    const powCanvas  = document.getElementById('sqe-pow-canvas');
    const powCb      = document.getElementById('sqe-pow-cb');
    const dosCanvas  = document.getElementById('dos-canvas');

    let ydata     = null;
    let powResult = null;
    let dosResult = null;

    // ── Perf instrumentation (logs each phase to DevTools console) ───────
    // Filter the console for "[sqe perf]" to see the breakdown.
    const __perf = { plog: (label, ms) =>
        console.log(`[sqe perf] ${label.padEnd(32)} ${ms.toFixed(1).padStart(8)} ms`) };
    window.__sqePerf = __perf;

    // ── Compute worker ────────────────────────────────────────────────────
    // S(Q,E) / DOS compute runs off the main thread so parameter sweeps don't
    // freeze the UI.  `pendingId` ignores stale replies if the user rapidly
    // changes parameters; only the latest result is rendered.
    let worker     = null;
    let nextId     = 0;
    let pendingId  = -1;
    try {
        worker = new Worker('sqeworker.js');
        worker.onmessage = (ev) => {
            const tRecv = performance.now();
            const m = ev.data;

            // 'loaded' reply: YAML parse + THz finished in the worker.
            if (m.type === 'loaded') {
                if (m.error) {
                    statusEl.textContent = '✗ ' + m.error;
                    console.error('sqeworker load:', m.error);
                    return;
                }
                const s = m.stats || {};
                ydata = s;   // stand-in so existing `if (!ydata)` gates still work
                statusEl.textContent =
                    `✓ ${s.natom} atoms · ${s.nqpoint} q-pts · ${s.nModes} modes`;
                if (m.timings && __perf.tLoadSend) {
                    __perf.plog('worker YAML parse', m.timings.parse);
                    __perf.plog('worker THz → meV', m.timings.thz);
                }
                return;
            }

            // 'compute' reply
            if (m.id !== pendingId) return;            // stale; drop
            if (m.error) {
                statusEl.textContent = '✗ ' + m.error;
                console.error('sqeworker:', m.error);
                return;
            }
            if (__perf.tComputeReq) {
                const rt = tRecv - __perf.tComputeReq;
                __perf.plog('worker round-trip', rt);
                if (m.timings) {
                    __perf.plog('  ├─ compute (worker total)', m.timings.total);
                    __perf.plog('  │   ├─ S(Q,E) powder',       m.timings.pow);
                    __perf.plog('  │   └─ DOS',                  m.timings.dos);
                    __perf.plog('  └─ overhead (post + IPC)',    rt - m.timings.total);
                }
                __perf.tComputeReq = 0;
            }
            powResult = m.powResult;
            dosResult = m.dosResult;
            const tR0 = performance.now();
            resizeCanvases();
            redraw();
            const tR1 = performance.now();
            __perf.plog('render (heatmap+DOS+cb)', tR1 - tR0);
            if (__perf.tFile) {
                __perf.plog('TOTAL file→S(Q,E) shown', tR1 - __perf.tFile);
                __perf.tFile = 0;
            }
            const Qmx = powResult ? powResult.xMax.toFixed(2) : '?';
            statusEl.textContent = `Done · Q_max ${Qmx} Å⁻¹`;

            // Deferred phononwebsite load: now that S(Q,E) is painted, kick off
            // the 3D viewer in the background. Main thread will still block
            // while phononwebsite parses + builds the supercell mesh, but the
            // user already has S(Q,E) to look at.
            if (pendingFile && !phononLoaded) {
                setTimeout(loadDeferredPhonon, 0);
            }
        };
        worker.onerror = (e) => {
            statusEl.textContent = '✗ worker: ' + (e.message || 'error');
            console.error('sqeworker error:', e);
        };
    } catch (err) {
        statusEl.textContent = '✗ worker unavailable: ' + err.message;
        console.error(err);
    }

    // ── Deferred phononwebsite loading ────────────────────────────────────
    // Phononwebsite's own file-input handler synchronously builds the 3D
    // structure viewer when a file is loaded — for a large supercell that's
    // tens of seconds of blocked main thread. We intercept the first change
    // event (capture phase + stopImmediatePropagation), do our YAML parse +
    // S(Q,E) compute, and only let phononwebsite see the event AFTER the
    // S(Q,E) heatmap has rendered.
    let pendingFile      = null;   // first-load file, queued for phononwebsite
    let phononLoaded     = false;  // becomes true after phononwebsite has seen the file
    let syntheticDispatch = false; // suppress our handler for the re-dispatched event
    let postLoadSnap     = null;   // DOM snapshot to restore after phononwebsite resets it

    function restoreSnap(snap) {
        Object.entries(snap).forEach(([id, val]) => {
            const el = document.getElementById(id);
            if (el) el.value = val;
        });
        document.getElementById('update')?.click();
        // Re-fit camera after supercell update so the full structure is visible.
        // Multiply cameraDistance by 1.35 so the 2×2×1 structure is fully visible.
        setTimeout(() => {
            document.getElementById('cameraz')?.click();
            setTimeout(() => { if (typeof v !== 'undefined') { v.cameraDistance *= 1.35; v.setCameraDirection('z'); } }, 80);
        }, 150);
        document.getElementById('modeselect')?.click();
    }

    function loadDeferredPhonon() {
        if (phononLoaded || !pendingFile) return;
        phononLoaded = true;
        const file = pendingFile;
        pendingFile = null;
        statusEl.textContent += ' · loading 3D…';

        // Re-fire the change event so phononwebsite's bubble-phase listener
        // picks up the file. Our capture-phase handler will see
        // syntheticDispatch=true and bail out, leaving the event to bubble.
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
        syntheticDispatch = true;
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));

        // After phononwebsite settles, restore the DOM values it wiped.
        if (postLoadSnap) {
            const snap = postLoadSnap;
            setTimeout(() => restoreSnap(snap), 300);
            setTimeout(() => restoreSnap(snap), 700);
        }
    }

    // Capture-phase listener fires BEFORE phononwebsite's bubble-phase one,
    // so we can stopImmediatePropagation to block phononwebsite on the
    // initial load.
    fileInput.addEventListener('change', (e) => {
        if (syntheticDispatch) {
            // Re-dispatched event from loadDeferredPhonon — let phononwebsite
            // handle it, skip our YAML parse (we already did it).
            syntheticDispatch = false;
            return;
        }
        const file = e.target.files[0];
        if (!file) return;

        __perf.tFile = performance.now();
        __perf.plog('--- file selected ---', 0);

        // Snapshot all user-controlled settings; phononwebsite resets them
        // when it eventually runs (either deferred or on subsequent loads).
        const snap = {};
        ['nx','ny','nz','kindex','nindex',
         'sqe-emin','sqe-emax','sqe-temp','sqe-sigma','sqe-ei'].forEach(id => {
            const el = document.getElementById(id);
            if (el) snap[id] = el.value;
        });
        postLoadSnap = snap;

        if (!phononLoaded) {
            // First load — defer phononwebsite by blocking its bubble listener.
            pendingFile = file;
            e.stopImmediatePropagation();
        }

        const reader = new FileReader();
        reader.onload = (evt) => {
            const tRead = performance.now();
            __perf.plog('FileReader.readAsText', tRead - __perf.tFile);
            try {
                // Ship the raw text to the worker. YAML parse + THz happen
                // off the main thread; we get a 'loaded' reply with stats.
                const t0 = performance.now();
                worker?.postMessage({ type: 'load', id: ++nextId, text: evt.target.result });
                const t1 = performance.now();
                __perf.plog('postMessage(load) text', t1 - t0);
                __perf.tLoadDone = t1;
                __perf.tLoadSend = t1;
                ydata = { __loading: true };   // gate so triggerCompute() will fire
                statusEl.textContent = 'Parsing…';

                if (phononLoaded) {
                    // Subsequent loads: phononwebsite runs too; restore DOM after it resets.
                    setTimeout(() => {
                        restoreSnap(snap);
                        if (panelBody.style.display !== 'none') triggerCompute();
                    }, 300);
                    setTimeout(() => {
                        restoreSnap(snap);
                        if (panelBody.style.display !== 'none') triggerCompute();
                    }, 700);
                } else {
                    // First load: phononwebsite deferred. Trigger compute
                    // immediately — the worker queues 'compute' behind 'load'
                    // and processes them in order.
                    if (panelBody.style.display !== 'none') triggerCompute();
                }
            } catch(err) {
                statusEl.textContent = '✗ ' + err.message;
                console.error(err);
            }
        };
        reader.readAsText(file);
    }, true);  // CAPTURE phase

    computeBtn.addEventListener('click', triggerCompute);

    // Compute-affecting inputs: recompute on change
    ['sqe-temp','sqe-sigma','sqe-emin','sqe-emax','sqe-ei'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => { if (ydata) triggerCompute(); });
    });
    // Render-only inputs: redraw without recomputing
    ['sqe-cmap','sqe-log'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => { if (powResult) redraw(); });
    });

    toggleBtn.addEventListener('click', () => {
        const showing = panelBody.style.display !== 'none';
        panelBody.style.display = showing ? 'none' : '';
        toggleBtn.textContent   = showing ? '▼ S(Q,E)' : '▲ S(Q,E)';
        if (!showing && ydata && !powResult) triggerCompute();
        if (!showing && powResult) { resizeCanvases(); redraw(); }
    });

    window.addEventListener('resize', () => {
        if (powResult && panelBody.style.display !== 'none') { resizeCanvases(); redraw(); }
    });

    // ── Responsive nav ────────────────────────────────────────────────────
    const sidebarEl  = document.querySelector('.flex-menu');
    const backdropEl = document.getElementById('sidebar-backdrop');
    const rootEl     = document.getElementById('rmcph-root');

    function closeSidebar() {
        sidebarEl?.classList.remove('open');
        backdropEl?.classList.remove('open');
    }
    document.getElementById('nav-toggle')?.addEventListener('click', () => {
        const isOpen = sidebarEl.classList.toggle('open');
        backdropEl.classList.toggle('open', isOpen);
    });
    backdropEl?.addEventListener('click', closeSidebar);

    // Phone tab bar: 3D / Band / S(Q,E) tabs
    document.querySelectorAll('#tab-bar [data-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#tab-bar [data-tab]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            rootEl.dataset.tab = btn.dataset.tab;
            // Resize canvases when switching to S(Q,E) tab
            if (btn.dataset.tab === 'sqe' && powResult) { resizeCanvases(); redraw(); }
        });
    });
    // Settings tab toggles sidebar drawer
    document.getElementById('tab-settings')?.addEventListener('click', () => {
        const isOpen = sidebarEl.classList.toggle('open');
        backdropEl.classList.toggle('open', isOpen);
    });

    function triggerCompute() {
        if (!ydata || !worker) return;
        const T     = parseFloat(document.getElementById('sqe-temp').value)  || 5;
        const sigma = parseFloat(document.getElementById('sqe-sigma').value) || 0.5;
        const Emin  = parseFloat(document.getElementById('sqe-emin').value);
        const Emax  = parseFloat(document.getElementById('sqe-emax').value);
        if (isNaN(Emin) || isNaN(Emax) || Emin >= Emax) {
            statusEl.textContent = '✗ Invalid energy range'; return;
        }
        const Ei_in = parseFloat(document.getElementById('sqe-ei').value) || 0;
        statusEl.textContent = 'Computing…';
        const tReq = performance.now();
        if (__perf.tLoadDone) {
            __perf.plog('idle gap (load→1st compute)', tReq - __perf.tLoadDone);
            __perf.tLoadDone = 0;
        }
        __perf.tComputeReq = tReq;
        pendingId = ++nextId;
        worker.postMessage({
            type: 'compute',
            id:   pendingId,
            params: { T, sigma, Emin, Emax, nE: 300, nQbins: 100, Ei: Ei_in }
        });
    }

    function redraw() {
        const cmap = document.getElementById('sqe-cmap').value || 'inferno';
        const log  = document.getElementById('sqe-log').checked;
        if (powResult) { renderHeatmap(powCanvas, powResult, cmap, log); drawColorbar(powCb, cmap); }
        if (dosResult && dosCanvas) renderDOS(dosCanvas, dosResult);
    }

    function resizeCanvases() {
        const wrap = document.getElementById('sqe-canvas-wrap');
        if (!wrap) return;
        const H = wrap.clientHeight || 280;

        const qWrap = document.getElementById('sqe-pow-wrap');
        if (qWrap) {
            powCanvas.width  = Math.max(10, qWrap.clientWidth - (powCb.width || 36) - 6);
            powCanvas.height = Math.max(10, H);
            powCb.height = powCanvas.height;
        }
        const dWrap = document.getElementById('dos-wrap');
        if (dWrap && dosCanvas) {
            dosCanvas.width  = Math.max(10, dWrap.clientWidth - 2);
            dosCanvas.height = Math.max(10, H);
        }
    }
});
