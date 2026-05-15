// rmcph.js  —  S(Q,E) computation and rendering for rmcph.html
'use strict';

// ── Neutron coherent scattering lengths b (fm) ─────────────────────────────
const B_COH = {
    H:-3.739,He:3.26,Li:-1.90,Be:7.79,B:5.30,C:6.646,N:9.36,O:5.803,
    F:5.654,Na:3.63,Mg:5.375,Al:3.449,Si:4.1491,P:5.13,S:2.847,Cl:9.577,
    K:3.67,Ca:4.70,Sc:12.29,Ti:-3.370,V:-0.3824,Cr:3.635,Mn:-3.73,
    Fe:9.45,Co:2.49,Ni:10.3,Cu:7.718,Zn:5.680,Ga:7.288,Ge:8.185,
    As:6.58,Se:7.970,Br:6.795,Rb:7.09,Sr:7.02,Y:7.75,Zr:7.16,Nb:7.054,
    Mo:6.715,Tc:6.80,Ru:7.03,Rh:5.88,Pd:5.91,Ag:5.922,Cd:4.87,In:4.065,
    Sn:6.225,Sb:5.57,Te:5.80,I:5.28,Cs:5.42,Ba:5.07,La:8.24,Ce:4.84,
    Pr:4.58,Nd:7.69,Sm:0.80,Eu:7.22,Gd:6.5,Tb:7.38,Dy:16.9,Ho:8.01,
    Er:7.79,Tm:7.07,Yb:12.43,Lu:7.21,Hf:7.77,Ta:6.91,W:4.86,Re:9.2,
    Os:10.7,Ir:10.6,Pt:9.60,Au:7.63,Hg:12.692,Tl:8.776,Pb:9.405,Bi:8.532
};

const KB        = 0.08617333;   // meV / K
// ℏ²/(2m_n) — recoil parabola E_recoil(Q) = HBAR2_2MN * Q²
// Q in Å⁻¹ (2π convention), E in meV.  ℏ=1.0546e-34 J·s, m_n=1.6749e-27 kg
const HBAR2_2MN = 2.0723;       // meV · Å²
// phonopy band.yaml stores frequencies in THz; all our physics uses meV
const THZ_TO_MEV = 4.135667696; // h·10¹²/e  (h=6.62607e-34 J·s, e=1.60218e-19 C)

// Hook Highcharts (loaded by phononwebsite) to display band frequencies in meV
(function hookHighcharts() {
    if (typeof Highcharts === 'undefined') { setTimeout(hookHighcharts, 50); return; }
    const origSetData = Highcharts.Series.prototype.setData;
    Highcharts.Series.prototype.setData = function(data, ...args) {
        const conv = Array.isArray(data)
            ? data.map(p => Array.isArray(p) ? [p[0], p[1] * THZ_TO_MEV] : p)
            : data;
        const r = origSetData.call(this, conv, ...args);
        try {
            this.chart?.yAxis?.[0]?.update({ title: { text: 'Energy (meV)' }, min: 0, max: 100 }, false);
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
})();

// ── Physics helpers ─────────────────────────────────────────────────────────

function bose(omega, T) {
    if (T <= 0 || omega <= 0) return 0;
    const x = omega / (KB * T);
    return x > 100 ? 0 : 1.0 / (Math.exp(x) - 1.0);
}

function gauss(x, mu, sigma) {
    const d = x - mu;
    return Math.exp(-0.5 * d * d / (sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
}

// ── Shared accumulator ───────────────────────────────────────────────────────
// Adds S(q, E) contributions from all modes at one q-point into grid[xIdx*nE+Ei].
// Passing xIdx=0 and a plain Float64Array(nE) as grid gives a 1-D scratch buffer.
function accumulateModes(bands, atoms, b2m, T, sigma, Emin, dE, nE, xIdx, grid) {
    const cutoff = 4 * sigma;
    for (let mi = 0; mi < bands.length; mi++) {
        const mode  = bands[mi];
        const omega = mode.frequency;
        const ev    = mode.eigenvector;
        if (!ev || Math.abs(omega) < 1e-6) continue;

        let F2 = 0;
        for (let ai = 0; ai < atoms.length; ai++) {
            let m2 = 0;
            for (let ci = 0; ci < 3; ci++) {
                const re = ev[ai][ci][0], im = ev[ai][ci][1];
                m2 += re*re + im*im;
            }
            F2 += b2m[ai] * m2;
        }

        const absOmega  = Math.abs(omega);
        const n         = bose(absOmega, T);
        const occ       = omega > 0 ? n + 1 : n;
        const prefactor = F2 * occ / absOmega;

        const iCenter = (omega - Emin) / dE;
        const iLo = Math.max(0,    Math.floor(iCenter - cutoff / dE));
        const iHi = Math.min(nE-1, Math.ceil( iCenter + cutoff / dE));
        const base = xIdx * nE;
        for (let Ei = iLo; Ei <= iHi; Ei++) {
            grid[base + Ei] += prefactor * gauss(Emin + Ei * dE, omega, sigma);
        }
    }
}

// ── Powder S(|Q|,E) ──────────────────────────────────────────────────────────
//
// Correct physics:
//   - Phonons at reduced wavevector q scatter at Q = q + G (G = reciprocal
//     lattice vector), so many Brillouin zones contribute to a given |Q|.
//   - Powder cross-section ∝ Q² · Σ_λ F²(q,λ) · (n+1)/ω · G(E−ω, σ)
//     where F² = Σ_α (b²/m) |e_{α,λ}|²  (no Debye-Waller for simplicity).
//   - Kinematic upper limit: only E ≤ ℏ²Q²/(2m_n) is accessible.
//
// Implementation:
//   For each q-point on the path, compute S(q,E); then for every G-shell
//   up to Q_max, distribute to the Q = |q+G| bin with Gaussian Q-smearing.
//
function computePowderSqE(ydata, T, sigma, Emin, Emax, nE, nQbins, Ei) {
    const phonon = ydata.phonon;
    const atoms  = ydata.points;
    const rl     = ydata.reciprocal_lattice;   // rows = a*,b*,c* WITHOUT 2π
    if (!rl) return null;

    const b2m = atoms.map(a => { const b = B_COH[a.symbol]||0; return a.mass>0 ? b*b/a.mass : 0; });

    // Reciprocal lattice vectors WITH 2π (Å⁻¹)
    const bv   = rl.map(v => [2*Math.PI*v[0], 2*Math.PI*v[1], 2*Math.PI*v[2]]);
    const bmag = bv.map(v => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]));

    // Q display range: if Ei provided, max accessible Q = 2*ki (at E=0); else kinematic estimate from Emax
    const ki_plot  = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
    const Q_max_plot = Ei > 0
        ? Math.max(1.5, 2 * ki_plot * 1.05)
        : Math.max(1.5, Math.sqrt(Math.max(Emax, 1) / HBAR2_2MN) * 1.2);
    const dQ   = Q_max_plot / nQbins;
    const sigQ = 1.8 * dQ;   // Gaussian Q-smearing width

    // Precompute G-vectors within reach of the plot range
    const b_min = Math.min(...bmag);
    const n_max = Math.min(6, Math.ceil(Q_max_plot / b_min) + 1);
    const G_buf = [];
    for (let n1 = -n_max; n1 <= n_max; n1++)
    for (let n2 = -n_max; n2 <= n_max; n2++)
    for (let n3 = -n_max; n3 <= n_max; n3++) {
        const Gx = n1*bv[0][0] + n2*bv[1][0] + n3*bv[2][0];
        const Gy = n1*bv[0][1] + n2*bv[1][1] + n3*bv[2][1];
        const Gz = n1*bv[0][2] + n2*bv[1][2] + n3*bv[2][2];
        // Keep only if it can bring some q into [0, Q_max_plot]
        const Gm = Math.sqrt(Gx*Gx + Gy*Gy + Gz*Gz);
        if (Gm < Q_max_plot + Math.max(...bmag)) G_buf.push([Gx, Gy, Gz]);
    }

    const dE    = (Emax - Emin) / (nE - 1);
    const Eaxis = Float64Array.from({length: nE}, (_, i) => Emin + i * dE);
    const S     = new Float64Array(nQbins * nE);
    const norm  = new Float64Array(nQbins);
    const tmpS  = new Float64Array(nE);   // per-q scratch buffer
    const Qspan = Math.ceil(3.5 * sigQ / dQ);

    for (let qi = 0; qi < phonon.length; qi++) {
        // Compute S(q, E) for this path q-point into tmpS
        tmpS.fill(0);
        accumulateModes(phonon[qi].band, atoms, b2m, T, sigma, Emin, dE, nE, 0, tmpS);

        // Cartesian q (Å⁻¹, with 2π)
        const qf = phonon[qi]['q-position'];
        const qx = 2*Math.PI*(qf[0]*rl[0][0] + qf[1]*rl[1][0] + qf[2]*rl[2][0]);
        const qy = 2*Math.PI*(qf[0]*rl[0][1] + qf[1]*rl[1][1] + qf[2]*rl[2][1]);
        const qz = 2*Math.PI*(qf[0]*rl[0][2] + qf[1]*rl[1][2] + qf[2]*rl[2][2]);

        for (const [Gx, Gy, Gz] of G_buf) {
            const Qvx = qx + Gx, Qvy = qy + Gy, Qvz = qz + Gz;
            const Q   = Math.sqrt(Qvx*Qvx + Qvy*Qvy + Qvz*Qvz);
            if (Q <= 0 || Q > Q_max_plot) continue;

            // |Q|² prefactor (powder cross-section ∝ Q²)
            const Q2   = Q * Q;
            const Qi_f = Q / dQ;
            const Qi_lo = Math.max(0,          Math.floor(Qi_f - Qspan));
            const Qi_hi = Math.min(nQbins - 1, Math.ceil( Qi_f + Qspan));

            for (let Qi = Qi_lo; Qi <= Qi_hi; Qi++) {
                const dQi = (Qi + 0.5) * dQ - Q;
                const wG  = Math.exp(-0.5 * dQi*dQi / (sigQ*sigQ));
                norm[Qi] += wG;                 // Gaussian-only — preserves Q² after division
                const base = Qi * nE;
                for (let Ei = 0; Ei < nE; Ei++) S[base + Ei] += Q2 * wG * tmpS[Ei];
            }
        }
    }

    // Normalize each Q-bin by accumulated weight
    for (let Qi = 0; Qi < nQbins; Qi++) {
        if (norm[Qi] > 0) {
            const base = Qi * nE;
            for (let Ei = 0; Ei < nE; Ei++) S[base + Ei] /= norm[Qi];
        }
    }

    let Smax = 0;
    for (let i = 0; i < S.length; i++) if (S[i] > Smax) Smax = S[i];

    return { S, nX: nQbins, nE, Eaxis, Smax,
             xMin: 0, xMax: Q_max_plot,
             xLabel: '|Q| (Å⁻¹)',
             labels: [],
             recoilA: HBAR2_2MN,
             Ei: Ei > 0 ? Ei : 0 };
}

// ── Phonon DOS ────────────────────────────────────────────────────────────────
function computePhononDOS(ydata, sigma, Emin, Emax, nE) {
    const phonon = ydata.phonon;
    const dE     = (Emax - Emin) / (nE - 1);
    const dos    = new Float64Array(nE);
    const cutoff = 4 * sigma;
    let count = 0;

    for (let qi = 0; qi < phonon.length; qi++) {
        for (const mode of phonon[qi].band) {
            const omega = mode.frequency;
            if (Math.abs(omega) < 1e-6) continue;
            const iCenter = (omega - Emin) / dE;
            const iLo = Math.max(0,    Math.floor(iCenter - cutoff / dE));
            const iHi = Math.min(nE-1, Math.ceil( iCenter + cutoff / dE));
            for (let Ei = iLo; Ei <= iHi; Ei++)
                dos[Ei] += gauss(Emin + Ei * dE, omega, sigma);
            count++;
        }
    }
    if (count > 0) for (let i = 0; i < nE; i++) dos[i] /= count;

    let dosMax = 0;
    for (let i = 0; i < nE; i++) if (dos[i] > dosMax) dosMax = dos[i];

    return { dos, nE, Emin, Emax, dosMax };
}

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

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            try {
                ydata = window.jsyaml.load(evt.target.result);
                // Convert all frequencies from THz (phonopy default) to meV
                if (ydata.phonon) {
                    for (const qpt of ydata.phonon)
                        if (qpt.band) for (const mode of qpt.band) mode.frequency *= THZ_TO_MEV;
                }
                const nM = ydata.phonon?.[0]?.band?.length ?? '?';
                statusEl.textContent =
                    `✓ ${ydata.natom} atoms · ${ydata.nqpoint} q-pts · ${nM} modes`;
                if (panelBody.style.display !== 'none') triggerCompute();
            } catch(err) {
                statusEl.textContent = '✗ ' + err.message;
                console.error(err);
            }
        };
        reader.readAsText(file);
    });

    computeBtn.addEventListener('click', triggerCompute);

    ['sqe-temp','sqe-sigma','sqe-emin','sqe-emax','sqe-ei','sqe-cmap','sqe-log'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => { if (ydata) triggerCompute(); });
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

    function triggerCompute() {
        if (!ydata) return;
        statusEl.textContent = 'Computing…';
        setTimeout(() => {
            try {
                const T     = parseFloat(document.getElementById('sqe-temp').value)  || 5;
                const sigma = parseFloat(document.getElementById('sqe-sigma').value) || 0.5;
                const Emin  = parseFloat(document.getElementById('sqe-emin').value);
                const Emax  = parseFloat(document.getElementById('sqe-emax').value);
                if (isNaN(Emin) || isNaN(Emax) || Emin >= Emax) {
                    statusEl.textContent = '✗ Invalid energy range'; return;
                }
                const Ei_in = parseFloat(document.getElementById('sqe-ei').value) || 0;
                powResult = computePowderSqE(ydata, T, sigma, Emin, Emax, 300, 100, Ei_in);
                dosResult = computePhononDOS(ydata, sigma, Emin, Emax, 300);
                resizeCanvases();
                redraw();
                const Qmx = powResult ? powResult.xMax.toFixed(2) : '?';
                statusEl.textContent = `Done · Q_max ${Qmx} Å⁻¹`;
            } catch(err) {
                statusEl.textContent = '✗ ' + err.message;
                console.error(err);
            }
        }, 15);
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
