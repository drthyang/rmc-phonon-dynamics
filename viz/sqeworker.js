// sqeworker.js  —  off-main-thread S(Q,E) and DOS compute for rmcph.html
//
// Protocol:
//   { type: 'load',    ydata }                 → store ydata, no reply
//   { type: 'compute', id, params }            → compute, reply { id, powResult, dosResult }
//
// ydata is cached so subsequent recomputes (parameter sweeps) skip the
// postMessage clone of the eigenvector array, which dominates transfer time.
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
const HBAR2_2MN = 2.0723;       // meV · Å²

// ── Gaussian LUT for exp(-x²/2) on |x| ∈ [0, LUT_RANGE] ────────────────────
// Inner loops in accumulateModes and the Q-smear in computePowderSqE call
// Math.exp millions of times; a 2048-point LUT with linear interp reproduces
// the Gaussian to ~1e-5 absolute accuracy and is ~5–10× faster than exp().
const LUT_N        = 2048;
const LUT_RANGE    = 5;                // covers 4σ cutoff with margin
const LUT_INV_STEP = LUT_N / LUT_RANGE;
const GAUSS_LUT    = new Float64Array(LUT_N + 2);
for (let i = 0; i <= LUT_N + 1; i++) {
    const x = i / LUT_INV_STEP;
    GAUSS_LUT[i] = Math.exp(-0.5 * x * x);
}

// ── Physics helpers ─────────────────────────────────────────────────────────
function bose(omega, T) {
    if (T <= 0 || omega <= 0) return 0;
    const x = omega / (KB * T);
    return x > 100 ? 0 : 1.0 / (Math.exp(x) - 1.0);
}

// ── Shared accumulator ───────────────────────────────────────────────────────
// Adds S(q, E) contributions from all modes at one q-point into grid[xIdx*nE+Ei].
// Inner E-loop uses the Gaussian LUT instead of calling Math.exp per bin.
function accumulateModes(bands, atoms, b2m, T, sigma, Emin, dE, nE, xIdx, grid) {
    const cutoff      = 4 * sigma;
    const inv_sigma   = 1 / sigma;
    const norm        = 1 / (sigma * Math.sqrt(2 * Math.PI));
    const cutoff_dE   = cutoff / dE;
    const inv_step    = LUT_INV_STEP;
    const lut         = GAUSS_LUT;
    const nAtoms      = atoms.length;

    for (let mi = 0; mi < bands.length; mi++) {
        const mode  = bands[mi];
        const omega = mode.frequency;
        const ev    = mode.eigenvector;
        if (!ev || Math.abs(omega) < 1e-6) continue;

        let F2 = 0;
        for (let ai = 0; ai < nAtoms; ai++) {
            const b2 = b2m[ai];
            if (b2 === 0) continue;
            const eva = ev[ai];
            const re0 = eva[0][0], im0 = eva[0][1];
            const re1 = eva[1][0], im1 = eva[1][1];
            const re2 = eva[2][0], im2 = eva[2][1];
            F2 += b2 * (re0*re0 + im0*im0 + re1*re1 + im1*im1 + re2*re2 + im2*im2);
        }
        if (F2 === 0) continue;

        const absOmega  = Math.abs(omega);
        const n         = bose(absOmega, T);
        const occ       = omega > 0 ? n + 1 : n;
        const prefactor = F2 * occ / absOmega;
        const amp       = prefactor * norm;

        const iCenter = (omega - Emin) / dE;
        const iLo = Math.max(0,    Math.floor(iCenter - cutoff_dE));
        const iHi = Math.min(nE-1, Math.ceil( iCenter + cutoff_dE));
        const base = xIdx * nE;
        for (let Ei = iLo; Ei <= iHi; Ei++) {
            // gauss((Emin + Ei*dE) − ω, σ) via LUT: x = |(E−ω)/σ|
            let d = (Emin + Ei * dE - omega) * inv_sigma;
            if (d < 0) d = -d;
            if (d >= LUT_RANGE) continue;
            const fi = d * inv_step;
            const i0 = fi | 0;
            const frac = fi - i0;
            const g = lut[i0] + frac * (lut[i0 + 1] - lut[i0]);
            grid[base + Ei] += amp * g;
        }
    }
}

// ── Powder S(|Q|,E) ──────────────────────────────────────────────────────────
function computePowderSqE(ydata, T, sigma, Emin, Emax, nE, nQbins, Ei) {
    const phonon = ydata.phonon;
    const atoms  = ydata.points;
    const rl     = ydata.reciprocal_lattice;
    if (!rl) return null;

    const b2m = atoms.map(a => { const b = B_COH[a.symbol]||0; return a.mass>0 ? b*b/a.mass : 0; });

    const bv   = rl.map(v => [2*Math.PI*v[0], 2*Math.PI*v[1], 2*Math.PI*v[2]]);
    const bmag = bv.map(v => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]));

    const ki_plot  = Ei > 0 ? Math.sqrt(Ei / HBAR2_2MN) : 0;
    const Q_max_plot = Ei > 0
        ? Math.max(1.5, 2 * ki_plot * 1.05)
        : Math.max(1.5, Math.sqrt(Math.max(Emax, 1) / HBAR2_2MN) * 1.2);
    const dQ   = Q_max_plot / nQbins;

    const b_min = Math.min(...bmag);
    const sigQ  = Math.max(1.5 * dQ, 0.4 * b_min);
    const inv_sigQ = 1 / sigQ;

    const n_max = Math.min(20, Math.ceil(Q_max_plot / b_min) + 1);
    const G_buf = [];
    const G_reach = Q_max_plot + Math.max(...bmag);
    for (let n1 = -n_max; n1 <= n_max; n1++)
    for (let n2 = -n_max; n2 <= n_max; n2++)
    for (let n3 = -n_max; n3 <= n_max; n3++) {
        const Gx = n1*bv[0][0] + n2*bv[1][0] + n3*bv[2][0];
        const Gy = n1*bv[0][1] + n2*bv[1][1] + n3*bv[2][1];
        const Gz = n1*bv[0][2] + n2*bv[1][2] + n3*bv[2][2];
        if (Math.sqrt(Gx*Gx + Gy*Gy + Gz*Gz) < G_reach) G_buf.push([Gx, Gy, Gz]);
    }
    const nG = G_buf.length;
    // Flat typed array for tighter inner loop (avoids tuple destructuring)
    const Gflat = new Float64Array(nG * 3);
    for (let i = 0; i < nG; i++) {
        Gflat[3*i  ] = G_buf[i][0];
        Gflat[3*i+1] = G_buf[i][1];
        Gflat[3*i+2] = G_buf[i][2];
    }

    const dE    = (Emax - Emin) / (nE - 1);
    const Eaxis = new Float64Array(nE);
    for (let i = 0; i < nE; i++) Eaxis[i] = Emin + i * dE;
    const S     = new Float64Array(nQbins * nE);
    const norm  = new Float64Array(nQbins);
    const tmpS  = new Float64Array(nE);
    const Qspan = Math.ceil(3.5 * sigQ / dQ);

    const wq_acc = new Float64Array(nQbins);
    const wn_acc = new Float64Array(nQbins);

    const lut      = GAUSS_LUT;
    const inv_step = LUT_INV_STEP;

    for (let qi = 0; qi < phonon.length; qi++) {
        tmpS.fill(0);
        accumulateModes(phonon[qi].band, atoms, b2m, T, sigma, Emin, dE, nE, 0, tmpS);

        const qf = phonon[qi]['q-position'];
        const qx = 2*Math.PI*(qf[0]*rl[0][0] + qf[1]*rl[1][0] + qf[2]*rl[2][0]);
        const qy = 2*Math.PI*(qf[0]*rl[0][1] + qf[1]*rl[1][1] + qf[2]*rl[2][1]);
        const qz = 2*Math.PI*(qf[0]*rl[0][2] + qf[1]*rl[1][2] + qf[2]*rl[2][2]);

        wq_acc.fill(0);
        wn_acc.fill(0);
        for (let gi = 0; gi < nG; gi++) {
            const Qvx = qx + Gflat[3*gi  ];
            const Qvy = qy + Gflat[3*gi+1];
            const Qvz = qz + Gflat[3*gi+2];
            const Q   = Math.sqrt(Qvx*Qvx + Qvy*Qvy + Qvz*Qvz);
            if (Q <= 0 || Q > Q_max_plot) continue;

            const Q2   = Q * Q;
            const Qi_f = Q / dQ;
            const Qi_lo = Math.max(0,          Math.floor(Qi_f - Qspan));
            const Qi_hi = Math.min(nQbins - 1, Math.ceil( Qi_f + Qspan));
            for (let Qi = Qi_lo; Qi <= Qi_hi; Qi++) {
                // exp(-0.5*dQi²/σ_Q²) via LUT: x = |dQi/σ_Q|
                let d = ((Qi + 0.5) * dQ - Q) * inv_sigQ;
                if (d < 0) d = -d;
                if (d >= LUT_RANGE) continue;
                const fi = d * inv_step;
                const i0 = fi | 0;
                const wG = lut[i0] + (fi - i0) * (lut[i0 + 1] - lut[i0]);
                wq_acc[Qi] += Q2 * wG;
                wn_acc[Qi] += wG;
            }
        }

        for (let Qi = 0; Qi < nQbins; Qi++) {
            if (wq_acc[Qi] === 0) continue;
            norm[Qi] += wn_acc[Qi];
            const base = Qi * nE;
            const w = wq_acc[Qi];
            for (let Ei = 0; Ei < nE; Ei++) S[base + Ei] += w * tmpS[Ei];
        }
    }

    for (let Qi = 0; Qi < nQbins; Qi++) {
        if (norm[Qi] > 0) {
            const base = Qi * nE;
            const invN = 1 / norm[Qi];
            for (let Ei = 0; Ei < nE; Ei++) S[base + Ei] *= invN;
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

// ── Phonon DOS via histogram + Gaussian convolution ─────────────────────────
// Was O(nQ × nModes × kernelLen) per call; now O(nQ × nModes) for the
// histogram + O(nE × kernelLen) for one convolution.  ~30–60× faster for
// typical inputs while producing the same continuous-Gaussian profile to
// second order in dE/σ (linear-interp deposit reproduces the sub-bin centre).
function computePhononDOS(ydata, sigma, Emin, Emax, nE) {
    const phonon    = ydata.phonon;
    const dE        = (Emax - Emin) / (nE - 1);
    const halfWidth = Math.ceil(4 * sigma / dE);
    const kLen      = 2 * halfWidth + 1;

    // Pad histogram so modes within 4σ of either boundary still smear into
    // the visible range (matches the clamped-window behaviour of the
    // direct-sum implementation).
    const pad     = halfWidth;
    const histLen = nE + 2 * pad;
    const hist    = new Float64Array(histLen);
    let count = 0;

    for (let qi = 0; qi < phonon.length; qi++) {
        const band = phonon[qi].band;
        for (let mi = 0; mi < band.length; mi++) {
            const omega = band[mi].frequency;
            if (Math.abs(omega) < 1e-6) continue;
            count++;
            const fpos = (omega - Emin) / dE + pad;
            const i0   = Math.floor(fpos);
            const f    = fpos - i0;
            if (i0     >= 0 && i0     < histLen) hist[i0]     += (1 - f);
            if (i0 + 1 >= 0 && i0 + 1 < histLen) hist[i0 + 1] += f;
        }
    }

    const kernel = new Float64Array(kLen);
    const knorm  = 1 / (sigma * Math.sqrt(2 * Math.PI));
    const inv2s2 = 0.5 / (sigma * sigma);
    for (let i = 0; i < kLen; i++) {
        const x = (i - halfWidth) * dE;
        kernel[i] = knorm * Math.exp(-x * x * inv2s2);
    }

    const dos = new Float64Array(nE);
    for (let i = 0; i < histLen; i++) {
        const hi = hist[i];
        if (hi === 0) continue;
        // dos[j] += hi * kernel[j - (i - pad) + halfWidth]
        const jBase = i - pad - halfWidth;   // dos index when k = 0
        const kLo = Math.max(0,        -jBase);
        const kHi = Math.min(kLen - 1, nE - 1 - jBase);
        for (let k = kLo; k <= kHi; k++) {
            dos[jBase + k] += hi * kernel[k];
        }
    }

    if (count > 0) { const inv = 1 / count; for (let i = 0; i < nE; i++) dos[i] *= inv; }

    let dosMax = 0;
    for (let i = 0; i < nE; i++) if (dos[i] > dosMax) dosMax = dos[i];

    return { dos, nE, Emin, Emax, dosMax };
}

// ── Message dispatch ────────────────────────────────────────────────────────
let cachedYdata = null;

self.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'load') {
        cachedYdata = msg.ydata;
        return;
    }
    if (msg.type === 'compute') {
        if (!cachedYdata) {
            self.postMessage({ id: msg.id, error: 'no ydata loaded' });
            return;
        }
        try {
            const t0 = performance.now();
            const p = msg.params;
            const powResult = computePowderSqE(
                cachedYdata, p.T, p.sigma, p.Emin, p.Emax, p.nE, p.nQbins, p.Ei
            );
            const t1 = performance.now();
            const dosResult = computePhononDOS(
                cachedYdata, p.sigma, p.Emin, p.Emax, p.nE
            );
            const t2 = performance.now();
            const transfer = [];
            if (powResult) transfer.push(powResult.S.buffer, powResult.Eaxis.buffer);
            if (dosResult) transfer.push(dosResult.dos.buffer);
            self.postMessage({
                id: msg.id, powResult, dosResult,
                timings: { pow: t1 - t0, dos: t2 - t1, total: t2 - t0 }
            }, transfer);
        } catch (err) {
            self.postMessage({ id: msg.id, error: err.message || String(err) });
        }
    }
};
