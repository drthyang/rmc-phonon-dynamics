'use strict';
// Fit Quality panel — X-ray F(Q), X-ray PDF G(r), and G(r) partial pairs.
// Requires Chart.js loaded globally (window.Chart) before this module runs.

async function sqgrFetch(url) {
    const r = await fetch(url);
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail ?? r.statusText); }
    return r.json();
}

// Tableau-10 palette for the 6 partial pairs
const PAIR_COLORS = ['#e15759','#f28e2b','#edc948','#4e79a7','#76b7b2','#59a14f'];
const TOTAL_COLOR = '#0f172a';

const R_MAX = 10; // Å cutoff for G(r) panels
const GRID_COLOR = 'rgba(148, 163, 184, 0.22)';
const TICK_COLOR = '#64748b';

let pluginsReady = false;

function ensureChartPlugins() {
    if (pluginsReady || !window.Chart) return;
    if (window.ChartZoom) window.Chart.register(window.ChartZoom);
    pluginsReady = true;
}

// ── Chart factory ─────────────────────────────────────────────────────────────

function makeChart(canvas, datasets, xLabel, yLabel, bounds) {
    ensureChartPlugins();

    // Destroy any existing chart on this canvas before creating a new one
    const existing = window.Chart?.getChart(canvas);
    if (existing) existing.destroy();

    return new window.Chart(canvas, {
        type: 'line',
        data: { datasets },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            parsing: false,
            normalized: true,
            interaction: { mode: 'nearest', intersect: false, axis: 'x' },
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.92)',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderWidth: 1,
                    displayColors: true,
                    padding: 9,
                    callbacks: {
                        title(items) {
                            const x = items[0]?.parsed?.x;
                            return Number.isFinite(x) ? `${xLabel}: ${x.toFixed(4)}` : '';
                        },
                        label(item) {
                            const raw = item.raw || {};
                            const value = Number.isFinite(raw.residual) ? raw.residual : item.parsed.y;
                            return `${item.dataset.label}: ${formatValue(value)}`;
                        },
                    },
                },
                zoom: {
                    pan: { enabled: true, mode: 'x', modifierKey: 'shift' },
                    zoom: {
                        drag: { enabled: true, backgroundColor: 'rgba(37, 99, 235, 0.12)' },
                        pinch: { enabled: true },
                        wheel: { enabled: true, modifierKey: 'ctrl' },
                        mode: 'x',
                    },
                },
            },
            elements: {
                point: { radius: 0 },
                line:  { borderWidth: 1.25, tension: 0 },
            },
            scales: {
                x: {
                    type: 'linear',
                    min: bounds?.xMin,
                    max: bounds?.xMax,
                    title: { display: true, text: xLabel, color: TICK_COLOR, font: { size: 11, weight: 600 }, padding: { top: 6 } },
                    ticks: { color: TICK_COLOR, font: { size: 10 }, maxTicksLimit: 7, maxRotation: 0 },
                    grid: { color: GRID_COLOR, tickColor: GRID_COLOR },
                    border: { color: GRID_COLOR },
                },
                y: {
                    min: bounds?.yMin,
                    max: bounds?.yMax,
                    title: { display: true, text: yLabel, color: TICK_COLOR, font: { size: 11, weight: 600 } },
                    ticks: { color: TICK_COLOR, font: { size: 10 }, maxTicksLimit: 6, callback: formatTick },
                    grid: { color: GRID_COLOR, tickColor: GRID_COLOR },
                    border: { color: GRID_COLOR },
                },
            },
        },
    });
}

function mkDataset(label, xyData, color, extra = {}) {
    return { label, data: xyData, borderColor: color, backgroundColor: color,
             parsing: false, ...extra };
}

function formatTick(value) {
    if (!Number.isFinite(value)) return value;
    const abs = Math.abs(value);
    if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    return value.toFixed(2);
}

function formatValue(value) {
    if (!Number.isFinite(value)) return 'n/a';
    const abs = Math.abs(value);
    if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(3);
    return value.toPrecision(5);
}

function paddedBounds(min, max, fraction) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: undefined, max: undefined };
    const span = Math.max(max - min, Math.abs(max) * 0.01, 1);
    const pad = span * fraction;
    return { min: min - pad, max: max + pad };
}

// Colors
const C_EXPT = '#dc2626';   // observed (red)
const C_RMC  = '#2563eb';   // calculated (blue)
const C_DIFF = '#059669';   // residual (green)
const C_BASE = '#cbd5e1';   // zero baseline (slate)

// Build the four datasets for a Rietveld-style fit plot: observed, calculated,
// a residual (obs − calc) offset below the data, and a zero baseline for
// that residual.
function fitDatasets(xs, exptY, rmcY) {
    const expt = [];
    const rmc  = [];
    const resid = [];

    let xMin = Infinity, xMax = -Infinity;
    let dataMin = Infinity, dataMax = -Infinity, residMin = Infinity, residMax = -Infinity;
    let sumSq = 0, sumDataSq = 0, n = 0;
    for (let i = 0; i < xs.length; i++) {
        const e = exptY[i], r = rmcY[i], dv = e - r;
        if (![xs[i], e, r, dv].every(Number.isFinite)) continue;
        expt.push({ x: xs[i], y: e });
        rmc.push({ x: xs[i], y: r });
        resid.push({ x: xs[i], value: dv });
        if (xs[i] < xMin) xMin = xs[i];
        if (xs[i] > xMax) xMax = xs[i];
        if (e < dataMin) dataMin = e;
        if (r < dataMin) dataMin = r;
        if (e > dataMax) dataMax = e;
        if (r > dataMax) dataMax = r;
        if (dv < residMin) residMin = dv;
        if (dv > residMax) residMax = dv;
        sumSq += dv * dv;
        sumDataSq += e * e;
        n += 1;
    }

    // Place the residual band in a gap below the data; top of band sits one gap
    // below the data minimum.
    const span = (dataMax - dataMin) || 1;
    const gap  = 0.12 * span;
    const offset = dataMin - gap - Math.max(0, residMax);

    const diff = resid.map(p => ({ x: p.x, y: p.value + offset, residual: p.value }));
    const diffMin = offset + residMin;
    const diffMax = offset + residMax;
    const xBounds = paddedBounds(xMin, xMax, 0.01);
    const yBounds = paddedBounds(Math.min(dataMin, diffMin), Math.max(dataMax, diffMax), 0.04);

    const base = expt.length
        ? [{ x: expt[0].x, y: offset }, { x: expt[expt.length - 1].x, y: offset }]
        : [];

    // Array order = draw order (later draws on top): Expt under RMC; base under Diff.
    return {
        datasets: [
            mkDataset('Exp', expt, C_EXPT, { borderWidth: 1.05 }),
            mkDataset('RMC',  rmc,  C_RMC,  { borderWidth: 1.55 }),
            mkDataset('_residual zero', base, C_BASE, { borderWidth: 1 }),
            mkDataset('Diff', diff, C_DIFF, { borderWidth: 1 }),
        ],
        metrics: {
            rw: Math.sqrt(sumSq / Math.max(sumDataSq, Number.EPSILON)) * 100,
        },
        bounds: {
            xMin: xBounds.min,
            xMax: xBounds.max,
            yMin: yBounds.min,
            yMax: yBounds.max,
        },
        residualRange: [residMin, residMax],
    };
}

function renderMetrics(el, metrics) {
    if (!el || !metrics) return;
    el.innerHTML = `
      <span>Rw ${formatValue(metrics.rw)}%</span>
    `;
}

// ── Render helpers ────────────────────────────────────────────────────────────

function renderData(d, charts, container) {
    // Panel 1: X-ray F(Q) — observed vs calculated + residual
    if (d.xfq) {
        const fit = fitDatasets(d.xfq.q, d.xfq.expt, d.xfq.rmc);
        charts[0] = updateOrCreate(charts[0], container.querySelector('#sqgr-c0'),
                                   fit.datasets, 'Q (Å⁻¹)', 'F(Q)', fit.bounds);
        renderMetrics(container.querySelector('#sqgr-m0'), fit.metrics);
    }

    // Panel 2: X-ray PDF G(r) — observed vs calculated + residual, full range
    if (d.xpdf) {
        const fit = fitDatasets(d.xpdf.r, d.xpdf.expt, d.xpdf.rmc);
        charts[1] = updateOrCreate(charts[1], container.querySelector('#sqgr-c1'),
                                   fit.datasets, 'r (Å)', 'G(r)', fit.bounds);
        renderMetrics(container.querySelector('#sqgr-m1'), fit.metrics);
    }
}

function updateOrCreate(chart, canvas, datasets, xLabel, yLabel, bounds) {
    if (chart) {
        chart.destroy();
    }
    return makeChart(canvas, datasets, xLabel, yLabel, bounds);
}

// ── Public mount function ─────────────────────────────────────────────────────

export async function mountSqgrPanel(container, folderPath) {
    // Destroy any existing Chart.js instances before replacing the DOM
    container.querySelectorAll('canvas').forEach(c => {
        const ch = window.Chart?.getChart(c);
        if (ch) ch.destroy();
    });
    container.innerHTML = '';

    let cfgInfo;
    try {
        cfgInfo = await sqgrFetch(`/api/sqgr/configs?folder=${encodeURIComponent(folderPath)}`);
    } catch (_) { return; }
    if (!cfgInfo.configs?.length) return;

    const configs = cfgInfo.configs;

    container.innerHTML = `
      <div class="fitq">
        <div class="fitq-head">
          <h3>Fit Quality</h3>
          <label class="sqgr-label">
            Configuration
            <select id="sqgr-cfg">${configs.map(n => `<option value="${n}">${n}</option>`).join('')}</select>
          </label>
          <span class="muted sqgr-status" id="sqgr-status">loading…</span>
        </div>
        <div class="sqgr-charts">
          <div class="sqgr-chart-wrap">
            <div class="sqgr-chart-head">
              <div>
                <div class="sqgr-chart-title">X-ray F(Q)</div>
                <div class="sqgr-metrics" id="sqgr-m0"></div>
              </div>
              <div class="sqgr-chart-tools">
                <div class="sqgr-legend" aria-label="Figure legend">
                  <span><i class="sqgr-key observed"></i>Exp</span>
                  <span><i class="sqgr-key rmc"></i>RMC</span>
                  <span><i class="sqgr-key residual"></i>Diff</span>
                </div>
                <button class="sqgr-reset" type="button" data-reset-chart="0">Reset</button>
              </div>
            </div>
            <div class="sqgr-canvas-box"><canvas id="sqgr-c0"></canvas></div>
          </div>
          <div class="sqgr-chart-wrap">
            <div class="sqgr-chart-head">
              <div>
                <div class="sqgr-chart-title">X-ray PDF G(r)</div>
                <div class="sqgr-metrics" id="sqgr-m1"></div>
              </div>
              <div class="sqgr-chart-tools">
                <div class="sqgr-legend" aria-label="Figure legend">
                  <span><i class="sqgr-key observed"></i>Exp</span>
                  <span><i class="sqgr-key rmc"></i>RMC</span>
                  <span><i class="sqgr-key residual"></i>Diff</span>
                </div>
                <button class="sqgr-reset" type="button" data-reset-chart="1">Reset</button>
              </div>
            </div>
            <div class="sqgr-canvas-box"><canvas id="sqgr-c1"></canvas></div>
          </div>
        </div>
      </div>
    `;

    const sel      = container.querySelector('#sqgr-cfg');
    const statusEl = container.querySelector('#sqgr-status');
    const charts   = [null, null];

    container.querySelectorAll('[data-reset-chart]').forEach(btn => {
        btn.addEventListener('click', () => {
            const chart = charts[+btn.dataset.resetChart];
            chart?.resetZoom?.();
        });
    });

    async function load(configNum) {
        statusEl.textContent = `loading config ${configNum}…`;
        try {
            const d = await sqgrFetch(`/api/sqgr/data?folder=${encodeURIComponent(folderPath)}&config=${configNum}`);
            renderData(d, charts, container);
            statusEl.textContent = `config ${configNum}  ·  ${configs.length} available`;
        } catch (err) {
            statusEl.textContent = `✗ ${err.message}`;
        }
    }

    sel.addEventListener('change', () => load(+sel.value));
    load(+sel.value);
}
