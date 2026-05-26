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

function makeChart(canvas, datasets, xLabel, yLabel, bounds, chartOptions = {}) {
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
                            if (!Number.isFinite(x)) return '';
                            if (xLabel === 'Configuration #') return `Configuration #${Math.round(x)}`;
                            return `${xLabel}: ${x.toFixed(4)}`;
                        },
                        label(item) {
                            if (item.dataset.label.startsWith('_')) return null;
                            const raw = item.raw || {};
                            const value = Number.isFinite(raw.residual) ? raw.residual : item.parsed.y;
                            const suffix = yLabel.includes('%') ? '%' : '';
                            return `${item.dataset.label}: ${formatValue(value)}${suffix}`;
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
                    ticks: {
                        color: TICK_COLOR,
                        font: { size: 10 },
                        maxTicksLimit: 6,
                        callback(value) {
                            const label = formatTick(value);
                            return yLabel.includes('%') ? `${label}%` : label;
                        },
                    },
                    grid: { color: GRID_COLOR, tickColor: GRID_COLOR },
                    border: { color: GRID_COLOR },
                },
            },
            ...chartOptions,
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
    if (!el) return;
    if (!metrics) {
        el.innerHTML = '';
        return;
    }
    el.innerHTML = `
      <span>Rw ${formatValue(metrics.rw)}%</span>
    `;
}

function rwSummaryDatasets(summary, selectedConfig) {
    const xfq = [];
    const xpdf = [];
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;

    for (const row of summary.points || []) {
        const config = row.config;
        if (!Number.isFinite(config)) continue;
        if (Number.isFinite(row.xfq)) {
            xfq.push({ x: config, y: row.xfq });
            yMin = Math.min(yMin, row.xfq);
            yMax = Math.max(yMax, row.xfq);
        }
        if (Number.isFinite(row.xpdf)) {
            xpdf.push({ x: config, y: row.xpdf });
            yMin = Math.min(yMin, row.xpdf);
            yMax = Math.max(yMax, row.xpdf);
        }
        xMin = Math.min(xMin, config);
        xMax = Math.max(xMax, config);
    }

    const yBounds = paddedBounds(0, yMax, 0.08);
    const selectedLine = Number.isFinite(selectedConfig)
        ? [mkDataset('_selected config', [
            { x: selectedConfig, y: 0 },
            { x: selectedConfig, y: yBounds.max },
        ], 'rgba(15, 23, 42, 0.38)', {
            borderWidth: 1,
            pointRadius: 0,
            pointHoverRadius: 0,
        })]
        : [];
    return {
        datasets: [
            ...selectedLine,
            mkDataset('F(Q)', xfq, 'rgba(37, 99, 235, 0.86)', {
                borderWidth: 1.25,
                pointRadius: 0,
                pointHoverRadius: 3,
                pointHitRadius: 9,
                spanGaps: true,
            }),
            mkDataset('G(r)', xpdf, 'rgba(5, 150, 105, 0.86)', {
                borderWidth: 1.25,
                pointRadius: 0,
                pointHoverRadius: 3,
                pointHitRadius: 9,
                spanGaps: true,
            }),
        ],
        bounds: {
            xMin,
            xMax,
            yMin: 0,
            yMax: yBounds.max,
        },
    };
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

function renderSummary(summary, charts, container, onConfigClick, selectedConfig) {
    const rw = rwSummaryDatasets(summary, selectedConfig);
    charts.rw = updateOrCreate(charts.rw, container.querySelector('#sqgr-rw'),
                               rw.datasets, 'Configuration #', 'Rw (%)', rw.bounds, {
                                   onClick(event, _items, chart) {
                                       const points = chart.getElementsAtEventForMode(
                                           event, 'nearest', { intersect: false, axis: 'x' }, false
                                       );
                                       const point = points[0];
                                       const config = point
                                           ? chart.data.datasets[point.datasetIndex].data[point.index]?.x
                                           : null;
                                       if (Number.isFinite(config)) onConfigClick(Math.round(config));
                                   },
                                   onHover(event, _items, chart) {
                                       const points = chart.getElementsAtEventForMode(
                                           event, 'nearest', { intersect: false, axis: 'x' }, false
                                       );
                                       chart.canvas.style.cursor = points.length ? 'pointer' : 'default';
                                   },
                               });
}

function updateOrCreate(chart, canvas, datasets, xLabel, yLabel, bounds, chartOptions = {}) {
    if (chart) {
        chart.destroy();
    }
    return makeChart(canvas, datasets, xLabel, yLabel, bounds, chartOptions);
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
          <div class="sqgr-chart-wrap sqgr-summary-card">
            <div class="sqgr-chart-head">
              <div>
                <div class="sqgr-chart-title">Rw Across Configurations</div>
              </div>
              <div class="sqgr-chart-tools">
                <div class="sqgr-legend" aria-label="Figure legend">
                  <span><i class="sqgr-key rmc"></i>F(Q)</span>
                  <span><i class="sqgr-key residual"></i>G(r)</span>
                </div>
                <button class="sqgr-reset" type="button" data-reset-chart="rw">Reset</button>
              </div>
            </div>
            <div class="sqgr-summary-box"><canvas id="sqgr-rw"></canvas></div>
          </div>

          <details class="sqgr-detail">
            <summary>
              <span>X-ray F(Q)</span>
              <span class="sqgr-metrics" id="sqgr-m0"></span>
            </summary>
          <div class="sqgr-chart-wrap">
            <div class="sqgr-chart-head">
              <div class="sqgr-chart-title">Selected Configuration</div>
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
          </details>

          <details class="sqgr-detail">
            <summary>
              <span>X-ray PDF G(r)</span>
              <span class="sqgr-metrics" id="sqgr-m1"></span>
            </summary>
          <div class="sqgr-chart-wrap">
            <div class="sqgr-chart-head">
              <div class="sqgr-chart-title">Selected Configuration</div>
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
          </details>
        </div>
      </div>
    `;

    const sel      = container.querySelector('#sqgr-cfg');
    const statusEl = container.querySelector('#sqgr-status');
    const charts   = { rw: null, 0: null, 1: null };
    let selectedData = null;
    let summaryData = null;

    container.querySelectorAll('[data-reset-chart]').forEach(btn => {
        btn.addEventListener('click', () => {
            const chart = charts[btn.dataset.resetChart];
            chart?.resetZoom?.();
        });
    });

    container.querySelectorAll('.sqgr-detail').forEach(detail => {
        detail.addEventListener('toggle', () => {
            if (detail.open && selectedData) renderData(selectedData, charts, container);
        });
    });

    async function load(configNum) {
        statusEl.textContent = `loading config ${configNum}…`;
        try {
            const d = await sqgrFetch(`/api/sqgr/data?folder=${encodeURIComponent(folderPath)}&config=${configNum}`);
            selectedData = d;
            renderMetrics(container.querySelector('#sqgr-m0'),
                          d.xfq ? fitDatasets(d.xfq.q, d.xfq.expt, d.xfq.rmc).metrics : null);
            renderMetrics(container.querySelector('#sqgr-m1'),
                          d.xpdf ? fitDatasets(d.xpdf.r, d.xpdf.expt, d.xpdf.rmc).metrics : null);
            container.querySelectorAll('.sqgr-detail').forEach(detail => {
                if (detail.open) renderData(d, charts, container);
            });
            statusEl.textContent = `config ${configNum}  ·  ${configs.length} available`;
            if (summaryData) {
                renderSummary(summaryData, charts, container, openConfigFromSummary, configNum);
            }
        } catch (err) {
            statusEl.textContent = `✗ ${err.message}`;
        }
    }

    async function openConfigFromSummary(configNum) {
        if (![...sel.options].some(option => +option.value === configNum)) return;
        sel.value = String(configNum);
        container.querySelectorAll('.sqgr-detail').forEach(detail => { detail.open = true; });
        await load(configNum);
        container.querySelector('.sqgr-detail')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    sqgrFetch(`/api/sqgr/rw-summary?folder=${encodeURIComponent(folderPath)}`)
        .then(summary => {
            summaryData = summary;
            renderSummary(summary, charts, container, openConfigFromSummary, +sel.value);
        })
        .catch(err => { statusEl.textContent = `✗ ${err.message}`; });

    sel.addEventListener('change', () => load(+sel.value));
    load(+sel.value);
}
