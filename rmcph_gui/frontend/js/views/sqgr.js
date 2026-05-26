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

// ── Chart factory ─────────────────────────────────────────────────────────────

function makeChart(canvas, datasets, xLabel, yLabel) {
    // Destroy any existing chart on this canvas before creating a new one
    const existing = window.Chart?.getChart(canvas);
    if (existing) existing.destroy();

    return new window.Chart(canvas, {
        type: 'line',
        data: { datasets },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.4,
            parsing: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 14, padding: 8, font: { size: 10 } },
                },
                tooltip: { enabled: false },
            },
            elements: {
                point: { radius: 0 },
                line:  { borderWidth: 1.5, tension: 0 },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: xLabel, font: { size: 11 }, padding: { top: 4 } },
                    ticks: { font: { size: 10 }, maxTicksLimit: 8 },
                },
                y: {
                    title: { display: true, text: yLabel, font: { size: 11 } },
                    ticks: { font: { size: 10 }, maxTicksLimit: 6 },
                },
            },
        },
    });
}

function mkDataset(label, xyData, color, extra = {}) {
    return { label, data: xyData, borderColor: color, backgroundColor: color,
             parsing: false, ...extra };
}

function toXY(xs, ys, xmax) {
    const out = [];
    for (let i = 0; i < xs.length; i++) {
        if (xmax !== undefined && xs[i] > xmax) break;
        out.push({ x: xs[i], y: ys[i] });
    }
    return out;
}

// ── Render helpers ────────────────────────────────────────────────────────────

function renderData(d, charts, container) {
    // Panel 1: X-ray F(Q) — experiment vs RMC
    if (d.xfq) {
        const ds = [
            mkDataset('Expt', toXY(d.xfq.q, d.xfq.expt), '#ef4444'),
            mkDataset('RMC',  toXY(d.xfq.q, d.xfq.rmc),  '#2563eb'),
        ];
        charts[0] = updateOrCreate(charts[0], container.querySelector('#sqgr-c0'),
                                   ds, 'Q (Å⁻¹)', 'F(Q)');
    }

    // Panel 2: X-ray PDF G(r) — experiment vs RMC, full range
    if (d.xpdf) {
        const ds = [
            mkDataset('Expt', toXY(d.xpdf.r, d.xpdf.expt), '#ef4444'),
            mkDataset('RMC',  toXY(d.xpdf.r, d.xpdf.rmc),  '#2563eb'),
        ];
        charts[1] = updateOrCreate(charts[1], container.querySelector('#sqgr-c1'),
                                   ds, 'r (Å)', 'G(r)');
    }
}

function updateOrCreate(chart, canvas, datasets, xLabel, yLabel) {
    if (chart) {
        chart.data.datasets = datasets;
        chart.update('none');
        return chart;
    }
    return makeChart(canvas, datasets, xLabel, yLabel);
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
      <section class="panel sqgr-panel">
        <h2>Fit Quality</h2>
        <div class="sqgr-controls">
          <label class="sqgr-label">
            Configuration
            <select id="sqgr-cfg">${configs.map(n => `<option value="${n}">${n}</option>`).join('')}</select>
          </label>
          <span class="muted sqgr-status" id="sqgr-status">loading…</span>
        </div>
        <div class="sqgr-charts">
          <div class="sqgr-chart-wrap">
            <div class="sqgr-chart-title">X-ray F(Q)</div>
            <canvas id="sqgr-c0"></canvas>
          </div>
          <div class="sqgr-chart-wrap">
            <div class="sqgr-chart-title">X-ray PDF G(r)</div>
            <canvas id="sqgr-c1"></canvas>
          </div>
        </div>
      </section>
    `;

    const sel      = container.querySelector('#sqgr-cfg');
    const statusEl = container.querySelector('#sqgr-status');
    const charts   = [null, null];

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
