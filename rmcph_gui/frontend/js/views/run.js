// Phase 5 — run a calculation.
// Renders the runner's parameter form from /api/runners, summarizes the active
// dataset + k-path (kept live via state.subscribe), submits a job, then polls
// /api/jobs/{id} for a progress bar + message with a Cancel button. The k-path
// is sent as state.kpath.segments (conventional-cell coords + npoints + labels);
// the dataset (structure file, configs, reference) is read server-side from the
// session. On success it reports the band.yaml path (viewer hand-off = Phase 6).
'use strict';

import { api } from '../api.js';
import { state } from '../state.js';

let pollTimer = null;
let unsub = null;

export function mountRunView(root) {
    root.innerHTML = `
      <section class="panel">
        <h2>4 · Run calculation</h2>
        <div id="run-summary" class="run-summary"></div>
        <div id="run-form"></div>
        <div class="next"><button id="run-go" class="primary">Run phonon bands</button></div>
        <div id="run-job"></div>
      </section>
    `;

    const summaryEl = root.querySelector('#run-summary');
    const formEl = root.querySelector('#run-form');
    const goBtn = root.querySelector('#run-go');
    const jobEl = root.querySelector('#run-job');

    let schema = { fields: [] };

    api.listRunners()
        .then((runners) => {
            const r = runners.find((x) => x.name === 'phonon_bands') || runners[0];
            schema = (r && r.param_schema) || { fields: [] };
            renderForm();
        })
        .catch((err) => { formEl.innerHTML = `<p class="err">✗ ${err.message}</p>`; });

    function renderForm() {
        formEl.innerHTML = schema.fields.map((f) => `
          <label class="run-field">
            <span class="run-flabel">${f.label}</span>
            <input type="number" data-key="${f.key}" value="${f.default}"
              ${f.min != null ? `min="${f.min}"` : ''}
              step="${f.type === 'int' ? '1' : 'any'}">
            ${f.help ? `<span class="run-fhelp">${f.help}</span>` : ''}
          </label>`).join('');
    }

    function readParams() {
        const params = {};
        formEl.querySelectorAll('input[data-key]').forEach((inp) => {
            params[inp.dataset.key] = Number(inp.value);
        });
        return params;
    }

    function renderSummary() {
        const ds = state.get('dataset');
        const kp = state.get('kpath');
        const okDs = !!(ds && ds.structure_file && ds.n_configs);
        const okKp = !!(kp && kp.segments && kp.segments.length);
        const dsTxt = okDs
            ? `${ds.n_configs} configs · ${ds.natom ?? '?'} atoms/cell`
            : '<span class="err">no dataset — open a folder first</span>';
        const kpTxt = okKp
            ? `${kp.segments.length} segment(s) · ${kp.totalPoints} k-points`
            : '<span class="err">no k-path — build one in the Brillouin-zone view</span>';
        summaryEl.innerHTML = `
          <table class="kv">
            <tr><td>Dataset</td><td>${dsTxt}</td></tr>
            <tr><td>k-path</td><td>${kpTxt}</td></tr>
          </table>`;
        goBtn.disabled = !(okDs && okKp);
    }

    if (unsub) unsub();
    unsub = state.subscribe((key) => {
        if (key === 'dataset' || key === 'kpath') renderSummary();
    });
    renderSummary();

    goBtn.addEventListener('click', submit);

    async function submit() {
        const kp = state.get('kpath');
        const params = { ...readParams(), segments: kp.segments };
        goBtn.disabled = true;
        jobEl.innerHTML = '<p class="muted">submitting…</p>';
        try {
            const job = await api.submitJob('phonon_bands', params);
            state.set('job', job);
            renderJob(job);
            startPolling(job.id);
        } catch (err) {
            jobEl.innerHTML = `<p class="err">✗ ${err.message}</p>`;
            renderSummary();
        }
    }

    function startPolling(id) {
        stopPolling();
        pollTimer = setInterval(async () => {
            try {
                const job = await api.getJob(id);
                state.set('job', job);
                renderJob(job);
                if (['done', 'error', 'cancelled'].includes(job.status)) {
                    stopPolling();
                    renderSummary();
                }
            } catch (err) {
                stopPolling();
                jobEl.innerHTML = `<p class="err">✗ ${err.message}</p>`;
            }
        }, 500);
    }

    function stopPolling() {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }

    function renderJob(job) {
        const p = job.progress || { done: 0, total: 0, fraction: 0, message: '' };
        const pct = Math.round((p.fraction || 0) * 100);
        const running = job.status === 'running' || job.status === 'queued';
        const statusClass = { done: 'ok', error: 'err', cancelled: 'muted' }[job.status] || '';

        let resultHtml = '';
        if (job.status === 'done' && job.result) {
            const name = (job.result.band_yaml || '').split('/').pop();
            const viewerUrl = '/viz/rmcph.html?band=' + encodeURIComponent('/results/' + name);
            resultHtml = `<div class="run-result">
              <div>✓ ${job.result.n_qpoints} k-points · ${job.result.n_segments} segment(s)</div>
              <div>band.yaml: <code>${job.result.band_yaml}</code></div>
              <a class="run-open" href="${viewerUrl}" target="_blank" rel="noopener">Open in S(Q,E) viewer →</a>
            </div>`;
        } else if (job.status === 'error') {
            resultHtml = `<p class="err">✗ ${job.error || 'failed'}</p>`;
        }

        jobEl.innerHTML = `
          <div class="run-jobhead">
            <span class="run-status ${statusClass}">${job.status}</span>
            <span class="muted">${p.message || ''}</span>
            ${running ? '<button id="run-cancel">Cancel</button>' : ''}
          </div>
          <div class="run-bar"><div class="run-bar-fill" style="width:${pct}%"></div></div>
          <div class="run-bar-label muted">${p.done}/${p.total} (${pct}%)</div>
          ${resultHtml}
        `;

        const cancelBtn = jobEl.querySelector('#run-cancel');
        if (cancelBtn) cancelBtn.addEventListener('click', async () => {
            cancelBtn.disabled = true;
            try {
                const j = await api.cancelJob(job.id);
                state.set('job', j);
                renderJob(j);
            } catch (err) {
                cancelBtn.disabled = false;
            }
        });
    }
}
