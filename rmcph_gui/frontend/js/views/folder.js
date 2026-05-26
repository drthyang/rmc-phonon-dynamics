// Phase 1 — data folder selection & parsing.
// A server-side directory browser + dataset inspector. On "Open", the backend
// finds the .rmc6f + Frac files, parses atoms/cell via src_gpu/Readers, and we
// render the detected metadata. The result is stored in app state for later
// phases (structure, k-path, jobs).
'use strict';

import { api } from '../api.js';
import { state } from '../state.js';
import { openFilePicker } from './filepicker.js';

let curPath = null;
let onContinueCb = null;

export function mountFolderView(root, opts = {}) {
    onContinueCb = opts.onContinue || null;
    return _mount(root);
}

function _mount(root) {
    root.innerHTML = `
      <section class="panel">
        <h2><span class="step-badge">1</span>Data folder</h2>
        <p class="hint">Browse to the folder containing your <code>Frac*.txt</code>
          configurations (or its parent). The matching <code>.rmc6f</code> is
          auto-detected nearby.</p>

        <div class="browser">
          <div class="browser-bar">
            <button id="fb-up" title="Up one level">↑</button>
            <code id="fb-path">…</code>
            <button id="fb-open" class="primary">Open this folder</button>
          </div>
          <ul id="fb-list" class="browser-list"></ul>
        </div>

        <div id="fb-result" class="result"></div>
      </section>
    `;

    root.querySelector('#fb-up').addEventListener('click', () => {
        const list = root.querySelector('#fb-list');
        const parent = list.dataset.parent;
        if (parent && parent !== 'null') navigate(parent, root);
    });
    root.querySelector('#fb-open').addEventListener('click', () => openFolder(curPath, null, root));

    navigate(null, root);
}

async function navigate(path, root) {
    const listEl = root.querySelector('#fb-list');
    const pathEl = root.querySelector('#fb-path');
    listEl.innerHTML = '<li class="muted">loading…</li>';
    try {
        const d = await api.browseDir(path);
        curPath = d.path;
        pathEl.textContent = d.path;
        listEl.dataset.parent = d.parent ?? 'null';
        listEl.innerHTML = '';
        if (!d.subdirs.length) {
            listEl.innerHTML = '<li class="muted">(no subfolders)</li>';
        }
        for (const name of d.subdirs) {
            const li = document.createElement('li');
            li.textContent = '📁 ' + name;
            li.addEventListener('click', () => navigate(joinPath(d.path, name), root));
            listEl.appendChild(li);
        }
        // Hint chips when the current folder already looks like a dataset
        const tags = [];
        if (d.has_frac)  tags.push('<span class="chip ok">Frac*.txt here</span>');
        if (d.has_rmc6f) tags.push('<span class="chip ok">.rmc6f here</span>');
        if (tags.length) {
            const li = document.createElement('li');
            li.className = 'tags';
            li.innerHTML = tags.join(' ');
            listEl.prepend(li);
        }
    } catch (err) {
        listEl.innerHTML = `<li class="err">${err.message}</li>`;
    }
}

async function openFolder(path, structureFile, root) {
    const resEl = root.querySelector('#fb-result');
    resEl.innerHTML = '<p class="muted">inspecting…</p>';
    try {
        const r = await api.openFolder(path, structureFile);
        state.set('dataset', r);
        renderResult(r, root);
    } catch (err) {
        resEl.innerHTML = `<p class="err">✗ ${err.message}</p>`;
    }
}

function renderResult(r, root) {
    const resEl = root.querySelector('#fb-result');
    const warn = (r.warnings || []).map(w => `<li>⚠️ ${w}</li>`).join('');

    const atomsRows = (r.atoms || [])
        .map(a => `<tr><td>${a.symbol}</td><td>${a.count}</td></tr>`).join('');

    // Structure-file picker when several .rmc6f were found nearby
    let structPicker = '';
    if ((r.structure_candidates || []).length > 1) {
        const opts = r.structure_candidates
            .map(p => `<option value="${p}" ${p === r.structure_file ? 'selected' : ''}>${p}</option>`)
            .join('');
        structPicker = `<label class="eq-pick">choose:
          <select id="fb-struct">${opts}</select></label>`;
    }

    const ref = r.reference || { mode: 'average', file: null };

    resEl.innerHTML = `
      <h3>Detected dataset</h3>
      ${warn ? `<ul class="warnings">${warn}</ul>` : ''}
      <table class="kv">
        <tr><td>Configs dir</td><td><code>${r.configs_dir ?? '—'}</code></td></tr>
        <tr><td>Configurations</td><td>${r.n_configs}${r.config_family && r.config_family !== 'none' ? ` <span class="muted">(${r.config_family === 'rmc6f' ? '.rmc6f' : 'Frac*.txt'})</span>` : ''}</td></tr>
        <tr><td>Structure file</td>
            <td><code>${r.structure_file ?? '—'}</code> ${structPicker}
            <div class="sub">provides atom types + lattice (required)</div></td></tr>
        <tr><td>Atoms / cell</td><td>${r.natom ?? '—'}</td></tr>
        <tr><td>Supercell dim</td><td>${r.dim ? r.dim.join(' × ') : '—'}</td></tr>
      </table>
      ${atomsRows ? `<table class="atoms"><thead><tr><th>Element</th><th>Count</th></tr></thead><tbody>${atomsRows}</tbody></table>` : ''}

      <h3>Displacement reference (hsym)</h3>
      <p class="hint">Reference positions subtracted from each configuration:
        <code>u = config − hsym</code>.</p>
      <div class="ref" id="fb-ref">
        <label class="radio"><input type="radio" name="refmode" value="average"
          ${ref.mode === 'average' ? 'checked' : ''}> Average of all configurations
          <span class="muted">(default)</span></label>
        <label class="radio"><input type="radio" name="refmode" value="file"
          ${ref.mode === 'file' ? 'checked' : ''}> Equilibrium file:</label>
        <div class="ref-file">
          <code id="fb-ref-path">${ref.file ?? '(none selected)'}</code>
          <button id="fb-ref-browse">Browse…</button>
        </div>
      </div>

      <div class="next">
        <button id="fb-continue" class="primary" ${r.structure_file && r.n_configs ? '' : 'disabled'}>
          Continue to structure →
        </button>
      </div>
    `;

    // Structure-file override
    const structSel = resEl.querySelector('#fb-struct');
    if (structSel) structSel.addEventListener('change', () => openFolder(r.path, structSel.value, root));

    // ── Displacement-reference control ──────────────────────────────────
    const refRoot   = resEl.querySelector('#fb-ref');
    const refPathEl = resEl.querySelector('#fb-ref-path');
    const browseBtn = resEl.querySelector('#fb-ref-browse');

    function persistRef() {
        const mode = refRoot.querySelector('input[name="refmode"]:checked').value;
        const file = (mode === 'file') ? (r.reference.file || null) : null;
        r.reference = { mode, file };
        state.set('dataset', { ...r });
    }

    refRoot.querySelectorAll('input[name="refmode"]').forEach(radio =>
        radio.addEventListener('change', persistRef));

    browseBtn.addEventListener('click', async () => {
        const start = r.reference.file || r.structure_file || r.path;
        const startDir = start && start.includes('/') ? start.slice(0, start.lastIndexOf('/')) : r.path;
        const picked = await openFilePicker({
            title: 'Select equilibrium structure',
            exts: ['rmc6f', 'cif'],
            startPath: startDir,
        });
        if (picked) {
            r.reference.file = picked;
            refPathEl.textContent = picked;
            // selecting a file implies "file" mode
            refRoot.querySelector('input[value="file"]').checked = true;
            persistRef();
        }
    });

    const cont = resEl.querySelector('#fb-continue');
    if (cont) cont.addEventListener('click', () => {
        if (onContinueCb) onContinueCb(state.get('dataset'));
    });
}

// ── helpers ──────────────────────────────────────────────────────────────────
function joinPath(base, name) {
    return base.endsWith('/') ? base + name : base + '/' + name;
}
