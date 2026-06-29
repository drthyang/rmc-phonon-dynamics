// Reusable modal file picker. Browses server-side directories and lists files
// matching the given extensions; resolves to the chosen absolute path (or null
// if cancelled). Used to select an equilibrium structure; reusable later for
// output dirs etc.
'use strict';

import { api } from '../api.js';

export function openFilePicker({ title = 'Select a file', exts = [], startPath = null } = {}) {
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'fp-overlay';
        overlay.innerHTML = `
          <div class="fp-modal">
            <div class="fp-head">
              <span>${title}</span>
              <button class="fp-x" title="Cancel">✕</button>
            </div>
            <div class="fp-bar">
              <button class="fp-up" title="Up one level">↑</button>
              <code class="fp-path">…</code>
            </div>
            <ul class="fp-list"></ul>
            <div class="fp-foot">
              <span class="fp-sel muted">No file selected</span>
              <span class="fp-actions">
                <button class="fp-cancel">Cancel</button>
                <button class="fp-ok primary" disabled>Select</button>
              </span>
            </div>
          </div>
        `;
        document.body.appendChild(overlay);

        const listEl = overlay.querySelector('.fp-list');
        const pathEl = overlay.querySelector('.fp-path');
        const selEl  = overlay.querySelector('.fp-sel');
        const okBtn  = overlay.querySelector('.fp-ok');
        let chosen = null;

        function close(result) { overlay.remove(); resolve(result); }
        overlay.querySelector('.fp-x').onclick = () => close(null);
        overlay.querySelector('.fp-cancel').onclick = () => close(null);
        overlay.addEventListener('click', (e) => { if (e.target === overlay) close(null); });
        okBtn.onclick = () => close(chosen);
        overlay.querySelector('.fp-up').onclick = () => {
            const p = listEl.dataset.parent;
            if (p && p !== 'null') nav(p);
        };

        async function nav(path) {
            listEl.innerHTML = '<li class="muted">loading…</li>';
            chosen = null; okBtn.disabled = true; selEl.textContent = 'No file selected';
            selEl.classList.add('muted');
            try {
                const d = await api.browseDir(path, exts);
                pathEl.textContent = d.path;
                listEl.dataset.parent = d.parent ?? 'null';
                listEl.innerHTML = '';
                for (const name of d.subdirs) {
                    const li = document.createElement('li');
                    li.className = 'dir';
                    li.textContent = '📁 ' + name;
                    li.onclick = () => nav(join(d.path, name));
                    listEl.appendChild(li);
                }
                for (const name of (d.files || [])) {
                    const li = document.createElement('li');
                    li.className = 'file';
                    li.textContent = '📄 ' + name;
                    const full = join(d.path, name);
                    li.onclick = () => {
                        listEl.querySelectorAll('li.file.active').forEach(x => x.classList.remove('active'));
                        li.classList.add('active');
                        chosen = full; okBtn.disabled = false;
                        selEl.textContent = full; selEl.classList.remove('muted');
                    };
                    li.ondblclick = () => close(full);
                    listEl.appendChild(li);
                }
                if (!d.subdirs.length && !(d.files || []).length) {
                    listEl.innerHTML = '<li class="muted">(empty)</li>';
                }
            } catch (err) {
                listEl.innerHTML = `<li class="err">${err.message}</li>`;
            }
        }

        nav(startPath);
    });
}

function join(base, name) {
    return base.endsWith('/') ? base + name : base + '/' + name;
}
