// Entry point. Phase 0: confirm the backend is reachable and show its status.
// Later phases mount their views into #view-root.
'use strict';

import { api } from './api.js';
import { mountFolderView } from './views/folder.js';
import { mountStructureView } from './views/structure3d.js';
import { mountBZView } from './views/bz.js';

const statusEl = document.getElementById('backend-status');
const viewRoot = document.getElementById('view-root');

async function init() {
    try {
        const info = await api.ping();
        statusEl.textContent = `✓ ${info.service} v${info.version}`;
        statusEl.classList.add('ok');
    } catch (err) {
        statusEl.textContent = '✗ backend unreachable';
        statusEl.classList.add('err');
        console.error('ping failed:', err);
        return;
    }

    // Step containers stack here: folder → structure → reciprocal/k-path.
    viewRoot.innerHTML =
        '<div id="step-folder"></div><div id="step-structure"></div><div id="step-reciprocal"></div>';
    const stepFolder = document.getElementById('step-folder');
    const stepStructure = document.getElementById('step-structure');
    const stepReciprocal = document.getElementById('step-reciprocal');

    mountFolderView(stepFolder, {
        onContinue: (dataset) => {
            mountStructureView(stepStructure, dataset, {
                onContinue: () => {
                    mountBZView(stepReciprocal);
                    stepReciprocal.scrollIntoView({ behavior: 'smooth', block: 'start' });
                },
            });
            stepStructure.scrollIntoView({ behavior: 'smooth', block: 'start' });
        },
    });
}

init();
