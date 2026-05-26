'use strict';

import { api } from './api.js';
import { mountFolderView } from './views/folder.js';
import { mountStructureView } from './views/structure3d.js';
import { mountBZView } from './views/bz.js';
import { mountRunView } from './views/run.js';

const statusEl = document.getElementById('backend-status');
const viewRoot = document.getElementById('view-root');

function stepDone(n) {
    const el = document.getElementById(`snav-${n}`);
    if (!el) return;
    el.classList.remove('active');
    el.classList.add('done');
    el.querySelector('.step-num').textContent = '✓';
}

function stepActive(n) {
    const el = document.getElementById(`snav-${n}`);
    if (el) el.classList.add('active');
}

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

    viewRoot.innerHTML =
        '<div id="step-folder"></div><div id="step-structure"></div>'
        + '<div id="step-reciprocal"></div><div id="step-run"></div>';
    const stepFolder     = document.getElementById('step-folder');
    const stepStructure  = document.getElementById('step-structure');
    const stepReciprocal = document.getElementById('step-reciprocal');
    const stepRun        = document.getElementById('step-run');

    mountFolderView(stepFolder, {
        onContinue: (dataset) => {
            stepDone(1); stepActive(2);
            mountStructureView(stepStructure, dataset, {
                onContinue: () => {
                    stepDone(2); stepActive(3);
                    mountBZView(stepReciprocal);
                    mountRunView(stepRun);
                    stepDone(3); stepActive(4);
                    stepReciprocal.scrollIntoView({ behavior: 'smooth', block: 'start' });
                },
            });
            stepStructure.scrollIntoView({ behavior: 'smooth', block: 'start' });
        },
    });
}

init();
