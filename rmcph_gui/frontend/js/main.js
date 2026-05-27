'use strict';

import { api } from './api.js';
import { mountFolderView } from './views/folder.js';
import { mountStructureView } from './views/structure3d.js';
import { mountBZView } from './views/bz.js?v=confirm-kpath-3-20260526';
import { mountRunView } from './views/run.js?v=run-slower-active-20260526';

const statusEl = document.getElementById('backend-status');
const viewRoot = document.getElementById('view-root');
const stepRoots = new Map();

function stepDone(n) {
    const el = document.getElementById(`snav-${n}`);
    if (!el) return;
    el.classList.remove('active');
    el.classList.add('done');
    el.querySelector('.step-num').textContent = '✓';
}

function stepActive(n) {
    document.querySelectorAll('.step-item.active').forEach((item) => item.classList.remove('active'));
    const el = document.getElementById(`snav-${n}`);
    if (el) el.classList.add('active');
}

function stepAvailable(n) {
    const el = document.getElementById(`snav-${n}`);
    if (el) el.classList.add('available');
}

function expandStep(root) {
    const panel = root?.querySelector('.panel');
    const toggle = panel?.querySelector('.panel-toggle');
    if (!panel) return false;
    panel.classList.remove('collapsed');
    if (toggle) toggle.textContent = 'Collapse';
    return true;
}

function collapseStep(root) {
    const panel = root.querySelector('.panel');
    const title = panel?.querySelector('h2');
    if (!panel || !title) return;

    panel.classList.add('collapsed');
    if (!title.querySelector('.panel-status')) {
        const status = document.createElement('span');
        status.className = 'panel-status';
        status.textContent = 'Completed';
        title.appendChild(status);
    }
    if (!title.querySelector('.panel-toggle')) {
        const toggle = document.createElement('button');
        toggle.className = 'panel-toggle';
        toggle.type = 'button';
        toggle.textContent = 'Edit';
        toggle.addEventListener('click', () => {
            const isCollapsed = panel.classList.toggle('collapsed');
            toggle.textContent = isCollapsed ? 'Edit' : 'Collapse';
        });
        title.appendChild(toggle);
    }
}

function jumpToStep(n) {
    const root = stepRoots.get(n);
    if (!root || !root.querySelector('.panel')) return;
    stepActive(n);
    expandStep(root);
    root.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
    stepRoots.set(1, stepFolder);
    stepRoots.set(2, stepStructure);
    stepRoots.set(3, stepReciprocal);
    stepRoots.set(4, stepRun);
    for (let n = 1; n <= 4; n++) {
        const nav = document.getElementById(`snav-${n}`);
        if (nav) nav.addEventListener('click', () => jumpToStep(n));
    }
    stepAvailable(1);

    mountFolderView(stepFolder, {
        onContinue: (dataset) => {
            collapseStep(stepFolder);
            stepDone(1); stepActive(2); stepAvailable(2);
            mountStructureView(stepStructure, dataset, {
                onContinue: () => {
                    collapseStep(stepStructure);
                    stepDone(2); stepActive(3); stepAvailable(3);
                    mountBZView(stepReciprocal, {
                        onContinue: () => {
                            collapseStep(stepReciprocal);
                            stepDone(3); stepActive(4); stepAvailable(4);
                            mountRunView(stepRun);
                            stepRun.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        },
                    });
                    stepReciprocal.scrollIntoView({ behavior: 'smooth', block: 'start' });
                },
            });
            stepStructure.scrollIntoView({ behavior: 'smooth', block: 'start' });
        },
    });
}

init();
