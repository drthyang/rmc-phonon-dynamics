// Entry point. Phase 0: confirm the backend is reachable and show its status.
// Later phases mount their views into #view-root.
'use strict';

import { api } from './api.js';

const statusEl = document.getElementById('backend-status');

async function init() {
    try {
        const info = await api.ping();
        statusEl.textContent = `✓ ${info.service} v${info.version}`;
        statusEl.classList.add('ok');
    } catch (err) {
        statusEl.textContent = '✗ backend unreachable';
        statusEl.classList.add('err');
        console.error('ping failed:', err);
    }
}

init();
