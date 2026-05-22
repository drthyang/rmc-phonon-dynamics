// Entry point. Phase 0: confirm the backend is reachable and show its status.
// Later phases mount their views into #view-root.
'use strict';

import { api } from './api.js';
import { mountFolderView } from './views/folder.js';

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
    mountFolderView(viewRoot);
}

init();
