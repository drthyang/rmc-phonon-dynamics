// Thin wrappers around the backend REST API. All calls go through here so the
// FE↔BE contract lives in one place.
'use strict';

async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail ?? detail; } catch (_) {}
        throw new Error(`${r.status} ${detail}`);
    }
    return r.json();
}

export const api = {
    ping: () => getJSON('/api/ping'),

    browseDir: (path, fileExts) => {
        const qs = new URLSearchParams();
        if (path) qs.set('path', path);
        if (fileExts && fileExts.length) qs.set('files', fileExts.join(','));
        const q = qs.toString();
        return getJSON('/api/data/browse' + (q ? `?${q}` : ''));
    },

    openFolder: (path, structureFile) =>
        getJSON('/api/data/open', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, structure_file: structureFile ?? null }),
        }),

    getStructure: () => getJSON('/api/structure'),

    getReciprocal: () => getJSON('/api/reciprocal'),

    listRunners: () => getJSON('/api/runners'),

    submitJob: (runner, params) =>
        getJSON('/api/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ runner, params }),
        }),

    getJob: (id) => getJSON(`/api/jobs/${id}`),

    cancelJob: (id) => getJSON(`/api/jobs/${id}/cancel`, { method: 'POST' }),
};
