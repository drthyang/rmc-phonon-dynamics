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
};
