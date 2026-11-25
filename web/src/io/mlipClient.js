// web/src/io/mlipClient.js
//
// Optional client for the local `rmcph-mlip serve` companion server.
// Fails silently when offline so the static GitHub Pages app works standalone.

export const DEFAULT_MLIP_URL = 'http://127.0.0.1:8765';

export async function checkMlipServer(baseUrl = DEFAULT_MLIP_URL, timeoutMs = 1200) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: ctrl.signal });
    if (!res.ok) return { connected: false, url: baseUrl };
    const data = await res.json();
    return { connected: !!data.ok, url: baseUrl, device: data.device, version: data.version };
  } catch {
    return { connected: false, url: baseUrl };
  } finally {
    clearTimeout(timer);
  }
}

export async function runMlipJob(baseUrl, payload, onStatus) {
  const res = await fetch(`${baseUrl}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `MLIP server error (${res.status})`);
  }
  const { id } = await res.json();

  while (true) {
    await new Promise(r => setTimeout(r, 900));
    const poll = await fetch(`${baseUrl}/jobs/${id}`);
    if (!poll.ok) throw new Error(`Job poll failed (${poll.status})`);
    const job = await poll.json();
    if (job.message && onStatus) onStatus(job.message);
    if (job.status === 'done') return job.result;
    if (job.status === 'error') throw new Error(job.message || 'MLIP calculation failed');
  }
}
