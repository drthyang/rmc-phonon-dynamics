// web/src/io/mlipClient.js
//
// Client for the optional local `rmcph-mlip serve` companion server.
// Discovers installed universal MLIPs via `/health` and fails silently when
// offline so the static GitHub Pages app works standalone.

export const DEFAULT_MLIP_URL = 'http://127.0.0.1:8765';

export const FALLBACK_MLIP_MODELS = [
  {
    id: 'mace-mp-medium',
    short: 'MACE-MP-0',
    label: 'MACE-MP-0 Medium (PBE · MPtrj)',
    family: 'mace',
    functional: 'PBE',
    installed: true,
    cached_locally: true,
  },
  {
    id: 'mace-mpa-0',
    short: 'MACE-MPA-0',
    label: 'MACE-MPA-0 Medium (PBE · MPtrj + Alexandria)',
    family: 'mace',
    functional: 'PBE',
    installed: true,
    cached_locally: false,
  },
  {
    id: 'mace-omat-0',
    short: 'MACE-OMAT-0',
    label: 'MACE-OMAT-0 Medium (PBE · OMAT-24 Rattled Phonons)',
    family: 'mace',
    functional: 'PBE (OMAT-24)',
    installed: true,
    cached_locally: false,
  },
  {
    id: 'mace-matpes-r2scan-0',
    short: 'MACE-r²SCAN',
    label: 'MACE-MatPES-0 (r²SCAN meta-GGA)',
    family: 'mace',
    functional: 'r²SCAN',
    installed: true,
    cached_locally: false,
  },
  {
    id: 'mace-mp-0b3',
    short: 'MACE-0b3',
    label: 'MACE-MP-0b3 Medium (PBE + ZBL Repulsion)',
    family: 'mace',
    functional: 'PBE+ZBL',
    installed: true,
    cached_locally: false,
  },
  {
    id: 'mace-mp-small',
    short: 'MACE-Small',
    label: 'MACE-MP-0 Small (Fast Screening)',
    family: 'mace',
    functional: 'PBE',
    installed: true,
    cached_locally: false,
  },
  {
    id: 'mattersim-5m',
    short: 'MatterSim-5M',
    label: 'MatterSim-v1 5M (Microsoft Research)',
    family: 'mattersim',
    functional: 'PBE',
    installed: false,
    cached_locally: false,
    install_hint: 'uv sync --extra mattersim',
  },
  {
    id: 'orb-v3',
    short: 'ORB-v3',
    label: 'ORB-v3 Conservative OMAT (Orbital Materials)',
    family: 'orb',
    functional: 'PBE (OMAT)',
    installed: false,
    cached_locally: false,
    install_hint: 'uv sync --extra orb',
  },
  {
    id: 'sevennet-0',
    short: 'SevenNet-0',
    label: 'SevenNet-0 (Equivariant NequIP/Allegro)',
    family: 'sevennet',
    functional: 'PBE',
    installed: false,
    cached_locally: false,
    install_hint: 'uv sync --extra sevennet',
  },
  {
    id: 'chgnet',
    short: 'CHGNet',
    label: 'CHGNet (Crystal Hamiltonian Graph Net)',
    family: 'chgnet',
    functional: 'PBE/r²SCAN',
    installed: false,
    cached_locally: false,
    install_hint: 'uv sync --extra chgnet',
  },
];

export function shortModelName(modelId) {
  if (!modelId) return 'MLIP';
  const hit = FALLBACK_MLIP_MODELS.find(m => m.id === modelId);
  if (hit) return hit.short;
  return String(modelId).replace(/^mace-mp-/, 'MACE-').replace(/^mace-/, 'MACE-');
}

export async function checkMlipServer(baseUrl = DEFAULT_MLIP_URL, timeoutMs = 1500) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: ctrl.signal });
    if (!res.ok) return { connected: false, url: baseUrl, models: FALLBACK_MLIP_MODELS };
    const data = await res.json();
    return {
      connected: Boolean(data.ok),
      url: baseUrl,
      device: data.device,
      version: data.version,
      models: Array.isArray(data.models) && data.models.length ? data.models : FALLBACK_MLIP_MODELS,
    };
  } catch {
    return { connected: false, url: baseUrl, models: FALLBACK_MLIP_MODELS };
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
