// web/src/io/sessionSnapshot.js
//
// Portable `.rmcph.json` Workspace Snapshot Serializer & Hydrator
// ===============================================================
// Allows saving and restoring an entire multi-candidate RMC Phonon Dynamics
// session (pinned Parent & Local subgroup structures, diagonalized phonon bands,
// eigenvectors / compact force constants, neutron GDOS, powder S(|Q|, E), and
// loaded experimental 1D / 2D INS spectra) in a single self-contained JSON file.
//
// Handles transparent conversion between JavaScript `TypedArray` buffers
// (`Float64Array`, `Float32Array`, `Int32Array`) and compact JSON arrays, while
// encoding `NaN` kinematic detector masks in 2D experimental `.spe` grids cleanly.

export const WORKSPACE_SNAPSHOT_FORMAT = 'rmc-phonon-workspace-v1';

function serializeValue(val) {
  if (val === null || val === undefined) return val;
  if (typeof val === 'number') {
    return Number.isFinite(val) ? Math.round(val * 1e8) / 1e8 : null;
  }
  if (ArrayBuffer.isView(val) && !(val instanceof DataView)) {
    const out = new Array(val.length);
    for (let i = 0; i < val.length; i++) {
      const x = val[i];
      out[i] = Number.isFinite(x) ? Math.round(x * 1e7) / 1e7 : null;
    }
    return { __typed: val.constructor.name, data: out };
  }
  if (Array.isArray(val)) {
    return val.map(serializeValue);
  }
  if (typeof val === 'object') {
    const obj = {};
    for (const [k, v] of Object.entries(val)) {
      // Skip transient runtime caches like `_asr` on dynamicalMatrixPayload
      if (k.startsWith('_')) continue;
      obj[k] = serializeValue(v);
    }
    return obj;
  }
  return val;
}

function hydrateValue(val) {
  if (val === null || val === undefined) return val;
  if (Array.isArray(val)) return val.map(hydrateValue);
  if (typeof val === 'object') {
    if (typeof val.__typed === 'string' && Array.isArray(val.data)) {
      const nums = val.data.map(x => (x === null ? NaN : Number(x)));
      if (val.__typed === 'Float32Array') return Float32Array.from(nums);
      if (val.__typed === 'Int32Array') return Int32Array.from(nums);
      return Float64Array.from(nums);
    }
    const out = {};
    for (const [k, v] of Object.entries(val)) {
      out[k] = hydrateValue(v);
    }
    return out;
  }
  return val;
}

function hydrateExp2D(exp2D) {
  if (!exp2D || !Array.isArray(exp2D.sqe)) return exp2D;
  return {
    ...exp2D,
    sqe: exp2D.sqe.map(row => row.map(v => (v === null ? NaN : v))),
    sigma: Array.isArray(exp2D.sigma)
      ? exp2D.sigma.map(row => row.map(v => (v === null ? NaN : v)))
      : null,
  };
}

/**
 * Serialize workspace state into a `.rmcph.json` string.
 */
export function exportWorkspaceSnapshot({
  candidates = [],
  activeCandidateId = null,
  page = 'symmetry',
  exp1D = null,
  exp2D = null,
}) {
  // If a candidate already carries `dynamicalMatrixPayload`, we can omit redundant
  // full `eigvecs` arrays when serializing if desired, or keep compact round-trip fidelity.
  const cleanCandidates = candidates
    .filter(c => !c.mlipRunning)
    .map(c => ({
      id: c.id,
      label: c.label,
      color: c.color,
      spaceGroup: c.spaceGroup || null,
      kind: c.kind || null,
      source: c.source || 'snapshot',
      filename: c.filename || null,
      cellMult: c.cellMult || null,
      cellLabel: c.cellLabel || null,
      symprec: c.symprec ?? null,
      supercell: c.supercell || null,
      cifText: c.cifText || null,
      mlipModelId: c.mlipModelId || null,
      mlipSpaceGroup: c.mlipSpaceGroup || null,
      mlipPrimAtoms: c.mlipPrimAtoms || null,
      mlipStatus: c.mlipStatus || null,
      model: serializeValue(c.model || null),
    }));

  const payload = {
    format: WORKSPACE_SNAPSHOT_FORMAT,
    version: 1,
    savedAt: new Date().toISOString(),
    page,
    activeCandidateId,
    candidates: cleanCandidates,
    exp1D: serializeValue(exp1D),
    exp2D: serializeValue(exp2D),
  };
  return JSON.stringify(payload);
}

/**
 * Parse and hydrate a `.rmcph.json` workspace snapshot string.
 */
export function importWorkspaceSnapshot(text) {
  let raw;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    throw new Error(`Invalid JSON in workspace snapshot: ${err.message}`);
  }
  if (!raw || raw.format !== WORKSPACE_SNAPSHOT_FORMAT || !Array.isArray(raw.candidates)) {
    throw new Error('Not a valid `.rmcph.json` workspace snapshot (`rmc-phonon-workspace-v1`).');
  }
  const candidates = raw.candidates.map(c => ({
    ...c,
    mlipRunning: false,
    mlipError: null,
    model: hydrateValue(c.model),
  }));
  return {
    page: raw.page || (raw.exp1D || raw.exp2D ? 'ins' : candidates.some(c => c.model?.bands) ? 'phonons' : 'symmetry'),
    activeCandidateId: raw.activeCandidateId || candidates[0]?.id || null,
    candidates,
    exp1D: hydrateValue(raw.exp1D) || null,
    exp2D: hydrateExp2D(hydrateValue(raw.exp2D)) || null,
  };
}

export function isWorkspaceSnapshotObject(obj) {
  return Boolean(obj && obj.format === WORKSPACE_SNAPSHOT_FORMAT && Array.isArray(obj.candidates));
}
