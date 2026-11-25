import { useState, useRef, useEffect, useCallback } from 'react';
import { PhononPipeline } from './compute/pipeline';
import { fromResults } from './io/viewermodel';
import { fromPhonopyDM } from './io/phonopyDM';
import { checkMlipServer, runMlipJob, DEFAULT_MLIP_URL } from './io/mlipClient';
import { generatePhonopyBandYaml, downloadString } from './io/writers';
import ThemeCue from './components/ThemeCue';
import SymmetryPage from './pages/SymmetryPage';
import ViewerPage from './pages/ViewerPage';
import InsComparePage from './pages/InsComparePage';
import RunnerPage from './pages/RunnerPage';

const CANDIDATE_PALETTE = [
  '#2563eb', // cobalt
  '#c95d3a', // terracotta
  '#157347', // forest
  '#7b4b94', // plum
  '#b7791f', // ochre
  '#0f766e', // teal
];

export default function App() {
  const [page, setPage] = useState('symmetry');   // 'symmetry' | 'phonons' | 'ins' | 'legacy'
  const [candidates, setCandidates] = useState([]);
  const [activeCandidateId, setActiveCandidateId] = useState(null);
  const [ready, setReady] = useState(false);
  const [mlipServer, setMlipServer] = useState({ connected: false, url: DEFAULT_MLIP_URL });
  const pipelineRef = useRef(null);
  const nextId = useRef(1);

  const missingLegacyFeatures = [
    typeof navigator !== 'undefined' && navigator.gpu ? null : 'WebGPU',
    typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function' ? null : 'folder access (File System Access API)',
  ].filter(Boolean);

  useEffect(() => {
    const p = new PhononPipeline(() => {});
    pipelineRef.current = p;
    p.initWorkers(4)
      .then(() => setReady(true))
      .catch(e => { console.error('Init failed:', e); setReady(true); });
    checkMlipServer(DEFAULT_MLIP_URL).then(setMlipServer);
  }, []);

  const activeCandidate = candidates.find(c => c.id === activeCandidateId)
    || candidates.find(c => c.model)
    || null;
  const activeModel = activeCandidate?.model || null;

  const pinCandidate = useCallback((spec) => {
    const id = `cand-${nextId.current++}`;
    setCandidates(prev => {
      const color = spec.color || CANDIDATE_PALETTE[prev.length % CANDIDATE_PALETTE.length];
      return [...prev, { id, color, ...spec }];
    });
    if (spec.model) setActiveCandidateId(id);
    return id;
  }, []);

  const removeCandidate = useCallback((id) => {
    setCandidates(prev => prev.filter(c => c.id !== id));
    setActiveCandidateId(cur => (cur === id ? null : cur));
  }, []);

  const loadModelFile = useCallback((model, filename = 'Loaded model', opts = {}) => {
    const label = model.name || filename.replace(/\.(json|ya?ml)$/i, '');
    const id = pinCandidate({
      label,
      spaceGroup: model.mlipMetadata?.spacegroup || null,
      source: 'file',
      filename,
      model,
    });
    setActiveCandidateId(id);
    if (!opts.stayOnPage) setPage('phonons');
  }, [pinCandidate]);

  const pinDosCandidate = useCallback((neutronDos, filename = 'mlip_ndos.csv') => {
    pinCandidate({
      label: filename.replace(/_ndos\.csv$|\.csv$/i, ''),
      source: 'file',
      filename,
      model: { neutronDos },
    });
  }, [pinCandidate]);

  const runMlipForId = useCallback(async (candId, cifText, filename, params) => {
    if (!mlipServer.connected || !cifText) return;
    setCandidates(prev => prev.map(c => (
      c.id === candId ? { ...c, mlipRunning: true, mlipError: null, mlipStatus: 'Starting MLIP…' } : c
    )));
    try {
      const dmPayload = await runMlipJob(
        mlipServer.url,
        {
          cif_text: cifText,
          filename: filename || 'candidate.cif',
          model: params.model || 'mace-mp-medium',
          supercell: params.supercell?.length === 3 ? params.supercell : [2, 2, 2],
          temperature: params.temperature ?? 250,
          symprec: params.symprec ?? 'auto',
          relax: params.relax ?? true,
        },
        (msg) => setCandidates(prev => prev.map(c => (c.id === candId ? { ...c, mlipStatus: msg } : c)))
      );
      const model = fromPhonopyDM(dmPayload);
      const usedSg = model.mlipMetadata?.spacegroup || '';
      const nPrim = model.baseStructure?.uniqueRN?.length || '';
      setCandidates(prev => prev.map(c => (
        c.id === candId
          ? {
              ...c,
              model,
              mlipRunning: false,
              mlipError: null,
              mlipSpaceGroup: usedSg,
              mlipPrimAtoms: nPrim,
              mlipStatus: usedSg ? `${usedSg} · ${nPrim} prim atoms` : 'Ready ✓',
            }
          : c
      )));
      setActiveCandidateId(cur => cur || candId);
    } catch (err) {
      setCandidates(prev => prev.map(c => (
        c.id === candId
          ? { ...c, mlipRunning: false, mlipError: err.message, mlipStatus: `Error: ${err.message}` }
          : c
      )));
    }
  }, [mlipServer]);

  /**
   * One-click batch runner from the Symmetry studio:
   * Upserts each selected structure spec as a candidate and runs MLIP sequentially.
   */
  const runMlipBatch = useCallback(async (specs, params) => {
    if (!mlipServer.connected || !specs?.length) return;
    const queued = specs.map((spec, idx) => {
      const id = `cand-${nextId.current++}`;
      const color = CANDIDATE_PALETTE[(candidates.length + idx) % CANDIDATE_PALETTE.length];
      return { id, color, ...spec, mlipRunning: true, mlipError: null, mlipStatus: 'Queued…' };
    });
    setCandidates(prev => [...prev, ...queued]);
    for (const item of queued) {
      const itemParams = { ...params, symprec: item.symprec ?? params.symprec ?? 'auto' };
      await runMlipForId(item.id, item.cifText, item.filename, itemParams);
    }
    setActiveCandidateId(queued[0].id);
  }, [mlipServer, candidates.length, runMlipForId]);

  const onLegacyResults = (results, kpathMeta, dos = null) => {
    const m = fromResults(results, kpathMeta);
    if (dos) m.dos = dos;
    const id = pinCandidate({
      label: `RMC Ensemble (${m.temperature ?? '?'} K)`,
      source: 'legacy-rmc',
      model: m,
    });
    setActiveCandidateId(id);
    setPage('phonons');
    try {
      const yaml = generatePhonopyBandYaml(m.baseStructure, m.qPoints, m.bands, m.eigvecs, m.kpathMeta);
      downloadString(yaml, 'band_gpu.yaml', 'text/yaml');
    } catch (e) { console.error('band.yaml auto-save failed:', e); }
  };

  const nPhononModels = candidates.filter(c => c.model?.bands).length;
  const anyRunning = candidates.some(c => c.mlipRunning);

  const stepBadge = (num, isActive, isDone) => (
    <span
      style={{
        width: 19, height: 19, borderRadius: '50%',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)',
        background: isActive ? 'var(--ink)' : isDone ? 'var(--accent)' : 'var(--muted)',
        color: isActive ? 'var(--page)' : 'var(--ink)',
        border: isDone && !isActive ? '1px solid var(--ink)' : '1px solid transparent',
        flex: 'none',
      }}
    >
      {isDone && !isActive ? '✓' : num}
    </span>
  );

  return (
    <div className="m-shell">
      {/* ── Topbar: 1-2-3 guided pipeline stepper + quiet Legacy link ── */}
      <header className="m-topbar">
        <div className="m-brand">
          <span className="m-mark">RMC PHONON</span>
          <span className="m-submark">supercell symmetry → MLIP → INS</span>
        </div>

        <nav className="m-nav" aria-label="Workflow steps" style={{ gap: 14 }}>
          <button type="button" className={page === 'symmetry' ? 'on' : ''} onClick={() => setPage('symmetry')}>
            {stepBadge(1, page === 'symmetry', candidates.length > 0)}
            <span>Symmetry</span>
          </button>
          <span style={{ alignSelf: 'center', color: 'var(--control)', fontSize: 12, userSelect: 'none' }}>→</span>
          <button type="button" className={page === 'phonons' ? 'on' : ''} onClick={() => setPage('phonons')}>
            {stepBadge(2, page === 'phonons', nPhononModels > 0)}
            <span>Phonons</span>
            {nPhononModels > 0 && (
              <span className="tabular-nums" style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                ({nPhononModels})
              </span>
            )}
          </button>
          <span style={{ alignSelf: 'center', color: 'var(--control)', fontSize: 12, userSelect: 'none' }}>→</span>
          <button type="button" className={page === 'ins' ? 'on' : ''} onClick={() => setPage('ins')}>
            {stepBadge(3, page === 'ins', false)}
            <span>INS Compare</span>
          </button>
        </nav>

        <div className="m-actions">
          <button
            type="button"
            className={`m-btn sm ${page === 'legacy' ? 'active' : 'subtle'}`}
            onClick={() => setPage('legacy')}
            title="Legacy brute-force 500+ config RMC covariance runner (WebGPU)"
            style={{ fontSize: 11.5 }}
          >
            Legacy RMC
          </button>
          <ThemeCue />
          <span
            className="m-status"
            title={mlipServer.connected ? 'Local MLIP server connected (1-click phonon runs enabled)' : 'Static browser mode — start `uv run rmcph-mlip serve` for 1-click MLIP runs'}
          >
            <span className={anyRunning ? 'm-status-dot' : (mlipServer.connected || ready) ? 'm-status-dot ready' : 'm-status-dot'} style={anyRunning ? { background: 'var(--fig-accent)', animation: 'blip 1s infinite' } : undefined} />
            {anyRunning ? 'MLIP running…' : mlipServer.connected ? 'MLIP local' : 'Browser'}
          </span>
        </div>
      </header>

      {/* ── Browser notice ONLY on Legacy RMC tab ── */}
      {page === 'legacy' && missingLegacyFeatures.length > 0 && (
        <div className="m-notice bad" role="alert">
          <span>
            <strong>Legacy RMC Runner requires Chrome or Edge</strong> ({missingLegacyFeatures.join(' and ')}). Steps <b>1 · Symmetry</b>, <b>2 · Phonons</b>, and <b>3 · INS Compare</b> work in any browser.
          </span>
        </div>
      )}

      {/* ── Pages stay mounted to preserve datasets & WebGL canvases ── */}
      <div style={{ display: page === 'symmetry' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <SymmetryPage
          candidates={candidates}
          onPinCandidate={pinCandidate}
          onRemoveCandidate={removeCandidate}
          onLoadModelFile={loadModelFile}
          onGoToPage={setPage}
          mlipServer={mlipServer}
          onRunMlipBatch={runMlipBatch}
          onRetryCandidate={(c, params) => runMlipForId(c.id, c.cifText, c.filename, params)}
        />
      </div>

      <div style={{ display: page === 'phonons' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <ViewerPage
          model={activeModel}
          candidates={candidates}
          activeCandidateId={activeCandidate?.id || null}
          onSelectCandidate={setActiveCandidateId}
          onRemoveCandidate={removeCandidate}
          onLoadModel={(m, name) => loadModelFile(m, name)}
          onGoToIns={() => setPage('ins')}
        />
      </div>

      <div style={{ display: page === 'ins' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <InsComparePage
          candidates={candidates}
          onLoadModelFile={loadModelFile}
          onPinDosCandidate={pinDosCandidate}
        />
      </div>

      <div style={{ display: page === 'legacy' ? 'flex' : 'none', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <RunnerPage
          pipeline={pipelineRef.current}
          ready={ready}
          onResults={onLegacyResults}
          onLoadResult={(m) => loadModelFile(m, 'band')}
        />
      </div>

      {/* ── Footer ── */}
      <footer className="m-foot">
        <span>
          RMC Phonon Dynamics · © 2026 Tsung-Han Yang ·{' '}
          <a href="https://github.com/drthyang/rmc-phonon-dynamics/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
            AGPL-3.0
          </a>
        </span>
        <span style={{ display: 'inline-flex', gap: 14 }}>
          <a href="https://github.com/drthyang/rmc-phonon-dynamics" target="_blank" rel="noopener noreferrer">
            Source code
          </a>
        </span>
      </footer>
    </div>
  );
}
