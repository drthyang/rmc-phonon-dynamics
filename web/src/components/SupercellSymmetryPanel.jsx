import React, { useState, useMemo } from 'react';

const DOMAIN_PALETTE = [
  '#c95d3a', '#2b6cb0', '#2f855a', '#b7791f',
  '#7b4b94', '#1d8a8a', '#b83280', '#5a67d8',
  '#9c4221', '#2c5282', '#276749', '#975a16',
];

/**
 * Supercell symmetry analysis panel — MOTIF editorial style.
 * Surfaces direct supercell symmetry metrics without folding to a single unit cell:
 *   - Local cell-by-cell subgroup projection (R², F-stat, ΔF orthogonal to parent G0, NN clustering)
 *   - Commensurate k-space mode power spectrum P(k) + mode-filtered minimal supercell ladder
 *   - Interactive 2D z-slice domain orientation map across the supercell
 */
export default function SupercellSymmetryPanel({
  result,
  busy,
  error,
  canRun,
  sampleSize,
  onSampleSizeChange,
  onRun,
  onAlignSubgroup,
  onDownloadCIF,
  helpOpen,
  onToggleHelp,
}) {
  const [tab, setTab] = useState('subgroups');
  const [sliceZ, setSliceZ] = useState(0);
  const [selectedSgIdx, setSelectedSgIdx] = useState(null);

  // Best symmetry-breaking subgroup (highest ΔF orthogonal to parent G0)
  const topBreaking = useMemo(() => {
    if (!result?.local?.subgroups?.length) return null;
    const breaking = result.local.subgroups.filter(s => s.breakingDoF > 0 && s.dof < result.local.internalDoF);
    if (!breaking.length) return null;
    return [...breaking].sort((a, b) => b.breakingFStat - a.breakingFStat)[0];
  }, [result]);

  // Subgroups that have a 3D domainGrid (nVariants > 1)
  const domainSubgroups = useMemo(() => {
    if (!result?.local?.subgroups?.length) return [];
    return result.local.subgroups
      .filter(s => s.domainGrid && s.nVariants > 1)
      .sort((a, b) => b.breakingFStat - a.breakingFStat);
  }, [result]);

  const activeDomainSg = useMemo(() => {
    if (!domainSubgroups.length) return null;
    if (selectedSgIdx !== null && domainSubgroups[selectedSgIdx]) return domainSubgroups[selectedSgIdx];
    return domainSubgroups[0];
  }, [domainSubgroups, selectedSgIdx]);

  const maxDeltaF = useMemo(() => {
    if (!result?.local?.subgroups?.length) return 1;
    return Math.max(1, ...result.local.subgroups.map(s => s.breakingFStat || 0));
  }, [result]);

  const maxSnr = useMemo(() => {
    if (!result?.spectrum?.topK?.length) return 1;
    return Math.max(1, ...result.spectrum.topK.map(k => k.snr || 0));
  }, [result]);

  return (
    <div style={{ paddingTop: 14, borderTop: '1px solid var(--line)' }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', marginBottom: 8 }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 'var(--fz-small)', color: 'var(--secondary)', fontWeight: 600 }}>
            Supercell symmetry analysis
          </span>
          <button
            type="button"
            className={`m-info ${helpOpen ? 'on' : ''}`}
            onClick={onToggleHelp}
            aria-label="About supercell symmetry analysis"
          >?</button>
        </div>

        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 'var(--fz-small)', color: 'var(--secondary)' }}>
            <span>configs</span>
            <input
              type="number"
              min={1}
              max={200}
              step={1}
              value={sampleSize}
              onChange={e => {
                const v = parseInt(e.target.value, 10);
                if (Number.isFinite(v)) onSampleSizeChange(Math.max(1, Math.min(200, v)));
              }}
              className="m-input mono"
              style={{ width: 54, height: 26, textAlign: 'center' }}
            />
          </label>
          <button
            type="button"
            onClick={onRun}
            disabled={busy || !canRun}
            className={`m-btn sm ${result ? '' : 'active'}`}
            title="Analyze local subgroups, k-space mode power, and domain structure directly on the supercell without unit-cell folding."
          >
            {busy ? 'Analyzing supercell…' : result ? 'Re-analyze supercell' : 'Analyze supercell'}
          </button>
        </div>
      </div>

      {helpOpen && (
        <p className="m-explain">
          Analyzes symmetry directly on the full RMC supercell without collapsing it into a single averaged unit cell.
          <b> Local subgroups</b> project every individual unit cell onto each closed subgroup of the parent space group and rank pure symmetry breaking by ΔF (variance per DoF orthogonal to the parent group).
          <b> k-Space modes</b> detect zone-boundary and low-q correlated distortions that cancel in 1×1×1 folding.
        </p>
      )}

      {error && (
        <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--warn-bg, #fdf2f2)', color: 'var(--warn-ink, #9b1c1c)', fontSize: 'var(--fz-small)', marginBottom: 8 }}>
          {error}
        </div>
      )}

      {!result && !busy && !error && (
        <div style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
          Run on the loaded supercell ensemble to detect local multi-domain and zone-boundary symmetry breaking that averages out in a folded unit cell.
        </div>
      )}

      {result && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 6 }}>
          {/* Compact summary strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 2 }}>Top local breaking (ΔF ⊥ G₀)</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
                <span style={{ fontFamily: 'var(--font-sym)', fontSize: 16, fontWeight: 700, color: 'var(--ink)' }}>
                  {topBreaking ? topBreaking.spaceGroup : result.local.parentSpaceGroup}
                </span>
                {topBreaking && (
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--secondary)' }}>
                    ΔF={topBreaking.breakingFStat.toFixed(2)} · R²={(100 * topBreaking.bestDomainR2).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>

            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 2 }}>Strongest k-mode</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink)', fontWeight: 600 }}>
                {result.spectrum.topK[0]
                  ? `[${result.spectrum.topK[0].kFrac.map(x => x.toFixed(2)).join(', ')}] · ${result.spectrum.topK[0].snr.toFixed(1)}×`
                  : '—'}
              </div>
            </div>

            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 2 }}>Folded mean / local RMS</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink)', fontWeight: 600 }}>
                {result.statistical.rmsMeanDisp.toFixed(4)} / {result.statistical.rmsLocalDisp.toFixed(4)} Å
                <span style={{ color: 'var(--faint)', fontWeight: 400 }}> ({(100 * result.statistical.meanToLocalRatio).toFixed(1)}%)</span>
              </div>
            </div>

            <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)' }}>
              <div style={{ fontSize: 11, color: 'var(--faint)', marginBottom: 2 }}>Supercell sample</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--ink)', fontWeight: 600 }}>
                {result.dim.join('×')} · {result.nConfigs} cfg · G₀={result.local.parentSpaceGroup}
              </div>
            </div>
          </div>

          {/* Tab selector */}
          <div style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap' }}>
            {[
              ['subgroups', 'Local subgroups (ΔF)'],
              ['aligned', 'Aligned structure & CIF'],
              ['kspace', 'k-Space modes'],
              ['domains', 'Domain slice map'],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={`m-btn sm ${tab === id ? 'active' : 'subtle'}`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Tab 1: Local Subgroups table */}
          {tab === 'subgroups' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ overflowX: 'auto', border: '1px solid var(--line)', borderRadius: 6 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                  <thead>
                    <tr style={{ background: 'var(--muted)', color: 'var(--secondary)', textAlign: 'left', borderBottom: '1px solid var(--line)' }}>
                      <th style={{ padding: '6px 8px' }}>Subgroup</th>
                      <th style={{ padding: '6px 6px' }}>PG</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>Vars</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>DoF</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>best R²</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>null R²</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>F</th>
                      <th style={{ padding: '6px 8px', minWidth: 110 }}>ΔF (breaking)</th>
                      <th style={{ padding: '6px 8px', textAlign: 'right' }}>NN cluster</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.local.subgroups.slice(0, 12).map((s, idx) => {
                      const isTopBrk = topBreaking && s.spaceGroup === topBreaking.spaceGroup && s.order === topBreaking.order && s.dof === topBreaking.dof;
                      const barW = Math.min(100, (s.breakingFStat / maxDeltaF) * 100);
                      return (
                        <tr
                          key={idx}
                          style={{
                            borderBottom: '1px solid var(--line)',
                            background: isTopBrk ? 'var(--note-bg, rgba(201,93,58,0.08))' : 'transparent',
                          }}
                        >
                          <td style={{ padding: '5px 8px', fontFamily: 'var(--font-sym)', fontWeight: isTopBrk ? 700 : 500, color: 'var(--ink)' }}>
                            {s.spaceGroup}
                          </td>
                          <td style={{ padding: '5px 6px', color: 'var(--secondary)' }}>{s.pointGroup}</td>
                          <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--secondary)' }}>{s.nVariants}</td>
                          <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--secondary)' }}>{s.dof}</td>
                          <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--ink)' }}>{(100 * s.bestDomainR2).toFixed(1)}%</td>
                          <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--faint)' }}>{(100 * s.nullR2).toFixed(1)}%</td>
                          <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--secondary)' }}>{s.fStat.toFixed(2)}</td>
                          <td style={{ padding: '5px 8px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span style={{ width: 34, textAlign: 'right', fontWeight: isTopBrk ? 700 : 400, color: isTopBrk ? 'var(--ink)' : 'var(--secondary)' }}>
                                {s.breakingDoF > 0 ? s.breakingFStat.toFixed(2) : '—'}
                              </span>
                              {s.breakingDoF > 0 && (
                                <div style={{ flex: 1, height: 6, background: 'var(--muted)', borderRadius: 3, overflow: 'hidden' }}>
                                  <div style={{ width: `${barW}%`, height: '100%', background: isTopBrk ? '#c95d3a' : 'var(--secondary)' }} />
                                </div>
                              )}
                            </div>
                          </td>
                          <td style={{ padding: '5px 8px', textAlign: 'right', color: s.neighborExcess > 0.01 ? 'var(--ink)' : 'var(--faint)' }}>
                            {s.neighborMatchRate !== null
                              ? `${(100 * s.neighborMatchRate).toFixed(1)}% (${s.neighborExcess >= 0 ? '+' : ''}${(100 * s.neighborExcess).toFixed(1)}%)`
                              : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {topBreaking && topBreaking.nVariants > 1 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--secondary)' }}>
                  <span>{topBreaking.spaceGroup} domain populations:</span>
                  {topBreaking.domainFractions.map((f, i) => (
                    <span
                      key={i}
                      style={{
                        display: 'inline-flex', alignItems: 'center', gap: 4,
                        padding: '1px 6px', borderRadius: 4, background: 'var(--muted)',
                      }}
                    >
                      <span style={{ width: 8, height: 8, borderRadius: 2, background: DOMAIN_PALETTE[i % DOMAIN_PALETTE.length] }} />
                      D{i + 1}: {(100 * f).toFixed(1)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Domain-Aligned Local Structure & CIF Export */}
          {tab === 'aligned' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--secondary)' }}>
                  <span>Align domains to</span>
                  <select
                    value={result.aligned ? `${result.aligned.targetSpaceGroup}|${result.aligned.verifiedOrder}` : ''}
                    onChange={e => {
                      const [sg, ord] = e.target.value.split('|');
                      const match = result.local.subgroups.find(s => s.spaceGroup === sg && String(s.order) === ord)
                        || result.local.subgroups.find(s => s.spaceGroup === sg);
                      if (match && onAlignSubgroup) onAlignSubgroup(match);
                    }}
                    className="m-input"
                    style={{ height: 28, padding: '0 8px' }}
                  >
                    {result.local.subgroups
                      .filter(s => s.breakingDoF > 0 && s.dof < result.local.internalDoF)
                      .sort((a, b) => b.breakingFStat - a.breakingFStat)
                      .map((s, i) => (
                        <option key={i} value={`${s.spaceGroup}|${s.order}`}>
                          {s.spaceGroup} ({s.nVariants} twins · ΔF={s.breakingFStat.toFixed(2)})
                        </option>
                      ))}
                  </select>
                </label>

                {result.aligned && onDownloadCIF && (
                  <div style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      onClick={() => onDownloadCIF(result.aligned, true)}
                      className="m-btn sm active"
                      title="Download domain-aligned unit cell projected onto exact subgroup symmetry (ready for VESTA / FINDSYM / Rietveld)."
                    >
                      Download {result.aligned.verifiedSpaceGroup} .cif
                    </button>
                    <button
                      type="button"
                      onClick={() => onDownloadCIF(result.aligned, false)}
                      className="m-btn sm subtle"
                      title="Download raw unprojected domain-aligned unit cell."
                    >
                      Raw aligned .cif
                    </button>
                  </div>
                )}
              </div>

              {result.aligned ? (
                <>
                  <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)', fontSize: 12, fontFamily: 'var(--font-mono)', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                    <span>
                      Verified SG (0.001 Å): <b style={{ fontFamily: 'var(--font-sym)', color: 'var(--ink)' }}>{result.aligned.verifiedSpaceGroup}</b>
                    </span>
                    <span>
                      Aligned RMS: <b style={{ color: 'var(--ink)' }}>{result.aligned.symRms.toFixed(5)} Å</b>
                    </span>
                    <span>
                      Standard folded RMS: <b style={{ color: 'var(--faint)' }}>{result.aligned.foldedRms.toFixed(5)} Å</b>
                    </span>
                    <span>
                      Max atomic shift: <b style={{ color: 'var(--ink)' }}>{result.aligned.maxShift.toFixed(5)} Å</b>
                    </span>
                    <span style={{ color: 'var(--secondary)' }}>
                      ({result.aligned.nVariants} twin{result.aligned.nVariants === 1 ? '' : 's'} × {result.aligned.nNormalizerCosets} polarity coset{result.aligned.nNormalizerCosets === 1 ? '' : 's'} aligned over {result.aligned.nSamples} cells)
                    </span>
                  </div>

                  <div style={{ overflowX: 'auto', border: '1px solid var(--line)', borderRadius: 6 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                      <thead>
                        <tr style={{ background: 'var(--muted)', color: 'var(--secondary)', textAlign: 'left', borderBottom: '1px solid var(--line)' }}>
                          <th style={{ padding: '6px 8px' }}>Element</th>
                          <th style={{ padding: '6px 6px', textAlign: 'right' }}>Sites</th>
                          <th style={{ padding: '6px 6px', textAlign: 'right' }}>Aligned RMS (Å)</th>
                          <th style={{ padding: '6px 6px', textAlign: 'right' }}>Max shift (Å)</th>
                          <th style={{ padding: '6px 8px', textAlign: 'right' }}>Standard folded RMS (Å)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.aligned.elementStats.map((es, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid var(--line)' }}>
                            <td style={{ padding: '5px 8px', fontWeight: 700, color: 'var(--ink)' }}>{es.el}</td>
                            <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--secondary)' }}>{es.count}</td>
                            <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--ink)', fontWeight: 600 }}>{es.symRms.toFixed(5)}</td>
                            <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--ink)' }}>{es.maxShift.toFixed(5)}</td>
                            <td style={{ padding: '5px 8px', textAlign: 'right', color: 'var(--faint)' }}>{es.foldedRms.toFixed(5)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
                  No proper symmetry-breaking subgroup available for domain alignment.
                </div>
              )}
            </div>
          )}

          {/* Tab 3: k-Space Mode Power & Minimal Supercell */}
          {tab === 'kspace' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ overflowX: 'auto', border: '1px solid var(--line)', borderRadius: 6 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                  <thead>
                    <tr style={{ background: 'var(--muted)', color: 'var(--secondary)', textAlign: 'left', borderBottom: '1px solid var(--line)' }}>
                      <th style={{ padding: '6px 8px' }}>k (rlu)</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>P(k) Å²</th>
                      <th style={{ padding: '6px 8px', minWidth: 110 }}>SNR vs BZ</th>
                      <th style={{ padding: '6px 6px', textAlign: 'right' }}>Coherent</th>
                      <th style={{ padding: '6px 8px' }}>Element breakdown</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.spectrum.topK.slice(0, 10).map((kp, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--line)' }}>
                        <td style={{ padding: '5px 8px', color: 'var(--ink)' }}>
                          [{kp.kFrac.map(x => x.toFixed(3)).join(', ')}]
                        </td>
                        <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--ink)' }}>{kp.power.toFixed(4)}</td>
                        <td style={{ padding: '5px 8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 42, textAlign: 'right', color: 'var(--ink)' }}>{kp.snr.toFixed(1)}×</span>
                            <div style={{ flex: 1, height: 6, background: 'var(--muted)', borderRadius: 3, overflow: 'hidden' }}>
                              <div style={{ width: `${Math.min(100, (kp.snr / maxSnr) * 100)}%`, height: '100%', background: '#2b6cb0' }} />
                            </div>
                          </div>
                        </td>
                        <td style={{ padding: '5px 6px', textAlign: 'right', color: 'var(--secondary)' }}>
                          {(100 * kp.coherentPower / Math.max(1e-12, kp.power)).toFixed(1)}%
                        </td>
                        <td style={{ padding: '5px 8px', color: 'var(--secondary)', fontSize: 11 }}>
                          {Object.entries(kp.byEl).map(([el, v]) => `${el}:${v.toFixed(2)}`).join(' · ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ padding: '8px 10px', borderRadius: 6, background: 'var(--muted)', border: '1px solid var(--line)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                <span style={{ color: 'var(--secondary)' }}>Mode-filtered minimal supercell: </span>
                <b style={{ color: 'var(--ink)' }}>{result.minimal.miniDim.join('×')}</b>
                <span style={{ color: 'var(--secondary)' }}> ({result.minimal.nActiveK} active coherent k-modes) · Ladder: </span>
                <span style={{ color: 'var(--ink)' }}>
                  {result.minimal.ladder.length
                    ? result.minimal.ladder.map(b => `${b.spaceGroup}[${b.from.toFixed(3)}–${b.to.toFixed(3)}Å]`).join(' → ')
                    : '—'}
                </span>
              </div>
            </div>
          )}

          {/* Tab 3: Interactive 2D z-Slice Domain Map */}
          {tab === 'domains' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {activeDomainSg ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: 'var(--secondary)' }}>
                      <span>Subgroup</span>
                      <select
                        value={selectedSgIdx ?? 0}
                        onChange={e => setSelectedSgIdx(parseInt(e.target.value, 10))}
                        className="m-input"
                        style={{ height: 28, padding: '0 8px' }}
                      >
                        {domainSubgroups.map((sg, i) => (
                          <option key={i} value={i}>
                            {sg.spaceGroup} ({sg.nVariants} domains · ΔF={sg.breakingFStat.toFixed(2)})
                          </option>
                        ))}
                      </select>
                    </label>

                    <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color: 'var(--secondary)', fontFamily: 'var(--font-mono)' }}>
                      <span>z-layer: <b style={{ color: 'var(--ink)' }}>{sliceZ}</b> / {result.dim[2] - 1}</span>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(0, result.dim[2] - 1)}
                        value={Math.min(sliceZ, result.dim[2] - 1)}
                        onChange={e => setSliceZ(parseInt(e.target.value, 10))}
                        className="m-range"
                        style={{ width: 120 }}
                      />
                    </label>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: `repeat(${result.dim[1]}, 24px)`,
                        gap: 3,
                        padding: 8,
                        borderRadius: 6,
                        background: 'var(--muted)',
                        border: '1px solid var(--line)',
                      }}
                    >
                      {Array.from({ length: result.dim[0] * result.dim[1] }, (_, idx) => {
                        const x = Math.floor(idx / result.dim[1]);
                        const y = idx % result.dim[1];
                        const z = Math.min(sliceZ, result.dim[2] - 1);
                        const c = (x * result.dim[1] + y) * result.dim[2] + z;
                        const dom = activeDomainSg.domainGrid[c] ?? 0;
                        return (
                          <div
                            key={idx}
                            title={`Cell (${x}, ${y}, ${z}) → Domain D${dom + 1}`}
                            style={{
                              width: 24,
                              height: 24,
                              borderRadius: 4,
                              background: DOMAIN_PALETTE[dom % DOMAIN_PALETTE.length],
                              color: '#fff',
                              fontSize: 10,
                              fontFamily: 'var(--font-mono)',
                              fontWeight: 700,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            {dom + 1}
                          </div>
                        );
                      })}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                      <div style={{ color: 'var(--secondary)', fontWeight: 600 }}>
                        {activeDomainSg.spaceGroup} twin domains (last config)
                      </div>
                      {activeDomainSg.domainFractions.map((f, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ width: 12, height: 12, borderRadius: 3, background: DOMAIN_PALETTE[i % DOMAIN_PALETTE.length] }} />
                          <span style={{ color: 'var(--ink)' }}>Domain {i + 1}</span>
                          <span style={{ color: 'var(--secondary)' }}>{(100 * f).toFixed(1)}%</span>
                        </div>
                      ))}
                      {activeDomainSg.neighborMatchRate !== null && (
                        <div style={{ marginTop: 4, fontSize: 11, color: 'var(--secondary)' }}>
                          NN same-domain: <b style={{ color: 'var(--ink)' }}>{(100 * activeDomainSg.neighborMatchRate).toFixed(1)}%</b>
                          {' '}({activeDomainSg.neighborExcess >= 0 ? '+' : ''}{(100 * activeDomainSg.neighborExcess).toFixed(1)}% vs random)
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 'var(--fz-small)', color: 'var(--faint)' }}>
                  No multi-variant subgroups available for this parent space group.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
