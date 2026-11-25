import React, { useMemo } from 'react';
import { DEFAULT_COLORS } from '../constants';

/**
 * Selected-mode readout footer in MOTIF's quiet warm-paper style.
 */
export default function ModeInspector({ results, selectedK, selectedMode, unit = 'meV', scale = 1, colors = {} }) {
  const info = useMemo(() => {
    if (!results) return null;
    const { bands, qPoints, eigvecs, baseStructure } = results;
    const e = bands[selectedK]?.[selectedMode];
    const q = qPoints[selectedK];
    const ev = eigvecs?.[selectedK]?.[selectedMode];
    if (e === undefined || !ev) return null;

    const { atomDic, uniqueRN } = baseStructure;
    const rnToRow = new Map((uniqueRN || []).map((rn, r) => [rn, r]));
    const weights = [];
    let total = 0;
    for (const el of Object.keys(atomDic)) {
      let w = 0;
      for (const rn of atomDic[el]) {
        const r = rnToRow.get(rn);
        if (r === undefined) continue;
        let s = 0;
        for (let c = 0; c < 3; c++) {
          const re = ev.real[r * 3 + c], im = ev.imag[r * 3 + c];
          s += re * re + im * im;
        }
        w += Math.sqrt(s);
      }
      weights.push({ el, w });
      total += w;
    }
    weights.forEach(x => { x.frac = total > 0 ? x.w / total : 0; });
    weights.sort((a, b) => b.frac - a.frac);
    return { e, q, weights };
  }, [results, selectedK, selectedMode]);

  if (!info) return null;
  const colorOf = (el) => colors[el] || DEFAULT_COLORS[el] || '#9a958d';

  return (
    <div style={{ borderTop: '1px solid var(--line)', padding: '10px 14px', background: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
      <div className="tabular-nums" style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 162, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ color: 'var(--faint)' }}>mode</span>
          <span style={{ color: 'var(--ink)' }}>band {selectedMode + 1} · k{selectedK + 1}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ color: 'var(--faint)' }}>energy</span>
          <b style={{ color: info.e < 0 ? 'var(--warn-ink)' : 'var(--ink)' }}>{(info.e * scale).toFixed(3)} {unit}</b>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
          <span style={{ color: 'var(--faint)' }}>q</span>
          <span style={{ color: 'var(--secondary)', fontSize: 10.5 }}>[{info.q.map(v => v.toFixed(3)).join(', ')}]</span>
        </div>
      </div>

      <div style={{ flex: 1, minWidth: 170, display: 'flex', flexDirection: 'column', gap: 5 }}>
        <div style={{ fontSize: 11, color: 'var(--secondary)', fontWeight: 600 }}>Band character</div>
        {info.weights.map(({ el, frac }) => (
          <div key={el} className="tabular-nums" style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
            <span style={{ width: 22, fontWeight: 600, color: 'var(--ink)' }}>{el}</span>
            <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--surface)', border: '1px solid var(--line)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${frac * 100}%`, background: colorOf(el) }} />
            </div>
            <span style={{ width: 32, textAlign: 'right', color: 'var(--secondary)' }}>{(frac * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
