import React, { useMemo } from 'react';

const EL_COLOR = {
  H: '#e5e7eb', O: '#ef4444', C: '#6b7280', N: '#3b82f6',
  Pb: '#52525b', Te: '#d4aa00', Se: '#f59e0b', S: '#eab308',
  Ga: '#a67e5b', Ta: '#4da6ff',
};

/**
 * Selected-mode readout: energy, k-point, mode index, and per-element band
 * character (eigenvector weights — port of Calculators._archive_get_ph_weights:
 * weight_el = Σ_sites∈el sqrt(Σ_xyz |e|²)). Shows which species dominate the mode.
 */
export default function ModeInspector({ results, selectedK, selectedMode }) {
  const info = useMemo(() => {
    if (!results) return null;
    const { bands, qPoints, eigvecs, baseStructure } = results;
    const e = bands[selectedK]?.[selectedMode];
    const q = qPoints[selectedK];
    const ev = eigvecs[selectedK]?.[selectedMode];
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

  return (
    <div className="glass-panel px-4 py-3 rounded-lg text-xs font-mono text-gray-300 max-w-[280px]">
      <div className="flex justify-between gap-4 mb-1">
        <span className="text-gray-400">Mode</span><span>{selectedMode + 1}</span>
      </div>
      <div className="flex justify-between gap-4 mb-1">
        <span className="text-gray-400">Energy</span><span className={info.e < 0 ? 'text-red-400' : 'text-blue-300'}>{info.e.toFixed(3)} meV</span>
      </div>
      <div className="flex justify-between gap-4 mb-2">
        <span className="text-gray-400">k (frac)</span>
        <span>[{info.q.map(v => v.toFixed(3)).join(', ')}]</span>
      </div>
      <div className="text-gray-400 mb-1">Band character</div>
      <div className="space-y-1">
        {info.weights.map(({ el, frac }) => (
          <div key={el} className="flex items-center gap-2">
            <span className="w-6 text-right">{el}</span>
            <div className="flex-1 h-2 bg-white/10 rounded overflow-hidden">
              <div className="h-full rounded" style={{ width: `${frac * 100}%`, background: EL_COLOR[el] || '#9ca3af' }} />
            </div>
            <span className="w-9 text-right text-gray-400">{(frac * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
