import React, { useMemo, useRef, useState } from 'react';
import { conventionalLattice, reciprocalLattice, pathDistances } from '../math/reciprocal';

/**
 * Physical band-structure plot (custom SVG).
 *
 * x-axis: cumulative reciprocal-path distance (Å⁻¹) with high-symmetry tick
 * labels and vertical segment dividers. y-axis: energy (meV) with a 0 line.
 * Click (or hover) selects the nearest (k, mode) and reports it via onPick;
 * the selected point is marked.
 */
const W = 820, H = 420;
const M = { l: 56, r: 16, t: 18, b: 40 };

export default function BandStructurePlot({ bands, qPoints, baseStructure, kpathMeta, selected, onPick, eMin, eMax }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const model = useMemo(() => {
    if (!bands?.length || !qPoints?.length) return null;
    const recip = reciprocalLattice(conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim));
    const dist = pathDistances(qPoints, recip, kpathMeta?.segSizes);
    const xMax = dist[dist.length - 1] || 1;
    let yMin = 0, yMax = 1;
    for (const row of bands) for (const v of row) { if (!isFinite(v)) continue; if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    yMax *= 1.05; if (yMin < 0) yMin *= 1.05;
    if (Number.isFinite(eMin)) yMin = eMin;
    if (Number.isFinite(eMax)) yMax = eMax;
    const nModes = bands[0].length;

    const plotW = W - M.l - M.r, plotH = H - M.t - M.b;
    const xOf = (d) => M.l + (d / xMax) * plotW;
    const yOf = (e) => M.t + plotH - ((e - yMin) / (yMax - yMin)) * plotH;

    // One polyline per mode.
    const paths = [];
    for (let m = 0; m < nModes; m++) {
      let dStr = '';
      for (let k = 0; k < bands.length; k++) {
        const v = bands[k][m];
        if (!isFinite(v)) continue;
        dStr += (dStr ? 'L' : 'M') + xOf(dist[k]).toFixed(1) + ' ' + yOf(v).toFixed(1) + ' ';
      }
      paths.push(dStr);
    }

    // High-symmetry ticks (label + divider) from kpathMeta.hsymIndex.
    const ticks = [];
    const hsym = kpathMeta?.hsymIndex || {};
    for (const k of Object.keys(hsym)) {
      const ki = +k;
      ticks.push({ x: xOf(dist[ki]), label: hsym[k] });
    }

    // y gridlines
    const yticks = [];
    const nY = 6;
    for (let i = 0; i <= nY; i++) {
      const e = yMin + (i / nY) * (yMax - yMin);
      yticks.push({ y: yOf(e), label: e.toFixed(0) });
    }

    return { dist, xOf, yOf, paths, ticks, yticks, nModes, yMin, yMax };
  }, [bands, qPoints, baseStructure, kpathMeta, eMin, eMax]);

  if (!model) return null;

  const pickNearest = (evt) => {
    // Map client coords to viewBox user space via the SVG CTM so the picked
    // point matches the cursor exactly regardless of preserveAspectRatio
    // letterboxing (a naive rect ratio is wrong when aspect ratios differ).
    const svg = svgRef.current;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const loc = pt.matrixTransform(ctm.inverse());
    const px = loc.x, py = loc.y;
    let best = null, bestD = Infinity;
    for (let k = 0; k < bands.length; k++) {
      const x = model.xOf(model.dist[k]);
      for (let m = 0; m < model.nModes; m++) {
        const v = bands[k][m];
        if (!isFinite(v)) continue;
        const y = model.yOf(v);
        const dd = (x - px) * (x - px) + (y - py) * (y - py);
        if (dd < bestD) { bestD = dd; best = { k, m, x, y, e: v }; }
      }
    }
    return bestD < 400 ? best : null; // ~20px radius
  };

  const selPt = selected && isFinite(bands[selected.k]?.[selected.m])
    ? { x: model.xOf(model.dist[selected.k]), y: model.yOf(bands[selected.k][selected.m]) }
    : null;

  return (
    <div className="w-full h-full p-2">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-full"
        style={{ cursor: 'crosshair' }}
        onMouseMove={(e) => setHover(pickNearest(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => { const p = pickNearest(e); if (p && onPick) onPick(p.k, p.m); }}
      >
        {/* y grid + labels */}
        {model.yticks.map((t, i) => (
          <g key={`y${i}`}>
            <line x1={M.l} x2={W - M.r} y1={t.y} y2={t.y} stroke="rgba(255,255,255,0.07)" />
            <text x={M.l - 8} y={t.y + 3} textAnchor="end" fontSize="11" fill="#9ca3af">{t.label}</text>
          </g>
        ))}
        {/* zero line */}
        <line x1={M.l} x2={W - M.r} y1={model.yOf(0)} y2={model.yOf(0)} stroke="rgba(255,255,255,0.25)" strokeDasharray="3 3" />

        {/* high-symmetry dividers + labels */}
        {model.ticks.map((t, i) => (
          <g key={`x${i}`}>
            <line x1={t.x} x2={t.x} y1={M.t} y2={H - M.b} stroke="rgba(255,255,255,0.18)" />
            <text x={t.x} y={H - M.b + 16} textAnchor="middle" fontSize="12" fill="#e5e7eb">{t.label}</text>
          </g>
        ))}

        {/* bands */}
        {model.paths.map((d, m) => (
          <path key={m} d={d} fill="none" stroke={`hsl(${(m * 47) % 360} 65% 62%)`} strokeWidth="1.1" opacity="0.85" />
        ))}

        {/* hover + selection markers */}
        {hover && <circle cx={hover.x} cy={hover.y} r="4" fill="#fff" opacity="0.6" />}
        {selPt && <circle cx={selPt.x} cy={selPt.y} r="5" fill="#ef4444" stroke="#fff" strokeWidth="1.5" />}

        {/* axis labels */}
        <text x={16} y={H / 2} fontSize="12" fill="#9ca3af" transform={`rotate(-90 16 ${H / 2})`} textAnchor="middle">Energy (meV)</text>
        <text x={(M.l + W - M.r) / 2} y={H - 4} fontSize="11" fill="#6b7280" textAnchor="middle">Wave vector (high-symmetry path)</text>

        {/* hover readout */}
        {hover && (
          <text x={W - M.r} y={M.t + 12} textAnchor="end" fontSize="12" fill="#fbbf24">
            {hover.e.toFixed(2)} meV · band {hover.m + 1} · k{hover.k + 1}
          </text>
        )}
      </svg>
    </div>
  );
}
