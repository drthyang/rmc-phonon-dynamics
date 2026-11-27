import React, { useMemo, useRef, useState, useEffect, useId } from 'react';
import { conventionalLattice, reciprocalLattice, pathDistances } from '../math/reciprocal';
import { computeSpectralFunctionGrid } from '../math/unfolding';
import { clientToSvg } from './svgCoords';

/**
 * Physical band-structure plot (custom SVG, Cobalt light theme).
 *
 * x-axis: cumulative reciprocal-path distance (Å⁻¹) with high-symmetry tick
 * labels (Noto Sans) and vertical segment dividers. y-axis: energy (`unit`)
 * with adaptive gridlines and a dashed zero line. Branches are drawn in a red
 * family; sub-zero portions are overdrawn dashed (soft-mode / imaginary freq).
 * A blue marker flags the selected (k, mode).
 *
 * Interactions: drag a box *inside the plot frame* to zoom — the boxed data
 * region is remapped to fill the whole frame and the axis tick values recompute
 * for the zoomed range (a true data-domain zoom, not a pixel crop). A single
 * click (no drag) selects the nearest (k, mode). `resetSignal` (a changing
 * nonce) clears the zoom back to the full range.
 *
 * Axis math (reciprocalLattice / pathDistances / kpathMeta.hsymIndex) is the
 * same as before — only the styling/zoom changed.
 */
const W = 760, H = 880;
const XL = 64, XR = 740, YT = 30, YB = 810;   // plot frame (pixel/user space)
const PW = XR - XL, PH = YB - YT;
// Red family — distinct hue/lightness per branch while reading as "red".
const branchColor = (m) => `hsl(${(m * 6) % 16}, 72%, ${40 + (m % 7) * 3.2}%)`;

// A "nice" gridline step (1/2/5 × 10ⁿ) for ~`target` divisions over `range`.
function niceStep(range, target = 6) {
  const raw = Math.max(range, 1e-9) / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

// Perceptual warm-editorial spectral ramp (transparent -> amber -> terracotta -> deep crimson-ink)
function spectralRGBA(normVal) {
  if (!(normVal > 0.012)) return [0, 0, 0, 0];
  const t = Math.pow(Math.min(1, normVal), 0.58);
  // 3-stop interpolation: 0 -> (245, 198, 132), 0.5 -> (201, 93, 58), 1.0 -> (98, 22, 18)
  let r, g, b;
  if (t < 0.5) {
    const u = t * 2;
    r = 245 + (201 - 245) * u;
    g = 198 + (93 - 198) * u;
    b = 132 + (58 - 132) * u;
  } else {
    const u = (t - 0.5) * 2;
    r = 201 + (98 - 201) * u;
    g = 93 + (22 - 93) * u;
    b = 58 + (18 - 58) * u;
  }
  const alpha = Math.round(Math.min(1, t * 1.08) * 238);
  return [Math.round(r), Math.round(g), Math.round(b), alpha];
}

function renderSpectralHeatmapUrl(specGrid, dist, xLo, xHi, nXPixels = 340) {
  if (typeof document === 'undefined' || !specGrid || !(specGrid.maxIntensity > 0)) return null;
  const { intensity, nQ, nEnergyBins, maxIntensity } = specGrid;
  const canvas = document.createElement('canvas');
  canvas.width = nXPixels;
  canvas.height = nEnergyBins;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const img = ctx.createImageData(nXPixels, nEnergyBins);
  const data = img.data;
  const xSpan = (xHi - xLo) || 1;
  const invMax = 1 / maxIntensity;

  // Map each horizontal pixel column px -> fractional index along dist[0..nQ-1]
  let qCursor = 0;
  for (let px = 0; px < nXPixels; px++) {
    const targetDist = xLo + ((px + 0.5) / nXPixels) * xSpan;
    while (qCursor + 1 < nQ - 1 && dist[qCursor + 1] < targetDist) qCursor++;
    const d0 = dist[qCursor];
    const d1 = dist[Math.min(nQ - 1, qCursor + 1)];
    const frac = d1 > d0 + 1e-12 ? Math.max(0, Math.min(1, (targetDist - d0) / (d1 - d0))) : 0;
    const col0 = qCursor * nEnergyBins;
    const col1 = Math.min(nQ - 1, qCursor + 1) * nEnergyBins;

    for (let bin = 0; bin < nEnergyBins; bin++) {
      const val = (intensity[col0 + bin] * (1 - frac) + intensity[col1 + bin] * frac) * invMax;
      const [r, g, b, a] = spectralRGBA(val);
      if (a === 0) continue;
      // Flip y so bin 0 (yMin) is at bottom of image and bin nEnergyBins-1 (yMax) is at top
      const row = (nEnergyBins - 1) - bin;
      const off = (row * nXPixels + px) * 4;
      data[off] = r;
      data[off + 1] = g;
      data[off + 2] = b;
      data[off + 3] = a;
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}

function buildSegmentStarts(segSizes, nTotal) {
  const starts = new Set([0]);
  let off = 0;
  for (const sz of (segSizes || [nTotal])) {
    starts.add(off);
    off += sz;
  }
  return starts;
}

export default function BandStructurePlot({
  bands,
  weights = null,
  spectralStyle = 'both', // 'both' | 'heatmap' | 'weighted'
  qPoints,
  baseStructure,
  kpathMeta,
  selected,
  onPick,
  eMin,
  eMax,
  unit = 'meV',
  resetSignal = 0,
  overlays = [],
}) {
  const svgRef = useRef(null);
  const clipId = useId();
  const [domain, setDomain] = useState(null);    // zoom window {xMin,xMax,yMin,yMax} in DATA coords, or null = full
  const [sel, setSel] = useState(null);          // live drag box {x0,y0,x1,y1} in pixel space
  const [hover, setHover] = useState(null);      // nearest point under the cursor {k,m,x,y}
  const dragRef = useRef(null);

  // Reset zoom whenever the parent bumps resetSignal.
  useEffect(() => { setDomain(null); }, [resetSignal]);

  const model = useMemo(() => {
    if (!bands?.length || !qPoints?.length) return null;
    // q-points are in conventional coords; prefer the conventional reciprocal the
    // runner attaches (bandRecip) so path distances are right even when the 3D
    // viewer cell is a custom/primitive computation cell.
    const recip = baseStructure.bandRecip || reciprocalLattice(conventionalLattice(baseStructure.v1, baseStructure.v2, baseStructure.v3, baseStructure.dim));
    const dist = pathDistances(qPoints, recip, kpathMeta?.segSizes);
    const xMax = dist[dist.length - 1] || 1;
    let yMin = 0, yMax = 1;
    for (const row of bands) for (const v of row) { if (!isFinite(v)) continue; if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    yMax *= 1.05; if (yMin < 0) yMin *= 1.05;
    if (Number.isFinite(eMin)) yMin = eMin;
    if (Number.isFinite(eMax)) yMax = eMax;
    const nModes = bands[0].length;

    // Effective (possibly zoomed) data window mapped onto the full plot frame.
    const xLo = domain ? domain.xMin : 0, xHi = domain ? domain.xMax : xMax;
    const yLo = domain ? domain.yMin : yMin, yHi = domain ? domain.yMax : yMax;
    const xSpan = (xHi - xLo) || 1, ySpan = (yHi - yLo) || 1;

    const xOf = (d) => XL + ((d - xLo) / xSpan) * PW;
    const yOf = (e) => YB - ((e - yLo) / ySpan) * PH;
    // Inverse (pixel → data) for converting the drag box into a data window.
    const invX = (px) => xLo + ((px - XL) / PW) * xSpan;
    const invY = (py) => yLo + ((YB - py) / PH) * ySpan;

    const segStarts = buildSegmentStarts(kpathMeta?.segSizes, bands.length);

    // Unfolded mode vs. standard unweighted band-line mode
    const hasWeights = Array.isArray(weights) && weights.length === bands.length;
    let spectralImageUrl = null;
    const branches = [];
    const weightedDots = [];

    if (hasWeights) {
      if (spectralStyle !== 'weighted') {
        const specGrid = computeSpectralFunctionGrid(bands, weights, {
          eMin: yLo,
          eMax: yHi,
          nEnergyBins: 240,
        });
        spectralImageUrl = renderSpectralHeatmapUrl(specGrid, dist, xLo, xHi, 360);
      }
      if (spectralStyle !== 'heatmap') {
        const minW = spectralStyle === 'both' ? 0.06 : 0.025;
        for (let k = 0; k < bands.length; k++) {
          const dx = dist[k];
          if (dx < xLo - xSpan * 0.02 || dx > xHi + xSpan * 0.02) continue;
          const X = xOf(dx);
          const wRow = weights[k];
          const eRow = bands[k];
          for (let m = 0; m < nModes; m++) {
            const w = wRow[m];
            if (!(w >= minW)) continue;
            const v = eRow[m];
            if (!isFinite(v) || v < yLo - ySpan * 0.02 || v > yHi + ySpan * 0.02) continue;
            weightedDots.push({
              x: X,
              y: yOf(v),
              r: spectralStyle === 'both' ? (1.1 + 1.8 * Math.min(1, w)) : (1.3 + 2.4 * Math.min(1, w)),
              opacity: spectralStyle === 'both'
                ? Math.min(0.88, 0.22 + 0.66 * w)
                : Math.min(0.95, 0.16 + 0.82 * Math.pow(w, 0.75)),
              neg: v < 0,
            });
          }
        }
      }
    } else {
      // Standard unweighted polyline per branch (broken cleanly at discontinuous segment jumps)
      for (let m = 0; m < nModes; m++) {
        let d = '', neg = '', pen = false, penNeg = false;
        for (let k = 0; k < bands.length; k++) {
          if (k > 0 && segStarts.has(k)) { pen = false; penNeg = false; }
          const v = bands[k][m];
          if (!isFinite(v)) { pen = false; penNeg = false; continue; }
          const X = xOf(dist[k]).toFixed(1), Y = yOf(v).toFixed(1);
          d += (pen ? 'L' : 'M') + X + ' ' + Y + ' ';
          pen = true;
          if (v < 0) { neg += (penNeg ? 'L' : 'M') + X + ' ' + Y + ' '; penNeg = true; }
          else penNeg = false;
        }
        branches.push({ d, neg, color: branchColor(m) });
      }
    }

    // High-symmetry ticks — only those whose distance falls in the x-window.
    // Ticks that land at the same x (a break/discontinuity, where two labels
    // would overlap) are merged into one, e.g. "X,L".
    const hsym = kpathMeta?.hsymIndex || {};
    const eps = xSpan * 1e-6;
    const xmap = new Map();
    for (const k of Object.keys(hsym).map(Number).sort((a, b) => a - b)) {
      const dx = dist[k];
      if (!(dx >= xLo - eps && dx <= xHi + eps)) continue;
      const x = xOf(dx), key = Math.round(x);
      const e = xmap.get(key) || { x, labels: [] };
      if (!e.labels.includes(hsym[k])) e.labels.push(hsym[k]);
      xmap.set(key, e);
    }
    const xticks = [...xmap.values()].map(e => ({ x: e.x, label: e.labels.join(',') }));

    // Adaptive y gridlines across the (zoomed) y-window, with matching decimals.
    const step = niceStep(yHi - yLo);
    const dec = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
    const ygrid = [];
    for (let e = Math.ceil(yLo / step) * step; e <= yHi + step * 1e-6; e += step) {
      ygrid.push({ y: yOf(e), label: e.toFixed(dec) });
    }

    const zeroIn = 0 >= yLo && 0 <= yHi;

    // Optional overlay candidate bands (supports both standard line overlays and unfolded weighted overlays!)
    const overlayBranches = [];
    for (const ov of (overlays || [])) {
      if (!ov?.bands?.length || !ov?.qPoints?.length || !ov?.baseStructure) continue;
      const ovRecip = ov.baseStructure.bandRecip || reciprocalLattice(conventionalLattice(
        ov.baseStructure.v1, ov.baseStructure.v2, ov.baseStructure.v3, ov.baseStructure.dim
      ));
      const ovSegStarts = buildSegmentStarts(ov.kpathMeta?.segSizes, ov.bands.length);
      const ovDist = pathDistances(ov.qPoints, ovRecip, ov.kpathMeta?.segSizes);
      const ovMax = ovDist[ovDist.length - 1] || 1;
      const xScale = ov.exactParentPath ? 1 : (xMax / ovMax);
      const ovModes = ov.bands[0]?.length || 0;

      if (ov.weights) {
        const dots = [];
        for (let k = 0; k < ov.bands.length; k++) {
          const X = xOf(ovDist[k] * xScale);
          for (let m = 0; m < ovModes; m++) {
            const w = ov.weights[k]?.[m] ?? 0;
            if (w < 0.08) continue;
            const v = ov.bands[k][m];
            if (!isFinite(v)) continue;
            dots.push({ x: X, y: yOf(v), r: 1.2 + 2.0 * Math.min(1, w), opacity: Math.min(0.85, 0.2 + 0.65 * w) });
          }
        }
        overlayBranches.push({ id: ov.id, label: ov.label, color: ov.color || '#2563eb', paths: [], dots });
      } else {
        const paths = [];
        for (let m = 0; m < ovModes; m++) {
          let d = '', pen = false;
          for (let k = 0; k < ov.bands.length; k++) {
            if (k > 0 && ovSegStarts.has(k)) pen = false;
            const v = ov.bands[k][m];
            if (!isFinite(v)) { pen = false; continue; }
            const X = xOf(ovDist[k] * xScale).toFixed(1), Y = yOf(v).toFixed(1);
            d += (pen ? 'L' : 'M') + X + ' ' + Y + ' ';
            pen = true;
          }
          if (d) paths.push(d);
        }
        overlayBranches.push({ id: ov.id, label: ov.label, color: ov.color || '#2563eb', paths, dots: [] });
      }
    }

    return {
      dist, xOf, yOf, invX, invY,
      hasWeights, spectralImageUrl, branches, weightedDots, overlayBranches,
      xticks, ygrid, nModes, zeroY: yOf(0), zeroIn,
    };
  }, [bands, weights, spectralStyle, qPoints, baseStructure, kpathMeta, eMin, eMax, domain, overlays]);

  if (!model) return null;

  // Map client coords → viewBox user space (== pixel space here, viewBox is the
  // full frame). Uses getBoundingClientRect, not getScreenCTM, so it stays
  // aligned with the cursor under the app's ancestor `zoom` (see svgCoords.js).
  const toUser = (evt) => clientToSvg(svgRef.current, evt.clientX, evt.clientY);
  const clampX = (x) => Math.max(XL, Math.min(XR, x));
  const clampY = (y) => Math.max(YT, Math.min(YB, y));

  // Nearest selectable (k, mode) to a point, skipping zero-weight shadow ghosts when unfolded.
  const pickNearest = (loc) => {
    let best = null, bestD = Infinity;
    for (let k = 0; k < bands.length; k++) {
      const x = model.xOf(model.dist[k]);
      for (let m = 0; m < model.nModes; m++) {
        if (model.hasWeights && (weights[k]?.[m] ?? 0) < 0.02) continue;
        const v = bands[k][m];
        if (!isFinite(v)) continue;
        const y = model.yOf(v);
        const dd = (x - loc.x) ** 2 + (y - loc.y) ** 2;
        if (dd < bestD) { bestD = dd; best = { k, m, x, y, w: model.hasWeights ? weights[k][m] : null }; }
      }
    }
    return best;
  };

  const onDown = (e) => { const p = toUser(e); if (p) dragRef.current = { x0: clampX(p.x), y0: clampY(p.y), moved: false }; };
  const onMove = (e) => {
    const p = toUser(e); if (!p) return;
    if (dragRef.current) {
      const d = dragRef.current;
      const x1 = clampX(p.x), y1 = clampY(p.y);
      if (Math.abs(x1 - d.x0) > 4 || Math.abs(y1 - d.y0) > 4) d.moved = true;
      if (d.moved) { setSel({ x0: d.x0, y0: d.y0, x1, y1 }); setHover(null); }
      return;
    }
    // Not dragging: preview the nearest selectable point under the cursor.
    setHover(pickNearest(p));
  };
  const onUp = (e) => {
    const d = dragRef.current; dragRef.current = null; setSel(null);
    const p = toUser(e); if (!p || !d) return;
    if (d.moved) {
      // Box-zoom: pixel box (clamped to frame) → data window. Require a usable size.
      const xa = clampX(d.x0), xb = clampX(p.x), ya = clampY(d.y0), yb = clampY(p.y);
      if (Math.abs(xb - xa) > 12 && Math.abs(yb - ya) > 12) {
        setDomain({
          xMin: model.invX(Math.min(xa, xb)), xMax: model.invX(Math.max(xa, xb)),
          yMin: model.invY(Math.max(ya, yb)), yMax: model.invY(Math.min(ya, yb)),   // pixel-y is inverted
        });
      }
    } else {
      const hit = pickNearest(p);
      if (hit && onPick) onPick(hit.k, hit.m);
    }
  };

  const selPt = selected && isFinite(bands[selected.k]?.[selected.m])
    ? { x: model.xOf(model.dist[selected.k]), y: model.yOf(bands[selected.k][selected.m]) }
    : null;
  const inFrame = (p) => p && p.x >= XL - 1 && p.x <= XR + 1 && p.y >= YT - 1 && p.y <= YB + 1;
  const hoverLabel = hover && !sel && isFinite(bands[hover.k]?.[hover.m])
    ? `band ${hover.m + 1} · ${bands[hover.k][hover.m].toFixed(2)} ${unit}${hover.w !== null && hover.w !== undefined ? ` · W=${hover.w.toFixed(2)}` : ''} · q=[${(qPoints[hover.k] || []).map(v => v.toFixed(2)).join(', ')}]`
    : null;

  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', width: '100%', height: 'auto', cursor: 'crosshair', userSelect: 'none' }}
      onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onDoubleClick={() => setDomain(null)}
      onMouseLeave={() => { dragRef.current = null; setSel(null); setHover(null); }}>

      <defs><clipPath id={clipId}><rect x={XL} y={YT} width={PW} height={PH} /></clipPath></defs>

      {/* y grid + labels */}
      {model.ygrid.map((g, i) => (
        <g key={`y${i}`}>
          <line x1={XL} x2={XR} y1={g.y} y2={g.y} stroke="var(--border)" strokeWidth="1" />
          <text x={56} y={g.y + 4} textAnchor="end" fontFamily="STIX Two Text, serif" fontSize="11" fill="var(--faint)">{g.label}</text>
        </g>
      ))}

      {/* zero line (only when 0 is in range) */}
      {model.zeroIn && <line x1={XL} x2={XR} y1={model.zeroY} y2={model.zeroY} stroke="var(--dim)" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.6" />}

      {/* high-symmetry dividers + labels */}
      {model.xticks.map((t, i) => (
        <g key={`x${i}`}>
          <line x1={t.x} x2={t.x} y1={YT} y2={YB} stroke="var(--border)" strokeWidth="1" />
          <text x={t.x} y={832} textAnchor="middle" fontFamily="Noto Sans, sans-serif" fontSize="16" fontWeight="700" fill="var(--ink)">{t.label}</text>
        </g>
      ))}

      {/* axis frame */}
      <line x1={XL} x2={XL} y1={YT} y2={YB} stroke="var(--dim)" strokeWidth="1.1" />
      <line x1={XL} x2={XR} y1={YB} y2={YB} stroke="var(--dim)" strokeWidth="1.1" />

      {/* bands & unfolded spectral density (clipped to plot frame) */}
      <g clipPath={`url(#${clipId})`}>
        {model.spectralImageUrl && (
          <image
            href={model.spectralImageUrl}
            x={XL}
            y={YT}
            width={PW}
            height={PH}
            preserveAspectRatio="none"
            style={{ imageRendering: 'auto', pointerEvents: 'none' }}
          />
        )}
        {model.overlayBranches.map((ov) => (
          <g key={ov.id} opacity="0.68">
            {ov.paths.map((d, m) => (
              <path key={m} d={d} fill="none" stroke={ov.color} strokeWidth="1.45" strokeDasharray="4 3" strokeLinejoin="round" />
            ))}
            {ov.dots?.map((pt, idx) => (
              <circle key={idx} cx={pt.x} cy={pt.y} r={pt.r} fill={ov.color} opacity={pt.opacity} />
            ))}
          </g>
        ))}
        {model.weightedDots.map((pt, idx) => (
          <circle
            key={idx}
            cx={pt.x}
            cy={pt.y}
            r={pt.r}
            fill={pt.neg ? '#dc2626' : '#8f2d1e'}
            opacity={pt.opacity}
          />
        ))}
        {model.branches.map((b, m) => (
          <g key={m}>
            <path d={b.d} fill="none" stroke={b.color} strokeWidth="1.6" strokeLinejoin="round" opacity="0.9" />
            {b.neg && <path d={b.neg} fill="none" stroke="#e0564b" strokeWidth="2" strokeLinejoin="round" strokeDasharray="3 2" />}
          </g>
        ))}
        {/* hover marker — translucent spot on the nearest selectable point */}
        {hover && !sel && <circle cx={hover.x} cy={hover.y} r="7" fill="rgba(47,109,240,0.22)" stroke="rgba(47,109,240,0.5)" strokeWidth="1" style={{ pointerEvents: 'none' }} />}

        {/* selection marker (blue, for contrast against the red branches) */}
        {inFrame(selPt) && <circle cx={selPt.x} cy={selPt.y} r="6" fill="#2f6df0" stroke="#fff" strokeWidth="2" />}
      </g>

      {/* drag-to-zoom selection box */}
      {sel && (
        <rect x={Math.min(sel.x0, sel.x1)} y={Math.min(sel.y0, sel.y1)} width={Math.abs(sel.x1 - sel.x0)} height={Math.abs(sel.y1 - sel.y0)}
          fill="rgba(47,109,240,0.10)" stroke="#2f6df0" strokeWidth="1" strokeDasharray="4 3" />
      )}

      {/* hover readout pill */}
      {hoverLabel && (() => {
        const bw = Math.max(168, hoverLabel.length * 6.5 + 20);
        const bx = Math.min(XR - bw - 4, Math.max(XL + 4, hover.x + 12 > XR - bw ? hover.x - bw - 12 : hover.x + 12));
        const by = Math.max(YT + 6, Math.min(YB - 28, hover.y - 30 < YT + 6 ? hover.y + 12 : hover.y - 30));
        return (
          <g pointerEvents="none">
            <rect x={bx} y={by} width={bw} height={24} rx="6" fill="var(--card)" stroke="var(--border-strong)" strokeWidth="1" opacity="0.97" />
            <text x={bx + 10} y={by + 16} fontFamily="IBM Plex Mono, monospace" fontSize="10.5" fill="var(--ink)">{hoverLabel}</text>
          </g>
        );
      })()}

      {/* axis title */}
      <text x={20} y={(YT + YB) / 2} fontFamily="STIX Two Text, serif" fontSize="15" fill="var(--dim)" textAnchor="middle" transform={`rotate(-90 20 ${(YT + YB) / 2})`}>Energy ({unit})</text>

      {/* zoom hint */}
      <text x={XR} y={YT + 4} textAnchor="end" fontFamily="IBM Plex Mono, monospace" fontSize="11" fill="var(--faint)">{domain ? 'zoomed · dbl-click or Reset' : 'drag a box to zoom'}</text>
    </svg>
  );
}
