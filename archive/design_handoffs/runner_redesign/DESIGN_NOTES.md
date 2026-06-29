# Design notes — UI redesign spec

This is the behavioral contract for the redesign. **Reskin `pages/` and
`components/`; do not change the data/compute layer.** When the new UI feeds the
same props/shapes documented here, everything keeps working.

## The boundary (do not edit the behavior of)

Treat these as a stable, tested API (`npm run validate` guards them):

| Layer | Modules | Role |
|---|---|---|
| compute | `compute/pipeline.js`, `engine.js`, `ins.js`, `Sk_kernel.wgsl` | WebGPU S(k) → eigh → bands/DOS; INS data build |
| io | `io/readers.js`, `worker.js`, `writers.js`, `viewermodel.js`, `sqgr.js`, `sqeworker.js`, `vaspexport.js` | parse, save/load, fit-quality |
| math | `math/diagonalize.js`, `band_connection.js`, `reciprocal.js`, `bravais.js`, `brillouin.js`, `highsym.js`, `dos.js` | linear algebra, k-path, BZ, DOS |
| constants | `constants.js` | physical constants + element tables |

Free to rewrite: `App.jsx`, everything in `pages/` and `components/`, CSS.

## App structure

`App.jsx` is a 2-page shell (state router, no react-router):

- holds `page` (`'runner' | 'viewer'`), the shared `model` (viewer model), and a
  single `PhononPipeline` instance (`pipelineRef`).
- `onResults(results, kpathMeta)` → builds the viewer model (`fromResults`),
  switches to viewer, auto-saves `band.yaml`.
- `loadModel(m)` / `setModel` → set the viewer model (used by "load saved result").

Pages:
- **RunnerPage** — data → structure → k-path → run; plus DOS + fit-quality.
- **ViewerPage** — bands + 3D modes (side by side) and S(Q,E); exports.

## Page responsibilities (preserve these capabilities)

### RunnerPage `({ pipeline, onResults, onLoadResult })`
1. **Data folder** — `showDirectoryPicker` → `listConfigs`; for `frac` family also `findStructureFile`. Structure-file override (`listRmc6f`). "Load saved result" (`fromBandText` → `onLoadResult`).
2. **Displacement reference** — radio: ensemble average | equilibrium `.rmc6f` file.
3. **Crystal structure preview** — static `CrystalViewer` from the basis sites (no run needed).
4. **Brillouin zone k-path** — `BrillouinZoneViewer`; editable per-segment npoints + default density.
5. **Run** — T, degenerate tol, points/segment; Run + **Cancel** (`pipeline.cancel()`); progress. Calls `pipeline.runCalculation(...) → onResults`.
6. **Phonon DOS** — `PhononDOS` panel (q-grid).
7. **Fit quality** — `FitQuality` panel (uses the selected folder).

### ViewerPage `({ model, onLoadModel })`
- Tabs: **Bands + Mode** (default) and **S(Q,E)**.
- Bands+Mode: `BandStructurePlot` (left) + `CrystalViewer` (right) side by side; click a band point → updates `selK/selM` → 3D viewer shows that mode. `ModeInspector` overlay. All 3D controls (supercell, amplitude, speed, play, WebM/GIF, appearance, per-element color/radius, bonds, camera, vectors, tables).
- S(Q,E): `InsPanel`.
- Toolbar: mode k/band number inputs, **load band.yaml/.json** (`fromBandText`), export **band.yaml / band.json / VASP**.

## Component prop contracts

```
BandStructurePlot({ bands, qPoints, baseStructure, kpathMeta,
                    selected:{k,m}, onPick(k,m), eMin?, eMax? })
CrystalViewer({ baseStructure, eigenvector|null, qPoint,
                isPlaying, amplitude, speed, supercell:[nx,ny,nz],
                showVectors, showCell, atomScale, cameraAxis,
                elementColors:{el:hex}, elementRadii:{el:Å}, displayStyle,
                showBonds, bondScale, bondRules:{'A-B':Å}, shading,
                recording, gifSignal, vectorScale })
BrillouinZoneViewer({ bzModel, system, onPathChange(segments, pointsConv) })
ModeInspector({ results:model, selectedK, selectedMode })
InsPanel({ results:model, temperature })
DatasetInspector({ directoryName, filesList, configFamily, baseStructure })
LineChart({ series:[{name,points:[[x,y]],color,dashed,width}], xLabel, yLabel, onPick?, height?, markerSeries? })
FitQuality({ dirHandle })
PhononDOS({ pipeline, files, family, baseStructure, temperature, referenceHandle })
```

## Core data shapes

**Viewer model** (`io/viewermodel.js` — from a run or a loaded band file):
```
{ bands: number[q][mode] (meV),
  eigvecs: ({real:Float64Array, imag:Float64Array} length 3·nSites)[q][mode] | null,
  qPoints: [q1,q2,q3][] (fractional, conventional),
  kpathMeta: { segSizes:number[], hsymIndex:{flatQIndex: label} },
  temperature, source:'runner'|'file',
  baseStructure: { atomDic:{el:[refNums]}, dim:[1,1,1], v1,v2,v3,
                   uniqueRN:number[] (eigvec row order), atomType:number[],
                   hsym_xyz:Float64Array, cellIdx:Float64Array } }
```

**BZ model** (`math/highsym.js buildBZModel(bravais)`):
```
{ code, variant,
  points: { LABEL: { frac, fracConv, cart:[x,y,z], display } },
  path: [[fromLabel,toLabel], ...],
  bz: { vertices:[xyz], faces:[[xyz]], edges:[[xyz,xyz]] } }
```

**Pipeline** (`new PhononPipeline(onProgress)`): `initWorkers()`,
`runCalculation(files, family, baseStructure, qFrac, T, batch, {referenceHandle, degenerateTol})`,
`computeDOSGrid(files, family, baseStructure, gridN, T, batch, {referenceHandle})`,
`cancel()`. `onProgress(percent, message)`.

## Run / verify
```
cd web && npm run dev        # http://localhost:5173 (Chromium only)
npm run validate             # science + UI-mapping tests — run after UI changes
npm run build                # static → dist/
```

## Design guidance
- Edit only `web/src/**`; keep `compute/io/math` behavior unchanged.
- 3D viewers (`CrystalViewer`, `BrillouinZoneViewer`) are bespoke three.js canvases — a design tool won't regenerate them; reskin the controls *around* them. Memoize array/object props (`supercell`, `elementColors`, `bzModel`) so the canvas doesn't rebuild/reset the camera on every render.
- Browser constraints: WebGPU + `showDirectoryPicker` ⇒ Chromium only; secure context.
- GitHub Pages: `VITE_BASE=/<repo>/ npm run build`.
