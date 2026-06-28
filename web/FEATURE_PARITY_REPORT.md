# Backend / Scientific Feature‑Parity Report

**Date:** 2026‑06‑28
**Branch:** `backend-parity-audit`
**Scope:** Audit of the migrated browser/WebGPU app (`web/`) against the legacy
Python + viewer codebase (`src/`, `src_gpu/`, `rmcph_gui/`, `viz/`).

This is an **audit + critical-fix** report. It does *not* claim full parity. The
legacy canonical scientific path is `src_gpu/runner.py` (see
`PHYSICS_ALGORITHM_AUDIT.md`). Findings below were verified against the real
`GTS_5K` dataset in `data/`.

---

## TL;DR

The migration is **partial and the core physics is currently incorrect.** Three
independent, individually fatal numerical regressions mean the band structure
produced by `web/` does not match the legacy code:

1. **S(k) is grouped by chemical element instead of crystallographic basis
   site (RMC reference number).** For the `GTS_5K` test data this produces **9
   bands instead of the correct 156** (3 × 52 basis sites). Verified directly
   from the `.rmc6f` file.
2. **The 2π Bloch‑phase factor is missing.** k‑points are passed to the kernel
   as raw fractional coordinates; the legacy code applies `kvec = 2π·q_frac`
   (`src_gpu/kpath.py`, `src_gpu/constants.py`). Without it `S(k=G) ≠ S(Γ)` and
   all phases are wrong.
3. **`ENERGY_CONV` is wrong by a factor of ~350,623.** The meV conversion
   constant in `web/src/constants.js` does not match
   `src_gpu/constants.py` (0.6002 vs 210438). All energies are off scale.

Several whole subsystems are **absent**: simulated INS / S(Q,E) + DOS,
seekpath‑based high‑symmetry path detection, the full `band.yaml` writer, and
the VASP export. The `Frac*.txt` data path is **stubbed/broken**.

Per the project's stated failure rule, full parity **cannot** be achieved in one
pass without these being addressed; the high‑confidence numerical fixes are
implemented on this branch and the remaining gaps are itemized with
recommendations.

---

## Feature‑Parity Matrix

Status legend: ✅ Fully migrated · 🟡 Partially migrated · ❌ Missing ·
🔴 Broken · 🧪 Needs numerical validation · 🎨 UI‑only gap (backend OK)

### 1. Data loading & data‑selector logic

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| `.rmc6f` ensemble detection (numbered, excl. AVERAGE/`_0`/base) | `Readers.list_configs` | `io/readers.js listConfigs` | ✅ | Logic matches, incl. `configs/` Frac fallback. |
| `.rmc6f` per‑frame parse (within‑cell frame, mod 1.0) | `read_rmc6f_atom_ph` | `io/worker.js parseRMC6F` | 🟡 | Math matches but uses **Float32** for coords/displacement accumulation (legacy uses float64 for the mean). |
| `Frac*.txt` parse | `read_frac_atom_ph` | `io/worker.js parseFrac` | 🟡 | Parser exists, but base structure is **faked** (see below). |
| Base structure (cell vectors, dim, atom_dic) | `read_cell_vec` + `get_atom_idx` | `io/readers.js readBaseStructure` | 🔴 | rmc6f path OK; **Frac path sets `{atomDic:{Fake:[1]}, dim:[1,1,1]}`** in `App.jsx` → Frac runs are physically meaningless. |
| Displacement reference: ensemble average | `avg_frac_atom_ph` | `pipeline.js` (inline mean) | ✅ | Mean over frames; accumulates in Float64. |
| Displacement reference: from file (`.rmc6f`/CIF) | `_hsym_from_file` (rmc6f) | — | ❌ | No file‑reference option in UI/pipeline. |
| Ensemble‑average frame invariants (cell idx / RN order checks) | `avg_frac_atom_ph` | — | ❌ | No per‑frame consistency validation. |

### 2. Interactive k‑path selection

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| High‑symmetry point detection (any Bravais lattice) | `reciprocal.py` (**seekpath**) | `BrillouinZoneViewer.jsx` | 🔴 | Hardcoded `cubic`/`fcc` templates; **always forced to `cubic`** regardless of lattice. No spglib/seekpath. |
| Primitive↔conventional k transform | `reciprocal.py` (`k_conv = k_prim·T`) | — | ❌ | Not implemented; only one (cubic) frame assumed. |
| BZ facet geometry | `lattice.get_brillouin_zone()` | template / box fallback | 🟡 | Cube wireframe only. |
| Build/edit path, segment it | `kpath.segments_from_path/build_kpath` | `App.jsx` inline interp | 🟡 | Interpolates, but **omits 2π** and does not duplicate segment junction points (phonopy convention). |
| High‑sym labels on band axis | `Writers` `hsym_qi` | `BandChart.jsx` | 🟡 | Labels by segment index; works for uniform segments. |

### 3. Eigenvalue / eigenvector computation

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| Mass‑weighted U(k) accumulation (segment sum) | `process_batch_kernel` (JAX) | `Sk_kernel.wgsl` + `engine.js` | 🔴 | Kernel math is faithful **but grouped by element, not reference number** (`pipeline.js`). Wrong matrix dimension → wrong bands. |
| `S(k) = A^TA+B^TB + i(B^TA−A^TB)` recombine | `Calculators.py` | `engine.js` | ✅ | Real/imag split identical to legacy Metal path. |
| Hermitian diagonalization | `np.linalg.eigh` | `diagonalize.js eigh` (2N real‑sym trick) | ✅ | Correct method; symmetrized. |
| Eigenvalue → meV | `eigenvalues_to_meV` | `diagonalize.js eigenvaluesToMev` | 🔴 | Logic (threshold 1e‑4, signed) matches, but **`ENERGY_CONV` constant is wrong by ~3.5e5×**. |
| 2π phase convention | `kpath.build_kpath`/`constants.TWO_PI_PHASE` | — | 🔴 | **Missing entirely.** |
| Per‑dataset mass table | `get_mass_array` (global cache bug noted in audit) | `pipeline.js` (per‑run) | ✅ | New code rebuilds per run — actually avoids the legacy global‑cache bug. |

### 4. 3D phonon mode pipeline

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| Eigenvector → per‑atom displacement | `viz` + `_archive_get_ph_weights` | `CrystalViewer.jsx` | 🔴 | Maps eigvec by **element index** (tied to bug #1); no per‑cell `exp(i k·n)` phase, so finite‑k modes are not spatially modulated. |
| Equilibrium atom positions in Cartesian Å | viewer | `CrystalViewer.jsx` | 🟡 | Uses within‑cell fractional×dim directly as Cartesian (no `v_super` transform). OK‑ish for cubic, wrong in general. |
| Animated mode (phase sweep) | `viz` | `CrystalViewer.jsx` | ✅ | `Re(eig·e^{iωt})` animation present and clickable‑driven. |

### 5. Phonon band structure + interaction

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| Band calculation along path | `runner.run_bands_segments` | `pipeline.runCalculation` | 🔴 | Blocked by bugs #1–#3. |
| Band connection (Hungarian on eigvec overlap) | `Writers.connect_bands` | `band_connection.js connectBands` | 🟡 | Hungarian + multi‑pass ported; **SVD degenerate‑subspace rotation NOT ported**. |
| Band plot + axis labels | `Visualization` / viewer | `BandChart.jsx` | 🟡 | Functional; x‑axis is point index, not cumulative reciprocal distance. |
| Click band point → mode data | viewer | `App.jsx` + `BandChart` + `CrystalViewer` | ✅ | Wiring works (kIndex/modeIndex → 3D viewer). |
| `band.yaml` export (phonopy) | `gen_phonopy_band_yaml(_segments)` | `io/writers.js` | 🔴 | Stub: no atom points, **no eigenvectors**, diagonal‑only reciprocal lattice, no distances; **not wired to any UI button**. |
| Units (meV vs THz `freq_factor`) | configurable | meV only | 🟡 | THz factor not exposed. |

### 6. Simulated INS / S(Q,E)

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| Powder S(Q,E) from band.yaml + eigenvectors | `viz/sqeworker.js` | — | ❌ | **Entirely absent.** |
| Phonon DOS | `viz/sqeworker.js` | — | ❌ | Absent. |
| Coherent scattering lengths / cross sections | `viz/sqeworker.js` (`B_COH`), `constants.py` (`NEUTRON_SCATT_SIGMA`) | — | ❌ | Not ported; `NEUTRON_SCATT_SIGMA` missing from `web/src/constants.js`. |
| Bose factor, Debye‑Waller, Q/E grids, smearing | `viz/sqeworker.js` | — | ❌ | Absent. |
| Export S(Q,E) | viewer | — | ❌ | Absent. |

### 7. Other legacy backend features

| Feature | Legacy | New (`web/`) | Status | Notes |
|---|---|---|---|---|
| VASP phonon export (POSCAR/INCAR/KPOINTS + displaced modes) | `Writers.gen_vasp_phonon` | — | ❌ | Absent. |
| Eigenvector MCIF export | `Writers.gen_ev_mcif` | — | ❌ | Intentionally removed earlier (commit `4c42b57`). |
| Per‑element phonon weights (band character) | `_archive_get_ph_weights` | — | ❌ | Absent. |
| Partial (single‑type) S(k) | `Partial_Sk_avg` | — | ❌ | Absent (legacy version also flagged buggy in audit). |
| q‑grid generator | `gen_grid` | — | ❌ | Absent. |
| Fit‑quality / RMCProfile CSV preview (F(Q), G(r), Rw) | `rmcph_gui` `sqgr.js` | — | ❌ | Absent (out of phonon core, but part of old GUI). |
| Cache of `S(k)` sums | `Sk_sum_kvec_*.csv` | — | ❌ | No caching (acceptable for browser; recompute). |
| `NEUTRON_SCATT_SIGMA` / `B_COH` tables | `constants.py` / `sqeworker.js` | — | ❌ | Needed for INS port. |

---

## Critical numerical findings (with evidence)

### F1 — Band dimensionality (element vs basis site) — **fatal**
`pipeline.js` uses `numTypes = Object.keys(atomDic).length` (elements) and
`typeIndices[i] = atomTypeIdxMap.get(rn)` (atom→element). Legacy `Sk_avg` uses
`np.unique(per_atom_reference_number)` → groups by basis site.
Evidence (`GTS_5K.rmc6f`): Ga has reference numbers {4,8,12,16} (4 sites), Ta 16
sites, Se 32 sites ⇒ **52 basis sites ⇒ 156 bands**. New code yields **9**.

### F2 — Missing 2π Bloch phase — **fatal**
Legacy: `kvec = 2π·q_frac` (validated by `S(G)=S(Γ)` in
`validate_kpath_2pi.py`). New: fractional coords flow straight into the WGSL
`dot(cellIdx, kvec)`. No 2π anywhere in `web/`.

### F3 — `ENERGY_CONV` wrong by ~3.5×10⁵ — **fatal**
Legacy `ENERGY_CONV = ħ·√(kB/(amu·Å²)) = 0.600181…`.
New `= 13.6057e3 / (0.529177·√(2·13.6057/1822.888)) = 210438…`.
Ratio ≈ 350623. (The source even carries a "let's just use a known value"
comment — it is a guess.)

---

## Architectural / hosting notes

- **WebGPU**: the S(k) kernel runs on WebGPU and is structurally faithful — good.
  It accumulates in Float32 on‑GPU then recombines/eigensolves in Float64 on CPU
  (`ml-matrix`), matching the legacy Metal‑float32 strategy. Suitable.
- **GitHub Pages**: `vite build` produces static assets — compatible. Two
  caveats: (a) the File System Access API (`showDirectoryPicker`) is
  Chromium‑only and requires a secure context (Pages is HTTPS — OK, but
  Firefox/Safari users cannot select folders); (b) `vite.config.js` `base` must
  be set to the repo subpath for project Pages.
- **`pyodide` is a dependency but unused.** It was presumably intended to run
  `seekpath` in‑browser for k‑paths but is not wired up. Either wire it
  (heavy: ~10 MB download) or port a JS spglib/seekpath equivalent.

---

## What is implemented on this branch (full-parity pass)

All items below are committed and validated by `npm run validate` (passes):

1. **F1 — basis-site grouping.** `pipeline.js` now groups S(k) by sorted unique
   reference number (`np.unique` semantics) → `3 × N_basis_sites` bands (156 for
   GTS_5K). `masses`/`counts`/`typeIndices` rebuilt accordingly.
2. **F2 — 2π Bloch phase.** Applied as `kvec = 2π·q_frac` inside the pipeline;
   `q_frac` is retained for band.yaml/INS. Validated by `S(G)==S(Γ)`.
3. **F3 — `ENERGY_CONV`.** `constants.js` now derives the exact value
   `0.600181852836787`; dead guess removed from `diagonalize.js`.
4. **Frac base structure.** `findStructureFile` locates a companion `.rmc6f`
   for the lattice + RN→element map; the `{Fake:[1]}` stub is gone.
5. **Lattice-aware k-path** (`math/reciprocal.js`): proper 3×3 inverse reciprocal
   lattice, crystal-system detection from the metric, standard conventional-cell
   high-symmetry points + default paths (cubic/tetragonal/orthorhombic/hexagonal
   /rhombohedral/monoclinic/triclinic). BZ viewer is now driven by these.
6. **Band connection SVD** degenerate-subspace rotation ported into
   `band_connection.js` (complex SVD via the existing eigh), with a safe fallback.
7. **Full `band.yaml` writer** (`io/writers.js`): points, per-q distances,
   eigenvectors, correct reciprocal lattice; wired to an **Export band.yaml** button.
8. **INS S(Q,E) + DOS** ported from `viz/sqeworker.js` into `io/sqeworker.js`
   (operating on in-memory meV `ydata` built by `compute/ins.js`), with the
   `InsPanel` UI (heatmap + DOS + CSV export). Added `NEUTRON_SCATT_SIGMA` and
   `B_COH` tables to `constants.js`.
9. **3D viewer** now maps eigenvectors by reference number, applies the per-cell
   `exp(i k·n)` phase, and places atoms at true Cartesian supercell positions.
10. **Validation + build + Pages.** `test/reference_sk.py` (independent reference)
    + `test/validate.mjs`; `vite.config.js` `base` via `VITE_BASE` for project Pages.

## Still NOT migrated (lower priority / out of phonon core)

- **VASP export** (`Writers.gen_vasp_phonon`), **eigenvector MCIF**
  (intentionally removed earlier), **per-element band-character weights**,
  **partial single-type S(k)**, **file-reference displacement mode**, and the
  **RMCProfile fit-quality CSV preview** from the old GUI.
- True **spglib/seekpath** symmetry (centering F/I/C detection, exact BZ
  polyhedron). The JS module detects the crystal *system* and uses simple-setting
  (P) points — correct for the GTS cubic test case and a large improvement over
  "always cubic," but not a full replacement for seekpath on centered lattices.

## Remaining scientific / architectural risks

- **No browser-run numerical comparison against the JAX path.** Validation is
  against an independent pure-Python re-implementation of the same algorithm and
  internal invariants (2π periodicity, eigh reconstruction), not against a live
  `src_gpu` run on a full dataset (JAX/Metal not runnable in CI here). A
  full-dataset A/B against `src_gpu/runner.py` output is the recommended next
  validation step.
- **Float32 on-GPU accumulation** (matching the legacy Metal strategy) recombined
  in Float64 — fine, but worth a tolerance check on large ensembles.

## Frontend stage (Stage 2) — done on this branch

The minimal validation UI from Stage 1 was upgraded to a usable scientific UI:

- **Band-structure plot** (`BandStructurePlot.jsx`): custom SVG with a physical
  cumulative reciprocal-distance x-axis, high-symmetry tick labels + segment
  dividers, energy y-axis, 0 line, hover readout, and click-to-select with a
  marker. Replaces the recharts point-index chart.
- **INS panel** (`InsPanel.jsx`): S(|Q|,E) heatmap with real Q/E axes, a
  matching 4-stop colormap + colorbar, and an axed DOS plot.
- **Mode inspector** (`ModeInspector.jsx`): selected mode energy, k (frac),
  index, and **per-element band character** (eigenvector weights — also fills the
  legacy `_archive_get_ph_weights` gap).
- Results summary chips in the header (bands / k-points / energy range);
  error/empty states; lattice-aware BZ path picker; `DatasetInspector` label
  corrected to "basis sites"; Frac selection reads the companion `.rmc6f`.
- Removed now-unused `recharts` and `pyodide` dependencies (bundle 1.32 MB → 0.97 MB).

## Remaining UI limitations (future polish)

- BZ geometry is a schematic reference box, not the exact polyhedron.
- No zoom/pan or per-mode highlight on the band plot; no responsive/mobile layout pass.
- INS heatmap aspect is fixed; no interactive cursor readout on S(Q,E).
