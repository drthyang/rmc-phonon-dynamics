# Handoff: RMC Phonon Dynamics — Runner page redesign

## Overview
A ground-up visual + UX redesign of the **Runner** page of the RMC Phonon Dynamics
web app (WebGPU phonon-band calculator driven by RMC configurations). The Runner
takes the user from raw data → structure/fit assessment → reciprocal-space k-path →
launching the calculation. This redesign reorganizes that flow into three clear,
numbered groups and modernizes the entire visual language.

The original behavioral contract for the rebuild lives in `DESIGN_NOTES.md` (bundled
here). **That contract still holds**: reskin `App.jsx`, `pages/`, and `components/`
only; do **not** change the `compute/`, `io/`, `math/`, or `constants` layers. Feed
the new UI the same props/data shapes documented there and everything keeps working.

## About the Design Files
The files in this bundle (`Runner.dc.html`, `ConsoleRunner.dc.html`) are **design
references authored in HTML/JS** — a streaming "Design Component" prototype showing
the intended look and behavior. They are **not** production code to paste in.

Your task: **recreate this design in the app's existing React + Vite environment**,
reusing its established patterns (the real `CrystalViewer`, `BrillouinZoneViewer`,
`BandStructurePlot`, `LineChart`, `PhononDOS`, `FitQuality`, `DatasetInspector`
components, three.js canvases, and the live `PhononPipeline`). The prototype fakes
data/interactions purely to demonstrate UI; wire the real ones per `DESIGN_NOTES.md`.

> The two bespoke three.js canvases (`CrystalViewer`, `BrillouinZoneViewer`) are
> represented in the prototype by static SVG stand-ins. **Keep the real three.js
> canvases** — only restyle the chrome/controls around them.

## Fidelity
**High-fidelity.** Final colors, typography, spacing, and interactions are specified
below and are present in the HTML. Recreate the UI pixel-accurately using the
codebase's component library, then bind real data.

---

## Layout overview

App shell: sticky top nav (60px) + a centered content column, `max-width: 1320px`,
`padding: 22px`, vertical `gap: 28px` between the three groups.

Each group = a header row (numbered badge + title + one-line description + hairline
rule) followed by its cards. Cards: white, `1px` border `#e3e7ef`, `border-radius:10px`,
`padding:18px`. Card rows use flex/grid with `gap:14px`.

### Top nav (60px tall, white, bottom border `#e3e7ef`)
- **Left:** logo tile (30px, `border-radius:7px`, fill `#2f6df0`) containing a white
  **atom glyph** — three overlapping ellipses (rx10/ry4.1, rotated 0/60/120°) + a
  center dot, **slowly rotating** (`@keyframes spin` 9s linear infinite) to imply
  atomic motion. Then wordmark "RMC Phonon Dynamics" (Space Grotesk 700, 17px,
  letter-spacing −.02em). Then a segmented **Runner / Viewer** toggle (pill group,
  bg `#eef1f7`, active segment fill `#2f6df0` white text).
- **Right:** "WebGPU ready" status chip — `#e9f0ff` bg, 7px accent dot pulsing
  (`@keyframes blip` 2s), Space Mono 600 12px `#2257cf`.

### GROUP 1 — "Data & assessment"
Description: "Inspect the configuration & fit before calculating".
- **Row A** (`flex`, gap 14): **Data folder** card (fixed `width:340px`) + **Crystal
  structure** card (`flex:1`). Cards stretch to equal height.
- **Row B** (full width): **Fit quality** card.

**Data folder card**
- Title "Data folder".
- Primary button **"Select directory"** (full width, fill `#2f6df0`, white, Space
  Grotesk 600 14px, `border-radius:8px`, padding 11px). → `window.showDirectoryPicker`.
- Secondary button **"Load saved result (.yaml / .json)"** (bg `#f4f6f8`, border,
  `#5a6373`, 12px). → file input, `fromBandText`.
- **Dataset readout** panel (bg `#f7f9fc`, `border-radius:8px`, padding 12px, Space
  Mono 12px, line-height 2.05) — label left (`#6b7488`) / value right (`#101826`):
  `dir → srtio3_rmc6f/`, `configs → 64 · rmc6f` (value `#2257cf`),
  `formula → Sr Ti O₃ · 5 sites`. Hairline divider, then crystallographic block:
  `a, b, c (Å) → 3.905, 3.905, 3.905`, `α, β, γ → 90°, 90°, 90°`,
  `supercell → 10 × 10 × 10`. (Populate from `DatasetInspector` / `baseStructure`.)

**Crystal structure card** (`overflow:hidden`, relative)
- Absolute title top-left "Crystal structure"; absolute top-right hint "1×1×1 cell"
  (Space Mono 10px `#aab1bc`).
- Canvas area `flex:1; min-height:236px`, bg `#f7f9fa` → **mount the real
  `CrystalViewer`** here (static preview of basis sites, no animation — this is the
  input structure, not a mode). Prototype shows an SVG perovskite cell placeholder.
- Footer legend (top border): CPK swatches `● Sr` `#13a07f`, `● Ti` `#5b677a`,
  `● O` `#e06a3b` (Space Mono 11px).

**Fit quality card**
- Title "Fit quality" + muted sub "how well the model reproduces the measured data".
- **R-value overview** (the assessment driver):
  - Header row: left label "R-value per configuration (click a bar to inspect)"
    (Space Mono 11px, `white-space:nowrap`); right a **config picker**: text
    "config", a `<input type=number>` (width 54px, Space Mono 13px, centered,
    `#2257cf`), "/ N", and "Rw X.X%" (`#2257cf` 700).
  - **Bar chart**: one bar per configuration (here 28; real = number of configs,
    e.g. 64). `display:flex; align-items:flex-end; gap:3px; height:92px`. Each bar
    `flex:1`, height ∝ that config's R-value, `border-radius:2px 2px 0 0`. Selected
    bar fill `#2f6df0`; others `#d5dae4`. **Click a bar OR type in the box** →
    selects that configuration and updates the two fit charts below + the caption.
- **Two fit charts** side by side (`flex:1` each, bg `#f7f9fa`, `border-radius:9px`,
  padding 13/15): **Structure factor S(Q)** and **Pair distribution G(r)**. These are
  proper scientific plots — see "Charts" below. (Back with the real `LineChart`
  component fed S(Q)/G(r) data + fit for the selected config; `FitQuality` provides it.)

### GROUP 2 — "Reciprocal space & k-path"
Description: "Click high-symmetry points to build the path". Row: **Brillouin zone**
card (`flex:1`) + **k-path segments** card (`flex:1`), stretched to equal height.

**Brillouin zone card** (flex column so footer pins to bottom)
- Absolute title "Brillouin zone"; top-right hint "cP · cubic" (the Bravais code —
  from `analyzeBravais`).
- Canvas `flex:1; min-height:268px`, bg `#f7f9fa` → **mount the real
  `BrillouinZoneViewer`**. Prototype shows an SVG cube + Γ→X→M→Γ→R path stand-in;
  high-symmetry labels use **Noto Sans** (clean Greek Γ). Path lines fade
  (`opacity` 0/1) when the path is cleared.
- Footer (top border): "path Γ→X→M→Γ→R" (Noto Sans 600) on the left; on the right
  two buttons — **"Default path"** (soft `#e9f0ff` / `#2257cf`) and **"Clear"**
  (outline). Default → restore standard seekpath; Clear → empty the path.

**k-path segments card** (flex column)
- Title "k-path segments".
- One row per segment (bg `#f7f9fa`, `border-radius:8px`, padding 8/12): segment label
  e.g. "Γ → X" (Noto Sans 700 14px) + "npoints" + a **stepper** = value (Space Mono
  13px `#2257cf`, width 42px centered) with stacked ▲/▼ buttons (24×16px each, hover
  → soft bg + accent). ▲/▼ adjust that segment's npoints (min 2).
- When the path is cleared: show a centered empty prompt instead of rows ("No path
  defined — press **Default path** or click points on the zone.").
- **Bottom bar** (top border, `margin-top:auto`): an **overall density** stepper —
  "density [N] ▲▼ pts/Å⁻¹" — on the left, and "M seg · K k-pts" summary on the right.
  Changing density **recomputes every segment's npoints** = `round(density × Lᵢ)`,
  where `Lᵢ` is the segment's length in reciprocal space (Å⁻¹). Per-row ▲▼ still
  override individual segments.

### GROUP 3 — "Run calculation"
Row: **Displacement reference** card (fixed `width:340px`) + **Run** card (`flex:1`),
stretched to equal height.

**Displacement reference card** (flex column; note pinned to bottom)
- Title "Displacement reference".
- Two radio options (custom radios: 16px circle, 2px border → accent when selected,
  7px accent inner dot; selected row bg `#e9f0ff`): **"Ensemble average"** (+ muted
  "default") and **"Equilibrium .rmc6f file"**.
- When "Equilibrium…" is selected: reveal a **file `<select>`** (bg `#f7f9fc`, Space
  Mono 12px) listing the available `.rmc6f` files (from `listRmc6f`), and below it a
  line "selected <filename>" (`#2257cf`). Maps to `refMode` + `referenceHandle`.
- Bottom note (muted): "Sets the equilibrium positions r₀ for the displacement field
  u = r − r₀ that builds the dynamical matrix."

**Run card**
- Title "Run" + right "Bravais cP cubic".
- **PARAMETERS** eyebrow (Space Mono 10px, letter-spacing .16em, `#9aa1b2`), then a
  2-column grid of fixed-width (150px) fields: **T (K)** = 5, **degen tol** = 5e-3.
  (Field = label + value box bg `#f7f9fc`, border, `border-radius:7px`, Space Mono 13px.)
- **OPTIONS** eyebrow, then a full-width toggle row (bg `#f7f9fc`): an 18px checkbox
  (fill `#2f6df0` + white ✓ when on), label **"Run Phonon DOS"**, right side
  "q-grid" + a `20³` chip. This is the `PhononDOS` option, computed as part of the run.
- **Launch block** (top border): left "M segments · K k-points" (nowrap); right the
  primary **"▶ Run phonon bands"** button (fill `#2f6df0`). While running, that button
  is replaced by **"■ Cancel"** (fill `#e0564b`).
- **Progress row**: status message (Space Mono 11px, ellipsized) + percent (`#2257cf`
  700). Then a **progress bar**: track bg `#eef1f7` 8px, fill `#2f6df0`,
  `transition: width .4s`.
- **Run log**: a dark console panel (bg `#0f1623`, border `#1c2740`,
  `border-radius:9px`, height 120px, `overflow-y:auto`, Space Mono 11px, text
  `#9fb3d1`, prompt glyph `›` `#4a6b8a`). Lines stream as the run progresses;
  auto-scrolls to bottom. Empty state: "› console output will appear here…".
  Bind to `pipeline.onProgress(percent, message)`.

---

## Charts (S(Q) & G(r)) — scientific style
Each chart is an SVG with proper axes (not a bare sparkline):
- Plot rect inside margins (left 44 for y-axis, bottom ~34 for x-axis); axis lines
  `#7b8494` 1.1px; light gridlines `#e4e7ec` 0.7px at each tick.
- **Tick marks + numeric labels** and **axis titles with units**, all set in
  **STIX Two Text** (the scientific/journal serif), tick labels 9px, axis titles 10px,
  color `#7b8494`. S(Q): x = "Q (Å⁻¹)" (0–12), y = "S(Q) (arb.)" (0–1, rotated).
  G(r): x = "r (Å)" (0–10), y = "G(r) (Å⁻²)" (−1…1, rotated) with a dashed zero line.
- Two series: **data** (dashed, `#b6bcc6`, `stroke-dasharray:4 2.5`) and **fit**
  (solid, `#2f6df0`, 1.8px). In-plot legend top-right ("-- data  — fit").
- Use `vector-effect: non-scaling-stroke` if you scale a path group, so strokes stay
  crisp. (In the real app, render these with the existing `LineChart` component and
  add the axis/tick/label treatment to match.)

---

## Interactions & Behavior
- **Select directory / Load saved result** → existing handlers (`showDirectoryPicker`,
  `fromBandText`).
- **R-value overview**: click a bar OR type a number in the config box → selects that
  configuration; the S(Q) & G(r) charts and the "Rw X.X%" caption update to it.
- **k-path steppers**: ▲/▼ change a segment's npoints (min 2); totals recompute.
- **Overall density** ▲/▼: recompute all npoints from reciprocal-segment lengths.
- **Default path / Clear**: set or empty the path (BZ path fades, segment list shows
  empty prompt, counts → 0).
- **Displacement reference**: radios switch mode; file mode reveals the `.rmc6f`
  selector and echoes the selection.
- **Run phonon bands**: starts the calculation; button becomes **Cancel**; progress
  bar + percent + status update; log streams step messages; on completion the app
  hands off to the Viewer (`onResults` → `fromResults`, auto-save `band.yaml`).
- Hover states: steppers and small buttons get a soft `#e9f0ff` bg + accent color.
- Transitions: progress bar width `.4s ease`; BZ path opacity `.2s`.

## State Management
Mirror the existing `RunnerPage` state (see `DESIGN_NOTES.md`), plus what the UI needs:
- `dirHandle, filesList, configFamily, rmc6fList, structureName, baseStructure`
- `refMode ('average' | 'file')`, `refName / referenceHandle`
- `temperature (T), degenerateTol`, `density` (pts/Å⁻¹) → derives per-segment npoints;
  `bzSegments`, `pointsConv`, `segNpoints` overrides; `cleared` (path empty)
- DOS option flag (compute DOS with the run)
- `selectedConfig` for the fit overview
- `isProcessing, progress, progressText` + a **log line array** for the console
- Bravais/`bzModel` memoized (stable identity — don't reset the three.js cameras).

## Design Tokens

**Theme (Cobalt) — drive via CSS variables:**
| token | value | use |
|---|---|---|
| `--bg` | `#f3f5f9` | page background |
| `--card` | `#ffffff` | card surface |
| `--inset` | `#f7f9fc` | inset panels / fields |
| `--inset2` | `#eef1f7` | nav pill / progress track |
| `--border` | `#e3e7ef` | borders / hairlines |
| `--ink` | `#101826` | primary text |
| `--dim` | `#6b7488` | secondary text |
| `--faint` | `#9aa1b2` | muted labels / ticks |
| `--accent` | `#2f6df0` | primary actions, fit line, selected bar |
| `--accentInk` | `#2257cf` | accent text / values |
| `--soft` | `#e9f0ff` | accent-tint backgrounds |
| `--warn` | `#f0663b` | secondary BZ points |
| `--warnInk` | `#c8502b` | secondary point labels |
| `--bar` | `#d5dae4` | unselected R-value bars |

**Fixed (non-theme):** CPK atoms Sr `#13a07f`, Ti `#5b677a`, O `#e06a3b`; chart data
line `#b6bcc6`; console panel bg `#0f1623`, border `#1c2740`, text `#9fb3d1`, prompt
`#4a6b8a`; Cancel button `#e0564b`.

**Typography**
- **Space Grotesk** — wordmark, group/card titles, buttons, badges. Group title
  15px/600 (−.01em); card title 13px/600 (+.01em); buttons 14–15px/600–700.
- **Spline Sans** — body text, radio labels, descriptions (12–13px).
- **Space Mono** — all numeric/data, readouts, steppers, eyebrows (10–13px); eyebrow
  labels 10px uppercase, letter-spacing .16em.
- **Noto Sans** — Brillouin-zone & k-path high-symmetry labels (clean Greek Γ).
- **STIX Two Text** — chart axes, ticks, axis titles (scientific serif).

**Scale/spacing:** radius 6–10px (cards 10, fields/steppers 7, chips 6); card padding
18px; row gap 14px; group gap 28px; content max-width 1320px.

## Assets
No raster assets. The logo atom glyph, crystal cell, Brillouin zone, and S(Q)/G(r)
plots are inline SVG in the prototype — in the app, the crystal & BZ come from the
real three.js components and the plots from `LineChart`/`FitQuality`. Icons elsewhere
in the app use **lucide-react** (keep using it).

## Files
- `Runner.dc.html` — themed entry: applies the Cobalt CSS variables and mounts the
  component. (Open in a browser to view the design.)
- `ConsoleRunner.dc.html` — the full Runner UI (template + logic). All markup,
  styling, and the mock interactions live here. Primary reference.
- `support.js` — runtime needed to open the `.dc.html` files locally (not app code).
- `DESIGN_NOTES.md` — the original behavioral contract: module boundary, page
  responsibilities, **component prop contracts**, and core data shapes. Authoritative
  for wiring real data.

### How to view
Open `Runner.dc.html` in a Chromium browser (the files load fonts from Google Fonts).
