# RMC Phonon Viewer (`viz/`)

A browser viewer for phonon results computed from RMC ensembles. It renders, from
a single phonopy-style `band.yaml`:

- **Band structure** — dispersion along the chosen k-path (meV).
- **Neutron-weighted S(Q,E)** — a powder inelastic-neutron-scattering map.
- **Phonon DOS** — total density of states.
- **3D phonon modes** — interactive structure with animated mode displacements.

Live demo (GitHub Pages):
<https://drthyang.github.io/rmc-phonon-dynamics/viz/rmcph.html>

The 3D rendering, band plot, and mode animation are adapted from the
[phononwebsite](https://github.com/henriquemiranda/phononwebsite) project; the
S(Q,E) map, DOS, and the `band.yaml` parsing/encoding are custom to this repo.

## Input

A phonopy-compatible **`band.yaml`** (frequencies in **meV**) — exactly what
`src_gpu` / the RMC Phonon Runner writes. A **`band.json`** (THz-encoded, valid
YAML 1.2) is also accepted and parses ~10× faster; use **Save JSON** after a YAML
load to produce one.

## Loading

- **File picker / drag-and-drop** — choose a `band.yaml` / `band.json`.
- **Deep link** — `rmcph.html?band=<url>` fetches and loads a file
  automatically (optionally `&name=<filename>`). The RMC Phonon Runner uses this
  for its "Open in RMC Phonon Viewer" hand-off (`/viz/rmcph.html?band=/results/…`).

## Controls

- **S(Q,E)**: temperature, energy-resolution σ, incident energy Eᵢ, energy
  range, log scale, colormap. Compute-affecting inputs recompute; render-only
  inputs (colormap, log) just redraw.
- **Band structure**: y-axis (energy) range; per-(k, band) selection.
- **3D / modes**: supercell size, mode amplitude and animation speed, atom/bond
  styling, displacement vectors; GIF / WebM export.

## Performance

File-selection → S(Q,E) shown was cut ~**141×** (≈38 s → ≈0.27 s on a large
8×8×8 / 52-atom cell). Key moves (see `../CHANGELOG.md`): S(Q,E)/DOS run in a Web
Worker (`sqeworker.js`); the phononwebsite 3D supercell build is deferred until
after the heatmap paints; YAML is parsed off the main thread; and `band.json`
replaces YAML for fast reloads. Numerical output is unchanged.

> Note: a `band.yaml` with eigenvectors for a long, dense k-path can be very
> large (hundreds of MB). Prefer `band.json`, or a coarser k-path, for big runs.

## Files

```
rmcph.html              page shell + S(Q,E)/band/structure layout
rmcph.js                file load (incl. ?band= deep link), S(Q,E)/DOS UI, worker glue
sqeworker.js            Web Worker: YAML/JSON parse, THz→meV, S(Q,E) + DOS compute
phonon_assets/          vendored phononwebsite build (main.min.js, css, libs)
install.sh              copies the phononwebsite build into phonon_assets/
```

## Run locally

```bash
# one-time: vendor the phononwebsite build (needs a sibling phononwebsite-local repo)
cd viz && bash install.sh

# serve it (any static server)
python3 -m http.server 8080      # → http://localhost:8080/viz/rmcph.html
```

Or let the **RMC Phonon Runner** serve it: with `rmcph_gui` running, the viewer
is at `http://localhost:7236/viz/rmcph.html`, and finished jobs link straight to
it.
