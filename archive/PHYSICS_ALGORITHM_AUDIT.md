# Physics & Algorithm Audit (May 27, 2026)

This note records the current physics implementation, algorithm assumptions, and development/debug priorities for `rmc-phonon-dynamics`. It is intended as a future handoff for implementation, validation, and regression testing.

## Scope

- Reviewed the active phonon-band calculation path used by `rmcph_gui`.
- Compared the newer `src_gpu/` JAX/GPU implementation with the legacy `src/` CPU implementation.
- Checked data conventions for `.rmc6f`, `Frac*.txt`, displacement references, k-paths, energy conversion, and `band.yaml` output.
- This is a code/algorithm audit, not a complete scientific validation against experiment or an external phonon reference calculation.

## Executive Summary

- The canonical calculation path should be `src_gpu/runner.py` through `run_bands()` or `run_bands_segments()`. This path applies the validated `2π` phase convention through `src_gpu/kpath.py`.
- The main physical model is a mass-weighted displacement-correlation matrix `S(k)`, diagonalized at each k point, with mode energies inferred from the classical equipartition relation.
- The GPU implementation correctly avoids complex numbers inside the JAX/Metal kernel by splitting real and imaginary parts, then recombining on CPU.
- Several legacy or edge-case paths can produce inconsistent results if used directly, especially `src_gpu/main.py`, the legacy `src/` CPU path, and `Partial_Sk_avg()`.
- The highest-priority fixes are to fence off legacy k-vector paths, fix the global GPU mass-cache behavior, validate/fix partial `S(k)`, strengthen cache metadata, and add regression tests for the k-path and reader conventions.

## Active Calculation Flow

The GUI run path is:

1. `rmcph_gui/backend/core/runners/phonon_bands.py`
2. `src_gpu/runner.py`
3. `src_gpu/kpath.py`
4. `src_gpu/Calculators.py`
5. `src_gpu/Writers.py`

For GUI-driven band calculations:

- The frontend/GUI builds per-segment k-paths.
- `rmcph_gui/backend/core/reciprocal.py` stores high-symmetry points in both primitive reciprocal coordinates for display and conventional reciprocal coordinates for calculation.
- `PhononBandsRunner` passes `from_frac_conv` and `to_frac_conv` into `runner.run_bands_segments()`.
- `kpath.build_kpath()` converts conventional fractional reciprocal coordinates into the actual phase vector:

```text
kvec = 2π * q_frac
```

- `Calculators.Sk_avg()` computes the ensemble-averaged `S(k)`.
- `np.linalg.eigh()` diagonalizes the Hermitian symmetrized matrix.
- `Calculators.eigenvalues_to_meV()` converts covariance eigenvalues to energies.
- `Writers.connect_bands()` reorders bands by eigenvector continuity.
- `Writers.gen_phonopy_band_yaml_segments()` writes the phonopy-style `band.yaml`.

## Physics Model

The implemented object is a mass-weighted, Fourier-transformed displacement covariance.

For each atom/reference-site ID `t` and Cartesian component:

```text
u_j = (x_j - <x_j>) / dim @ V_super
```

where:

- `x_j` and `<x_j>` are within-unit-cell fractional coordinates scaled by the supercell dimensions.
- `dim` is the supercell repeat vector.
- `V_super` is the supercell lattice matrix in Å.
- `u_j` is therefore Cartesian displacement in Å.

For each reference-site ID `t`:

```text
U_t(k) = (1 / sqrt(N_t)) * Σ_j sqrt(m_t) * u_j * exp(i k · n_j)
```

where:

- `j` runs over all supercell replicas belonging to reference-site ID `t`.
- `N_t` is the number of replicas of that reference-site ID.
- `m_t` is the atomic mass in amu.
- `n_j` is the integer unit-cell index from the RMC configuration.
- `k` is in radians per conventional cell.

The covariance matrix is:

```text
S(k) = < U(k)† U(k) >
```

after ensemble averaging over configurations.

Important convention note: variable names sometimes say "type", but the full `S(k)` path groups by RMC reference number / crystallographic basis-site ID, not merely by chemical element. This gives `3 * N_basis_sites` modes, which is the intended band-structure dimensionality.

## Energy Conversion

Eigenvalues of `S(k)` have units of `amu Å²`. The code uses the classical equipartition relation:

```text
E [meV] = ENERGY_CONV * sqrt(T [K] / λ [amu Å²])
```

with:

```text
ENERGY_CONV = ℏ * sqrt(kB / (amu_to_kg * Å²_to_m²))
```

Negative eigenvalues are converted to negative energies to keep soft/unstable modes visually distinct. Eigenvalues with absolute value below `1e-4` are mapped to `0`.

Development note: the fixed `1e-4 amu Å²` threshold should become a named parameter or at least be logged in outputs, because it can affect low-energy/acoustic behavior.

## Data and Coordinate Conventions

### Structure and Configurations

- `Readers.read_cell_vec()` reads supercell lattice vectors and repeat dimensions from `.rmc6f`.
- `Readers.get_atom_idx()` maps element symbols to sorted RMC reference numbers.
- `Readers.list_configs()` prefers numbered `.rmc6f` ensemble files over derived `Frac*.txt` files, excluding `AVERAGE`, unnumbered base files, and `_0`.
- `Readers.read_frac_atom_ph()` reads `Frac*.txt` snapshots.
- `Readers.read_rmc6f_atom_ph()` reads `.rmc6f` snapshots and converts global supercell fractional coordinates into the same within-cell coordinate frame used by `Frac*.txt`.

The unified reader convention is:

```text
(atom_type_or_RN, within-cell xyz, integer cell_idx)
```

This is essential because `Sk_avg()` subtracts the displacement reference row-by-row. Any mismatch in row order, reference number, or cell index silently corrupts the displacement.

### Displacement Reference

Two reference modes exist:

- Ensemble average: `Readers.avg_frac_atom_ph()`.
- File reference: `_hsym_from_file()` in `src_gpu/runner.py`, currently `.rmc6f` only.

The file-reference path verifies reference number and cell-index layout against the first configuration. CIF reference support is intentionally not implemented yet because CIF unit-cell sites must be tiled and matched to the RMC supercell reference-number ordering.

## K-Vector Convention

The current GPU path uses:

```text
phase = exp(i * cell_idx · kvec)
kvec = 2π * q_frac
```

where `q_frac` is a conventional-cell fractional reciprocal coordinate.

This is documented and validated in `src_gpu/constants.py` and `src_gpu/validate_kpath_2pi.py`. The validation checks reciprocal-lattice periodicity:

```text
S(k = G) == S(Γ)
```

only when integer reciprocal vectors are scaled by `2π`.

Important warning: `src_gpu/main.py` and parts of the legacy `src/` CPU implementation can pass raw fractional k-points directly into `Calculators.Sk_avg()` without going through `kpath.build_kpath()`. Those paths can silently omit the `2π` factor and should not be used as authoritative physics results unless updated.

## Band Connection and Output

`Writers.connect_bands()` applies a Hungarian assignment based on eigenvector overlap between neighboring q-points. It also rotates nearly degenerate subspaces with SVD to improve continuity.

The GUI runner default relative degeneracy tolerance is `5e-3`.

`Writers._write_band_yaml()` writes:

- Unit-cell lattice vectors as `v_super / dim`.
- Reciprocal lattice as `inv(A).T`, without `2π`, which matches common phonopy-style distance conventions.
- Coordinates grouped by element.
- Eigenvectors in the basis-site ordering used by `np.linalg.eigh()`.
- Repeated segment endpoints and zero distance jumps at segment starts/breaks.

Development note: confirm downstream viewers expect the same no-`2π` reciprocal lattice convention for path-distance display.

## Findings and Risks

### High Priority

1. **Legacy k-vector paths can omit `2π`.**
   - `src_gpu/runner.py` is coherent because it always uses `kpath.build_kpath()`.
   - `src_gpu/main.py` still builds `current_k` directly from fractional coordinates and sends it to `Sk_avg()`.
   - The legacy CPU path in `src/Calculators.py` contains conflicting comments: the docstring says `exp(2πi k_frac · n_cell)`, while the implementation uses `exp(i * tmp_cell @ kvec)`.
   - Recommendation: make `src_gpu/runner.py` the only documented production path; update or deprecate direct scripts that bypass `kpath`.

2. **GPU mass lookup cache is global and not keyed by dataset.**
   - `src_gpu/Calculators.py` builds `_GPU_MASS_TABLE` once and reuses it for all later calls.
   - If a long-lived process runs different materials or different reference-number mappings, the cached mass table may be stale.
   - Recommendation: key the cache by `atom_dic` content, rebuild per material, or remove the global cache.

3. **`Partial_Sk_avg()` likely has incorrect reference indexing.**
   - It uses `hsym_ref_subset = hsym_config[1][target_indices]`.
   - `target_indices` are RMC reference numbers, not guaranteed zero-based row indices.
   - This is probably off-by-one or generally wrong when reference numbers do not match array row positions.
   - Recommendation: subset the reference by a boolean mask on `hsym_config[0]`, matching the same logic used when reading partial frames.

4. **Cache keys are too weak for scientific reproducibility.**
   - Current `Sk_sum_kvec_*.csv` cache names depend mostly on raw k-vector text.
   - They do not encode reference mode, reference file, temperature, atom mapping, code version, phase convention, displacement convention, or parser family.
   - The runner disables cache for `.rmc6f` family and file-reference mode in some cases, which avoids the worst contamination, but the general cache remains fragile.
   - Recommendation: add metadata sidecars or content-hash cache directories.

### Medium Priority

5. **`avg_frac_atom_ph()` checks cell indices but not atom/reference IDs after the first frame.**
   - It verifies cell-index consistency but does not explicitly verify that `atom_type` / reference-number ordering is unchanged.
   - Recommendation: also check reference-number sequence equality for every frame.

6. **GUI seekpath conventional-coordinate transform needs regression tests.**
   - `reciprocal.py` transforms primitive reciprocal coordinates to conventional reciprocal coordinates for `src_gpu`.
   - The cartesian k-vector preservation is mathematically sound, but `validate_kpath_2pi.py` warns that FCC primitive special points can fold/alias when fed directly to the conventional-cell tiling.
   - Recommendation: add a regression test for the exact GUI path-generation contract on the target material.

7. **Validation scripts contain local absolute paths.**
   - `src_gpu/validate_rmc6f_equiv.py` hardcodes `/Users/tt9/Research/GitHub/rmc-phonon-dynamics`.
   - Recommendation: make validation scripts repo-relative and CI-friendly.

8. **`.rmc6f` parser is correctness-oriented but Python-loop heavy.**
   - Current parsing is acceptable for validation and GUI use but may become a bottleneck for large ensembles.
   - Recommendation: profile before optimizing; if needed, add a structured intermediate cache.

### Lower Priority / Documentation

9. **Small-eigenvalue threshold is hidden in code.**
   - `eigenvalues_to_meV()` maps `|λ| < 1e-4` to zero.
   - Recommendation: expose or log this threshold.

10. **`band.yaml` path-distance convention should be documented for viewers.**
    - Calculation phases require `2π`.
    - Output reciprocal-lattice distances omit `2π`.
    - This can be correct, but it should be documented so plotting tools do not mix conventions.

## Recommended Implementation TODOs

1. **Canonicalize the runner path.**
   - Document `src_gpu/runner.py` as the production API.
   - Mark `src_gpu/main.py` as legacy or update it to call `runner.run_bands()`.
   - Add a guard or helper so direct `Sk_avg()` calls require explicit radians-per-cell k-vectors.

2. **Fix mass-cache correctness.**
   - Replace `_GPU_MASS_TABLE` with a cache keyed by a stable tuple of `(symbol, reference_numbers)` pairs.
   - Add a unit test that calls `get_mass_array()` with two different `atom_dic` mappings in the same Python process.

3. **Fix and test `Partial_Sk_avg()`.**
   - Subset `hsym_config` by matching reference numbers, not direct numeric indexing.
   - Add a small deterministic test where reference numbers are nonzero/noncontiguous.

4. **Strengthen reader invariants.**
   - In `avg_frac_atom_ph()`, check both `atom_type` sequence and `cell_idx`.
   - Keep `validate_rmc6f_equiv.py` and make it repo-relative.

5. **Add k-convention regression tests.**
   - Keep the `S(G) == S(Γ)` test as a fast fixture.
   - Add a test that `run_bands_segments()` applies `2π` exactly once.
   - Add a GUI-level test for `frac_conv` path generation from seekpath output.

6. **Improve cache reproducibility.**
   - Include metadata for phase convention, reference source, source file fingerprint, atom mapping, `dim`, and code/cache schema version.
   - Consider storing caches under `results/skcache/<dataset_hash>/<reference_hash>/`.

7. **Record numerical diagnostics.**
   - Log min/max eigenvalues before thresholding.
   - Log number of eigenvalues below the small-mode threshold.
   - Optionally warn if Hermitian symmetrization changes `S(k)` above a tolerance.

8. **Add scientific smoke checks.**
   - At Γ, verify expected near-zero acoustic behavior or explicitly document why the RMC covariance model may not produce exact zero modes.
   - Compare a tiny subset against CPU/GPU results after aligning the k convention.
   - Track representative output spectra to detect accidental convention changes.

## Debug Checklist

When a band result looks wrong:

1. Confirm the run used `src_gpu/runner.py`, not direct `src_gpu/main.py`.
2. Confirm k-points passed to `Sk_avg()` are radians per conventional cell.
3. Delete old `Sk_sum_kvec_*.csv` caches after changing reference mode, k convention, parser family, atom mapping, or displacement convention.
4. Verify reference/config row alignment: same reference numbers and same cell indices.
5. Check whether `.rmc6f` or `Frac*.txt` configs were selected by `Readers.list_configs()`.
6. Inspect the minimum and maximum eigenvalues of Hermitian `S(k)` before conversion.
7. Check whether the `1e-4` eigenvalue threshold is suppressing low-energy modes.
8. If using GUI paths, confirm `frac_conv` values, not primitive `frac` values, are passed to the runner.

## Current Confidence

- **High confidence:** active GPU runner flow, displacement unit conversion, `2π` phase requirement, Hermitian diagonalization, and the basic equipartition conversion formula.
- **Medium confidence:** GUI seekpath-to-conventional k-path integration, `band.yaml` viewer conventions, and cache behavior during normal GUI runs.
- **Low confidence:** legacy CPU/direct scripts as currently written, partial `S(k)` implementation, and cache reuse across materials/reference modes.

