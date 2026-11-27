"""Exporters for web viewer (phonopy-dynamical-matrix-v1 JSON, band.yaml), neutron DOS CSV, and summary figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .phonons import PhononResult

THZ_TO_MEV = 4.135667696


def export_phonopy_dm_json(result: PhononResult, out_path: str | Path) -> Path:
    """Export compact force constants in `phonopy-dynamical-matrix-v1` JSON format.

    This format is natively loaded and diagonalized by `web/src/io/phonopyDM.js`.
    """
    ph = result.phonopy
    prim = ph.primitive

    fc = ph.force_constants
    fc_compact = fc[prim.p2s_map] if fc.shape[0] != len(prim) else fc

    svecs, multi = prim.get_smallest_vectors()
    if multi.ndim == 2:
        dense_svecs = []
        dense_multi = np.zeros((multi.shape[0], multi.shape[1], 2), dtype=int)
        for s in range(multi.shape[0]):
            for i in range(multi.shape[1]):
                cnt = int(multi[s, i])
                addr = len(dense_svecs)
                dense_multi[s, i, 0] = cnt
                dense_multi[s, i, 1] = addr
                for v in range(cnt):
                    dense_svecs.append(svecs[s, i, v].tolist())
        svecs_out = dense_svecs
        multi_out = dense_multi.tolist()
    else:
        svecs_out = svecs.tolist()
        multi_out = multi.tolist()

    p2p_map = prim.p2p_map
    s2pp_map = [int(p2p_map[s_idx]) for s_idx in prim.s2p_map]
    conv_factor = float(ph.unit_conversion_factor)
    masses = [float(m) for m in prim.masses]
    lattice = prim.cell.tolist()

    ndos = result.neutron_dos
    ins = result.powder_ins

    payload = {
        "name": f"{result.primitive_atoms.get_chemical_formula()} ({result.metadata['model']})",
        "format": "phonopy-dynamical-matrix-v1",
        "lattice": lattice,
        "atom_types": list(prim.symbols),
        "atom_pos_red": prim.scaled_positions.tolist(),
        "average_mass": float(np.mean(masses)),
        "qpoints": result.qpoints.tolist(),
        "highsym_qpts": [[int(idx), lbl] for idx, lbl in result.highsym_qpts],
        "mlip_metadata": result.metadata,
        "neutron_dos": {
            "energy_mev": np.round(ndos.energy_mev, 5).tolist(),
            "total_dos": np.round(ndos.total_dos, 7).tolist(),
            "neutron_gdos": np.round(ndos.neutron_gdos, 7).tolist(),
            "element_gdos": {
                el: np.round(arr, 7).tolist() for el, arr in ndos.element_gdos.items()
            },
            "element_weights": {el: round(float(w), 6) for el, w in ndos.element_weights.items()},
            "fwhm_mev": float(ndos.fwhm_mev),
        },
        "powder_ins": {
            "q_grid": np.round(ins.q_grid, 5).tolist(),
            "e_grid": np.round(ins.e_grid, 5).tolist(),
            "sqe": np.round(ins.sqe, 6).tolist(),
            **(
                {"sqe_coherent": np.round(ins.sqe_coherent, 6).tolist()}
                if ins.sqe_coherent is not None
                else {}
            ),
            **(
                {"sqe_incoherent": np.round(ins.sqe_incoherent, 6).tolist()}
                if ins.sqe_incoherent is not None
                else {}
            ),
            **(
                {"sqe_multiphonon": np.round(ins.sqe_multiphonon, 6).tolist()}
                if ins.sqe_multiphonon is not None
                else {}
            ),
            **(
                {"fwhm_vs_e": np.round(ins.fwhm_vs_e, 5).tolist()}
                if ins.fwhm_vs_e is not None
                else {}
            ),
            "instrument": ins.instrument,
            "temperature_k": float(ins.temperature_k),
            "ei_mev": float(ins.ei_mev) if ins.ei_mev is not None else None,
            "fwhm_mev": float(ins.fwhm_mev),
            "u_iso_by_element": {el: round(float(u), 6) for el, u in ins.u_iso_by_element.items()},
        },
        "dynamical_matrix": {
            "format": "phonopy-dynamical-matrix-v1",
            "primitive_lattice": lattice,
            "masses": masses,
            "frequency_conversion_factor": conv_factor,
            "acoustic_sum_rule": "translational",
            "force_constants_compact": np.round(fc_compact, 8).tolist(),
            "shortest_vectors": svecs_out,
            "multiplicity": multi_out,
            "s2pp_map": s2pp_map,
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def export_band_yaml(result: PhononResult, out_path: str | Path) -> Path:
    """Write phonopy band.yaml (also loadable in the web viewer)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.phonopy.write_yaml_band_structure(filename=str(out))
    return out


def export_neutron_dos_csv(result: PhononResult, out_path: str | Path) -> Path:
    """Export energy [meV], total DOS, neutron-weighted GDOS, and per-element GDOS to CSV."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ndos = result.neutron_dos
    els = list(ndos.element_gdos.keys())
    header = ["energy_meV", "total_dos", "neutron_gdos"] + [f"gdos_{el}" for el in els]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(ndos.energy_mev)):
            row = [
                f"{ndos.energy_mev[i]:.6f}",
                f"{ndos.total_dos[i]:.8f}",
                f"{ndos.neutron_gdos[i]:.8f}",
            ] + [f"{ndos.element_gdos[el][i]:.8f}" for el in els]
            w.writerow(row)
    return out


def _load_exp_ins(exp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 2-column (energy_meV, intensity) experimental INS spectrum (CSV or whitespace)."""
    raw = np.loadtxt(str(exp_path), comments="#", delimiter="," if exp_path.suffix.lower() == ".csv" else None)
    e_exp = raw[:, 0]
    y_exp = raw[:, 1]
    return e_exp, y_exp


def plot_summary(
    result: PhononResult,
    out_path: str | Path,
    exp_ins_path: Path | None = None,
) -> Path:
    """Save a 3-panel summary figure: Band Structure | Neutron-Weighted GDOS | Powder INS S(|Q|, E)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Reciprocal-space cumulative distance along qpoints
    recip = 2.0 * np.pi * np.linalg.inv(result.primitive_atoms.cell.array).T
    q_cart = result.qpoints @ recip
    diffs = np.linalg.norm(np.diff(q_cart, axis=0), axis=1)
    hsym_indices = {idx for idx, _ in result.highsym_qpts}
    x = np.zeros(len(q_cart))
    for i in range(1, len(q_cart)):
        step = diffs[i - 1]
        if (i - 1) in hsym_indices and i in hsym_indices and step > 0.25:
            x[i] = x[i - 1]
        else:
            x[i] = x[i - 1] + step

    bands_mev = result.bands_thz * THZ_TO_MEV
    ndos = result.neutron_dos
    ins = result.powder_ins
    meta = result.metadata

    fig, (ax_b, ax_d, ax_s) = plt.subplots(
        1, 3,
        figsize=(14.0, 5.0),
        gridspec_kw={"width_ratios": [2.6, 1.35, 1.85]},
        sharey=True,
    )

    # ── Panel 1: Band Structure ───────────────────────────────────────────────
    for b in range(bands_mev.shape[1]):
        ax_b.plot(x, bands_mev[:, b], color="#1d4ed8", lw=1.05, alpha=0.88)

    tick_pos: list[float] = []
    tick_lbl: list[str] = []
    for idx, lbl in result.highsym_qpts:
        xp = x[idx]
        if tick_pos and abs(xp - tick_pos[-1]) < 1e-6:
            if tick_lbl[-1] != lbl:
                tick_lbl[-1] = f"{tick_lbl[-1]}|{lbl}"
        else:
            tick_pos.append(xp)
            tick_lbl.append(lbl)
            ax_b.axvline(xp, color="#94a3b8", lw=0.7, ls="--")

    ax_b.axhline(0.0, color="#64748b", lw=0.8, ls=":")
    ax_b.set_xticks(tick_pos)
    ax_b.set_xticklabels(tick_lbl, fontsize=10)
    ax_b.set_xlim(x[0], x[-1])
    ax_b.set_ylabel("Energy (meV)", fontsize=11)
    ax_b.set_title("Phonon Dispersion", fontsize=11, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel 2: Neutron-Weighted GDOS vs Total DOS (+ element breakdown) ────
    el_palette = ["#d97706", "#059669", "#7c3aed", "#dc2626", "#0284c7"]
    for idx_el, (el, g_el) in enumerate(ndos.element_gdos.items()):
        col = el_palette[idx_el % len(el_palette)]
        ax_d.fill_betweenx(ndos.energy_mev, 0, g_el, color=col, alpha=0.18)
        ax_d.plot(g_el, ndos.energy_mev, color=col, lw=1.0, label=f"{el} (σ/M={ndos.element_weights[el]:.3f})")

    ax_d.plot(ndos.total_dos, ndos.energy_mev, color="#64748b", lw=1.0, ls="--", label="Total DOS")
    ax_d.plot(ndos.neutron_gdos, ndos.energy_mev, color="#0f172a", lw=1.6, label="Neutron GDOS")

    if exp_ins_path is not None and exp_ins_path.exists():
        e_exp, y_exp = _load_exp_ins(exp_ins_path)
        # Scale experimental curve to match peak of neutron_gdos for visual comparison
        scale = float(ndos.neutron_gdos.max()) / max(float(np.max(y_exp)), 1e-12)
        ax_d.plot(y_exp * scale, e_exp, color="#dc2626", lw=1.3, alpha=0.85, label="Exp. INS")

    ax_d.axhline(0.0, color="#64748b", lw=0.8, ls=":")
    ax_d.set_xlabel("Normalized DOS (arb.)", fontsize=10)
    ax_d.set_xlim(left=0)
    ax_d.set_title(f"Neutron GDOS (FWHM={ndos.fwhm_mev} meV)", fontsize=11, fontweight="bold")
    ax_d.legend(fontsize=7.5, loc="upper right", framealpha=0.88)
    ax_d.grid(axis="y", alpha=0.25)

    # ── Panel 3: Simulated Powder INS S(|Q|, E) ──────────────────────────────
    # Soft log/power stretch so higher-energy modes remain visible alongside acoustic modes
    sqe_disp = np.power(np.clip(ins.sqe, 0.0, 1.0), 0.45)
    im = ax_s.pcolormesh(
        ins.q_grid,
        ins.e_grid,
        sqe_disp,
        shading="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    ax_s.set_xlabel(r"$|Q|$ ($\mathrm{\AA}^{-1}$)", fontsize=10)
    ei_txt = f", $E_i$={ins.ei_mev:g} meV" if ins.ei_mev else ""
    ax_s.set_title(f"Powder INS $S(|Q|,E)$ ({ins.temperature_k:g} K{ei_txt})", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_s, pad=0.02, fraction=0.05)
    cbar.set_label(r"$S(|Q|,E)^{0.45}$ (norm.)", fontsize=9)

    y_lo = min(-1.5, float(bands_mev.min()) - 1.5)
    y_hi = max(float(bands_mev.max()) + 3.0, float(ndos.energy_mev.max()))
    ax_b.set_ylim(y_lo, y_hi)

    fig.suptitle(
        f"{meta['formula']} [{meta['spacegroup']}] — {meta['model']} "
        f"(supercell {'×'.join(map(str, meta['supercell']))}, input: {Path(meta['source_structure']).name})",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
