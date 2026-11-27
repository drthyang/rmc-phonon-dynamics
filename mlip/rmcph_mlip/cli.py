"""Command-line interface for rmcph-mlip."""

from __future__ import annotations

import argparse
from pathlib import Path

from .calculators import build_calculator, list_available_models
from .export import (
    export_band_yaml,
    export_neutron_dos_csv,
    export_phonopy_dm_json,
    plot_summary,
)
from .phonons import compute_harmonic_phonons


def _parse_symprec(val: str) -> float | None:
    if val.lower() == "auto":
        return None
    return float(val)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rmcph-mlip",
        description=(
            "Compute MLIP harmonic phonon bands, FixSymmetry subgroup relaxation, "
            "neutron-weighted DOS (GDOS), and coherent powder INS S(|Q|, E)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="List supported universal MLIP models and local installation status")

    run_p = sub.add_parser("run", help="Run MLIP phonon + neutron DOS/INS calculation")
    run_p.add_argument("structure", type=Path, help="Input .rmc6f supercell or .cif / POSCAR structure")
    run_p.add_argument("--out", "-o", type=Path, default=Path("results/mlip_phonons"), help="Output directory")
    run_p.add_argument(
        "--model",
        "-m",
        default="mace-mp-medium",
        help="MLIP model ID (e.g. mace-mp-medium, mace-mpa-0, mace-omat-0, mace-matpes-r2scan-0, mattersim-5m, orb-v3) or .model path",
    )
    run_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    run_p.add_argument("--dtype", default="float64", choices=["float64", "float32"], help="Precision")
    run_p.add_argument("--supercell", "-s", nargs=3, type=int, default=[2, 2, 2], metavar=("NX", "NY", "NZ"))
    run_p.add_argument(
        "--symprec",
        default="auto",
        help="spglib symmetry tolerance in Å, or 'auto' to pick highest symmetry from ladder (default: auto)",
    )
    run_p.add_argument("--disp", type=float, default=0.01, help="Finite displacement distance in Å (default: 0.01)")
    run_p.add_argument(
        "--relax-mode",
        default="fix_symmetry",
        choices=["fix_symmetry", "unconstrained", "frozen"],
        help="Relaxation strategy: 'fix_symmetry' (default, preserves candidate space group), 'unconstrained', or 'frozen'",
    )
    run_p.add_argument("--no-relax", action="store_true", help="Alias for --relax-mode frozen")
    run_p.add_argument("--relax-cell", action="store_true", help="Relax lattice parameters in addition to positions")
    run_p.add_argument("--fmax", type=float, default=1e-3, help="Force convergence threshold in eV/Å")
    run_p.add_argument("--npoints", type=int, default=41, help="k-points per high-symmetry segment")
    run_p.add_argument("--dos-mesh", nargs=3, type=int, default=[15, 15, 15], metavar=("MX", "MY", "MZ"))
    run_p.add_argument("--temperature", "-T", type=float, default=5.0, help="Temperature in K for INS (default: 5.0)")
    run_p.add_argument("--fwhm", type=float, default=0.8, help="Gaussian resolution FWHM in meV (default: 0.8)")
    run_p.add_argument(
        "--instrument",
        default="SEQUOIA",
        choices=["SEQUOIA", "CNCS", "ARCS", "LET", "constant"],
        help="Direct-geometry chopper resolution preset for powder S(|Q|, E) (default: SEQUOIA)",
    )
    run_p.add_argument("--qmax", type=float, default=8.0, help="Max |Q| in Å⁻¹ for powder INS (default: 8.0)")
    run_p.add_argument("--ei", type=float, default=50.0, help="Incident neutron energy Ei in meV (0 = no cutoff)")
    run_p.add_argument("--exp-ins", type=Path, default=None, help="Optional 2-col (E_meV, I) experimental INS file to overlay")
    run_p.add_argument("--write-yaml", action="store_true", help="Also export phonopy band.yaml")

    srv_p = sub.add_parser("serve", help="Start optional local HTTP companion server for 1-click MLIP runs from the web UI")
    srv_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    srv_p.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    srv_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Default compute device")

    args = parser.parse_args(argv)

    if args.command == "models":
        print("Available Universal MLIPs in rmcph-mlip:")
        for m in list_available_models():
            status = "INSTALLED (cached)" if m["cached_locally"] else ("INSTALLED" if m["installed"] else f"optional ({m['install_hint']})")
            print(f"  {m['id']:22s}  [{m['functional']:12s}]  {status:26s}  — {m['label']}")
        return 0

    if args.command == "serve":
        from .server import serve
        serve(host=args.host, port=args.port, device=args.device)
        return 0

    if args.command == "run":
        eff_relax_mode = "frozen" if args.no_relax else args.relax_mode
        calc = build_calculator(model=args.model, device=args.device, dtype=args.dtype)
        result = compute_harmonic_phonons(
            structure_path=args.structure,
            calculator=calc,
            supercell=args.supercell,
            symprec=_parse_symprec(args.symprec),
            displacement=args.disp,
            relax=(eff_relax_mode != "frozen"),
            relax_cell=args.relax_cell,
            relax_mode=eff_relax_mode,
            fmax=args.fmax,
            npoints=args.npoints,
            dos_mesh=args.dos_mesh,
            temperature_k=args.temperature,
            fwhm_mev=args.fwhm,
            q_max=args.qmax,
            ei_mev=args.ei if args.ei > 0 else None,
            instrument=args.instrument,
            model_name=args.model,
        )

        out_dir: Path = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = args.structure.stem

        dm_json = export_phonopy_dm_json(result, out_dir / f"{stem}_mlip_dm.json")
        ndos_csv = export_neutron_dos_csv(result, out_dir / f"{stem}_mlip_ndos.csv")
        png_path = plot_summary(result, out_dir / f"{stem}_mlip_summary.png", exp_ins_path=args.exp_ins)
        print(f"[mlip] Wrote viewer JSON:  {dm_json}")
        print(f"[mlip] Wrote neutron DOS:  {ndos_csv}")
        print(f"[mlip] Wrote summary PNG:  {png_path}")
        if args.write_yaml:
            yaml_path = export_band_yaml(result, out_dir / f"{stem}_mlip_band.yaml")
            print(f"[mlip] Wrote band.yaml:    {yaml_path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
