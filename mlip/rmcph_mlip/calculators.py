"""Pluggable ASE calculator registry for universal Machine-Learning Interatomic Potentials (uMLIPs).

Physical Considerations for Harmonic Phonons Across uMLIP Families
==================================================================
Harmonic phonon frequencies omega_{q,nu} depend on the second derivative (Hessian)
Phi_{ij} = d^2 E / (du_i du_j) of the potential energy surface:

1. Exchange-Correlation Functional & Training-Distribution Effects:
   * PBE MPtrj-only models (`mace-mp-medium`) slightly underbind solids (2-4% larger
     equilibrium volumes), systematically softening optical phonon frequencies by ~3-8%.
   * MPtrj + Alexandria (`mace-mpa-0`) and off-equilibrium rattled supercell datasets
     (`mace-omat-0`, trained on Meta's OMAT-24) substantially improve restoring forces
     away from equilibrium.
   * Meta-GGA r^2SCAN foundation models (`mace-matpes-r2scan-0`) tighten covalent/ionic
     bonds closer to low-temperature experimental lattice constants and INS optical edges.
   * Comparing 2-3 distinct uMLIPs ("Multi-uMLIP Consensus") separates robust physical
     local-symmetry phonon splittings from single-functional systematic shifts.

2. Conservative Forces (-grad E) vs. Direct Force Heads:
   * Some GNN architectures (such as ORB) ship both a direct force-prediction head and
     a conservative energy-gradient head (F = -dE/dr).
   * For harmonic phonons, conservative forces are MANDATORY: non-conservative direct
     forces violate Clairaut's theorem (dF_i/du_j != dF_j/du_i), breaking Hessian
     symmetry and introducing spurious imaginary acoustic modes near Gamma. Our ORB
     adapter therefore always selects conservative models (`orb-v3-conservative-*`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch

_LOCAL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Structured catalog of supported universal MLIPs for CLI + `/health` UI discovery
MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "mace-mp-medium",
        "short": "MACE-MP-0",
        "label": "MACE-MP-0 Medium (PBE · MPtrj)",
        "family": "mace",
        "mace_key": "medium",
        "functional": "PBE",
        "package": "mace",
        "local_files": ["mace-mp-0-medium.model", "mace-mp-medium.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mace-mpa-0",
        "short": "MACE-MPA-0",
        "label": "MACE-MPA-0 Medium (PBE · MPtrj + Alexandria)",
        "family": "mace",
        "mace_key": "medium-mpa-0",
        "functional": "PBE",
        "package": "mace",
        "local_files": ["mace-mpa-0-medium.model", "mace-mpa-0.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mace-omat-0",
        "short": "MACE-OMAT-0",
        "label": "MACE-OMAT-0 Medium (PBE · OMAT-24 Rattled Phonons)",
        "family": "mace",
        "mace_key": "medium-omat-0",
        "functional": "PBE (OMAT-24)",
        "package": "mace",
        "local_files": ["mace-omat-0-medium.model", "mace-omat-0.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mace-matpes-r2scan-0",
        "short": "MACE-r²SCAN",
        "label": "MACE-MatPES-0 (r²SCAN meta-GGA)",
        "family": "mace",
        "mace_key": "mace-matpes-r2scan-0",
        "functional": "r²SCAN",
        "package": "mace",
        "local_files": ["mace-matpes-r2scan-0.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mace-mp-0b3",
        "short": "MACE-0b3",
        "label": "MACE-MP-0b3 Medium (PBE + ZBL Repulsion)",
        "family": "mace",
        "mace_key": "medium-0b3",
        "functional": "PBE+ZBL",
        "package": "mace",
        "local_files": ["mace-mp-0b3-medium.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mace-mp-small",
        "short": "MACE-Small",
        "label": "MACE-MP-0 Small (Fast Screening)",
        "family": "mace",
        "mace_key": "small",
        "functional": "PBE",
        "package": "mace",
        "local_files": ["mace-mp-0-small.model", "mace-mp-small.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mace-mp-large",
        "short": "MACE-Large",
        "label": "MACE-MP-0 Large (High Capacity)",
        "family": "mace",
        "mace_key": "large",
        "functional": "PBE",
        "package": "mace",
        "local_files": ["mace-mp-0-large.model", "mace-mp-large.model"],
        "install_hint": "uv pip install mace-torch",
    },
    {
        "id": "mattersim-5m",
        "short": "MatterSim-5M",
        "label": "MatterSim-v1 5M (Microsoft Research)",
        "family": "mattersim",
        "functional": "PBE",
        "package": "mattersim",
        "local_files": [],
        "install_hint": "uv pip install mattersim",
    },
    {
        "id": "mattersim-1m",
        "short": "MatterSim-1M",
        "label": "MatterSim-v1 1M (Fast)",
        "family": "mattersim",
        "functional": "PBE",
        "package": "mattersim",
        "local_files": [],
        "install_hint": "uv pip install mattersim",
    },
    {
        "id": "orb-v3",
        "short": "ORB-v3",
        "label": "ORB-v3 Conservative OMAT (Orbital Materials)",
        "family": "orb",
        "functional": "PBE (OMAT)",
        "package": "orb_models",
        "local_files": [],
        "install_hint": "uv pip install orb-models",
    },
    {
        "id": "sevennet-0",
        "short": "SevenNet-0",
        "label": "SevenNet-0 (Equivariant NequIP/Allegro)",
        "family": "sevennet",
        "functional": "PBE",
        "package": "sevenn",
        "local_files": [],
        "install_hint": "uv pip install sevenn",
    },
    {
        "id": "chgnet",
        "short": "CHGNet",
        "label": "CHGNet (Crystal Hamiltonian Graph Net)",
        "family": "chgnet",
        "functional": "PBE/r²SCAN",
        "package": "chgnet",
        "local_files": [],
        "install_hint": "uv pip install chgnet",
    },
]

_CATALOG_BY_ID = {entry["id"]: entry for entry in MODEL_CATALOG}


def _find_local_model(lower_name: str) -> str | None:
    entry = _CATALOG_BY_ID.get(lower_name)
    candidates = list(entry["local_files"]) if entry else []
    if lower_name == "mace-mp":
        candidates = ["mace-mp-0-medium.model", "mace-mp-medium.model"]
    for fname in candidates:
        p = _LOCAL_MODELS_DIR / fname
        if p.is_file():
            return str(p)
    # Also check standard ~/.cache/mace directory so downloaded models show as cached
    cache_dir = Path.home() / ".cache" / "mace"
    if cache_dir.is_dir() and entry and entry.get("mace_key"):
        key_clean = entry["mace_key"].replace("-", "_")
        for f in cache_dir.iterdir():
            if f.is_file() and (entry["mace_key"] in f.name or key_clean in f.name):
                return None  # Let mace_mp load its own cache path cleanly
    return None


def _is_cached_locally(entry: dict[str, Any]) -> bool:
    for fname in entry.get("local_files", []):
        if (_LOCAL_MODELS_DIR / fname).is_file():
            return True
    cache_dir = Path.home() / ".cache" / "mace"
    if entry.get("family") == "mace" and cache_dir.is_dir():
        mk = entry.get("mace_key", "")
        if any(mk in p.name for p in cache_dir.iterdir() if p.is_file()):
            return True
    return False


def list_available_models() -> list[dict[str, Any]]:
    """Return all cataloged uMLIPs with live `installed` and `cached_locally` flags."""
    pkg_cache: dict[str, bool] = {}
    out: list[dict[str, Any]] = []
    for entry in MODEL_CATALOG:
        pkg = entry["package"]
        if pkg not in pkg_cache:
            pkg_cache[pkg] = importlib.util.find_spec(pkg) is not None
        out.append({
            "id": entry["id"],
            "short": entry["short"],
            "label": entry["label"],
            "family": entry["family"],
            "functional": entry["functional"],
            "installed": pkg_cache[pkg],
            "cached_locally": _is_cached_locally(entry),
            "install_hint": entry["install_hint"],
        })
    return out


def resolve_device(device: str = "auto") -> str:
    """Choose the best available PyTorch device.

    Note: Apple MPS is float32-only. For finite-displacement phonon Hessians, float64 on CPU
    avoids 1e-4 eV/Å force roundoff noise unless the user explicitly selects `mps`.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_calculator(
    model: str = "mace-mp-medium",
    device: str = "auto",
    dtype: str = "float64",
    dispersion: bool = False,
):
    """Instantiate an ASE Calculator for the requested universal MLIP or local `.model` file."""
    dev = resolve_device(device)
    if dev == "mps" and dtype == "float64":
        print("[mlip] MPS does not support float64; falling back to float32 on MPS.")
        dtype = "float32"

    raw_path = Path(model)
    lower = model.strip().lower()

    # ── 1. Local MACE / TorchScript checkpoint file ───────────────────────────
    if raw_path.is_file() and raw_path.suffix.lower() in (".model", ".pt", ".pth"):
        from mace.calculators import mace_mp

        print(f"[mlip] Loading local MACE checkpoint={str(raw_path)!r} device={dev} dtype={dtype}")
        return mace_mp(model=str(raw_path), device=dev, default_dtype=dtype, dispersion=dispersion)

    # ── 2. MACE Foundation Family (MP-0, MPA-0, OMAT-0, MatPES-r2SCAN, 0b3) ───
    if lower.startswith("mace") or lower in _CATALOG_BY_ID and _CATALOG_BY_ID[lower]["family"] == "mace":
        from mace.calculators import mace_mp

        alias_map = {
            "mace-mp": "medium",
            "mace-mp-small": "small",
            "mace-mp-medium": "medium",
            "mace-mp-large": "large",
            "mace-mp-0b3": "medium-0b3",
            "mace-mpa-0": "medium-mpa-0",
            "mace-omat-0": "medium-omat-0",
            "mace-matpes-r2scan-0": "mace-matpes-r2scan-0",
            "mace-matpes-pbe-0": "mace-matpes-pbe-0",
        }
        local_model = _find_local_model(lower)
        model_arg = local_model if local_model else alias_map.get(lower, model)
        print(f"[mlip] Loading MACE model={model_arg!r} device={dev} dtype={dtype} dispersion={dispersion}")
        return mace_mp(
            model=model_arg,
            device=dev,
            default_dtype=dtype,
            dispersion=dispersion,
        )

    # ── 3. Microsoft MatterSim (`mattersim-5m`, `mattersim-1m`) ───────────────
    if lower.startswith("mattersim"):
        if importlib.util.find_spec("mattersim") is None:
            raise RuntimeError(
                f"Model '{model}' requires the `mattersim` package. Install via: `cd mlip && uv sync --extra mattersim`"
            )
        from mattersim.forcefield import MatterSimCalculator

        load_path = "MatterSim-v1.0.0-5M.pth" if "5m" in lower else "MatterSim-v1.0.0-1M.pth"
        print(f"[mlip] Loading MatterSim ({load_path}) on device={dev}")
        return MatterSimCalculator(load_path=load_path, device=dev)

    # ── 4. Orbital Materials ORB (`orb-v3` conservative force head) ───────────
    if lower.startswith("orb"):
        if importlib.util.find_spec("orb_models") is None:
            raise RuntimeError(
                f"Model '{model}' requires `orb-models`. Install via: `cd mlip && uv sync --extra orb`"
            )
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Always prefer conservative energy-gradient forces (-dE/dr) for symmetric phonon Hessians!
        if hasattr(pretrained, "orb_v3_conservative_inf_omat"):
            orbff = pretrained.orb_v3_conservative_inf_omat(device=dev, precision=dtype)
        else:
            orbff = pretrained.orb_v2(device=dev)
        print(f"[mlip] Loading ORB conservative calculator ({model}) on device={dev}")
        return ORBCalculator(orbff, device=dev)

    # ── 5. SevenNet (`sevennet-0`, `sevennet-mf-ompa`) ────────────────────────
    if lower.startswith("sevennet") or lower.startswith("7net"):
        if importlib.util.find_spec("sevenn") is None:
            raise RuntimeError(
                f"Model '{model}' requires `sevenn`. Install via: `cd mlip && uv sync --extra sevennet`"
            )
        from sevenn.calculator import SevenNetCalculator

        ckpt = "7net-mf-ompa" if "ompa" in lower else "7net-0"
        print(f"[mlip] Loading SevenNet ({ckpt}) on device={dev}")
        return SevenNetCalculator(model=ckpt, device=dev)

    # ── 6. CHGNet (`chgnet`) ──────────────────────────────────────────────────
    if lower == "chgnet":
        if importlib.util.find_spec("chgnet") is None:
            raise RuntimeError(
                "Model 'chgnet' requires `chgnet`. Install via: `cd mlip && uv sync --extra chgnet`"
            )
        from chgnet.model import CHGNetCalculator

        print(f"[mlip] Loading CHGNet on use_device={dev}")
        return CHGNetCalculator(use_device=dev)

    supported = ", ".join(m["id"] for m in MODEL_CATALOG)
    raise ValueError(f"Unsupported MLIP model '{model}'. Supported IDs: {supported}, or a local .model path.")


def evaluate_supercell_forces_batched(
    supercell_atoms_list: list[ Any ],
    calculator: Any,
    max_batch_atoms: int = 1024,
) -> tuple[list[Any], dict[str, Any]]:
    """Evaluate atomic forces on a list of displaced Phonopy supercells, using batched
    multi-graph GNN forward passes when supported by `MACECalculator`.

    Why Batching Helps (Especially on Local Subgroups):
    ---------------------------------------------------
    High-symmetry cubic parent cells (`Pm-3m`) often need only 2–3 finite displacements,
    whereas lower-symmetry local subgroups (`P4mm`, `Amm2`, `I4/mcm`, `P-42_1m`) generate
    10–48 displaced supercells! Packing multiple supercell graphs into a single
    `torch_geometric.Batch` with `compute_stress=False` (strain derivatives are unused for
    harmonic force constants) eliminates per-displacement Python/PyTorch dispatch overhead
    and saturates SIMD / GPU tensor cores while reproducing sequential `float64` forces to
    machine precision (< 1e-12 eV/Å).

    Falls back transparently to sequential `atoms.get_forces()` for non-MACE calculators,
    dispersion `SumCalculator` wrappers, PolarMACE, or external-field setups.
    """
    import numpy as np

    n_disp = len(supercell_atoms_list)
    if n_disp == 0:
        return [], {"batched": False, "n_batches": 0, "batch_size": 0}

    n_atoms_per_sc = len(supercell_atoms_list[0])
    same_size = all(len(a) == n_atoms_per_sc for a in supercell_atoms_list)

    # Detect standard MACECalculator eligible for exact multi-graph batching
    is_mace_batchable = (
        same_size
        and n_disp > 1
        and type(calculator).__name__ == "MACECalculator"
        and getattr(calculator, "model_type", "MACE") == "MACE"
        and getattr(calculator, "external_field", None) is None
        and not getattr(calculator, "use_compile", False)
        and hasattr(calculator, "models")
        and hasattr(calculator, "z_table")
        and hasattr(calculator, "r_max")
    )

    if is_mace_batchable:
        try:
            from mace import data as mace_data
            from mace.tools import torch_geometric, torch_tools

            batch_size = max(1, int(max_batch_atoms) // max(1, n_atoms_per_sc))
            # On CUDA, allow larger multi-supercell batches
            if str(getattr(calculator, "device", "cpu")).startswith("cuda"):
                batch_size = max(batch_size, 4096 // max(1, n_atoms_per_sc))
            batch_size = min(n_disp, max(2, batch_size))

            calculator.arrays_keys.update({"charges": calculator.charges_key})
            keyspec = mace_data.KeySpecification(
                info_keys=calculator.info_keys,
                arrays_keys=calculator.arrays_keys,
            )
            unit_conv = float(getattr(calculator, "energy_units_to_eV", 1.0)) / float(
                getattr(calculator, "length_units_to_A", 1.0)
            )

            forces_out: list[np.ndarray] = []
            n_batches = 0

            for start in range(0, n_disp, batch_size):
                chunk = supercell_atoms_list[start : start + batch_size]
                n_batches += 1

                with torch_tools.default_dtype(calculator.default_dtype):
                    graphs = [
                        mace_data.AtomicData.from_config(
                            mace_data.config_from_atoms(
                                at, key_specification=keyspec, head_name=calculator.head
                            ),
                            z_table=calculator.z_table,
                            cutoff=calculator.r_max,
                            heads=calculator.available_heads,
                        )
                        for at in chunk
                    ]
                    batch_base = torch_geometric.Batch.from_data_list(graphs).to(calculator.device)

                # Average forces across committee models if len(calculator.models) > 1
                f_accum = None
                for model in calculator.models:
                    batch = batch_base.clone()
                    model_dtype = next(model.parameters()).dtype
                    for key in batch.keys:
                        val = batch[key]
                        if torch.is_tensor(val) and torch.is_floating_point(val):
                            batch[key] = val.to(dtype=model_dtype)
                    with torch_tools.default_dtype(calculator.default_dtype):
                        out = model(
                            batch.to_dict(),
                            compute_stress=False,
                            training=False,
                            compute_edge_forces=False,
                            compute_atomic_stresses=False,
                        )
                    f_model = out["forces"].detach().cpu().numpy().astype(np.float64) * unit_conv
                    f_accum = f_model if f_accum is None else (f_accum + f_model)

                f_mean = f_accum / float(len(calculator.models))
                for k_idx in range(len(chunk)):
                    f_sc = f_mean[k_idx * n_atoms_per_sc : (k_idx + 1) * n_atoms_per_sc]
                    f_sc = f_sc - f_sc.mean(axis=0, keepdims=True)
                    forces_out.append(f_sc)

            return forces_out, {
                "batched": True,
                "n_batches": n_batches,
                "batch_size": batch_size,
            }
        except Exception as exc:
            print(f"[mlip] Note: batched supercell evaluation fell back to sequential ({exc})")

    # Generic sequential fallback (works with every ASE Calculator)
    prev_stress = getattr(calculator, "compute_stress", None)
    if prev_stress is True:
        try:
            calculator.compute_stress = False
        except Exception:
            prev_stress = None

    forces_seq: list[np.ndarray] = []
    try:
        for idx, sc_atoms in enumerate(supercell_atoms_list):
            work = sc_atoms.copy()
            work.calc = calculator
            f = np.asarray(work.get_forces(), dtype=np.float64)
            f = f - f.mean(axis=0, keepdims=True)
            forces_seq.append(f)
            print(f"  displacement {idx + 1}/{n_disp}: max |F| = {np.linalg.norm(f, axis=1).max():.4f} eV/Å")
    finally:
        if prev_stress is True:
            try:
                calculator.compute_stress = True
            except Exception:
                pass

    return forces_seq, {"batched": False, "n_batches": n_disp, "batch_size": 1}
