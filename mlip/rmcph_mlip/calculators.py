"""Pluggable ASE calculator factory for universal MLIPs."""

from __future__ import annotations

from pathlib import Path

import torch

_LOCAL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_LOCAL_MODEL_FILES = {
    "mace-mp-medium": ["mace-mp-0-medium.model", "mace-mp-medium.model"],
    "mace-mp": ["mace-mp-0-medium.model", "mace-mp-medium.model"],
    "mace-mp-small": ["mace-mp-0-small.model", "mace-mp-small.model"],
    "mace-mp-large": ["mace-mp-0-large.model", "mace-mp-large.model"],
    "mace-mpa-0": ["mace-mpa-0-medium.model", "mace-mpa-0.model"],
}


def _find_local_model(lower_name: str) -> str | None:
    for fname in _LOCAL_MODEL_FILES.get(lower_name, []):
        p = _LOCAL_MODELS_DIR / fname
        if p.is_file():
            return str(p)
    return None


def resolve_device(device: str = "auto") -> str:
    """Choose the best available PyTorch device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    # Note: MACE float64 is not supported on Apple MPS (MPS is float32-only).
    # We keep CPU for float64 phonon forces unless the user explicitly asks for mps+float32.
    return "cpu"


def build_calculator(
    model: str = "mace-mp-medium",
    device: str = "auto",
    dtype: str = "float64",
    dispersion: bool = False,
):
    """Instantiate an ASE Calculator for the requested MLIP.

    Supported model names:
      - mace-mp-small / mace-mp-medium / mace-mp-large (MACE-MP-0)
      - mace-mpa-0 (MACE-MPA-0 medium)
      - any local path to a MACE .model file
    """
    dev = resolve_device(device)
    if dev == "mps" and dtype == "float64":
        print("[mlip] MPS does not support float64; falling back to float32 on MPS.")
        dtype = "float32"

    lower = model.lower()
    if lower.startswith("mace") or lower.endswith(".model"):
        from mace.calculators import mace_mp

        size_map = {
            "mace-mp-small": "small",
            "mace-mp-medium": "medium",
            "mace-mp-large": "large",
            "mace-mpa-0": "medium-mpa-0",
            "mace-mp": "medium",
        }
        local_model = _find_local_model(lower)
        model_arg = local_model if local_model else size_map.get(lower, model)
        print(f"[mlip] Loading MACE model={model_arg!r} device={dev} dtype={dtype} dispersion={dispersion}")
        return mace_mp(
            model=model_arg,
            device=dev,
            default_dtype=dtype,
            dispersion=dispersion,
        )

    raise ValueError(
        f"Unsupported MLIP model '{model}'. "
        "Supported: mace-mp-small, mace-mp-medium, mace-mp-large, mace-mpa-0, or a .model path."
    )
