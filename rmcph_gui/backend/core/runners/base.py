"""Runner plugin interface — the extensibility seam for calculation types.

Each calculation the GUI can launch (phonon bands, total DOS, partial DOS,
S(Q,E) maps, temperature sweeps, …) is implemented as a Runner subclass.

A Runner declares:
  - `name` / `label`    : identity for the API and UI
  - `param_schema()`    : declarative parameter spec so the frontend can render
                          the settings form WITHOUT bespoke per-runner UI code
  - `run(params, cb)`   : execute the calculation, calling `cb(Progress)` as it
                          advances; returns a result dict (paths, summary)

Phase 0 ships only this interface (plus the registry). The first concrete
runner (phonon bands) arrives in Phase 4.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Progress:
    """A single progress update emitted during a run."""
    done: int
    total: int
    message: str = ""
    fraction: float = field(init=False)

    def __post_init__(self):
        self.fraction = (self.done / self.total) if self.total else 0.0


ProgressCallback = Callable[[Progress], None]


class Runner(ABC):
    # Stable identifier used in the API (e.g. "phonon_bands").
    name: str = "runner"
    # Human-readable label for the UI.
    label: str = "Runner"

    @abstractmethod
    def param_schema(self) -> dict:
        """Return a JSON-serialisable parameter spec.

        Shape (kept deliberately simple; the frontend renders fields from it):
            {
              "fields": [
                {"key": "T", "label": "Temperature (K)", "type": "number",
                 "default": 5, "min": 0},
                {"key": "kstep", "label": "Steps / segment", "type": "int",
                 "default": 16, "min": 1},
                ...
              ]
            }
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, params: dict, progress_cb: Optional[ProgressCallback] = None) -> dict:
        """Execute the calculation. Return a result dict (e.g. output paths)."""
        raise NotImplementedError


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, Runner] = {}


def register(runner: Runner) -> Runner:
    _REGISTRY[runner.name] = runner
    return runner


def get_runner(name: str) -> Runner:
    if name not in _REGISTRY:
        raise KeyError(f"unknown runner: {name!r} (have {list(_REGISTRY)})")
    return _REGISTRY[name]


def list_runners() -> list[dict]:
    return [{"name": r.name, "label": r.label} for r in _REGISTRY.values()]
