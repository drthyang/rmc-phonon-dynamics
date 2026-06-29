"""Single-user in-memory session: the currently-open dataset.

This is a local single-user tool, so a module-level holder is sufficient.
Later phases (structure, reciprocal, jobs) read the active dataset from here
instead of re-parsing on every request.
"""
from __future__ import annotations

_state: dict = {"dataset": None}


def set_dataset(dataset: dict) -> None:
    _state["dataset"] = dataset


def get_dataset() -> dict | None:
    return _state["dataset"]
