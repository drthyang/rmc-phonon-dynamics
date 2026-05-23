"""Phonon-band Runner — the first concrete calculation (Phase 4).

Wraps src_gpu/runner.run_bands_segments. The k-path comes from the BZ view as
explicit per-segment data (conventional-cell coords + per-segment npoints +
labels), so it is passed in `params["segments"]`; the dataset (structure file +
configs dir + displacement reference) comes from the active session.

src_gpu uses bare top-level imports (`import Readers`), so it only resolves when
src_gpu is on sys.path. We add it lazily inside run() — importing this module
(and registering the runner) stays cheap and jax-free.
"""
from __future__ import annotations

import sys
from typing import Optional

from .base import Runner, Progress, ProgressCallback, register
from .. import session
from ... import config


class PhononBandsRunner(Runner):
    name = "phonon_bands"
    label = "Phonon bands"

    def param_schema(self) -> dict:
        return {
            "fields": [
                {"key": "T", "label": "Temperature (K)", "type": "number",
                 "default": 5, "min": 0,
                 "help": "Sample temperature; sets the meV scale via equipartition."},
                {"key": "degenerate_tol", "label": "Degenerate tolerance (rel.)",
                 "type": "number", "default": 5e-3, "min": 0,
                 "help": "Relative frequency window for the degenerate-subspace "
                         "rotation in band connection. 0 disables it."},
            ]
        }

    def run(self, params: dict, progress_cb: Optional[ProgressCallback] = None) -> dict:
        ds = session.get_dataset()
        if not ds:
            raise RuntimeError("No dataset open. Open a data folder first.")

        structure_file = ds.get("structure_file")
        configs_dir = ds.get("configs_dir")
        if not structure_file:
            raise RuntimeError("Dataset has no .rmc6f structure file "
                               "(needed for atom types + lattice).")
        if not configs_dir:
            raise RuntimeError("Dataset has no Frac*.txt configurations.")

        # The reference is chosen in the UI and submitted with the run; fall
        # back to the session dataset's, then to the ensemble average.
        reference = params.get("reference") or ds.get("reference") or {"mode": "average"}
        reference_file = None
        if reference.get("mode") == "file":
            reference_file = reference.get("file")
            if not reference_file:
                raise RuntimeError(
                    "Displacement reference is set to 'file' but no equilibrium "
                    "file was selected.")

        raw_segments = params.get("segments") or []
        if not raw_segments:
            raise ValueError(
                "No k-path. Build a path in the Brillouin-zone view first.")
        segments = [{
            "from_frac": s["from_frac_conv"],
            "to_frac":   s["to_frac_conv"],
            "npoints":   int(s["npoints"]),
            "from_label": s.get("from", ""),
            "to_label":   s.get("to", ""),
        } for s in raw_segments]

        T = float(params.get("T", 5))
        degenerate_tol = float(params.get("degenerate_tol", 5e-3))

        n_qpoints = sum(s["npoints"] for s in segments)

        def on_step(i, n, k_frac):
            if progress_cb:
                progress_cb(Progress(done=i, total=n,
                                     message=f"S(k) {i + 1}/{n}"))

        def on_phase(message):
            # Post-loop phases (band connection, file write) have no per-k
            # progress; keep the bar full and surface the current phase instead.
            if progress_cb:
                progress_cb(Progress(done=n_qpoints, total=n_qpoints,
                                     message=message))

        runner_mod = self._src_gpu_runner()
        result = runner_mod.run_bands_segments(
            structure_file=structure_file,
            configs_dir=configs_dir,
            segments=segments,
            T=T,
            out_dir=str(config.RESULTS_DIR) + "/",
            degenerate_tol=degenerate_tol,
            verbose=False,
            on_step=on_step,
            on_phase=on_phase,
            reference_file=reference_file,
        )

        if progress_cb:
            progress_cb(Progress(done=result["n_qpoints"],
                                 total=result["n_qpoints"],
                                 message="Wrote band.yaml"))

        return {
            "band_yaml": result["band_yaml"],
            "n_qpoints": result["n_qpoints"],
            "n_segments": len(segments),
            "T": T,
        }

    @staticmethod
    def _src_gpu_runner():
        """Import src_gpu/runner on first use (triggers the jax import)."""
        src = str(config.SRC_GPU_DIR)
        if src not in sys.path:
            sys.path.insert(0, src)
        import runner  # noqa: E402 — provided by src_gpu
        return runner


register(PhononBandsRunner())
