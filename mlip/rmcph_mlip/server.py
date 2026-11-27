"""Optional lightweight local HTTP companion server for one-click MLIP runs from the browser.

Usage:
    rmcph-mlip serve [--host 127.0.0.1] [--port 8765] [--device auto]

Uses Python's stdlib ThreadingHTTPServer (zero extra dependencies).
"""

from __future__ import annotations

import json
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .calculators import build_calculator, list_available_models
from .export import export_phonopy_dm_json
from .phonons import compute_harmonic_phonons

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_CALC_CACHE: dict[tuple[str, str, str], Any] = {}
_CALC_LOCK = threading.Lock()


def _get_calculator(model: str, device: str, dtype: str):
    key = (model, device, dtype)
    with _CALC_LOCK:
        if key not in _CALC_CACHE:
            _CALC_CACHE[key] = build_calculator(model=model, device=device, dtype=dtype)
        return _CALC_CACHE[key]


def _run_job(job_id: str, req: dict[str, Any], default_device: str) -> None:
    try:
        model = str(req.get("model", "mace-mp-medium"))
        device = str(req.get("device", default_device))
        dtype = str(req.get("dtype", "float64"))
        supercell = [int(x) for x in req.get("supercell", [2, 2, 2])]
        temperature = float(req.get("temperature", 250.0))
        fwhm = float(req.get("fwhm", 0.8))
        ei = req.get("ei", 50.0)
        ei_val = float(ei) if ei is not None and float(ei) > 0 else None
        instrument = str(req.get("instrument", "SEQUOIA"))
        symprec_raw = req.get("symprec", "auto")
        symprec = None if str(symprec_raw).lower() == "auto" else float(symprec_raw)
        relax = bool(req.get("relax", True))
        relax_cell = bool(req.get("relax_cell", False))
        relax_mode = str(req.get("relax_mode") or ("fix_symmetry" if relax else "frozen"))
        cif_text = str(req["cif_text"])
        filename = Path(str(req.get("filename", "candidate.cif"))).name

        with _JOBS_LOCK:
            _JOBS[job_id]["message"] = f"Loading {model} on {device}…"
        calc = _get_calculator(model, device, dtype)

        with tempfile.TemporaryDirectory(prefix="rmcph_mlip_") as td:
            td_path = Path(td)
            in_path = td_path / filename
            in_path.write_text(cif_text, encoding="utf-8")

            with _JOBS_LOCK:
                sp_tag = "auto" if symprec is None else f"{symprec}Å"
                _JOBS[job_id]["message"] = (
                    f"{model} · {'×'.join(map(str, supercell))} · {relax_mode} (tol={sp_tag})…"
                )

            result = compute_harmonic_phonons(
                structure_path=in_path,
                calculator=calc,
                supercell=supercell,
                symprec=symprec,
                relax=(relax_mode != "frozen"),
                relax_cell=relax_cell,
                relax_mode=relax_mode,
                temperature_k=temperature,
                fwhm_mev=fwhm,
                ei_mev=ei_val,
                instrument=instrument,
                model_name=model,
            )

            dm_path = export_phonopy_dm_json(result, td_path / "out_dm.json")
            dm_payload = json.loads(dm_path.read_text(encoding="utf-8"))

        with _JOBS_LOCK:
            elapsed = result.metadata.get("elapsed_seconds", "?")
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["message"] = f"Done · {result.spacegroup} · {elapsed}s"
            _JOBS[job_id]["result"] = dm_payload
    except Exception as exc:
        traceback.print_exc()
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["message"] = str(exc)


def serve(host: str = "127.0.0.1", port: int = 8765, device: str = "auto") -> None:
    """Start the local MLIP companion HTTP server."""

    class Handler(BaseHTTPRequestHandler):
        def _send_cors(self):
            origin = self.headers.get("Origin", "*")
            self.send_header("Access-Control-Allow-Origin", origin if origin else "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send_json(self, code: int, payload: dict[str, Any]):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors()
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "version": "0.2.0",
                        "device": device,
                        "models": list_available_models(),
                    },
                )
                return
            if self.path.startswith("/jobs/"):
                job_id = self.path.split("/jobs/", 1)[1].strip("/")
                with _JOBS_LOCK:
                    job = _JOBS.get(job_id)
                if not job:
                    self._send_json(404, {"error": f"Unknown job {job_id}"})
                    return
                self._send_json(200, job)
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/jobs":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                if "cif_text" not in req:
                    self._send_json(400, {"error": "Missing cif_text"})
                    return
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            job_id = uuid.uuid4().hex[:12]
            with _JOBS_LOCK:
                _JOBS[job_id] = {"id": job_id, "status": "running", "message": "Queued…"}
            t = threading.Thread(target=_run_job, args=(job_id, req, device), daemon=True)
            t.start()
            self._send_json(202, {"id": job_id, "status": "running"})

        def log_message(self, fmt, *args):
            print(f"[rmcph-mlip serve] {fmt % args}")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[rmcph-mlip] Local companion server v0.2.0 listening on http://{host}:{port} (device={device})")
    print("[rmcph-mlip] Open the web app and click 'Compute Phonons' on any symmetry candidate.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[rmcph-mlip] Server stopped.")
    finally:
        httpd.server_close()
