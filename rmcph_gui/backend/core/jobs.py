"""Persistent job manager for calculation runs (Phase 4).

Single-user, local tool: a module-level JobManager runs each submitted Runner on
its own daemon thread, tracks live progress, persists snapshots to disk, and
supports cooperative cancel.
Phase 5 layers the submission UI + live streaming on top of this.

Cancellation is cooperative: the progress callback raises JobCancelled on the
next progress tick (i.e. between k-points), so a cancelled run stops within one
k-point and never writes a partial band.yaml (the write happens after the loop).
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..config import JOBS_STATE_FILE
from .runners import get_runner


class JobCancelled(Exception):
    """Raised inside the progress callback to abort a running job."""


@dataclass
class Job:
    id: str
    runner: str
    params: dict
    status: str = "queued"           # queued | running | done | error | cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None

    def snapshot(self) -> dict:
        frac = (self.done / self.total) if self.total else 0.0
        return {
            "id": self.id,
            "runner": self.runner,
            "params": self.params,
            "status": self.status,
            "progress": {"done": self.done, "total": self.total,
                         "fraction": frac, "message": self.message},
            "result": self.result,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._last_persist = 0.0
        self._load()

    def submit(self, runner_name: str, params: dict) -> str:
        runner = get_runner(runner_name)   # KeyError if unknown
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, runner=runner_name, params=params or {})
        cancel = threading.Event()
        with self._lock:
            self._jobs[jid] = job
            self._cancels[jid] = cancel
            self._persist_locked(force=True)
        threading.Thread(target=self._run, args=(runner, job, cancel),
                         daemon=True, name=f"job-{jid}").start()
        return jid

    def _load(self):
        if not JOBS_STATE_FILE.exists():
            return
        try:
            payload = json.loads(JOBS_STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            traceback.print_exc()
            return

        now = time.time()
        loaded: dict[str, Job] = {}
        for raw in payload.get("jobs", []):
            progress = raw.get("progress") or {}
            status = raw.get("status") or "error"
            message = raw.get("message") or progress.get("message") or ""
            error = raw.get("error")
            finished = raw.get("finished")
            if status in ("queued", "running"):
                status = "error"
                error = error or "Interrupted: backend restarted before the job finished"
                message = error
                finished = finished or now
            job = Job(
                id=raw["id"],
                runner=raw.get("runner", ""),
                params=raw.get("params") or {},
                status=status,
                done=int(progress.get("done", raw.get("done", 0)) or 0),
                total=int(progress.get("total", raw.get("total", 0)) or 0),
                message=message,
                result=raw.get("result"),
                error=error,
                created=float(raw.get("created") or now),
                started=raw.get("started"),
                finished=finished,
            )
            loaded[job.id] = job
        self._jobs = loaded
        self._persist_locked(force=True)

    def _persist_locked(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_persist < 0.25:
            return
        self._last_persist = now
        JOBS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated": now,
            "jobs": [j.snapshot() for j in sorted(
                self._jobs.values(), key=lambda item: item.created
            )],
        }
        tmp_path = JOBS_STATE_FILE.with_suffix(JOBS_STATE_FILE.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, JOBS_STATE_FILE)

    def _run(self, runner, job: Job, cancel: threading.Event):
        with self._lock:
            job.status = "running"
            job.started = time.time()
            self._persist_locked(force=True)

        def progress_cb(p):
            if cancel.is_set():
                raise JobCancelled()
            with self._lock:
                job.done, job.total, job.message = p.done, p.total, p.message
                self._persist_locked()

        try:
            result = runner.run(job.params, progress_cb=progress_cb)
            with self._lock:
                job.result, job.status, job.message = result, "done", "Done"
                self._persist_locked(force=True)
        except JobCancelled:
            with self._lock:
                job.status, job.message = "cancelled", "Cancelled"
                self._persist_locked(force=True)
        except Exception as e:
            traceback.print_exc()
            with self._lock:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.message = job.error
                self._persist_locked(force=True)
        finally:
            with self._lock:
                job.finished = time.time()
                self._persist_locked(force=True)

    def get(self, jid: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(jid)
            return job.snapshot() if job else None

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
            return [j.snapshot() for j in jobs]

    def latest(self, active_only: bool = False) -> Optional[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
            if active_only:
                jobs = [j for j in jobs if j.status in ("queued", "running")]
            if not jobs:
                return None
            job = max(jobs, key=lambda j: j.created)
            return job.snapshot()

    def cancel(self, jid: str) -> bool:
        with self._lock:
            cancel = self._cancels.get(jid)
            job = self._jobs.get(jid)
            if not cancel or not job or job.status in ("done", "error", "cancelled"):
                return False
        cancel.set()
        with self._lock:
            self._persist_locked(force=True)
        return True


manager = JobManager()
