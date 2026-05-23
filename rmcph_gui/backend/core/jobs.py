"""In-memory job manager for calculation runs (Phase 4).

Single-user, local tool: a module-level JobManager runs each submitted Runner on
its own daemon thread, tracks live progress, and supports cooperative cancel.
Phase 5 layers the submission UI + live streaming on top of this.

Cancellation is cooperative: the progress callback raises JobCancelled on the
next progress tick (i.e. between k-points), so a cancelled run stops within one
k-point and never writes a partial band.yaml (the write happens after the loop).
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

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

    def submit(self, runner_name: str, params: dict) -> str:
        runner = get_runner(runner_name)   # KeyError if unknown
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, runner=runner_name, params=params or {})
        cancel = threading.Event()
        with self._lock:
            self._jobs[jid] = job
            self._cancels[jid] = cancel
        threading.Thread(target=self._run, args=(runner, job, cancel),
                         daemon=True, name=f"job-{jid}").start()
        return jid

    def _run(self, runner, job: Job, cancel: threading.Event):
        with self._lock:
            job.status = "running"
            job.started = time.time()

        def progress_cb(p):
            if cancel.is_set():
                raise JobCancelled()
            with self._lock:
                job.done, job.total, job.message = p.done, p.total, p.message

        try:
            result = runner.run(job.params, progress_cb=progress_cb)
            with self._lock:
                job.result, job.status, job.message = result, "done", "Done"
        except JobCancelled:
            with self._lock:
                job.status, job.message = "cancelled", "Cancelled"
        except Exception as e:
            traceback.print_exc()
            with self._lock:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.message = job.error
        finally:
            with self._lock:
                job.finished = time.time()

    def get(self, jid: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(jid)
            return job.snapshot() if job else None

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
            return [j.snapshot() for j in jobs]

    def cancel(self, jid: str) -> bool:
        with self._lock:
            cancel = self._cancels.get(jid)
            job = self._jobs.get(jid)
            if not cancel or not job or job.status in ("done", "error", "cancelled"):
                return False
        cancel.set()
        return True


manager = JobManager()
