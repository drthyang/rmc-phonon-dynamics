"""Jobs API: list runners, submit/poll/cancel calculation runs (Phase 4).

The live-streaming submission UI (WebSocket / progress bar / resume) lands in
Phase 5; this exposes the backend job manager so it can be driven and tested.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core import jobs as jobmod
from ..core.runners import list_runners as _list_runners, get_runner

router = APIRouter()


class SubmitJobRequest(BaseModel):
    runner: str = "phonon_bands"
    params: dict = Field(default_factory=dict)


@router.get("/runners")
def runners():
    """Available calculation types + their parameter schemas (for the form UI)."""
    return [{**r, "param_schema": get_runner(r["name"]).param_schema()}
            for r in _list_runners()]


@router.post("/jobs")
def submit_job(req: SubmitJobRequest):
    try:
        jid = jobmod.manager.submit(req.runner, req.params)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return jobmod.manager.get(jid)


@router.get("/jobs")
def list_jobs():
    return jobmod.manager.list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    snap = jobmod.manager.get(job_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    return snap


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if jobmod.manager.cancel(job_id):
        return jobmod.manager.get(job_id)
    snap = jobmod.manager.get(job_id)
    if not snap:
        raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
    raise HTTPException(status_code=409,
                        detail=f"job not cancellable (status={snap['status']})")
