"""Health-check endpoint — proves the frontend ↔ backend round-trip."""
from fastapi import APIRouter

from ..config import APP_NAME, APP_VERSION

router = APIRouter()


@router.get("/ping")
def ping():
    return {"ok": True, "service": APP_NAME, "version": APP_VERSION}
