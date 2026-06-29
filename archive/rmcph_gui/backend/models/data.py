"""Request schemas for the data-folder API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class OpenFolderRequest(BaseModel):
    path: str
    structure_file: Optional[str] = None   # override the auto-detected .rmc6f
