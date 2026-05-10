"""Serve evaluation plot PNGs — GET /plots/{filename}"""

import os
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from web.config import PLOTS_DIR

router = APIRouter()

_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9_.\-]+\.png$")


@router.get("/plots/{filename}")
async def serve_plot(filename: str):
    if not _SAFE_FILENAME.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = os.path.join(PLOTS_DIR, filename)
    resolved = os.path.realpath(path)
    if not resolved.startswith(os.path.realpath(PLOTS_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail="Plot not found")

    return FileResponse(resolved, media_type="image/png")
