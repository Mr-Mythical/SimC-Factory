"""Model history routes — GET /models, POST /models/restore"""

import os

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import Response

from web import data_service
from web.config import MODEL_DIR, SNAPSHOTS_DIR

router = APIRouter()


@router.get("/models")
async def models_page(request: Request):
    """Model history page showing current model and all snapshots."""
    templates = request.app.state.templates
    current = data_service.get_model_summary()
    snapshots = data_service.get_model_snapshots()
    return templates.TemplateResponse(request, "models/index.html", {
        "request": request,
        "current": current,
        "snapshots": snapshots,
    })


@router.post("/models/restore")
async def restore_snapshot(request: Request, dirname: str = Form(...)):
    """Restore a snapshot to the active model directory."""
    snap_dir = os.path.join(SNAPSHOTS_DIR, dirname)
    if not os.path.isdir(snap_dir):
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {dirname}")

    # Prevent path traversal
    real_snap = os.path.realpath(snap_dir)
    real_base = os.path.realpath(SNAPSHOTS_DIR)
    if not real_snap.startswith(real_base):
        raise HTTPException(status_code=400, detail="Invalid snapshot path")

    from sagemaker.snapshot import restore_snapshot as do_restore
    do_restore(snapshot_dir=snap_dir, model_dir=MODEL_DIR, snapshots_dir=SNAPSHOTS_DIR)

    # Invalidate the data_service cache for the metadata file
    data_service._cache.pop(
        os.path.join(MODEL_DIR, "nn_metadata.json"), None
    )

    return Response(
        status_code=303,
        headers={"HX-Redirect": "/models", "Location": "/models"},
    )
