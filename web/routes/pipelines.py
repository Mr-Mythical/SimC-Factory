"""Pipeline routes — list, launch, detail, cancel."""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from web import pipeline_engine
from web.config import PIPELINE_SHARED_FIELDS
from web.pipeline_defs import PIPELINES

router = APIRouter()


@router.get("/pipelines")
async def pipelines_index(request: Request):
    templates = request.app.state.templates
    runs = pipeline_engine.list_runs()
    schedule_state = {
        pid: pipeline_engine.is_schedule_enabled(pid)
        for pid in PIPELINES
    }
    return templates.TemplateResponse(request, "pipelines/index.html", {
        "request": request,
        "pipelines": PIPELINES,
        "runs": runs,
        "shared_fields": PIPELINE_SHARED_FIELDS,
        "schedule_state": schedule_state,
    })


@router.get("/pipelines/runs")
async def pipelines_runs_partial(request: Request):
    """HTMX partial: pipeline runs table for auto-refresh."""
    templates = request.app.state.templates
    runs = pipeline_engine.list_runs()
    return templates.TemplateResponse(request, "partials/_pipeline_runs.html", {
        "request": request,
        "runs": runs,
    })


@router.post("/pipelines/{pipeline_id}/toggle-schedule")
async def toggle_schedule(pipeline_id: str):
    """Enable or disable a pipeline's scheduled run."""
    if pipeline_id not in PIPELINES:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {pipeline_id}")
    currently_enabled = pipeline_engine.is_schedule_enabled(pipeline_id)
    pipeline_engine.set_schedule_enabled(pipeline_id, not currently_enabled)
    new_state = not currently_enabled
    color = "green" if new_state else "red"
    label = "Enabled" if new_state else "Disabled"
    return HTMLResponse(
        f'<span class="text-{color}-400 text-sm font-semibold">{label}</span>'
    )


@router.post("/pipelines/{pipeline_id}/launch")
async def launch_pipeline(request: Request, pipeline_id: str):
    """Manual trigger for a pipeline."""
    pipeline_def = PIPELINES.get(pipeline_id)
    if not pipeline_def:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline: {pipeline_id}")

    # Check if already running
    for run in pipeline_engine.list_runs():
        if run.pipeline_def.id == pipeline_id and run.status == "running":
            raise HTTPException(status_code=409, detail="Pipeline already running")

    # Collect param overrides from form
    form = await request.form()
    param_overrides = {}
    for field in PIPELINE_SHARED_FIELDS:
        val = form.get(field.arg_name, "")
        if val and str(val).strip():
            param_overrides[field.arg_name] = str(val).strip()

    loop = asyncio.get_running_loop()
    run = pipeline_engine.start_pipeline(pipeline_def, param_overrides, loop)

    return Response(
        status_code=303,
        headers={
            "HX-Redirect": f"/pipelines/run/{run.id}",
            "Location": f"/pipelines/run/{run.id}",
        },
    )


@router.get("/pipelines/run/{run_id}")
async def pipeline_detail(request: Request, run_id: str):
    templates = request.app.state.templates
    run = pipeline_engine.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return templates.TemplateResponse(request, "pipelines/detail.html", {
        "request": request,
        "run": run,
    })


@router.get("/pipelines/run/{run_id}/steps")
async def pipeline_steps_partial(request: Request, run_id: str):
    """HTMX partial: step timeline for auto-refresh."""
    templates = request.app.state.templates
    run = pipeline_engine.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return templates.TemplateResponse(request, "partials/_pipeline_steps.html", {
        "request": request,
        "run": run,
    })


@router.delete("/pipelines/run/{run_id}")
async def cancel_pipeline_run(run_id: str):
    run = pipeline_engine.cancel_pipeline(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return HTMLResponse(
        f'<span class="inline-block px-3 py-1 text-sm rounded badge-{run.status}">{run.status}</span>',
        headers={"HX-Trigger": "pipelineCancelled"},
    )
