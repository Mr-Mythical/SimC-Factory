"""Spec data routes — GET /specs, GET /specs/table, GET /specs/{spec}, POST /specs/launch"""

import asyncio
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from web import data_service, job_manager
from web.config import PRESETS, PROJECT_ROOT

router = APIRouter()

SPEC_LAUNCH_PRESETS = {
    "small_test_run",
    "full_sim_500",
    "quick_staleness_check",
    "regenerate_stale",
    "regenerate_stale_skip_check",
    "boost_worst_specs",
}


def _sort_specs(specs: list[dict], sort_col: str, sort_dir: str) -> list[dict]:
    reverse = sort_dir == "desc"
    if sort_col == "mae":
        return sorted(specs, key=lambda s: (s["mae"] is None, s["mae"] or 0), reverse=reverse)
    elif sort_col == "samples":
        return sorted(specs, key=lambda s: s["samples"], reverse=reverse)
    elif sort_col == "drift":
        return sorted(specs, key=lambda s: (s["staleness_drift_pct"] is None, s["staleness_drift_pct"] or 0), reverse=reverse)
    elif sort_col == "status":
        order = {"failed": 0, "stale": 1, "incomplete": 2, "fresh": 3}
        return sorted(specs, key=lambda s: order.get(s["status"], 9), reverse=reverse)
    else:  # spec name
        return sorted(specs, key=lambda s: s["display_name"], reverse=reverse)


@router.get("/specs")
async def specs_page(request: Request, sort: str = "mae", dir: str = "desc"):
    templates = request.app.state.templates
    specs = data_service.get_per_spec_data()
    specs = _sort_specs(specs, sort, dir)
    model = data_service.get_model_summary()

    maes = [s["mae"] for s in specs if s["mae"] is not None]
    avg_mae = sum(maes) / len(maes) if maes else None
    total_samples = sum(s["samples"] for s in specs)
    status_counts = {}
    for s in specs:
        status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1

    return templates.TemplateResponse(request, "specs.html", {
        "request": request,
        "specs": specs,
        "sort_col": sort,
        "sort_dir": dir,
        "avg_mae": avg_mae,
        "total_samples": total_samples,
        "status_counts": status_counts,
        "model": model,
    })


@router.get("/specs/table")
async def specs_table_partial(request: Request, sort: str = "mae", dir: str = "desc"):
    templates = request.app.state.templates
    specs = data_service.get_per_spec_data()
    specs = _sort_specs(specs, sort, dir)
    return templates.TemplateResponse(request, "partials/_spec_rows.html", {
        "request": request,
        "specs": specs,
    })


@router.get("/specs/{spec_name}")
async def spec_detail(request: Request, spec_name: str):
    """Detail page for a single spec."""
    templates = request.app.state.templates
    specs = data_service.get_per_spec_data()
    spec = next((s for s in specs if s["spec"] == spec_name), None)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Spec not found: {spec_name}")

    model = data_service.get_model_summary()
    drift_chart = data_service.get_spec_drift_chart(spec_name)

    return templates.TemplateResponse(request, "specs/detail.html", {
        "request": request,
        "spec": spec,
        "model": model,
        "drift_chart": drift_chart,
    })


@router.post("/specs/launch")
async def launch_spec_job(request: Request):
    """Launch a simulation job for specific spec(s) from the specs page."""
    form = await request.form()
    preset_id = form.get("preset_id", "small_test_run")
    specs_value = form.get("specs", "all")
    profile_source = str(form.get("profile_source", "simc")).strip().lower()

    preset = PRESETS.get(preset_id)
    if not preset:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_id}")
    if preset_id not in SPEC_LAUNCH_PRESETS:
        raise HTTPException(status_code=400, detail=f"Preset is not allowed from the specs page: {preset_id}")

    # Build args from preset defaults
    from web.config import build_args_for_preset
    args = build_args_for_preset(preset)

    # Override --specs with the requested value
    # Remove any existing --specs from args
    filtered = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "--specs":
            skip_next = True
            continue
        if a == "--skip-profile-refresh":
            continue
        filtered.append(a)
    filtered.extend(["--specs", specs_value])
    if profile_source == "custom":
        filtered.append("--skip-profile-refresh")

    loop = asyncio.get_running_loop()
    script_path = os.path.join(PROJECT_ROOT, preset.script)
    if not os.path.exists(script_path):
        raise HTTPException(status_code=400, detail=f"Script not found: {preset.script}")

    source_suffix = "custom-profile" if profile_source == "custom" else "simc-profile"
    name = f"{preset.name} ({specs_value}, {source_suffix})" if specs_value != "all" else f"{preset.name} ({source_suffix})"
    job = job_manager.launch_with_ecr_rebuild(
        script_path=script_path,
        args=filtered,
        name=name,
        preset_id=preset_id,
        loop=loop,
    )

    return Response(
        status_code=303,
        headers={"HX-Redirect": f"/jobs/{job.id}", "Location": f"/jobs/{job.id}"},
    )
