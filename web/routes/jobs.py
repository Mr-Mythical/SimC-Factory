"""Job routes — launch, list, detail, cancel."""

import asyncio
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from web import job_manager
from web.config import PRESETS, PROJECT_ROOT, PYTHON_EXE, get_presets_by_category

router = APIRouter()

DANGEROUS_PRESET_CONFIRMATIONS = {
    "wipe_all_data": "WIPE ALL DATA",
    "terraform_destroy": "DESTROY INFRASTRUCTURE",
}


@router.get("/jobs")
async def jobs_index(request: Request):
    templates = request.app.state.templates
    jobs = job_manager.list_jobs()
    return templates.TemplateResponse(request, "jobs/index.html", {
        "request": request,
        "jobs": jobs,
    })


@router.get("/jobs/list")
async def jobs_list_partial(request: Request):
    """HTMX partial: job table rows for auto-refresh."""
    templates = request.app.state.templates
    jobs = job_manager.list_jobs()
    return templates.TemplateResponse(request, "partials/_job_rows.html", {
        "request": request,
        "jobs": jobs,
    })


@router.get("/jobs/launch")
async def launch_page(request: Request, preset: str | None = None):
    templates = request.app.state.templates
    preset_groups = get_presets_by_category()
    selected_preset = PRESETS.get(preset) if preset else None
    return templates.TemplateResponse(request, "jobs/launch.html", {
        "request": request,
        "preset_groups": preset_groups,
        "selected_preset": selected_preset,
    })


@router.get("/jobs/launch/form")
async def launch_form_partial(request: Request, preset: str = ""):
    """HTMX partial: render the parameter form for a preset."""
    templates = request.app.state.templates
    preset_config = PRESETS.get(preset)
    return templates.TemplateResponse(request, "partials/_launch_form.html", {
        "request": request,
        "preset": preset_config,
    })


@router.post("/jobs/launch")
async def launch_job(request: Request):
    """Process launch form submission."""
    form = await request.form()
    preset_id = form.get("preset_id", "")
    preset = PRESETS.get(preset_id)
    if not preset:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset_id}")

    required_confirmation = DANGEROUS_PRESET_CONFIRMATIONS.get(str(preset_id))
    if required_confirmation:
        supplied_confirmation = str(form.get("confirm_action", "")).strip()
        if supplied_confirmation != required_confirmation:
            raise HTTPException(
                status_code=400,
                detail=f"Type {required_confirmation!r} to launch {preset.name}.",
            )

    # Build argument list
    args: list[str] = []

    # Add default args from preset
    for arg_name, arg_val in preset.default_args.items():
        if isinstance(arg_val, bool):
            if arg_val:
                args.append(arg_name)
        else:
            args.extend([arg_name, str(arg_val)])

    # Add form field values
    for field in preset.form_fields:
        form_val = form.get(field.arg_name, "")

        if field.field_type == "bool":
            # Checkbox: present in form only if checked
            if form_val == "true":
                args.append(field.arg_name)
        elif form_val and str(form_val).strip():
            val = str(form_val).strip()
            # Skip if same as default and not required (avoid clutter)
            # but always include for clarity
            if field.field_type == "int":
                try:
                    int(val)
                except ValueError:
                    continue
            elif field.field_type == "float":
                try:
                    float(val)
                except ValueError:
                    continue
            args.extend([field.arg_name, val])

    loop = asyncio.get_running_loop()

    SIMULATION_PRESETS_WITH_ECR = {
        "quick_staleness_check", "regenerate_stale", "regenerate_stale_skip_check",
        "boost_worst_specs", "full_sim_500", "small_test_run",
    }

    if preset.is_shell:
        # Shell command (e.g. terraform, aws) — use explicit subcommand
        subcmd_parts = preset.subcommand.split() if preset.subcommand else []
        raw_cmd = [preset.script] + subcmd_parts + args
        job = job_manager.launch(
            script_path=preset.script,
            args=args,
            name=preset.name,
            preset_id=preset_id,
            loop=loop,
            raw_cmd=raw_cmd,
            cwd=preset.cwd,
        )
    else:
        # Python script
        script_path = os.path.join(PROJECT_ROOT, preset.script)
        if not os.path.exists(script_path):
            raise HTTPException(status_code=400, detail=f"Script not found: {preset.script}")
        if preset_id in SIMULATION_PRESETS_WITH_ECR:
            job = job_manager.launch_with_ecr_rebuild(
                script_path=script_path,
                args=args,
                name=preset.name,
                preset_id=preset_id,
                loop=loop,
            )
        else:
            job = job_manager.launch(
                script_path=script_path,
                args=args,
                name=preset.name,
                preset_id=preset_id,
                loop=loop,
            )

    return Response(
        status_code=303,
        headers={"HX-Redirect": f"/jobs/{job.id}", "Location": f"/jobs/{job.id}"},
    )


@router.get("/jobs/{job_id}")
async def job_detail(request: Request, job_id: str):
    templates = request.app.state.templates
    job = job_manager.get_job(job_id)
    if not job:
        return templates.TemplateResponse(request, "jobs/job_missing.html", {
            "request": request,
            "job_id": job_id,
        }, status_code=404)

    command = f"{PYTHON_EXE} {job.script} {' '.join(job.args)}"

    return templates.TemplateResponse(request, "jobs/detail.html", {
        "request": request,
        "job": job,
        "command": command,
    })


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = job_manager.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return HTMLResponse(
        f'<span class="inline-block px-3 py-1 text-sm rounded badge-{job.status}">{job.status}</span>',
        headers={"HX-Trigger": "jobCancelled"},
    )
