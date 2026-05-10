"""Pipeline engine — chains jobs with conditional branching and parallel steps."""

import asyncio
import json
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from web import job_manager
from web.config import PRESETS, PROJECT_ROOT, build_args_for_preset

SCHEDULE_STATE_FILE = os.path.join(PROJECT_ROOT, "local", "pipeline_schedules.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineStepDef:
    """One step in a pipeline definition."""
    id: str
    preset_id: str
    name: str
    arg_overrides: dict = field(default_factory=dict)
    # Dynamic arg overrides resolved at runtime (receives PipelineRun, returns dict)
    dynamic_args: Callable[["PipelineRun"], dict] | None = None
    condition: Callable[["PipelineRun"], bool] | None = None
    output_parser: Callable[["job_manager.Job", "PipelineRun"], None] | None = None
    parallel_with: str | None = None  # step ID to run concurrently with


@dataclass
class PipelineDef:
    """A complete pipeline definition."""
    id: str
    name: str
    description: str
    steps: list[PipelineStepDef]
    schedule_cron: str | None = None  # "M H D Mo DoW"


@dataclass
class PipelineStepRun:
    """Runtime state of one pipeline step."""
    step_def: PipelineStepDef
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    job_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class PipelineRun:
    """Runtime state of a full pipeline execution."""
    id: str
    pipeline_def: PipelineDef
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    steps: list[PipelineStepRun] = field(default_factory=list)
    current_step_index: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    context: dict = field(default_factory=dict)  # shared state across steps
    param_overrides: dict = field(default_factory=dict)  # user overrides from UI


# ---------------------------------------------------------------------------
# In-memory registry
# ---------------------------------------------------------------------------

_runs: dict[str, PipelineRun] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Schedule state persistence
# ---------------------------------------------------------------------------

def _load_schedule_state() -> dict:
    if os.path.exists(SCHEDULE_STATE_FILE):
        try:
            with open(SCHEDULE_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_schedule_state(state: dict) -> None:
    os.makedirs(os.path.dirname(SCHEDULE_STATE_FILE), exist_ok=True)
    with open(SCHEDULE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def is_schedule_enabled(pipeline_id: str) -> bool:
    """Check if a pipeline's scheduled run is enabled. Defaults to True."""
    state = _load_schedule_state()
    return state.get(pipeline_id, {}).get("enabled", True)


def set_schedule_enabled(pipeline_id: str, enabled: bool) -> None:
    """Enable or disable a pipeline's scheduled run."""
    state = _load_schedule_state()
    state.setdefault(pipeline_id, {})["enabled"] = enabled
    _save_schedule_state(state)


def get_run(run_id: str) -> PipelineRun | None:
    with _lock:
        return _runs.get(run_id)


def list_runs() -> list[PipelineRun]:
    with _lock:
        return sorted(_runs.values(), key=lambda r: r.started_at, reverse=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def start_pipeline(pipeline_def: PipelineDef,
                   param_overrides: dict | None = None,
                   loop: asyncio.AbstractEventLoop | None = None) -> PipelineRun:
    """Create a new pipeline run and start the first step(s)."""
    run_id = uuid.uuid4().hex[:12]
    run = PipelineRun(
        id=run_id,
        pipeline_def=pipeline_def,
        steps=[PipelineStepRun(step_def=s) for s in pipeline_def.steps],
        param_overrides=param_overrides or {},
    )
    with _lock:
        _runs[run_id] = run

    _advance(run, loop)
    return run


def cancel_pipeline(run_id: str) -> PipelineRun | None:
    """Cancel a running pipeline and its active job(s)."""
    with _lock:
        run = _runs.get(run_id)
    if not run or run.status != "running":
        return run

    run.status = "cancelled"
    run.finished_at = datetime.now(UTC)

    # Cancel any running steps
    for step_run in run.steps:
        if step_run.status == "running" and step_run.job_id:
            job_manager.cancel(step_run.job_id)
            step_run.status = "cancelled"
            step_run.finished_at = datetime.now(UTC)

    return run


def _advance(run: PipelineRun, loop: asyncio.AbstractEventLoop | None) -> None:
    """Advance the pipeline to the next eligible step(s)."""
    if run.status != "running":
        return

    idx = run.current_step_index

    # Check if we've completed all steps
    if idx >= len(run.steps):
        run.status = "completed"
        run.finished_at = datetime.now(UTC)
        return

    step_run = run.steps[idx]

    # If current step is still running (parallel partner finished first), wait
    if step_run.status == "running":
        return

    # If current step already done (completed/failed/skipped), move forward
    if step_run.status in ("completed", "skipped"):
        run.current_step_index = idx + 1
        _advance(run, loop)
        return

    if step_run.status == "failed":
        run.status = "failed"
        run.finished_at = datetime.now(UTC)
        return

    # Step is pending — check condition
    if step_run.step_def.condition and not step_run.step_def.condition(run):
        step_run.status = "skipped"
        step_run.started_at = datetime.now(UTC)
        step_run.finished_at = datetime.now(UTC)
        run.current_step_index = idx + 1
        _advance(run, loop)
        return

    # Launch this step
    _launch_step(run, step_run, loop)

    # Check if next step should run in parallel with this one
    if idx + 1 < len(run.steps):
        next_step = run.steps[idx + 1]
        if (next_step.step_def.parallel_with == step_run.step_def.id
                and next_step.status == "pending"):
            # Check condition for the parallel step too
            if next_step.step_def.condition and not next_step.step_def.condition(run):
                next_step.status = "skipped"
                next_step.started_at = datetime.now(UTC)
                next_step.finished_at = datetime.now(UTC)
            else:
                _launch_step(run, next_step, loop)


def _launch_step(run: PipelineRun, step_run: PipelineStepRun,
                 loop: asyncio.AbstractEventLoop | None) -> None:
    """Launch a single pipeline step as a job."""
    step_def = step_run.step_def
    preset = PRESETS.get(step_def.preset_id)
    if not preset:
        step_run.status = "failed"
        step_run.started_at = datetime.now(UTC)
        step_run.finished_at = datetime.now(UTC)
        return

    # Merge overrides: preset defaults < step arg_overrides < dynamic_args < user param_overrides
    overrides = dict(step_def.arg_overrides)
    if step_def.dynamic_args:
        overrides.update(step_def.dynamic_args(run))
    # Apply user param_overrides that match this preset's form fields
    for ff in preset.form_fields:
        if ff.arg_name in run.param_overrides:
            overrides[ff.arg_name] = run.param_overrides[ff.arg_name]

    args = build_args_for_preset(preset, overrides)

    # Build the completion callback
    def on_complete(job: job_manager.Job, _run=run, _step=step_run, _loop=loop):
        _on_step_complete(job, _run, _step, _loop)

    step_run.status = "running"
    step_run.started_at = datetime.now(UTC)

    try:
        if preset.is_shell:
            subcmd_parts = preset.subcommand.split() if preset.subcommand else []
            raw_cmd = [preset.script] + subcmd_parts + args
            job = job_manager.launch(
                script_path=preset.script,
                args=args,
                name=f"[Pipeline] {step_def.name}",
                preset_id=step_def.preset_id,
                loop=loop,
                raw_cmd=raw_cmd,
                cwd=preset.cwd,
                on_complete=on_complete,
            )
        else:
            script_path = os.path.join(PROJECT_ROOT, preset.script)
            job = job_manager.launch(
                script_path=script_path,
                args=args,
                name=f"[Pipeline] {step_def.name}",
                preset_id=step_def.preset_id,
                loop=loop,
                on_complete=on_complete,
            )
    except Exception:
        step_run.status = "failed"
        step_run.finished_at = datetime.now(UTC)
        return

    step_run.job_id = job.id


def _on_step_complete(job: job_manager.Job, run: PipelineRun,
                      step_run: PipelineStepRun,
                      loop: asyncio.AbstractEventLoop | None) -> None:
    """Called from _reader_thread when a pipeline step's job finishes."""
    step_run.finished_at = datetime.now(UTC)

    if job.status == "completed":
        step_run.status = "completed"
        # Run output parser to extract data for downstream steps
        if step_run.step_def.output_parser:
            try:
                step_run.step_def.output_parser(job, run)
            except Exception:
                pass
    else:
        step_run.status = "failed"

    # Check if there's a parallel partner still running
    step_idx = run.steps.index(step_run)
    partner_still_running = False

    # Check if WE are a parallel partner (next step has parallel_with pointing to us)
    if step_idx + 1 < len(run.steps):
        next_step = run.steps[step_idx + 1]
        if next_step.step_def.parallel_with == step_run.step_def.id:
            if next_step.status == "running":
                partner_still_running = True
            elif next_step.status in ("completed", "failed", "skipped"):
                # Partner done too — advance past both
                pass

    # Check if OUR parallel_with partner is still running
    if step_run.step_def.parallel_with:
        for sr in run.steps:
            if sr.step_def.id == step_run.step_def.parallel_with:
                if sr.status == "running":
                    partner_still_running = True
                break

    if partner_still_running:
        return  # Wait for partner to finish

    # Find the furthest completed/skipped/failed step index to advance past
    advance_to = 0
    for i, sr in enumerate(run.steps):
        if sr.status in ("completed", "skipped", "failed"):
            advance_to = i + 1
        else:
            break

    # If any step failed, fail the pipeline
    for sr in run.steps[:advance_to]:
        if sr.status == "failed":
            run.status = "failed"
            run.finished_at = datetime.now(UTC)
            return

    run.current_step_index = advance_to
    _advance(run, loop)
