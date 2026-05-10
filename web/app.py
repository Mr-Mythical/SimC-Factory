"""FastAPI application for the Mr. Mythical: SimC Factory web interface."""

import asyncio
import base64
import os
import secrets
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import job_manager, pipeline_engine
from web.config import WEB_DIR
from web.credentials import load_dotenv
from web.data_service import format_spec_name
from web.pipeline_defs import PIPELINES
from web.routes import dashboard, jobs, logs, models, pipelines, plots, settings, specs

# Load .env before anything else touches os.environ
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_manager.recover_jobs()
    # Start the scheduler for pipeline cron jobs
    scheduler = AsyncIOScheduler()
    for pdef in PIPELINES.values():
        if pdef.schedule_cron:
            parts = pdef.schedule_cron.split()
            scheduler.add_job(
                _scheduled_pipeline_launch,
                CronTrigger(
                    minute=parts[0], hour=parts[1], day=parts[2],
                    month=parts[3], day_of_week=parts[4],
                ),
                id=f"pipeline_{pdef.id}",
                args=[pdef.id],
                replace_existing=True,
            )
    scheduler.start()
    app.state.scheduler = scheduler
    yield
    scheduler.shutdown(wait=False)
    job_manager.shutdown_all()


async def _scheduled_pipeline_launch(pipeline_id: str) -> None:
    """Called by APScheduler at the cron time."""
    pdef = PIPELINES.get(pipeline_id)
    if not pdef:
        return
    # Check if schedule is enabled
    if not pipeline_engine.is_schedule_enabled(pipeline_id):
        return
    # Skip if already running
    for run in pipeline_engine.list_runs():
        if run.pipeline_def.id == pipeline_id and run.status == "running":
            return
    loop = asyncio.get_running_loop()
    pipeline_engine.start_pipeline(pdef, param_overrides={}, loop=loop)


APP_NAME = "Mr. Mythical: SimC Factory"

app = FastAPI(title=APP_NAME, lifespan=lifespan)


@app.middleware("http")
async def require_basic_auth_when_configured(request: Request, call_next):
    username = os.environ.get("SIMC_DASHBOARD_USERNAME", "")
    password = os.environ.get("SIMC_DASHBOARD_PASSWORD", "")
    if not username and not password:
        return await call_next(request)
    if not username or not password:
        return Response(
            "SIMC_DASHBOARD_USERNAME and SIMC_DASHBOARD_PASSWORD must be configured together.",
            status_code=500,
        )

    authorization = request.headers.get("Authorization", "")
    scheme, _, encoded = authorization.partition(" ")
    valid = False
    if scheme.lower() == "basic" and encoded:
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
            supplied_username, _, supplied_password = decoded.partition(":")
            valid = secrets.compare_digest(supplied_username, username) and secrets.compare_digest(
                supplied_password, password
            )
        except (ValueError, UnicodeDecodeError):
            valid = False

    if valid:
        return await call_next(request)

    return Response(
        status_code=401,
        headers={"WWW-Authenticate": f'Basic realm="{APP_NAME}"'},
    )


# Static files and templates
app.mount("/static", StaticFiles(directory=f"{WEB_DIR}/static"), name="static")
templates = Jinja2Templates(directory=f"{WEB_DIR}/templates")
templates.env.filters["format_spec"] = format_spec_name
templates.env.globals["format_spec"] = format_spec_name
templates.env.globals["static_version"] = int(os.path.getmtime(os.path.join(WEB_DIR, "static", "app.css")))

# Store templates on app state so routes can access it
app.state.templates = templates

# Include routers
app.include_router(dashboard.router)
app.include_router(specs.router)
app.include_router(jobs.router)
app.include_router(logs.router)
app.include_router(plots.router)
app.include_router(pipelines.router)
app.include_router(models.router)
app.include_router(settings.router)
