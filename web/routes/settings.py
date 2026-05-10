"""Settings routes — AWS credentials management."""

import html

from fastapi import APIRouter, Request
from fastapi.responses import Response

from web import credentials
from web.config import AWS_REGION

router = APIRouter()


@router.get("/settings")
async def settings_page(request: Request):
    templates = request.app.state.templates
    creds = credentials.load()
    return templates.TemplateResponse(request, "settings.html", {
        "request": request,
        "creds": creds,
        "default_region": AWS_REGION,
    })


@router.post("/settings/aws")
async def save_aws_credentials(request: Request):
    form = await request.form()
    existing = credentials.load()
    aws_secret_access_key = str(form.get("aws_secret_access_key", "")).strip()
    aws_session_token = str(form.get("aws_session_token", "")).strip()
    creds = {
        "aws_access_key_id": str(form.get("aws_access_key_id", "")).strip() or existing.get("aws_access_key_id", ""),
        "aws_secret_access_key": aws_secret_access_key or existing.get("aws_secret_access_key", ""),
        "aws_session_token": aws_session_token or existing.get("aws_session_token", ""),
        "aws_region": str(form.get("aws_region", "")).strip(),
    }
    credentials.save(creds)

    # Return success indicator for HTMX
    return Response(
        status_code=303,
        headers={"Location": "/settings", "HX-Redirect": "/settings"},
    )


@router.post("/settings/aws/clear")
async def clear_aws_credentials():
    credentials.save({})
    return Response(
        status_code=303,
        headers={"Location": "/settings", "HX-Redirect": "/settings"},
    )


@router.get("/settings/aws/test")
async def test_aws_credentials(request: Request):
    """Quick check: can we call sts get-caller-identity with saved creds?"""
    import os
    import subprocess

    env = os.environ.copy()
    env.update(credentials.get_aws_env())

    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True, text=True, timeout=10, env=env,
        )
        if result.returncode == 0:
            stdout = html.escape(result.stdout.strip())
            return Response(
                content=f'<div class="text-green-400 text-sm mt-2 p-3 bg-green-900/30 rounded">'
                        f'<strong>Connected!</strong><pre class="mt-1 text-xs">{stdout}</pre></div>',
                media_type="text/html",
            )
        else:
            stderr = html.escape(result.stderr.strip())
            return Response(
                content=f'<div class="text-red-400 text-sm mt-2 p-3 bg-red-900/30 rounded">'
                        f'<strong>Failed:</strong><pre class="mt-1 text-xs">{stderr}</pre></div>',
                media_type="text/html",
            )
    except Exception as e:
        error = html.escape(str(e))
        return Response(
            content=f'<div class="text-red-400 text-sm mt-2 p-3 bg-red-900/30 rounded">'
                    f'<strong>Error:</strong> {error}</div>',
            media_type="text/html",
        )
