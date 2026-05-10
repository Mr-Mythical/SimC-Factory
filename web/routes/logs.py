"""SSE log streaming — GET /jobs/{id}/logs"""

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from web import job_manager

router = APIRouter()


@router.get("/jobs/{job_id}/logs")
async def job_logs_sse(job_id: str):
    job = job_manager.get_job(job_id)

    async def event_generator():
        if not job:
            yield {
                "event": "message",
                "data": (
                    '<div class="log-line">'
                    'Local job tracker state was lost, likely due to a web app reload. '
                    'The external job may still be running.'
                    '</div>'
                ),
            }
            yield {
                "event": "done",
                "data": "missing",
            }
            return

        async for line in job_manager.subscribe(job_id):
            yield {
                "event": "message",
                "data": f'<div class="log-line">{_escape(line)}</div>',
            }
        # Send final event so the client knows the stream is done
        final_job = job_manager.get_job(job_id)
        status = final_job.status if final_job else "unknown"
        yield {
            "event": "done",
            "data": status,
        }

    return EventSourceResponse(event_generator())


def _escape(text: str) -> str:
    """Basic HTML escaping for log lines."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
