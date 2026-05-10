"""Launch the Mr. Mythical: SimC Factory web interface."""

import os

import uvicorn


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host=os.environ.get("SIMC_DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIMC_DASHBOARD_PORT", "8000")),
        reload=_env_bool("SIMC_DASHBOARD_RELOAD", True),
    )
