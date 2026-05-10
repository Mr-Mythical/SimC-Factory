"""Dashboard route — GET /"""

from fastapi import APIRouter, Request

from web import data_service

router = APIRouter()


@router.get("/")
async def dashboard(request: Request):
    templates = request.app.state.templates
    model = data_service.get_model_summary()
    counts = data_service.get_spec_training_counts()
    staleness = data_service.get_profile_staleness()
    best_params = data_service.get_best_hyperparams()
    plots = [p for p in data_service.get_evaluation_plot_names() if p != "drift_over_time.png"]
    drift = data_service.get_drift_summary()
    training_summary = data_service.get_training_data_summary()
    profile_warnings = data_service.get_profile_warnings(limit=25)

    total_samples = sum(counts.values())
    spec_count = len(set(counts.keys()) | set(staleness.keys()))
    stale_count = sum(1 for v in staleness.values() if v.get("is_stale"))
    cv_mae = (model or {}).get("cv_mae")

    return templates.TemplateResponse(request, "dashboard.html", {
        "request": request,
        "model": model,
        "best_params": best_params,
        "drift": drift,
        "plots": plots,
        "total_samples": total_samples,
        "spec_count": spec_count,
        "stale_count": stale_count,
        "cv_mae": cv_mae,
        "training_summary": training_summary,
        "profile_warnings": profile_warnings,
    })
