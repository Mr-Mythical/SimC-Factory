"""Read-only data service for project files with mtime-based caching."""

import json
import os
import re
import urllib.parse
from collections import deque

from web.config import (
    ALL_SPECS_CSV,
    BEST_HYPERPARAMS_FILE,
    DRIFT_HISTORY_FILE,
    DRIFT_PLOT_FILE,
    ENSEMBLE_REPORT_FILE,
    LOCAL_DIR,
    METADATA_FILE,
    PLOTS_DIR,
    PROFILE_METADATA_FILE,
    SNAPSHOTS_DIR,
    SPEC_PROFILES_DIR,
    TARGET_SAMPLES_PER_SPEC,
    TRAINING_DATA_DIR,
)

_cache: dict[str, tuple[float, object]] = {}
_profile_lookup_cache: dict[str, str] | None = None


def _read_cached(path: str) -> object | None:
    """Return cached data if the file hasn't changed, else None."""
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    entry = _cache.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    return None


def _store_cache(path: str, data: object) -> object:
    _cache[path] = (os.path.getmtime(path), data)
    return data


def get_model_summary() -> dict | None:
    """Top-level fields from nn_metadata.json.

    Handles both the SageMaker format (flat per_spec_mae/test_mae at top level)
    and the legacy Optuna format (ensemble_candidates array with production trial).
    """
    cached = _read_cached(METADATA_FILE)
    if cached is not None:
        return cached
    if not os.path.exists(METADATA_FILE):
        return None
    with open(METADATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    # SageMaker format: per_spec_mae and metrics at top level
    per_spec_mae = raw.get("per_spec_mae")
    test_mae = raw.get("test_mae")

    # Legacy Optuna format: metrics inside ensemble_candidates
    if per_spec_mae is None:
        production_trial = raw.get("production_trial_number")
        for cand in raw.get("ensemble_candidates", []):
            if cand.get("trial_number") == production_trial:
                test_mae = cand.get("test_mae")
                per_spec_mae = cand.get("per_spec_mae")
                break

    summary = {
        "device": raw.get("device"),
        "used_gpu": raw.get("used_gpu"),
        "incremental": raw.get("incremental"),
        "best_params": raw.get("best_params"),
        "hidden_dims": raw.get("hidden_dims"),
        "train_mae": raw.get("train_mae"),
        "test_mae": test_mae,
        "train_rmse": raw.get("train_rmse"),
        "test_rmse": raw.get("test_rmse"),
        "train_r2": raw.get("train_r2"),
        "test_r2": raw.get("test_r2"),
        "cv_mae": raw.get("cv_mae"),
        "cv_folds": raw.get("cv_folds"),
        "training_time_seconds": raw.get("training_time_seconds"),
        "train_samples": raw.get("train_samples"),
        "test_samples": raw.get("test_samples"),
        "n_features": raw.get("n_features"),
        "n_specs": raw.get("n_specs"),
        "per_spec_mae": per_spec_mae or {},
    }
    return _store_cache(METADATA_FILE, summary)


def get_spec_training_counts() -> dict[str, int]:
    """Count rows (excluding header) in each per-spec CSV."""
    if not os.path.isdir(TRAINING_DATA_DIR):
        return {}
    counts = {}
    for fname in os.listdir(TRAINING_DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        spec_name = fname[:-4]
        fpath = os.path.join(TRAINING_DATA_DIR, fname)
        with open(fpath, "rb") as f:
            line_count = sum(1 for _ in f) - 1  # exclude header
        counts[spec_name] = max(0, line_count)
    return counts


def get_profile_staleness() -> dict[str, dict]:
    """Parse profile_metadata.json for staleness info."""
    cached = _read_cached(PROFILE_METADATA_FILE)
    if cached is not None:
        return cached
    if not os.path.exists(PROFILE_METADATA_FILE):
        return {}
    with open(PROFILE_METADATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for spec_name, meta in raw.items():
        result[spec_name] = {
            "is_stale": meta.get("is_stale", False),
            "last_generated": meta.get("last_generated_timestamp", ""),
            "staleness_drift_pct": meta.get("staleness_drift_pct"),
            "staleness_checked_at": meta.get("staleness_checked_at", ""),
        }
    return _store_cache(PROFILE_METADATA_FILE, result)


def get_per_spec_data() -> list[dict]:
    """Merged per-spec data: MAE, training counts, staleness."""
    model = get_model_summary()
    counts = get_spec_training_counts()
    staleness = get_profile_staleness()
    warning_map = {w["profile_name"]: w for w in get_profile_warnings(limit=200)}

    per_spec_mae = (model or {}).get("per_spec_mae") or {}
    all_specs = sorted(set(per_spec_mae.keys()) | set(counts.keys()) | set(staleness.keys()))

    rows = []
    for spec in all_specs:
        mae = per_spec_mae.get(spec)
        samples = counts.get(spec, 0)
        stale_info = staleness.get(spec, {})
        is_stale = stale_info.get("is_stale", False)
        warning = warning_map.get(spec, {})

        # Determine status: failed > stale > incomplete > fresh
        if samples == 0:
            status = "failed"
        elif is_stale:
            status = "stale"
        elif samples < TARGET_SAMPLES_PER_SPEC:
            status = "incomplete"
        else:
            status = "fresh"

        rows.append({
            "spec": spec,
            "display_name": format_spec_name(spec),
            "mae": round(mae, 1) if mae is not None else None,
            "samples": samples,
            "is_stale": is_stale,
            "status": status,
            "last_generated": stale_info.get("last_generated", ""),
            "staleness_drift_pct": stale_info.get("staleness_drift_pct"),
            "staleness_checked_at": stale_info.get("staleness_checked_at", ""),
            "error_message": warning.get("message"),
            "error_source": warning.get("source"),
            **get_profile_talent_info(spec),
        })
    return rows


def get_best_hyperparams() -> dict | None:
    """Read sagemaker/best_hyperparameters.json."""
    cached = _read_cached(BEST_HYPERPARAMS_FILE)
    if cached is not None:
        return cached
    if not os.path.exists(BEST_HYPERPARAMS_FILE):
        return None
    with open(BEST_HYPERPARAMS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return _store_cache(BEST_HYPERPARAMS_FILE, data)


def get_ensemble_report() -> dict | None:
    """Read minimal_ensemble_report.json."""
    cached = _read_cached(ENSEMBLE_REPORT_FILE)
    if cached is not None:
        return cached
    if not os.path.exists(ENSEMBLE_REPORT_FILE):
        return None
    with open(ENSEMBLE_REPORT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return _store_cache(ENSEMBLE_REPORT_FILE, data)


def get_evaluation_plot_names() -> list[str]:
    """List PNG filenames in the evaluation plots directory."""
    if not os.path.isdir(PLOTS_DIR):
        return []
    # Key plots first, then the rest
    key_plots = [
        "drift_over_time.png",
        "actual_vs_predicted.png",
        "per_spec_mae.png",
        "error_by_dps_range.png",
        "pct_error_distribution.png",
        "ensemble_comparison.png",
        "optimization_history.png",
    ]
    all_pngs = sorted(f for f in os.listdir(PLOTS_DIR) if f.endswith(".png"))
    ordered = [p for p in key_plots if p in all_pngs]
    ordered += [p for p in all_pngs if p not in key_plots]
    return ordered


def get_drift_summary() -> dict:
    """Return drift plot visibility and latest drift check metadata."""
    summary: dict[str, object] = {
        "plot_name": "drift_over_time.png",
        "has_plot": os.path.exists(DRIFT_PLOT_FILE),
        "latest_check_timestamp": None,
        "latest_spec": None,
        "latest_median_abs_drift_pct": None,
        "latest_max_abs_drift_pct": None,
        "latest_threshold_pct": None,
        "latest_is_stale": None,
        "latest_simc_version": None,
        "latest_simc_git_revision": None,
        "record_count": 0,
    }

    if not os.path.exists(DRIFT_HISTORY_FILE):
        return summary

    cached = _read_cached(DRIFT_HISTORY_FILE)
    if cached is not None:
        out = dict(cached)
        out["has_plot"] = os.path.exists(DRIFT_PLOT_FILE)
        return out

    tail: deque[str] = deque(maxlen=4000)
    try:
        with open(DRIFT_HISTORY_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    tail.append(line)
    except OSError:
        return summary

    latest_row: dict | None = None
    latest_summary_row: dict | None = None
    count = 0
    for line in tail:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += 1
        latest_row = row
        if str(row.get("entry_type", "")).strip().lower() == "check_summary":
            latest_summary_row = row

    summary["record_count"] = count
    chosen = latest_summary_row or latest_row
    if chosen:
        summary.update({
            "latest_check_timestamp": chosen.get("check_timestamp"),
            "latest_spec": chosen.get("spec_name"),
            "latest_median_abs_drift_pct": chosen.get("median_abs_drift_pct"),
            "latest_max_abs_drift_pct": chosen.get("max_abs_drift_pct"),
            "latest_threshold_pct": chosen.get("threshold_pct"),
            "latest_is_stale": chosen.get("is_stale"),
            "latest_simc_version": chosen.get("simc_version"),
            "latest_simc_git_revision": chosen.get("simc_git_revision"),
        })

    return _store_cache(DRIFT_HISTORY_FILE, summary)


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _polyline_points(values: list[float | None], width: int = 760, height: int = 260, pad: int = 20) -> str:
    if not values:
        return ""
    valid = [v for v in values if v is not None]
    if not valid:
        return ""

    ymin = min(valid)
    ymax = max(valid)
    span = (ymax - ymin) if ymax > ymin else 1.0
    inner_w = max(1, width - (2 * pad))
    inner_h = max(1, height - (2 * pad))
    denom = max(1, len(values) - 1)

    pts: list[str] = []
    for i, val in enumerate(values):
        if val is None:
            continue
        x = pad + (i * inner_w / denom)
        y = height - pad - (((val - ymin) / span) * inner_h)
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)


def _y_for_value(value: float, ymin: float, ymax: float, height: int = 260, pad: int = 20) -> float:
    span = (ymax - ymin) if ymax > ymin else 1.0
    inner_h = max(1, height - (2 * pad))
    return height - pad - (((value - ymin) / span) * inner_h)


def get_spec_drift_chart(spec_name: str, max_points: int = 60) -> dict:
    """Return SVG-ready chart data for a single spec's drift history."""
    out: dict[str, object] = {
        "has_data": False,
        "point_count": 0,
        "from_timestamp": None,
        "to_timestamp": None,
        "latest_median_abs_drift_pct": None,
        "latest_max_abs_drift_pct": None,
        "latest_threshold_pct": None,
        "latest_is_stale": None,
        "median_polyline": "",
        "max_polyline": "",
        "threshold_y": None,
        "y_min": None,
        "y_max": None,
    }

    if not os.path.exists(DRIFT_HISTORY_FILE):
        return out

    per_check: dict[str, dict] = {}
    try:
        with open(DRIFT_HISTORY_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("spec_name", "")).strip() != spec_name:
                    continue
                ts = str(row.get("check_timestamp", "")).strip()
                if not ts:
                    continue
                entry_type = str(row.get("entry_type", "")).strip().lower()
                prev = per_check.get(ts)
                if entry_type == "check_summary":
                    # Prefer explicit check summaries when present.
                    per_check[ts] = row
                    continue
                idx = int(row.get("sample_index", 999999))
                prev_idx = int(prev.get("sample_index", 999999)) if prev else 999999
                prev_is_summary = str((prev or {}).get("entry_type", "")).strip().lower() == "check_summary"
                if (prev is None or (not prev_is_summary and idx < prev_idx)):
                    per_check[ts] = row
    except OSError:
        return out

    if not per_check:
        return out

    checks = sorted(per_check.values(), key=lambda r: str(r.get("check_timestamp", "")))
    if len(checks) > max_points:
        checks = checks[-max_points:]

    median_vals = [_to_float(r.get("median_abs_drift_pct")) for r in checks]
    max_vals = [_to_float(r.get("max_abs_drift_pct")) for r in checks]
    threshold = _to_float(checks[-1].get("threshold_pct"))

    plot_vals = [v for v in (median_vals + max_vals) if v is not None]
    if threshold is not None:
        plot_vals.append(threshold)
    if not plot_vals:
        return out

    y_min = min(plot_vals)
    y_max = max(plot_vals)
    if y_max <= y_min:
        y_max = y_min + 1.0

    out.update({
        "has_data": True,
        "point_count": len(checks),
        "from_timestamp": checks[0].get("check_timestamp"),
        "to_timestamp": checks[-1].get("check_timestamp"),
        "latest_median_abs_drift_pct": _to_float(checks[-1].get("median_abs_drift_pct")),
        "latest_max_abs_drift_pct": _to_float(checks[-1].get("max_abs_drift_pct")),
        "latest_threshold_pct": threshold,
        "latest_is_stale": checks[-1].get("is_stale"),
        "median_polyline": _polyline_points(median_vals),
        "max_polyline": _polyline_points(max_vals),
        "threshold_y": _y_for_value(threshold, y_min, y_max) if threshold is not None else None,
        "y_min": y_min,
        "y_max": y_max,
    })
    return out


def get_training_data_summary() -> dict | None:
    """Summary stats from all_specs_training_data.csv."""
    if not os.path.exists(ALL_SPECS_CSV):
        return None
    cached = _read_cached(ALL_SPECS_CSV)
    if cached is not None:
        return cached
    with open(ALL_SPECS_CSV, "rb") as f:
        total_rows = sum(1 for _ in f) - 1
    return _store_cache(ALL_SPECS_CSV, {"total_rows": max(0, total_rows), "path": ALL_SPECS_CSV})


def _build_profile_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not os.path.isdir(SPEC_PROFILES_DIR):
        return lookup
    for fname in os.listdir(SPEC_PROFILES_DIR):
        if not fname.endswith(".simc"):
            continue
        path = os.path.join(SPEC_PROFILES_DIR, fname)
        stem = fname[:-5]
        lookup.setdefault(stem, path)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                head = f.read(2048)
        except OSError:
            continue
        match = re.search(r'^\s*\w+\s*=\s*"?([^"\s]+)"?\s*$', head, re.MULTILINE)
        if match:
            lookup.setdefault(match.group(1), path)
    return lookup


def _resolve_profile_path(profile_name: str) -> str | None:
    global _profile_lookup_cache
    if _profile_lookup_cache is None:
        _profile_lookup_cache = _build_profile_lookup()
    path = _profile_lookup_cache.get(profile_name)
    if path:
        return path
    for key, value in _profile_lookup_cache.items():
        if key.startswith(profile_name) or profile_name.startswith(key):
            return value
    return None


def get_profile_talent_info(profile_name: str) -> dict[str, str | None]:
    path = _resolve_profile_path(profile_name)
    if not path or not os.path.exists(path):
        return {
            "profile_file": None,
            "talent_string": None,
            "talent_embed_url": None,
        }
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return {
            "profile_file": None,
            "talent_string": None,
            "talent_embed_url": None,
        }
    match = re.search(r'^talents=(.+)$', text, re.MULTILINE)
    talents = match.group(1).strip() if match else None
    embed_url = None
    if talents:
        encoded = urllib.parse.quote(talents, safe="")
        embed_url = f"https://www.raidbots.com/simbot/render/talents/{encoded}?width=700&level=90&hideExport=1"
    return {
        "profile_file": os.path.basename(path),
        "talent_string": talents,
        "talent_embed_url": embed_url,
    }


def get_profile_warnings(limit: int = 20) -> list[dict]:
    """Collect recent profile compatibility warnings from logs.

    Sources:
    - local/web_jobs/*.log (runtime initialization errors)
    - local/generator_mismatch_log.jsonl (compatibility events)
    """
    warnings: list[dict] = []
    seen: set[tuple[str, str]] = set()

    init_re = re.compile(r"Initialization error: Player '([^']+)':\s*(.+)")
    ws_dir = os.path.join(LOCAL_DIR, "web_jobs")
    if os.path.isdir(ws_dir):
        logs = sorted(
            (
                os.path.join(ws_dir, name)
                for name in os.listdir(ws_dir)
                if name.endswith(".log")
            ),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for log_path in logs[:20]:
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue
            for line in reversed(lines[-1200:]):
                m = init_re.search(line)
                if not m:
                    continue
                player = m.group(1).strip()
                message = m.group(2).strip()
                profile = re.sub(r"_\d+$", "", player)
                key = (profile, message)
                if key in seen:
                    continue
                seen.add(key)
                warnings.append({
                    "profile_name": profile,
                    "player_name": player,
                    "message": message,
                    "source": "job-log",
                    **get_profile_talent_info(profile),
                })
                if len(warnings) >= limit:
                    return warnings

    mismatch_log = os.path.join(LOCAL_DIR, "generator_mismatch_log.jsonl")
    if os.path.exists(mismatch_log):
        try:
            with open(mismatch_log, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            lines = []
        for line in reversed(lines[-400:]):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            profile = str(row.get("profile_name", "")).strip()
            event = str(row.get("event", "")).strip()
            msg = str(row.get("error_preview", "")).strip() or event
            if not profile or not msg:
                continue
            key = (profile, msg)
            if key in seen:
                continue
            seen.add(key)
            warnings.append({
                "profile_name": profile,
                "player_name": profile,
                "message": msg,
                "source": "compat-log",
                **get_profile_talent_info(profile),
            })
            if len(warnings) >= limit:
                return warnings

    return warnings


def format_spec_name(raw: str) -> str:
    """Transform 'MID1_Death_Knight_Frost_Rider' -> 'Death Knight Frost (Rider)'."""
    name = raw
    if name.startswith("MID1_"):
        name = name[5:]
    parts = name.split("_")
    # Rejoin with spaces
    return " ".join(parts).replace("  ", " ")


def get_model_snapshots() -> list[dict]:
    """List all model snapshots, most recent first.

    Each entry includes metrics from snapshot_info.json (or falls back to
    nn_metadata.json in the snapshot directory).
    """
    if not os.path.isdir(SNAPSHOTS_DIR):
        return []

    snapshots = []
    for dirname in os.listdir(SNAPSHOTS_DIR):
        snap_dir = os.path.join(SNAPSHOTS_DIR, dirname)
        if not os.path.isdir(snap_dir):
            continue

        info_path = os.path.join(snap_dir, "snapshot_info.json")
        meta_path = os.path.join(snap_dir, "nn_metadata.json")

        info = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Fall back to nn_metadata.json for missing fields
        if os.path.exists(meta_path) and (not info or not info.get("per_spec_mae")):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if not info:
                    info = {
                        "test_mae": meta.get("test_mae"),
                        "train_mae": meta.get("train_mae"),
                        "test_rmse": meta.get("test_rmse"),
                        "train_r2": meta.get("train_r2"),
                        "test_r2": meta.get("test_r2"),
                        "n_specs": meta.get("n_specs"),
                        "train_samples": meta.get("train_samples"),
                        "test_samples": meta.get("test_samples"),
                        "incremental": meta.get("incremental"),
                        "per_spec_mae": meta.get("per_spec_mae", {}),
                    }
                else:
                    info.setdefault("per_spec_mae", meta.get("per_spec_mae", {}))
            except (json.JSONDecodeError, OSError):
                pass

        snapshots.append({
            "dirname": dirname,
            "path": snap_dir,
            "timestamp": info.get("timestamp", ""),
            "label": info.get("label", ""),
            "type": info.get("type", ""),
            "test_mae": info.get("test_mae"),
            "train_mae": info.get("train_mae"),
            "test_rmse": info.get("test_rmse"),
            "train_r2": info.get("train_r2"),
            "test_r2": info.get("test_r2"),
            "n_specs": info.get("n_specs"),
            "train_samples": info.get("train_samples"),
            "test_samples": info.get("test_samples"),
            "per_spec_mae": info.get("per_spec_mae", {}),
            "incremental": info.get("incremental", False),
        })

    # Sort by dirname descending (timestamp prefix ensures chronological order)
    snapshots.sort(key=lambda s: s["dirname"], reverse=True)
    return snapshots
