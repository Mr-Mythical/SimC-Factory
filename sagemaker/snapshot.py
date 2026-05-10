"""Snapshot the current model before overwriting with new training results.

Used by sagemaker/download_model.py and local/optimize_ensemble.py to
preserve every training run for comparison and rollback.
"""

import json
import os
import shutil
from datetime import datetime, timezone

MODEL_FILES = ["dps_net.pt", "scalers.pkl", "nn_metadata.json"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "local", "nn_website_model", "deep_nn")
DEFAULT_SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "local", "nn_website_model", "snapshots")


def snapshot_current_model(
    model_dir: str = DEFAULT_MODEL_DIR,
    snapshots_dir: str = DEFAULT_SNAPSHOTS_DIR,
    label: str = "pre_overwrite",
) -> str | None:
    """Copy current model files into a timestamped snapshot directory.

    Returns the snapshot directory path, or None if no model exists to snapshot.
    """
    metadata_path = os.path.join(model_dir, "nn_metadata.json")
    if not os.path.exists(metadata_path):
        return None

    # Read existing metadata for the snapshot dirname
    try:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        meta = {}

    test_mae = meta.get("test_mae")
    mae_tag = f"_mae{int(test_mae)}" if test_mae is not None else ""
    incremental = meta.get("incremental", False)
    type_tag = "incr" if incremental else "full"

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    dirname = f"{timestamp}_{type_tag}{mae_tag}"
    snap_dir = os.path.join(snapshots_dir, dirname)
    os.makedirs(snap_dir, exist_ok=True)

    # Copy model files
    copied = []
    for fname in MODEL_FILES:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(snap_dir, fname))
            copied.append(fname)

    if not copied:
        # Nothing to snapshot
        shutil.rmtree(snap_dir, ignore_errors=True)
        return None

    # Write snapshot_info.json
    info = {
        "timestamp": now.isoformat(),
        "label": label,
        "type": type_tag,
        "test_mae": meta.get("test_mae"),
        "train_mae": meta.get("train_mae"),
        "test_rmse": meta.get("test_rmse"),
        "train_r2": meta.get("train_r2"),
        "test_r2": meta.get("test_r2"),
        "n_specs": meta.get("n_specs"),
        "train_samples": meta.get("train_samples"),
        "test_samples": meta.get("test_samples"),
        "per_spec_mae": meta.get("per_spec_mae", {}),
        "cv_folds": meta.get("cv_folds"),
        "cv_mae": meta.get("cv_mae"),
        "incremental": incremental,
        "files": copied,
    }
    with open(os.path.join(snap_dir, "snapshot_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"  Snapshot saved: {snap_dir}")
    return snap_dir


def restore_snapshot(
    snapshot_dir: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    snapshots_dir: str = DEFAULT_SNAPSHOTS_DIR,
) -> str:
    """Restore a snapshot into the active model directory.

    Snapshots the current model first (so restore is reversible),
    then copies the snapshot files into model_dir.

    Returns the path of the pre-restore snapshot.
    """
    # Snapshot current state before restoring
    pre_restore = snapshot_current_model(
        model_dir=model_dir,
        snapshots_dir=snapshots_dir,
        label="pre_restore",
    )

    # Copy snapshot files into active model dir
    os.makedirs(model_dir, exist_ok=True)
    for fname in MODEL_FILES:
        src = os.path.join(snapshot_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(model_dir, fname))

    return pre_restore or ""
