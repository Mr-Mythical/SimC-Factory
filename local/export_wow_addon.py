"""
Export the trained PyTorch DPS model to Lua tables for Mr. Mythical: DPS Predictor.

Usage:
  python export_wow_addon.py

Outputs:
  wow_addon/SimcDpsPredictor/ModelData.lua
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch


import math


SCRIPT_DIR = Path(__file__).resolve().parent
NN_DIR = SCRIPT_DIR / "nn_website_model" / "deep_nn"
MODEL_PATH = NN_DIR / "dps_net.pt"
SCALER_PATH = NN_DIR / "scalers.pkl"
META_PATH = NN_DIR / "nn_metadata.json"
MIN_ENSEMBLE_REPORT_PATH = NN_DIR / "minimal_ensemble_report.json"
OUT_PATH = SCRIPT_DIR / "wow_addon" / "SimcDpsPredictor" / "ModelData.lua"


def _fmt(x: float) -> str:
    return f"{float(x):.8g}"


def _lua_array(values, indent: str = "") -> str:
    parts = ["{"]
    line = []
    for idx, value in enumerate(values, start=1):
        line.append(_fmt(value))
        if len(line) >= 8:
            parts.append(indent + "  " + ", ".join(line) + ",")
            line = []
    if line:
        parts.append(indent + "  " + ", ".join(line) + ",")
    parts.append(indent + "}")
    return "\n".join(parts)


def _lua_matrix(matrix, indent: str = "") -> str:
    rows = [indent + "{"]
    for row in matrix:
        rows.append(indent + "  " + _lua_array(row, indent + "  ").replace("\n", "\n" + indent + "  ") + ",")
    rows.append(indent + "}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# IIFE-based writers — each numeric literal ends up in the IIFE's own
# constant pool, not the file-level chunk, so we stay under Lua 5.1's
# 65,535-constants-per-chunk limit.
# ---------------------------------------------------------------------------
_ROWS_PER_CHUNK_TARGET = 30_000   # max constants per IIFE (well under 65,535)


def _write_array_iife(lines: list[str], var_expr: str, values: list[float]) -> None:
    """Emit a 1-D array using an IIFE to isolate its constants."""
    vals = ",".join(_fmt(v) for v in values)
    lines.append(";(function()")
    lines.append(f"  {var_expr} = {{{vals}}}")
    lines.append("end)()")


def _write_matrix_iife(lines: list[str], var_expr: str, matrix: list[list[float]]) -> None:
    """Emit a 2-D weight matrix using one IIFE per row-chunk (~30k constants each)."""
    if not matrix:
        lines.append(f"{var_expr} = {{}}")
        return
    n_cols = len(matrix[0])
    chunk_size = max(1, _ROWS_PER_CHUNK_TARGET // n_cols)
    lines.append(f"{var_expr} = {{}}")
    for start in range(0, len(matrix), chunk_size):
        chunk = matrix[start : start + chunk_size]
        lines.append(";(function()")
        lines.append(f"  local _t = {var_expr}")
        for j, row in enumerate(chunk, start=start + 1):
            vals = ",".join(_fmt(v) for v in row)
            lines.append(f"  _t[{j}] = {{{vals}}}")
        lines.append("end)()")


BN_EPS = 1e-5


def _prebake_first_layer(payload: dict[str, Any], feature_names: list[str],
                         x_mean: list[float], x_scale: list[float]) -> None:
    """Fold spec one-hot features into first-layer bias for each spec.

    For a given spec, the one-hot columns are constant (one is 1, rest 0).
    We precompute their contribution (including scaler transform) and merge
    it into the bias, then keep only the stat columns in the weight matrix.
    Result: layer-1 input drops from 55 to 5 features at runtime.
    """
    stat_indices: list[int] = []
    spec_features: dict[str, int] = {}          # spec_key -> absolute index
    for i, name in enumerate(feature_names):
        if name.startswith("spec_"):
            spec_features[name[5:]] = i          # strip "spec_" prefix
        else:
            stat_indices.append(i)

    first = payload["layers"][0]
    W = first["w"]                               # [hidden][input_size]
    b = first["b"]                               # [hidden]
    hidden = len(W)

    # Stat-only weight columns for layer 1
    first["w"] = [[W[n][j] for j in stat_indices] for n in range(hidden)]

    # Base contribution: all spec features = 0 (after scaling)
    base = [0.0] * hidden
    for n in range(hidden):
        for j_abs in spec_features.values():
            base[n] += W[n][j_abs] * (-x_mean[j_abs] / x_scale[j_abs])

    # Per-spec bias: original bias + base + active-spec delta
    prebaked: dict[str, list[float]] = {}
    for spec_key, spec_idx in spec_features.items():
        inv_scale = 1.0 / x_scale[spec_idx]
        prebaked[spec_key] = [
            b[n] + base[n] + W[n][spec_idx] * inv_scale
            for n in range(hidden)
        ]

    payload["prebaked"] = prebaked


def _precompute_bn(bn_w, bn_b, bn_rm, bn_rv):
    """Precompute BatchNorm into a simple scale + offset: y = scale * x + offset.

    This avoids sqrt and division at Lua runtime.
    Original: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Precomputed: scale = gamma / sqrt(var + eps), offset = beta - scale * mean
    """
    scale = []
    offset = []
    for i in range(len(bn_w)):
        s = bn_w[i] / math.sqrt(bn_rv[i] + BN_EPS)
        scale.append(s)
        offset.append(bn_b[i] - s * bn_rm[i])
    return scale, offset


def _checkpoint_to_model_payload(checkpoint: dict[str, Any], trial_number: int | None = None, per_spec_mae: dict[str, float] | None = None, test_mae: float | None = None) -> dict[str, Any]:
    state_dict = checkpoint["state_dict"]
    hidden_dims = checkpoint["hidden_dims"]

    # Each hidden layer occupies 4 Sequential indices: Linear, Act, BN, Dropout
    # Layer i: Linear at 4*i, BN at 4*i+2.  Output Linear at 4*n_layers.
    n_layers = len(hidden_dims)

    layers = []
    for i in range(n_layers):
        lin_idx = 4 * i
        bn_idx = 4 * i + 2
        w = state_dict[f"{lin_idx}.weight"].tolist()
        b = state_dict[f"{lin_idx}.bias"].tolist()
        bn_w = state_dict[f"{bn_idx}.weight"].tolist()
        bn_b = state_dict[f"{bn_idx}.bias"].tolist()
        bn_rm = state_dict[f"{bn_idx}.running_mean"].tolist()
        bn_rv = state_dict[f"{bn_idx}.running_var"].tolist()

        # Precompute BN into scale + offset (avoids sqrt at runtime)
        bn_scale, bn_offset = _precompute_bn(bn_w, bn_b, bn_rm, bn_rv)
        layers.append({"w": w, "b": b, "bn_s": bn_scale, "bn_o": bn_offset})

    output_idx = 4 * n_layers
    out_w = state_dict[f"{output_idx}.weight"].tolist()[0]
    out_b = float(state_dict[f"{output_idx}.bias"].tolist()[0])

    return {
        "trial_number": trial_number,
        "test_mae": test_mae,
        "layers": layers,
        "output": {"w": out_w, "b": out_b},
        "per_spec_mae": per_spec_mae or {},
    }


def _write_model_block(lines: list[str], model_var_name: str, model_payload: dict[str, Any]) -> None:
    """Write a model definition using assignment statements + IIFEs.

    Each weight matrix and each 1-D parameter array goes into its own IIFE so
    that no single Lua chunk exceeds the 65,535-constant limit.
    """
    lines.append(f"{model_var_name} = {{}}")
    if model_payload.get("trial_number") is not None:
        lines.append(f"{model_var_name}.trial_number = {int(model_payload['trial_number'])}")
    if model_payload.get("test_mae") is not None:
        lines.append(f"{model_var_name}.test_mae = {_fmt(float(model_payload['test_mae']))}")

    prebaked = model_payload.get("prebaked")

    lines.append(f"{model_var_name}.layers = {{}}")
    for i, layer in enumerate(model_payload["layers"], start=1):
        lvar = f"{model_var_name}.layers[{i}]"
        lines.append(f"{lvar} = {{}}")
        _write_matrix_iife(lines, f"{lvar}.w", layer["w"])
        if i == 1 and prebaked:
            pass   # layer-1 bias replaced by per-spec prebaked bias
        else:
            _write_array_iife(lines, f"{lvar}.b", layer["b"])
        _write_array_iife(lines, f"{lvar}.bn_s", layer["bn_s"])
        _write_array_iife(lines, f"{lvar}.bn_o", layer["bn_o"])

    if prebaked:
        lines.append(f"{model_var_name}.prebaked = {{}}")
        for spec_key in sorted(prebaked.keys()):
            _write_array_iife(lines, f'{model_var_name}.prebaked["{spec_key}"]', prebaked[spec_key])

    lines.append(f"{model_var_name}.output = {{}}")
    _write_array_iife(lines, f"{model_var_name}.output.w", model_payload["output"]["w"])
    lines.append(f"{model_var_name}.output.b = {_fmt(model_payload['output']['b'])}")

    lines.append(f"{model_var_name}.per_spec_mae = {{}}")
    for spec_name, spec_mae in sorted(model_payload.get("per_spec_mae", {}).items()):
        lines.append(f"{model_var_name}.per_spec_mae[\"{spec_name}\"] = {_fmt(float(spec_mae))}")

    lines.append("")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scalers: {SCALER_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    feature_names = checkpoint["feature_names"]

    scalers = joblib.load(SCALER_PATH)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    metadata = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))

    single_model_payload = _checkpoint_to_model_payload(
        checkpoint,
        trial_number=int(metadata.get("production_trial_number", -1)) if metadata else None,
        test_mae=float(metadata.get("test_mae", 0.0)) if metadata else None,
    )

    # Optional compact ensemble payload (recommended_size_4 from minimal ensemble report).
    deployment: dict[str, Any] = {"mode": "single"}
    ensemble_models: list[dict[str, Any]] = []
    report = {}
    if MIN_ENSEMBLE_REPORT_PATH.exists():
        report = json.loads(MIN_ENSEMBLE_REPORT_PATH.read_text(encoding="utf-8"))

    candidate_map: dict[int, dict[str, Any]] = {}
    for c in metadata.get("ensemble_candidates", []) if metadata else []:
        candidate_map[int(c["trial_number"])] = c

    selected_ensemble = report.get("recommended_size_4") if report else None
    if selected_ensemble and selected_ensemble.get("trials"):
        trials = [int(t) for t in selected_ensemble.get("trials", [])]
        strategy = str(selected_ensemble.get("strategy", "equal"))
        spec_specialists = selected_ensemble.get("spec_specialists") or {}
        for trial in trials:
            candidate = candidate_map.get(trial)
            if not candidate:
                continue
            ckpt_path = Path(candidate["model_path"])
            if not ckpt_path.exists():
                continue
            ckp = torch.load(ckpt_path, map_location="cpu")
            ensemble_models.append(
                _checkpoint_to_model_payload(
                    ckp,
                    trial_number=trial,
                    per_spec_mae={k: float(v) for k, v in candidate.get("per_spec_mae", {}).items()},
                    test_mae=float(candidate.get("test_mae", 0.0)),
                )
            )

        if ensemble_models:
            deployment = {
                "mode": "ensemble",
                "strategy": strategy,
                "trials": trials,
                "spec_specialists": spec_specialists,
                "source": "recommended_size_4",
                "size": len(ensemble_models),
            }

    feature_index = {name: i + 1 for i, name in enumerate(feature_names)}
    spec_feature_names = [name for name in feature_names if name.startswith("spec_")]
    stat_feature_names = [name for name in feature_names if not name.startswith("spec_")]

    # Pre-bake spec one-hot features into first-layer bias for each model.
    x_mean = x_scaler.mean_.tolist()
    x_sc = x_scaler.scale_.tolist()
    _prebake_first_layer(single_model_payload, feature_names, x_mean, x_sc)
    for emp in ensemble_models:
        _prebake_first_layer(emp, feature_names, x_mean, x_sc)

    lines = []
    lines.append("local M = {}")
    lines.append("")
    lines.append("M.model_version = \"v5\"")
    lines.append(f"M.input_size = {len(feature_names)}")
    lines.append(f"M.n_stat_features = {len(stat_feature_names)}")
    lines.append("")

    lines.append("M.feature_names = {")
    for name in feature_names:
        lines.append(f"  \"{name}\",")
    lines.append("}")
    lines.append("")

    lines.append("M.feature_index = {")
    for name, idx in feature_index.items():
        lines.append(f"  [\"{name}\"] = {idx},")
    lines.append("}")
    lines.append("")

    lines.append("M.spec_feature_names = {")
    for name in spec_feature_names:
        lines.append(f"  \"{name}\",")
    lines.append("}")
    lines.append("")

    lines.append("M.scaler = {}")
    _write_array_iife(lines, "M.scaler.x_mean",  x_scaler.mean_.tolist())
    _write_array_iife(lines, "M.scaler.x_scale", x_scaler.scale_.tolist())
    lines.append(f"M.scaler.y_mean = {_fmt(float(y_scaler.mean_[0]))}")
    lines.append(f"M.scaler.y_scale = {_fmt(float(y_scaler.scale_[0]))}")
    lines.append("")

    _write_model_block(lines, "M.single_model", single_model_payload)

    lines.append("M.deployment = {")
    lines.append(f"  mode = \"{deployment.get('mode', 'single')}\",")
    lines.append(f"  strategy = \"{deployment.get('strategy', 'equal')}\",")
    lines.append(f"  source = \"{deployment.get('source', 'production_model')}\",")
    lines.append(f"  size = {int(deployment.get('size', 1))},")
    lines.append("  trials = {")
    for t in deployment.get("trials", []):
        lines.append(f"    {int(t)},")
    lines.append("  },")
    lines.append("  spec_specialists = {")
    for spec_name, trial in sorted((deployment.get("spec_specialists") or {}).items()):
        lines.append(f"    [\"{spec_name}\"] = {int(trial)},")
    lines.append("  },")
    lines.append("}")
    lines.append("")

    lines.append("M.ensemble_models = {}")
    for model_idx, model_payload in enumerate(ensemble_models, start=1):
        _write_model_block(lines, f"M.ensemble_models[{model_idx}]", model_payload)
    lines.append("")

    if metadata:
        best_params = metadata.get("best_params", {})
        lines.append("M.training = {")
        lines.append(f"  study_name = \"{metadata.get('study_name', '')}\",")
        lines.append(f"  production_trial = {int(metadata.get('production_trial_number', -1))},")
        lines.append(f"  cv_mae = {_fmt(metadata.get('cv_mae', 0.0))},")
        lines.append(f"  test_mae = {_fmt(metadata.get('test_mae', 0.0))},")
        lines.append(f"  activation = \"{best_params.get('activation', 'relu')}\",")
        lines.append("}")
        lines.append("")

    lines.append("-- Backward compatibility aliases for older addon code paths.")
    lines.append("M.layers = M.single_model.layers")
    lines.append("M.output = M.single_model.output")
    lines.append("")

    lines.append("_G.SimcDpsModelData = M")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
