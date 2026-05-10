"""
Find a compact ensemble with near-best MAE using saved candidate models.

This script evaluates greedy ensemble growth paths for multiple strategies and then
selects the smallest ensemble that is within a configurable tolerance of the best
MAE found.

Usage:
  python evaluate_minimal_ensemble.py
  python evaluate_minimal_ensemble.py --tol-pct 0.5 --max-size 8 --top-k 20

Outputs:
  nn_website_model/deep_nn/minimal_ensemble_report.json
  nn_website_model/deep_nn/minimal_ensemble_paths.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "nn_website_model", "deep_nn")
META_PATH = os.path.join(MODEL_DIR, "nn_metadata.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scalers.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "minimal_ensemble_report.json")
PATHS_CSV_PATH = os.path.join(MODEL_DIR, "minimal_ensemble_paths.csv")

# Keep these aligned with optimize_ensemble.py defaults.
TEST_SIZE = 0.15
RANDOM_STATE = 42
GPU_ID = 0


def _import_torch_modules():
    import importlib

    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    return torch, nn


def _load_dataset():
    from optimize_ensemble import DATA_FILE, prepare_features, load_data

    df = load_data(DATA_FILE)
    X, y, _feature_names = prepare_features(df)
    specs = df["spec"].values

    split = train_test_split(
        X,
        y,
        specs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    _X_train, X_test, _y_train, y_test, _specs_train, specs_test = split
    return X_test, y_test, specs_test


def _build_model(nn: Any, input_dim: int, hidden_dims: list[int], activation_name: str, dropout: float):
    layers: list[Any] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if activation_name == "gelu":
            layers.append(nn.GELU())
        elif activation_name == "silu":
            layers.append(nn.SiLU())
        else:
            layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


@dataclass
class Candidate:
    trial_number: int
    test_mae: float
    per_spec_mae: dict[str, float]
    model_path: str


@dataclass
class EvalContext:
    y_test: np.ndarray
    specs_test: np.ndarray
    trial_numbers: list[int]
    preds: np.ndarray  # shape: (n_test, n_models)
    maes: np.ndarray   # shape: (n_models,)
    per_spec_mae: list[dict[str, float]]


def _load_candidates(top_k: int) -> list[Candidate]:
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    candidates_raw = list(meta.get("ensemble_candidates", []))
    if not candidates_raw:
        raise ValueError("No ensemble_candidates found in nn_metadata.json")

    candidates_raw.sort(key=lambda c: float(c.get("test_mae", float("inf"))))
    candidates_raw = candidates_raw[:top_k]

    out: list[Candidate] = []
    for c in candidates_raw:
        out.append(
            Candidate(
                trial_number=int(c["trial_number"]),
                test_mae=float(c["test_mae"]),
                per_spec_mae={k: float(v) for k, v in c.get("per_spec_mae", {}).items()},
                model_path=str(c["model_path"]),
            )
        )
    return out


def _predict_candidates(candidates: list[Candidate], X_test: np.ndarray) -> EvalContext:
    torch, nn = _import_torch_modules()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{GPU_ID}" if use_cuda else "cpu")

    scalers = joblib.load(SCALER_PATH)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)

    # Need y/specs from deterministic split.
    _X_test_ref, y_test, specs_test = _load_dataset()

    pred_cols: list[np.ndarray] = []
    trial_numbers: list[int] = []
    maes: list[float] = []
    per_spec_mae: list[dict[str, float]] = []

    for c in candidates:
        checkpoint = torch.load(c.model_path, map_location=device)
        hidden_dims = checkpoint["hidden_dims"]
        params = checkpoint.get("params", {})
        activation = str(params.get("activation", "relu"))
        dropout = float(params.get("dropout", 0.0))

        model = _build_model(
            nn=nn,
            input_dim=X_test_scaled.shape[1],
            hidden_dims=hidden_dims,
            activation_name=activation,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy().reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

        pred_cols.append(pred)
        trial_numbers.append(c.trial_number)
        maes.append(float(mean_absolute_error(y_test, pred)))
        per_spec_mae.append(c.per_spec_mae)

    all_preds = np.column_stack(pred_cols)

    return EvalContext(
        y_test=np.asarray(y_test),
        specs_test=np.asarray(specs_test),
        trial_numbers=trial_numbers,
        preds=all_preds,
        maes=np.asarray(maes),
        per_spec_mae=per_spec_mae,
    )


def _eval_combo(ctx: EvalContext, combo: list[int], strategy: str) -> tuple[np.ndarray, float, dict[str, int] | None]:
    combo_preds = ctx.preds[:, combo]
    combo_maes = ctx.maes[combo]

    if strategy == "equal":
        pred = np.mean(combo_preds, axis=1)
        return pred, float(mean_absolute_error(ctx.y_test, pred)), None

    if strategy == "global_inverse_mae":
        inv = 1.0 / np.maximum(combo_maes, 1e-8)
        w = inv / inv.sum()
        pred = (combo_preds * w[np.newaxis, :]).sum(axis=1)
        return pred, float(mean_absolute_error(ctx.y_test, pred)), None

    spec_names = sorted(np.unique(ctx.specs_test))

    if strategy == "spec_inverse_mae":
        pred = np.empty_like(ctx.y_test, dtype=float)
        specialists: dict[str, int] = {}
        for spec in spec_names:
            mask = ctx.specs_test == spec
            if not np.any(mask):
                continue
            spec_maes = np.array([
                max(ctx.per_spec_mae[idx].get(str(spec), float("inf")), 1e-8)
                for idx in combo
            ])
            inv = 1.0 / spec_maes
            w = inv / inv.sum()
            pred[mask] = (combo_preds[mask] * w[np.newaxis, :]).sum(axis=1)
            specialists[str(spec)] = int(ctx.trial_numbers[combo[int(np.argmax(w))]])
        return pred, float(mean_absolute_error(ctx.y_test, pred)), specialists

    if strategy == "spec_router":
        pred = np.empty_like(ctx.y_test, dtype=float)
        specialists = {}
        for spec in spec_names:
            mask = ctx.specs_test == spec
            if not np.any(mask):
                continue
            spec_maes = np.array([
                max(ctx.per_spec_mae[idx].get(str(spec), float("inf")), 1e-8)
                for idx in combo
            ])
            best_local = int(np.argmin(spec_maes))
            pred[mask] = combo_preds[mask, best_local]
            specialists[str(spec)] = int(ctx.trial_numbers[combo[best_local]])
        return pred, float(mean_absolute_error(ctx.y_test, pred)), specialists

    raise ValueError(f"Unknown strategy: {strategy}")


def _greedy_path(ctx: EvalContext, strategy: str, max_size: int) -> list[dict[str, Any]]:
    selected: list[int] = []
    remaining = set(range(len(ctx.trial_numbers)))
    path: list[dict[str, Any]] = []

    for size in range(1, max_size + 1):
        best_cand = None
        best_mae = float("inf")
        best_spec = None

        for idx in remaining:
            combo = selected + [idx]
            _pred, mae, spec_map = _eval_combo(ctx, combo, strategy)
            if mae < best_mae:
                best_mae = mae
                best_cand = idx
                best_spec = spec_map

        if best_cand is None:
            break

        selected.append(best_cand)
        remaining.remove(best_cand)

        path.append(
            {
                "strategy": strategy,
                "size": len(selected),
                "trial_indices": list(selected),
                "trials": [int(ctx.trial_numbers[i]) for i in selected],
                "test_mae": float(best_mae),
                "spec_specialists": best_spec,
            }
        )

    return path


def _pareto_min_size(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep only points where no smaller-or-equal size has lower-or-equal MAE.
    out: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda x: (x["size"], x["test_mae"])):
        dominated = False
        for q in out:
            if q["size"] <= r["size"] and q["test_mae"] <= r["test_mae"]:
                dominated = True
                break
        if not dominated:
            out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate compact ensemble candidates")
    parser.add_argument("--top-k", type=int, default=20, help="How many candidate models to include")
    parser.add_argument("--max-size", type=int, default=10, help="Max ensemble size for greedy search")
    parser.add_argument(
        "--tol-pct",
        type=float,
        default=0.5,
        help="Acceptable percent MAE above global best when choosing minimal-size recommendation",
    )
    args = parser.parse_args()

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing metadata: {META_PATH}")

    # 1) Load deterministic test split and candidates.
    X_test, y_test, specs_test = _load_dataset()
    candidates = _load_candidates(top_k=args.top_k)
    print(f"Loaded {len(candidates)} candidates from metadata")

    # 2) Predict each candidate on test split and build evaluation context.
    ctx = _predict_candidates(candidates, X_test)
    ctx = EvalContext(
        y_test=np.asarray(y_test),
        specs_test=np.asarray(specs_test),
        trial_numbers=ctx.trial_numbers,
        preds=ctx.preds,
        maes=ctx.maes,
        per_spec_mae=ctx.per_spec_mae,
    )

    # 3) Greedy growth path per strategy.
    strategies = ["equal", "global_inverse_mae", "spec_inverse_mae", "spec_router"]
    all_results: list[dict[str, Any]] = []
    for strategy in strategies:
        path = _greedy_path(ctx, strategy=strategy, max_size=min(args.max_size, len(ctx.trial_numbers)))
        all_results.extend(path)

    if not all_results:
        raise RuntimeError("No ensemble results generated")

    # 4) Pick best MAE and then smallest model count within tolerance.
    best = min(all_results, key=lambda r: r["test_mae"])
    mae_cutoff = float(best["test_mae"] * (1.0 + args.tol_pct / 100.0))
    eligible = [r for r in all_results if r["test_mae"] <= mae_cutoff]
    recommended = min(eligible, key=lambda r: (r["size"], r["test_mae"]))

    # 5) Pareto frontier for size vs MAE transparency.
    pareto = _pareto_min_size(all_results)

    # 6) Explicit fixed-size pick for addon deployment simplicity.
    size4_candidates = [r for r in all_results if int(r["size"]) == 4]
    recommended_size_4 = min(size4_candidates, key=lambda r: r["test_mae"]) if size4_candidates else None

    report = {
        "top_k_candidates": int(args.top_k),
        "max_size": int(args.max_size),
        "tolerance_percent": float(args.tol_pct),
        "best_overall": best,
        "recommended_minimal": recommended,
        "recommended_size_4": recommended_size_4,
        "mae_cutoff": mae_cutoff,
        "pareto_front": pareto,
        "all_results": sorted(all_results, key=lambda r: (r["test_mae"], r["size"])),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(PATHS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "size", "test_mae", "trials"])
        for r in sorted(all_results, key=lambda x: (x["strategy"], x["size"])):
            writer.writerow([r["strategy"], r["size"], f"{r['test_mae']:.6f}", "+".join(str(t) for t in r["trials"])])

    print("\nMinimal Ensemble Evaluation")
    print("=" * 70)
    print(f"Best overall: strategy={best['strategy']} size={best['size']} mae={best['test_mae']:.3f}")
    print(
        "Recommended minimal: "
        f"strategy={recommended['strategy']} size={recommended['size']} mae={recommended['test_mae']:.3f} "
        f"(within {args.tol_pct:.2f}% of best)"
    )
    print("Trials:", "+".join(str(t) for t in recommended["trials"]))
    if recommended_size_4 is not None:
        print(
            "Recommended size-4: "
            f"strategy={recommended_size_4['strategy']} size=4 mae={recommended_size_4['test_mae']:.3f}"
        )
        print("Trials:", "+".join(str(t) for t in recommended_size_4["trials"]))
    print(f"\nWrote report: {REPORT_PATH}")
    print(f"Wrote paths csv: {PATHS_CSV_PATH}")


if __name__ == "__main__":
    main()
