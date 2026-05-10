"""
Offline test harness for Mr. Mythical: DPS Predictor logic.

What this validates:
1. Exported model/scalers can be loaded.
2. Addon-equivalent forward pass (Linear -> ReLU -> BatchNorm) runs end-to-end.
3. Item replacement delta logic returns sensible output.
4. Numpy forward (addon math) matches torch state_dict forward.

Run:
  python test_addon_offline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
NN_DIR = ROOT / "nn_website_model" / "deep_nn"
MODEL_PATH = NN_DIR / "dps_net.pt"
SCALER_PATH = NN_DIR / "scalers.pkl"


@dataclass
class RuntimeModel:
    feature_names: list[str]
    feature_index: dict[str, int]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    layers: list[dict[str, np.ndarray]]
    out_w: np.ndarray
    out_b: float


def load_runtime_model() -> RuntimeModel:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    sd = ckpt["state_dict"]
    feature_names = list(ckpt["feature_names"])
    hidden_dims = list(ckpt.get("hidden_dims") or [])

    scalers = joblib.load(SCALER_PATH)
    xs = scalers["x_scaler"]
    ys = scalers["y_scaler"]

    def arr(name: str) -> np.ndarray:
        return sd[name].detach().cpu().numpy().astype(np.float64)

    layers = []
    for layer_idx in range(len(hidden_dims)):
        linear_idx = 4 * layer_idx
        batch_norm_idx = linear_idx + 2
        layers.append({
            "w": arr(f"{linear_idx}.weight"),
            "b": arr(f"{linear_idx}.bias"),
            "bn_w": arr(f"{batch_norm_idx}.weight"),
            "bn_b": arr(f"{batch_norm_idx}.bias"),
            "bn_rm": arr(f"{batch_norm_idx}.running_mean"),
            "bn_rv": arr(f"{batch_norm_idx}.running_var"),
        })

    output_idx = 4 * len(hidden_dims)

    return RuntimeModel(
        feature_names=feature_names,
        feature_index={k: i for i, k in enumerate(feature_names)},
        x_mean=np.asarray(xs.mean_, dtype=np.float64),
        x_scale=np.asarray(xs.scale_, dtype=np.float64),
        y_mean=float(ys.mean_[0]),
        y_scale=float(ys.scale_[0]),
        layers=layers,
        out_w=arr(f"{output_idx}.weight")[0],
        out_b=float(arr(f"{output_idx}.bias")[0]),
    )


def build_feature_vector(model: RuntimeModel, stats: Mapping[str, float], spec_key: str) -> np.ndarray:
    x = np.zeros((len(model.feature_names),), dtype=np.float64)
    x[model.feature_index["primary_stat"]] = float(stats.get("primary_stat", 0.0))
    x[model.feature_index["crit"]] = float(stats.get("crit", 0.0))
    x[model.feature_index["haste"]] = float(stats.get("haste", 0.0))
    x[model.feature_index["mastery"]] = float(stats.get("mastery", 0.0))
    x[model.feature_index["versatility"]] = float(stats.get("versatility", 0.0))

    spec_feature = f"spec_{spec_key}"
    if spec_feature in model.feature_index:
        x[model.feature_index[spec_feature]] = 1.0
    else:
        raise KeyError(f"Unknown spec key: {spec_key}")

    return x


def addon_forward_numpy(model: RuntimeModel, features: np.ndarray, eps: float = 1e-5) -> float:
    x = (features - model.x_mean) / model.x_scale

    # Mirrors addon order exactly: Linear -> ReLU -> BatchNorm
    for layer in model.layers:
        x = layer["w"] @ x + layer["b"]
        x = np.maximum(x, 0.0)
        x = layer["bn_w"] * ((x - layer["bn_rm"]) / np.sqrt(layer["bn_rv"] + eps)) + layer["bn_b"]

    y_scaled = float(model.out_w @ x + model.out_b)
    y = y_scaled * model.y_scale + model.y_mean
    return float(y)


def torch_forward_reference(model: RuntimeModel, features: np.ndarray) -> float:
    # Build tensors from model arrays to compare math path.
    x = torch.from_numpy(((features - model.x_mean) / model.x_scale).astype(np.float64))

    for layer in model.layers:
        w = torch.from_numpy(layer["w"])
        b = torch.from_numpy(layer["b"])
        bn_w = torch.from_numpy(layer["bn_w"])
        bn_b = torch.from_numpy(layer["bn_b"])
        bn_rm = torch.from_numpy(layer["bn_rm"])
        bn_rv = torch.from_numpy(layer["bn_rv"])

        x = w.matmul(x) + b
        x = torch.relu(x)
        x = bn_w * ((x - bn_rm) / torch.sqrt(bn_rv + 1e-5)) + bn_b

    out_w = torch.from_numpy(model.out_w)
    y_scaled = float(out_w.dot(x) + model.out_b)
    y = y_scaled * model.y_scale + model.y_mean
    return float(y)


def add_stats(a: Mapping[str, float], b: Mapping[str, float], sign: int) -> dict[str, float]:
    return {
        "primary_stat": a.get("primary_stat", 0.0) + sign * b.get("primary_stat", 0.0),
        "crit": a.get("crit", 0.0) + sign * b.get("crit", 0.0),
        "haste": a.get("haste", 0.0) + sign * b.get("haste", 0.0),
        "mastery": a.get("mastery", 0.0) + sign * b.get("mastery", 0.0),
        "versatility": a.get("versatility", 0.0) + sign * b.get("versatility", 0.0),
    }


def predict_item_delta(
    model: RuntimeModel,
    base_stats: Mapping[str, float],
    new_item_stats: Mapping[str, float],
    equipped_stats_candidates: Sequence[Mapping[str, float]],
    spec_key: str,
) -> dict[str, float]:
    base_pred = addon_forward_numpy(model, build_feature_vector(model, base_stats, spec_key))

    best_delta = None
    best_new = None
    best_slot = None
    for slot_idx, eq_stats in enumerate(equipped_stats_candidates, start=1):
        with_removed = add_stats(base_stats, eq_stats, -1)
        with_new = add_stats(with_removed, new_item_stats, 1)
        pred_new = addon_forward_numpy(model, build_feature_vector(model, with_new, spec_key))
        delta = pred_new - base_pred

        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_new = pred_new
            best_slot = slot_idx

    if best_delta is None or best_new is None or best_slot is None:
        raise ValueError("No equipped options were provided for item delta evaluation")

    return {
        "dps_base": base_pred,
        "dps_new": float(best_new),
        "dps_delta": float(best_delta),
        "slot_choice": int(best_slot),
    }


def main() -> None:
    model = load_runtime_model()

    spec_key = "MID1_Mage_Frost"

    # Mock player and inventory values to simulate addon logic without WoW runtime.
    base_stats = {
        "primary_stat": 30500.0,
        "crit": 12500.0,
        "haste": 11800.0,
        "mastery": 9800.0,
        "versatility": 7600.0,
    }

    # New item being hovered (example ring/trinket-like stat budget)
    new_item = {
        "primary_stat": 2100.0,
        "crit": 780.0,
        "haste": 620.0,
        "mastery": 400.0,
        "versatility": 350.0,
    }

    # Two currently equipped alternatives for same slot group.
    equipped_options = [
        {"primary_stat": 2000.0, "crit": 650.0, "haste": 540.0, "mastery": 380.0, "versatility": 280.0},
        {"primary_stat": 1960.0, "crit": 720.0, "haste": 500.0, "mastery": 330.0, "versatility": 320.0},
    ]

    # 1) Forward parity check: addon math vs torch-math reference.
    feat = build_feature_vector(model, base_stats, spec_key)
    y_np = addon_forward_numpy(model, feat)
    y_ref = torch_forward_reference(model, feat)
    abs_diff = abs(y_np - y_ref)

    # 2) End-to-end item delta logic check.
    result = predict_item_delta(model, base_stats, new_item, equipped_options, spec_key)

    print("OFFLINE ADDON TEST")
    print("=" * 60)
    print(f"Model features: {len(model.feature_names)}")
    print(f"Spec used: {spec_key}")
    print()
    print("Forward parity check")
    print(f"  addon_numpy: {y_np:.4f}")
    print(f"  torch_ref  : {y_ref:.4f}")
    print(f"  abs diff   : {abs_diff:.8f}")
    if abs_diff > 1e-6:
        raise AssertionError(f"Forward mismatch too large: {abs_diff}")

    print()
    print("Item delta simulation")
    print(f"  base DPS   : {result['dps_base']:.2f}")
    print(f"  new DPS    : {result['dps_new']:.2f}")
    print(f"  delta DPS  : {result['dps_delta']:+.2f}")
    print(f"  best slot  : {result['slot_choice']}")

    # Basic sanity check: with stronger mock item, delta should usually be positive.
    if result["dps_delta"] < -500:
        raise AssertionError(f"Unexpectedly large negative delta: {result['dps_delta']}")

    print()
    print("PASS: Offline addon test completed successfully.")


if __name__ == "__main__":
    main()
