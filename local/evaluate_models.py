"""
Model evaluation and visualization for trained DPS prediction neural nets.
Loads trained models and Optuna study data to produce detailed analysis plots.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import joblib
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(SCRIPT_DIR, "nn_website_model", "deep_nn")
EVAL_DIR = os.path.join(SCRIPT_DIR, "nn_website_model", "evaluation_plots")
DATA_FILE = os.path.join(PROJECT_ROOT, "all_specs_training_data.csv")
TEST_SIZE = 0.15
RANDOM_STATE = 42


def load_metadata():
    path = os.path.join(MODEL_DIR, "nn_metadata.json")
    with open(path) as f:
        return json.load(f)


def load_study_csv():
    path = os.path.join(MODEL_DIR, "nn_optuna_study.csv")
    return pd.read_csv(path)


def load_data_and_split():
    df = pd.read_csv(DATA_FILE)
    base_cols = ["primary_stat", "crit", "haste", "mastery", "versatility"]
    dummies = pd.get_dummies(df["spec"], prefix="spec", dtype=np.float32)
    feature_df = pd.concat([df[base_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    feature_cols = list(feature_df.columns)
    X = feature_df.values.astype(np.float32)
    y = df["dps"].values
    specs = df["spec"].values

    X_train, X_test, y_train, y_test, specs_train, specs_test = train_test_split(
        X, y, specs, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, specs_train, specs_test, feature_cols


def load_production_model(metadata, X_test, y_test, feature_cols):
    """Load the production .pt model and scalers, return predictions."""
    import torch
    import torch.nn as nn

    scalers = joblib.load(metadata["scalers_path"])
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    checkpoint = torch.load(metadata["model_path"], map_location="cpu", weights_only=False)
    hidden_dims = checkpoint["hidden_dims"]
    params = checkpoint["params"]
    activation_name = params.get("activation", "relu")
    dropout = float(params.get("dropout", 0.0))

    def build_net(input_dim, hidden_dims, activation_name, dropout):
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if activation_name == "gelu":
                layers.append(nn.GELU())
            elif activation_name == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    model = build_net(len(feature_cols), hidden_dims, activation_name, dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_test_scaled)).numpy()
    y_pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    return y_pred


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_optimization_history(study_df, out_dir):
    """Trial value over time with running best."""
    completed = study_df[study_df["state"] == "COMPLETE"].copy()
    completed = completed.sort_values("number")
    vals = completed["value"].values
    running_best = np.minimum.accumulate(vals)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(completed["number"], vals, s=8, alpha=0.5, label="Trial MAE")
    ax.plot(completed["number"], running_best, color="red", linewidth=1.5, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV MAE (DPS)")
    ax.set_title("Optuna Optimization History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "optimization_history.png"), dpi=150)
    plt.close(fig)


def plot_param_importances(study_df, out_dir):
    """Correlation-based parameter importance (Spearman rank with trial value)."""
    completed = study_df[study_df["state"] == "COMPLETE"].copy()
    param_cols = [c for c in completed.columns if c.startswith("params_")]
    numeric_params = []
    for col in param_cols:
        try:
            completed[col] = pd.to_numeric(completed[col], errors="raise")
            numeric_params.append(col)
        except (ValueError, TypeError):
            pass

    importances = {}
    for col in numeric_params:
        corr = completed[[col, "value"]].corr(method="spearman").iloc[0, 1]
        importances[col.replace("params_", "")] = abs(corr)

    if not importances:
        return

    params_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in params_sorted]
    vals = [p[1] for p in params_sorted]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    ax.barh(names, vals)
    ax.set_xlabel("|Spearman correlation| with CV MAE")
    ax.set_title("Hyperparameter Importance (rank correlation)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "param_importances.png"), dpi=150)
    plt.close(fig)


def plot_param_distributions(study_df, out_dir):
    """Value distributions for each param, colored by trial quality."""
    completed = study_df[study_df["state"] == "COMPLETE"].copy()
    param_cols = [c for c in completed.columns if c.startswith("params_")]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    last_used = -1
    for i, col in enumerate(sorted(param_cols)):
        if i >= len(axes):
            break
        last_used = i
        ax = axes[i]
        pname = col.replace("params_", "")
        try:
            vals = pd.to_numeric(completed[col], errors="raise")
            scatter = ax.scatter(vals, completed["value"], s=8, alpha=0.5,
                                 c=completed["value"], cmap="viridis_r")
            ax.set_xlabel(pname)
            ax.set_ylabel("CV MAE")
        except (ValueError, TypeError):
            cats = completed[col].astype(str)
            unique_cats = sorted(cats.unique())
            positions = {c: i for i, c in enumerate(unique_cats)}
            x = cats.map(positions)
            ax.scatter(x, completed["value"], s=8, alpha=0.5,
                       c=completed["value"], cmap="viridis_r")
            ax.set_xticks(range(len(unique_cats)))
            ax.set_xticklabels(unique_cats, rotation=45, ha="right")
            ax.set_xlabel(pname)
            ax.set_ylabel("CV MAE")
        ax.set_title(pname)

    # Hide unused axes
    for j in range(last_used + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Parameter vs CV MAE", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "param_distributions.png"), dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_test, y_pred, title, out_dir, filename):
    """Scatter plot of actual vs predicted DPS."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, s=10, alpha=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", linewidth=1, color="red")
    ax.set_xlabel("Actual DPS")
    ax.set_ylabel("Predicted DPS")
    ax.set_title(f"{title}\nMAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.5f}")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


def plot_residuals(y_test, y_pred, title, out_dir, filename):
    """Residual plot: predicted vs error."""
    residuals = y_pred - y_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, s=8, alpha=0.4)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted DPS")
    ax.set_ylabel("Residual (pred - actual)")
    ax.set_title(f"Residuals vs Predicted — {title}")

    # Residual histogram
    ax = axes[1]
    ax.hist(residuals, bins=60, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Count")
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    ax.set_title(f"Residual Distribution — mean={mean_r:.1f}, std={std_r:.1f}")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


def plot_per_spec_error(y_test, y_pred, specs_test, out_dir):
    """Bar chart of MAE per spec, sorted worst to best."""
    unique_specs = sorted(set(specs_test))
    spec_maes = []
    for spec in unique_specs:
        mask = specs_test == spec
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        spec_maes.append((spec, mae, int(mask.sum())))

    spec_maes.sort(key=lambda x: x[1], reverse=True)
    names = [s[0].replace("MID1_", "") for s in spec_maes]
    maes = [s[1] for s in spec_maes]
    counts = [s[2] for s in spec_maes]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), maes)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Test MAE (DPS)")
    ax.set_title("Per-Spec Test MAE (production model)")
    ax.invert_yaxis()

    for i, (bar, cnt) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"n={cnt}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_spec_mae.png"), dpi=150)
    plt.close(fig)


def plot_error_by_dps_range(y_test, y_pred, out_dir):
    """MAE bucketed by DPS range."""
    df = pd.DataFrame({"actual": y_test, "pred": y_pred})
    df["error"] = np.abs(df["pred"] - df["actual"])
    df["bucket"] = pd.qcut(df["actual"], q=10, duplicates="drop")
    grouped = df.groupby("bucket", observed=True).agg(
        mae=("error", "mean"),
        count=("error", "count"),
        mean_dps=("actual", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(len(grouped))
    bars = ax1.bar(x, grouped["mae"], alpha=0.7, label="MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(r)}" for r in grouped["mean_dps"]], rotation=45, ha="right")
    ax1.set_xlabel("Mean DPS in bucket")
    ax1.set_ylabel("MAE (DPS)")
    ax1.set_title("MAE by DPS Range (deciles)")

    ax2 = ax1.twinx()
    ax2.plot(x, grouped["count"], "ro-", markersize=5, label="Sample count")
    ax2.set_ylabel("Sample count")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_by_dps_range.png"), dpi=150)
    plt.close(fig)


def plot_ensemble_comparison(metadata, out_dir):
    """Bar chart comparing production model vs top ensembles."""
    entries = [("Production (single)", metadata["test_mae"])]
    for ens in metadata.get("top_ensembles", []):
        strategy = ens.get("strategy", "weighted" if ens.get("weighted") else "equal")
        trials_str = "+".join(str(t) for t in ens["trials"])
        label = f"[{strategy}] {trials_str}"
        entries.append((label, ens["test_mae"]))

    if len(entries) <= 1:
        return

    names = [e[0] for e in entries]
    maes = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    colors = ["#2196F3"]
    for ens in metadata.get("top_ensembles", []):
        strategy = ens.get("strategy", "weighted" if ens.get("weighted") else "equal")
        if strategy == "equal":
            colors.append("#4CAF50")
        elif strategy == "global_inverse_mae":
            colors.append("#FF9800")
        elif strategy == "spec_inverse_mae":
            colors.append("#9C27B0")
        else:
            colors.append("#E91E63")
    ax.barh(range(len(names)), maes, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Test MAE (DPS)")
    ax.set_title("Production Model vs Top Ensembles")
    ax.invert_yaxis()

    for i, v in enumerate(maes):
        ax.text(v + 2, i, f"{v:.1f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ensemble_comparison.png"), dpi=150)
    plt.close(fig)


def plot_ensemble_specialists(metadata, out_dir):
    """Heatmap showing which candidate model specializes in each spec."""
    if not metadata.get("spec_specialists"):
        return

    specialists = metadata["spec_specialists"]
    if not specialists:
        return

    spec_names = list(specialists.keys())
    trial_ids = sorted(set(s["trial_number"] for s in specialists.values()))

    if not trial_ids:
        return

    spec_matrix = np.zeros((len(spec_names), len(trial_ids)))
    for i, spec in enumerate(spec_names):
        for j, trial_id in enumerate(trial_ids):
            if specialists[spec]["trial_number"] == trial_id:
                spec_matrix[i, j] = specialists[spec]["test_mae"]
            else:
                spec_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(6, len(trial_ids) * 0.6), max(8, len(spec_names) * 0.35)))
    im = ax.imshow(spec_matrix, cmap="RdYlGn_r", aspect="auto")

    spec_labels = [s.replace("MID1_", "") for s in spec_names]
    ax.set_xticks(range(len(trial_ids)))
    ax.set_xticklabels([f"Trial {t}" for t in trial_ids], rotation=45, ha="right")
    ax.set_yticks(range(len(spec_names)))
    ax.set_yticklabels(spec_labels, fontsize=8)
    ax.set_title("Specialist Models: Which Trial Best Predicts Each Spec")
    ax.set_xlabel("Candidate Trial")
    ax.set_ylabel("Spec")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Test MAE (DPS)")

    for i in range(len(spec_names)):
        for j in range(len(trial_ids)):
            if not np.isnan(spec_matrix[i, j]):
                text = ax.text(j, i, f"{spec_matrix[i, j]:.0f}",
                              ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ensemble_specialists.png"), dpi=150)
    plt.close(fig)


def plot_candidate_model_wins(metadata, out_dir):
    """Bar chart showing which specs each candidate model specializes in."""
    if not metadata.get("ensemble_candidates"):
        return

    candidates = metadata["ensemble_candidates"]
    wins_data = {}
    for cand in candidates:
        trial_num = cand["trial_number"]
        spec_wins = cand.get("spec_wins", [])
        wins_data[f"Trial {trial_num}"] = len(spec_wins)

    if not wins_data:
        return

    names = list(wins_data.keys())
    win_counts = list(wins_data.values())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    bars = ax.bar(range(len(names)), win_counts, color="#2196F3")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Number of Specs Specialized In")
    ax.set_title("Candidate Model Specialization Count")
    ax.set_ylim(0, max(win_counts) + 1 if win_counts else 1)

    for i, (bar, cnt) in enumerate(zip(bars, win_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{cnt}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "candidate_specialization.png"), dpi=150)
    plt.close(fig)


def plot_ensemble_composition(metadata, out_dir):
    """Show composition and strategy of top 3 ensembles."""
    if not metadata.get("top_ensembles"):
        return

    top_ensembles = metadata["top_ensembles"][:3]
    if not top_ensembles:
        return

    fig, axes = plt.subplots(1, len(top_ensembles), figsize=(5 * len(top_ensembles), 5))
    if len(top_ensembles) == 1:
        axes = [axes]

    for idx, ens in enumerate(top_ensembles):
        ax = axes[idx]
        strategy = ens.get("strategy", "unknown")
        trials = ens["trials"]
        mae = ens["test_mae"]
        spec_spec = ens.get("spec_specialists", {})

        ax.text(0.5, 0.95, f"Strategy: {strategy}", ha="center", va="top",
                transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.text(0.5, 0.85, f"Size: {ens['size']} models | MAE: {mae:.1f} DPS",
                ha="center", va="top", transform=ax.transAxes, fontsize=10)

        trials_text = f"Trials: {', '.join(str(t) for t in trials)}"
        ax.text(0.5, 0.75, trials_text, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, family="monospace")

        if spec_spec:
            spec_text = f"Specialists: {len(spec_spec)} specs assigned"
        else:
            spec_text = "No per-spec routing"
        ax.text(0.5, 0.60, spec_text, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, style="italic")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.suptitle("Top 3 Ensembles: Configuration & Strategy", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ensemble_composition.png"), dpi=150)
    plt.close(fig)


def plot_pct_error_distribution(y_test, y_pred, out_dir):
    """Histogram of percentage errors."""
    pct_errors = 100.0 * (y_pred - y_test) / np.maximum(y_test, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pct_errors, bins=80, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    median_pct = np.median(np.abs(pct_errors))
    p95 = np.percentile(np.abs(pct_errors), 95)
    ax.set_xlabel("% Error ((pred-actual)/actual × 100)")
    ax.set_ylabel("Count")
    ax.set_title(f"Percentage Error Distribution — median |%err|={median_pct:.2f}%, 95th={p95:.2f}%")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pct_error_distribution.png"), dpi=150)
    plt.close(fig)


def print_summary_table(metadata, y_test, y_pred, specs_test):
    """Print a compact text summary table."""
    print(f"\n{'='*70}")
    print("MODEL EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Production trial: #{metadata['production_trial_number']}")
    print(f"  Optuna best trial: #{metadata['optuna_cv_best_trial_number']}")
    print(f"  Hidden dims: {metadata['hidden_dims']}")
    print(f"  Activation: {metadata['best_params'].get('activation', 'N/A')}")
    print(f"  Epochs: {metadata['best_params'].get('epochs', 'N/A')}")
    print(f"  LR: {metadata['best_params'].get('learning_rate', 'N/A')}")
    print(f"  Dropout: {metadata['best_params'].get('dropout', 'N/A')}")

    print(f"\n  {'Metric':<25} {'Train':>12} {'Test':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'MAE (DPS)':<25} {metadata['train_mae']:>12.2f} {metadata['test_mae']:>12.2f}")
    print(f"  {'RMSE (DPS)':<25} {metadata['train_rmse']:>12.2f} {metadata['test_rmse']:>12.2f}")
    print(f"  {'R²':<25} {metadata['train_r2']:>12.5f} {metadata['test_r2']:>12.5f}")

    pct_errors = 100.0 * np.abs(y_pred - y_test) / np.maximum(y_test, 1.0)
    print(f"\n  Percentage error (test set):")
    print(f"    Median: {np.median(pct_errors):.2f}%")
    print(f"    Mean:   {np.mean(pct_errors):.2f}%")
    print(f"    95th:   {np.percentile(pct_errors, 95):.2f}%")
    print(f"    99th:   {np.percentile(pct_errors, 99):.2f}%")

    # Per-spec summary (top 5 worst, top 5 best)
    unique_specs = sorted(set(specs_test))
    spec_maes = []
    for spec in unique_specs:
        mask = specs_test == spec
        if mask.sum() > 0:
            spec_maes.append((spec.replace("MID1_", ""), mean_absolute_error(y_test[mask], y_pred[mask])))
    spec_maes.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  Top 5 hardest specs (highest MAE):")
    for name, mae in spec_maes[:5]:
        print(f"    {name:<45} {mae:>8.1f} DPS")
    print(f"  Top 5 easiest specs (lowest MAE):")
    for name, mae in spec_maes[-5:]:
        print(f"    {name:<45} {mae:>8.1f} DPS")

    if metadata.get("top_ensembles"):
        print(f"\n  Top 3 ensembles:")
        for i, ens in enumerate(metadata["top_ensembles"][:3]):
            strategy = ens.get("strategy", "weighted" if ens.get("weighted") else "equal")
            trials_str = "+".join(str(t) for t in ens["trials"])
            print(f"    {i+1}. [{strategy:16s}] size={ens['size']} "
                  f"trials=[{trials_str}] MAE={ens['test_mae']:.2f}")

    if metadata.get("spec_specialists"):
        print(f"\n  Hard-spec specialists:")
        for name, mae in spec_maes[:5]:
            spec_key = f"MID1_{name}" if not name.startswith("MID1_") else name
            specialist = metadata["spec_specialists"].get(spec_key)
            if specialist:
                print(
                    f"    {name:<45} trial #{specialist['trial_number']:>4} "
                    f"best-model MAE={specialist['test_mae']:.1f}"
                )


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("Loading metadata and study data...")
    metadata = load_metadata()
    study_df = load_study_csv()

    print("Loading data and splitting...")
    X_train, X_test, y_train, y_test, specs_train, specs_test, feature_cols = load_data_and_split()

    print("Loading production model and generating predictions...")
    y_pred = load_production_model(metadata, X_test, y_test, feature_cols)

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\nGenerating evaluation plots...")

    plot_optimization_history(study_df, EVAL_DIR)
    print("  [OK] optimization_history.png")

    plot_param_importances(study_df, EVAL_DIR)
    print("  [OK] param_importances.png")

    plot_param_distributions(study_df, EVAL_DIR)
    print("  [OK] param_distributions.png")

    plot_actual_vs_predicted(y_test, y_pred, "Production Model", EVAL_DIR, "actual_vs_predicted.png")
    print("  [OK] actual_vs_predicted.png")

    plot_residuals(y_test, y_pred, "Production Model", EVAL_DIR, "residuals.png")
    print("  [OK] residuals.png")

    plot_per_spec_error(y_test, y_pred, specs_test, EVAL_DIR)
    print("  [OK] per_spec_mae.png")

    plot_error_by_dps_range(y_test, y_pred, EVAL_DIR)
    print("  [OK] error_by_dps_range.png")

    plot_pct_error_distribution(y_test, y_pred, EVAL_DIR)
    print("  [OK] pct_error_distribution.png")

    plot_ensemble_comparison(metadata, EVAL_DIR)
    print("  [OK] ensemble_comparison.png")

    plot_ensemble_specialists(metadata, EVAL_DIR)
    print("  [OK] ensemble_specialists.png")

    plot_candidate_model_wins(metadata, EVAL_DIR)
    print("  [OK] candidate_specialization.png")

    plot_ensemble_composition(metadata, EVAL_DIR)
    print("  [OK] ensemble_composition.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(metadata, y_test, y_pred, specs_test)

    print(f"\n{'='*70}")
    print(f"All plots saved to: {EVAL_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
