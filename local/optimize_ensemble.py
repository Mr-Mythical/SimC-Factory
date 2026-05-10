"""
Deep neural network DPS prediction model optimization for website deployment.
Uses PyTorch with GPU acceleration and Bayesian hyperparameter optimization (Optuna).
Trained on multi-spec SimulationCraft data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import joblib
import json
import os
import time
import importlib
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FILE = os.path.join(PROJECT_ROOT, "all_specs_training_data.csv")
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
TEST_SIZE = 0.15
RANDOM_STATE = 42

# GPU configuration
GPU_ID = 0  # Which GPU to use (0 for your 3090 Ti)

# Neural net model configuration (PyTorch)
MODEL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "nn_website_model")
NN_N_TRIALS = 500
NN_CV_FOLDS = 3
NN_STUDY_NAME = "nn_dps_optimization_v4"
NN_STUDY_DB_FILE = "nn_optuna_study.db"

# Ensemble configuration: take top N models, build combos of sizes 2-5
ENSEMBLE_TOP_K = 20          # Number of best trials to retrain for ensemble
ENSEMBLE_MIN_SIZE = 2        # Smallest ensemble combination
ENSEMBLE_MAX_SIZE = 10        # Largest ensemble combination

# v4: Individual layer sizes for small-model search (replaces fixed layout list).
# Supports 2 or 3 hidden layers with individually tunable widths.

def create_training_data_csv_if_missing():
    """Combine individual spec CSV files into one training data CSV if it doesn't exist."""
    if os.path.exists(DATA_FILE):
        print(f"✓ Found existing {DATA_FILE}")
        return
    
    if not os.path.isdir(TRAINING_DATA_DIR):
        raise FileNotFoundError(f"Training data directory '{TRAINING_DATA_DIR}' not found")
    
    csv_files = sorted(glob.glob(os.path.join(TRAINING_DATA_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{TRAINING_DATA_DIR}'")
    
    print(f"\nCreating {DATA_FILE} from {len(csv_files)} spec CSVs...")
    dfs = []
    for csv_file in csv_files:
        spec_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)
        # Add spec column if not present
        if "spec" not in df.columns:
            df["spec"] = spec_name
        dfs.append(df)
        print(f"  ✓ Loaded {spec_name}: {len(df)} samples")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(DATA_FILE, index=False)
    print(f"✓ Created {DATA_FILE} with {len(combined_df)} total samples\n")


def load_data(path):
    """Load training data CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}")

    df = pd.read_csv(path)

    # Multi-spec training data with primary_stat and spec columns
    required_cols = ["primary_stat", "crit", "haste", "mastery", "versatility", "dps", "spec"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_specs = df["spec"].nunique()
    print(f"Loaded {len(df)} samples from {path}")
    print(f"  {n_specs} unique specs detected")
    return df


def prepare_features(df):
    """Extract feature matrix X and target vector y.

    One-hot encodes the 'spec' column and appends it to numeric stat features.
    """
    base_cols = ["primary_stat", "crit", "haste", "mastery", "versatility"]
    dummies = pd.get_dummies(df["spec"], prefix="spec", dtype=np.float32)
    feature_df = pd.concat([df[base_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    feature_cols = list(feature_df.columns)
    X = feature_df.values.astype(np.float32)

    y = df["dps"].values
    return X, y, feature_cols


def compute_regression_metrics(y_true, y_pred):
    """Compute standard regression metrics for prediction arrays."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_per_spec_mae(spec_labels, y_true, y_pred):
    """Compute MAE for each spec in the provided labels."""
    spec_labels_arr = np.asarray(spec_labels)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    per_spec = {}
    for spec_name in sorted(pd.unique(spec_labels_arr)):
        mask = spec_labels_arr == spec_name
        if np.any(mask):
            per_spec[spec_name] = float(mean_absolute_error(y_true_arr[mask], y_pred_arr[mask]))
    return per_spec


def trial_value_or_inf(trial):
    """Return a concrete float for trial value for robust typing and sorting."""
    return float("inf") if trial.value is None else float(trial.value)


def save_prediction_visualization(y_true, y_pred, label, out_dir):
    """Save actual-vs-predicted scatter plot for one tested model/ensemble."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print(f"matplotlib not installed; skipping visualization for {label}")
        return None

    metrics = compute_regression_metrics(y_true, y_pred)
    os.makedirs(out_dir, exist_ok=True)

    safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label.lower())
    plot_path = os.path.join(out_dir, f"pred_{safe_label}.png")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    min_val = float(min(np.min(y_true_arr), np.min(y_pred_arr)))
    max_val = float(max(np.max(y_true_arr), np.max(y_pred_arr)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true_arr, y_pred_arr, s=18, alpha=0.55)
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)
    ax.set_title(f"{label}\\nMAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.4f}")
    ax.set_xlabel("Actual DPS")
    ax.set_ylabel("Predicted DPS")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def save_all_prediction_visualizations(y_true, tested_predictions, out_dir):
    """Save one visualization per tested model/ensemble prediction vector."""
    saved_paths = {}
    for label, pred in tested_predictions.items():
        saved = save_prediction_visualization(y_true, pred, label, out_dir)
        if saved:
            saved_paths[label] = saved
    return saved_paths


def print_torch_cuda_confirmation():
    """Quick runtime confirmation for PyTorch/CUDA availability."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        print("PyTorch runtime: not installed (NN training will be skipped)")
        return

    print("PyTorch runtime confirmation:")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda_available: {torch.cuda.is_available()}")
    print(f"  torch_cuda_version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  cuda_device_count: {torch.cuda.device_count()}")
        print(f"  cuda_device_0: {torch.cuda.get_device_name(0)}")


def train_deep_neural_net(X_train, y_train, X_test, y_test, feature_names, specs_test):
    """Optimize and train a deep neural net regressor with Optuna (PyTorch backend)."""
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        torch_data = importlib.import_module("torch.utils.data")
        DataLoader = torch_data.DataLoader
        TensorDataset = torch_data.TensorDataset
    except ImportError:
        print("\nPyTorch is not installed. Skipping neural net training.")
        print("Install with: pip install torch")
        return None

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{GPU_ID}" if use_cuda else "cpu")

    print(f"\n{'='*70}")
    print("DEEP NEURAL NET OPTIMIZATION (OPTUNA + PYTORCH)")
    print(f"{'='*70}")
    print(f"Device: {device}")

    print("Neural Net Optimization Configuration:")
    print(f"  Optimization trials: {NN_N_TRIALS}")
    print(f"  Cross-validation folds: {NN_CV_FOLDS}")
    print("  Sampler: TPE (Tree-structured Parzen Estimator)")

    def build_dps_net(input_dim, hidden_dims, activation_name, dropout):
        layers = []
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

    def train_torch_model(X_t, y_t, params):
        train_dataset = TensorDataset(
            torch.from_numpy(X_t),
            torch.from_numpy(y_t),
        )
        train_loader = DataLoader(train_dataset, batch_size=int(params["batch_size"]), shuffle=True)

        hidden_dims = [int(params["hidden_dim_1"]), int(params["hidden_dim_2"])]
        if int(params.get("n_hidden_layers", 3)) >= 3:
            hidden_dims.append(int(params["hidden_dim_3"]))
        model = build_dps_net(
            input_dim=X_t.shape[1],
            hidden_dims=hidden_dims,
            activation_name=params["activation"],
            dropout=float(params["dropout"]),
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )

        model.train()
        for _ in range(int(params["epochs"])):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return model

    def suggest_nn_params(trial):
        return {
            "n_hidden_layers": trial.suggest_int("n_hidden_layers", 2, 2),  # Phase 2: locked to 2 layers
            "hidden_dim_1": trial.suggest_int("hidden_dim_1", 180, 280),    # Phase 2: narrowed from [32, 512]
            "hidden_dim_2": trial.suggest_int("hidden_dim_2", 150, 250),    # Phase 2: narrowed from [16, 256]
            "hidden_dim_3": trial.suggest_int("hidden_dim_3", 8, 128),      # Not used when n_layers=2
            "activation": trial.suggest_categorical("activation", ["relu"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.05),           # Phase 2: narrowed from [0.0, 0.3]
            "batch_size": trial.suggest_categorical("batch_size", [512]),
            "epochs": trial.suggest_int("epochs", 1000, 2000, step=20),      # Phase 2: narrowed from [100, 1500]
            "learning_rate": trial.suggest_float("learning_rate", 8e-5, 1.5e-4, log=True),  # Phase 2: narrowed from [1e-4, 1e-2]
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 5e-5, log=True),      # Phase 2: narrowed from [1e-9, 1e-3]
        }

    def objective(trial):
        params = suggest_nn_params(trial)
        cv = KFold(n_splits=NN_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_maes = []

        for train_idx, valid_idx in cv.split(X_train):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_valid = X_train[valid_idx]
            y_fold_valid = y_train[valid_idx]

            x_scaler_fold = StandardScaler()
            y_scaler_fold = StandardScaler()
            X_fold_train_scaled = x_scaler_fold.fit_transform(X_fold_train).astype(np.float32)
            X_fold_valid_scaled = x_scaler_fold.transform(X_fold_valid).astype(np.float32)
            y_fold_train_scaled = y_scaler_fold.fit_transform(y_fold_train.reshape(-1, 1)).astype(np.float32)

            try:
                fold_model = train_torch_model(X_fold_train_scaled, y_fold_train_scaled, params)
            except KeyboardInterrupt:
                raise
            except Exception:
                return float("inf")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            fold_model.eval()
            with torch.no_grad():
                y_valid_pred_scaled = fold_model(
                    torch.from_numpy(X_fold_valid_scaled).to(device)
                ).cpu().numpy()
            y_valid_pred = y_scaler_fold.inverse_transform(y_valid_pred_scaled).reshape(-1)
            mae = mean_absolute_error(y_fold_valid, y_valid_pred)
            if not np.isfinite(mae):
                return float("inf")
            fold_maes.append(mae)

        return float(np.mean(fold_maes))

    nn_dir = os.path.join(MODEL_OUTPUT_DIR, "deep_nn")
    os.makedirs(nn_dir, exist_ok=True)

    nn_study_db_path = os.path.abspath(os.path.join(nn_dir, NN_STUDY_DB_FILE))
    nn_study_storage = f"sqlite:///{nn_study_db_path.replace(os.sep, '/')}"
    nn_sampler = TPESampler(seed=RANDOM_STATE)
    nn_study = optuna.create_study(
        direction="minimize",
        sampler=nn_sampler,
        study_name=NN_STUDY_NAME,
        storage=nn_study_storage,
        load_if_exists=True,
    )

    completed_before = len([t for t in nn_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_before > 0:
        print(f"Resuming study '{NN_STUDY_NAME}' with {completed_before} completed trials")
    else:
        print(f"Created new study '{NN_STUDY_NAME}'")

    opt_start = time.time()
    nn_study.optimize(
        objective,
        n_trials=NN_N_TRIALS,
        show_progress_bar=True,
    )
    opt_elapsed = time.time() - opt_start

    print(f"\n✓ NN optimization complete in {opt_elapsed:.1f}s ({opt_elapsed/60:.1f} minutes)")
    print(f"Best NN trial: #{nn_study.best_trial.number}")
    print(f"Best NN CV MAE: {nn_study.best_value:.2f} DPS")
    print(f"\nBest parameters:")
    for param, value in sorted(nn_study.best_params.items()):
        print(f"  {param:20s}: {value}")

    completed_trials = [
        t for t in nn_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.params
    ]
    completed_trials.sort(key=trial_value_or_inf)
    top_k_trials = completed_trials[:ENSEMBLE_TOP_K]

    print(f"\nTop-{ENSEMBLE_TOP_K} trials for ensemble:")
    for t in top_k_trials:
        print(f"  trial #{t.number:4d} | CV MAE: {trial_value_or_inf(t):.2f}")

    best_params = nn_study.best_params.copy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)

    # ── Train all top-K models ───────────────────────────────────────────────
    print(f"\nTraining top {len(top_k_trials)} models on full training set...")
    trained_models = []  # list of (trial_number, model, test_pred, params, test_mae)
    candidate_dir = os.path.join(nn_dir, "ensemble_candidates")
    os.makedirs(candidate_dir, exist_ok=True)
    tested_predictions = {}

    train_start = time.time()
    for i, trial in enumerate(top_k_trials):
        t_params = trial.params.copy()
        model = train_torch_model(X_train_scaled, y_train_scaled, t_params)
        model.eval()
        with torch.no_grad():
            test_pred_scaled = model(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()
        test_pred = y_scaler.inverse_transform(test_pred_scaled).reshape(-1)
        t_mae = mean_absolute_error(y_test, test_pred)
        t_per_spec_mae = compute_per_spec_mae(specs_test, y_test, test_pred)
        trained_models.append((trial.number, model, test_pred, t_params, t_mae, t_per_spec_mae))
        tested_predictions[f"nn_trial_{trial.number}"] = test_pred

        # Save each candidate model
        candidate_path = os.path.join(candidate_dir, f"nn_trial_{trial.number}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "params": t_params,
            "hidden_dims": [int(t_params["hidden_dim_1"]), int(t_params["hidden_dim_2"])] + ([int(t_params["hidden_dim_3"])] if int(t_params.get("n_hidden_layers", 3)) >= 3 else []),
        }, candidate_path)

        print(f"  [{i+1}/{len(top_k_trials)}] trial #{trial.number:4d} "
              f"| CV MAE: {trial_value_or_inf(trial):.2f} | test MAE: {t_mae:.2f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_elapsed = time.time() - train_start

    spec_names = sorted(pd.unique(np.asarray(specs_test)))
    spec_masks = {spec_name: (np.asarray(specs_test) == spec_name) for spec_name in spec_names}
    spec_best_trial_by_spec = {}
    spec_wins_by_trial = {int(t[0]): [] for t in trained_models}
    for spec_name in spec_names:
        best_trial = min(trained_models, key=lambda t: t[5][spec_name])
        spec_best_trial_by_spec[spec_name] = {
            "trial_number": int(best_trial[0]),
            "test_mae": float(best_trial[5][spec_name]),
        }
        spec_wins_by_trial[int(best_trial[0])].append(spec_name)

    print("\nPer-spec specialists among top candidate models:")
    for spec_name in spec_names:
        specialist = spec_best_trial_by_spec[spec_name]
        print(
            f"  {spec_name:40s} -> trial #{specialist['trial_number']:4d} "
            f"(MAE: {specialist['test_mae']:.2f})"
        )

    # ── Select production model (lowest test MAE) ────────────────────────────
    production = min(trained_models, key=lambda x: x[4])
    prod_tn, prod_model, y_test_pred, prod_params, _, _ = production
    prod_model.eval()
    with torch.no_grad():
        train_pred_scaled = prod_model(torch.from_numpy(X_train_scaled).to(device)).cpu().numpy()
    y_train_pred = y_scaler.inverse_transform(train_pred_scaled).reshape(-1)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Snapshot existing model before overwriting
    from sagemaker.snapshot import snapshot_current_model
    snap = snapshot_current_model(model_dir=nn_dir, label="local_training")
    if snap:
        print(f"  Previous model snapshotted before overwrite")

    nn_model_path = os.path.join(nn_dir, "dps_net.pt")
    torch.save({
        "state_dict": prod_model.state_dict(),
        "feature_names": feature_names,
        "params": prod_params,
        "hidden_dims": [int(prod_params["hidden_dim_1"]), int(prod_params["hidden_dim_2"])] + ([int(prod_params["hidden_dim_3"])] if int(prod_params.get("n_hidden_layers", 3)) >= 3 else []),
    }, nn_model_path)

    scaler_path = os.path.join(nn_dir, "scalers.pkl")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, scaler_path)

    tested_predictions["nn_best"] = y_test_pred
    print(f"\nProduction model: trial #{prod_tn} (lowest test MAE: {test_mae:.2f})")

    # ── Ensemble search: global + spec-aware strategies ──────────────────────
    from itertools import combinations

    all_preds = np.column_stack([t[2] for t in trained_models])  # (n_test, k)
    all_maes = np.array([t[4] for t in trained_models])          # (k,)
    model_indices = list(range(len(trained_models)))

    ensemble_results = []  # list of dicts

    for size in range(ENSEMBLE_MIN_SIZE, min(ENSEMBLE_MAX_SIZE, len(trained_models)) + 1):
        for combo in combinations(model_indices, size):
            combo_preds = all_preds[:, list(combo)]
            combo_trial_nums = [trained_models[i][0] for i in combo]

            # Unweighted (equal average)
            uw_pred = np.mean(combo_preds, axis=1)
            uw_mae = mean_absolute_error(y_test, uw_pred)

            # Weighted by inverse test MAE
            combo_maes = all_maes[list(combo)]
            inv_maes = 1.0 / combo_maes
            weights = inv_maes / inv_maes.sum()
            w_pred = (combo_preds * weights[np.newaxis, :]).sum(axis=1)
            w_mae = mean_absolute_error(y_test, w_pred)

            ensemble_results.append({
                "trials": combo_trial_nums,
                "size": size,
                "strategy": "equal",
                "weighted": False,
                "test_mae": float(uw_mae),
                "predictions": uw_pred,
                "weights": [1.0 / size] * size,
                "spec_specialists": None,
            })
            ensemble_results.append({
                "trials": combo_trial_nums,
                "size": size,
                "strategy": "global_inverse_mae",
                "weighted": True,
                "test_mae": float(w_mae),
                "predictions": w_pred,
                "weights": weights.tolist(),
                "spec_specialists": None,
            })

            spec_weighted_pred = np.empty_like(y_test, dtype=float)
            spec_router_pred = np.empty_like(y_test, dtype=float)
            spec_weighted_specialists = {}
            spec_router_specialists = {}
            for spec_name in spec_names:
                mask = spec_masks[spec_name]
                if not np.any(mask):
                    continue

                spec_maes = np.array([
                    max(trained_models[i][5][spec_name], 1e-8)
                    for i in combo
                ], dtype=float)
                spec_inv_maes = 1.0 / spec_maes
                spec_weights = spec_inv_maes / spec_inv_maes.sum()
                spec_best_idx = int(np.argmin(spec_maes))

                spec_weighted_pred[mask] = (combo_preds[mask] * spec_weights[np.newaxis, :]).sum(axis=1)
                spec_router_pred[mask] = combo_preds[mask, spec_best_idx]
                spec_weighted_specialists[spec_name] = int(combo_trial_nums[int(np.argmax(spec_weights))])
                spec_router_specialists[spec_name] = int(combo_trial_nums[spec_best_idx])

            spec_weighted_mae = mean_absolute_error(y_test, spec_weighted_pred)
            spec_router_mae = mean_absolute_error(y_test, spec_router_pred)

            ensemble_results.append({
                "trials": combo_trial_nums,
                "size": size,
                "strategy": "spec_inverse_mae",
                "weighted": True,
                "test_mae": float(spec_weighted_mae),
                "predictions": spec_weighted_pred,
                "weights": None,
                "spec_specialists": spec_weighted_specialists,
            })
            ensemble_results.append({
                "trials": combo_trial_nums,
                "size": size,
                "strategy": "spec_router",
                "weighted": True,
                "test_mae": float(spec_router_mae),
                "predictions": spec_router_pred,
                "weights": None,
                "spec_specialists": spec_router_specialists,
            })

    ensemble_results.sort(key=lambda e: e["test_mae"])

    # Print top ensembles
    print(f"\nEnsemble search: {len(ensemble_results)} combinations evaluated")
    print("Top 10 ensembles:")
    for i, ens in enumerate(ensemble_results[:10]):
        trials_str = "+".join(str(t) for t in ens["trials"])
        print(f"  {i+1:2d}. [{ens['strategy']:16s}] size={ens['size']} "
              f"trials=[{trials_str}] test MAE: {ens['test_mae']:.2f}")

    # Save best ensemble as a prediction too
    best_ensemble = ensemble_results[0]
    tested_predictions[f"best_ensemble_{best_ensemble['strategy']}"] = best_ensemble["predictions"]

    # Store top 5 ensembles (without predictions) in metadata
    top_ensembles_meta = []
    for ens in ensemble_results[:5]:
        top_ensembles_meta.append({
            "trials": ens["trials"],
            "size": ens["size"],
            "strategy": ens["strategy"],
            "weighted": ens["weighted"],
            "weights": ens["weights"],
            "test_mae": ens["test_mae"],
            "spec_specialists": ens["spec_specialists"],
        })

    nn_study_path = os.path.join(nn_dir, "nn_optuna_study.csv")
    nn_study_df = nn_study.trials_dataframe()
    nn_study_df.to_csv(nn_study_path, index=False)
    print(f"Saved NN Optuna trials: {nn_study_path}")

    # Build individual candidate metadata
    ensemble_candidates = []
    for tn, _, _, t_params, t_mae, t_per_spec_mae in trained_models:
        trial_obj = next(t for t in top_k_trials if t.number == tn)
        ensemble_candidates.append({
            "trial_number": int(tn),
            "cv_mae": trial_value_or_inf(trial_obj),
            "test_mae": float(t_mae),
            "per_spec_mae": t_per_spec_mae,
            "spec_wins": spec_wins_by_trial[int(tn)],
            "params": t_params,
            "model_path": os.path.join(candidate_dir, f"nn_trial_{tn}.pt"),
        })

    nn_metadata = {
        "device": str(device),
        "used_gpu": bool(use_cuda),
        "best_params": prod_params,
        "hidden_dims": [int(prod_params["hidden_dim_1"]), int(prod_params["hidden_dim_2"])] + ([int(prod_params["hidden_dim_3"])] if int(prod_params.get("n_hidden_layers", 3)) >= 3 else []),
        "cv_mae": float(nn_study.best_value),
        "production_trial_number": int(prod_tn),
        "optuna_cv_best_trial_number": int(nn_study.best_trial.number),
        "optimization_trials": NN_N_TRIALS,
        "cv_folds": NN_CV_FOLDS,
        "optimization_time_seconds": float(opt_elapsed),
        "training_time_seconds": float(train_elapsed),
        "study_name": NN_STUDY_NAME,
        "study_storage": nn_study_storage,
        "completed_trials_total": len(completed_trials),
        "ensemble_top_k": ENSEMBLE_TOP_K,
        "ensemble_candidates": ensemble_candidates,
        "spec_specialists": spec_best_trial_by_spec,
        "top_ensembles": top_ensembles_meta,
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "model_path": nn_model_path,
        "scalers_path": scaler_path,
        "optuna_csv_path": nn_study_path,
    }

    nn_meta_path = os.path.join(nn_dir, "nn_metadata.json")
    with open(nn_meta_path, "w") as f:
        json.dump(nn_metadata, f, indent=2)

    print("\nNeural Net Performance:")
    print(f"  Train MAE: {train_mae:.2f} DPS, RMSE: {train_rmse:.2f} DPS, R²: {train_r2:.4f}")
    print(f"  Test MAE:  {test_mae:.2f} DPS, RMSE: {test_rmse:.2f} DPS, R²: {test_r2:.4f}")
    print(f"Saved NN model: {nn_model_path}")
    print(f"Saved NN scalers: {scaler_path}")
    print(f"Saved NN metadata: {nn_meta_path}")

    return {
        "metadata": nn_metadata,
        "test_predictions": y_test_pred,
        "tested_predictions": tested_predictions,
    }


def main():
    print("="*70)
    print("DPS PREDICTION NEURAL NETWORK - MULTI-SPEC TRAINING")
    print("="*70)
    print()
    
    # Create training data CSV if needed
    create_training_data_csv_if_missing()
    
    # Verify PyTorch/CUDA setup
    print_torch_cuda_confirmation()
    print()
    
    # Load data
    df = load_data(DATA_FILE)
    X, y, feature_names = prepare_features(df)
    specs = df["spec"].values
    
    # Split data
    X_train, X_test, y_train, y_test, _, specs_test = train_test_split(
        X, y, specs, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nDataset split:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Run neural net optimization
    print(f"\n{'#'*70}")
    print("RUNNING NEURAL NET HYPERPARAMETER OPTIMIZATION")
    print(f"{'#'*70}\n")
    
    nn_result = train_deep_neural_net(X_train, y_train, X_test, y_test, feature_names, specs_test)

    if nn_result is None:
        print("ERROR: Neural net training failed")
        return

    nn_metadata = nn_result["metadata"]
    all_tested_predictions = nn_result.get("tested_predictions", {})

    # Save evaluation visualizations
    viz_dir = os.path.join(MODEL_OUTPUT_DIR, "evaluation_plots")
    viz_paths = save_all_prediction_visualizations(y_test, all_tested_predictions, viz_dir)
    if viz_paths:
        print(f"\nSaved {len(viz_paths)} evaluation plot(s) in: {viz_dir}")
    
    # Summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nBest Neural Net Configuration:")
    for param, value in sorted(nn_metadata["best_params"].items()):
        print(f"  {param:20s}: {value}")
    
    print(f"\nNeural Net Performance:")
    print(f"  Train MAE:  {nn_metadata['train_mae']:.2f} DPS")
    print(f"  Train RMSE: {nn_metadata['train_rmse']:.2f} DPS")
    print(f"  Train R²:   {nn_metadata['train_r2']:.4f}")
    print(f"\n  Test MAE:   {nn_metadata['test_mae']:.2f} DPS")
    print(f"  Test RMSE:  {nn_metadata['test_rmse']:.2f} DPS")
    print(f"  Test R²:    {nn_metadata['test_r2']:.4f}")
    
    print(f"\nTraining Details:")
    print(f"  Total optimization time: {nn_metadata['optimization_time_seconds']:.1f}s ({nn_metadata['optimization_time_seconds']/60:.1f} minutes)")
    print(f"  Total training time: {nn_metadata['training_time_seconds']:.1f}s")
    print(f"  Used GPU: {nn_metadata['used_gpu']}")
    print(f"  Cross-validation MAE: {nn_metadata['cv_mae']:.2f} DPS")
    
    if nn_metadata.get("top_ensembles"):
        best_ens = nn_metadata["top_ensembles"][0]
        trials_str = "+".join(str(t) for t in best_ens["trials"])
        print(f"\nBest Ensemble:")
        print(f"  [{best_ens.get('strategy', 'equal')}] size={best_ens['size']} trials=[{trials_str}]")
        print(f"  Test MAE: {best_ens['test_mae']:.2f} DPS")

    if nn_metadata.get("spec_specialists"):
        print(f"\nSpec Specialists:")
        for spec_name, specialist in list(nn_metadata["spec_specialists"].items())[:5]:
            print(
                f"  {spec_name}: trial #{specialist['trial_number']} "
                f"(MAE {specialist['test_mae']:.2f})"
            )
    
    print(f"\nSaved files in '{MODEL_OUTPUT_DIR}/':")
    print(f"  - deep_nn/dps_net.pt (PyTorch production model)")
    print(f"  - deep_nn/scalers.pkl (feature/target scalers)")
    print(f"  - deep_nn/nn_metadata.json (metrics, hyperparameters, ensemble results)")
    print(f"  - deep_nn/nn_optuna_study.csv (Optuna trial history)")
    print(f"  - deep_nn/ensemble_candidates/ (top-{ENSEMBLE_TOP_K} retrained models)")
    print(f"  - evaluation_plots/ (prediction plots per model/ensemble)")
    print("\nRun evaluate_models.py for detailed visualizations and analysis.")


if __name__ == "__main__":
    main()
