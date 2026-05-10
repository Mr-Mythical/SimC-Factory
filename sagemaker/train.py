"""
SageMaker-compatible training script for the DPS prediction neural network.

Adapts the core training logic from local/optimize_ensemble.py to follow
SageMaker conventions:
  - Reads data from SM_CHANNEL_TRAINING
  - Reads hyperparameters via argparse (SageMaker passes SM_HP_* as CLI args)
  - Saves model artifacts to SM_MODEL_DIR
  - Saves/resumes checkpoints from /opt/ml/checkpoints/ (synced to S3)
  - Prints metrics to stdout for AMT regex capture

Trains a single model per invocation. SageMaker AMT handles hyperparameter
search externally by launching multiple training jobs with different params.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

CHECKPOINT_FILENAME = "checkpoint.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPS prediction network")

    # SageMaker environment (set automatically by SageMaker, with local defaults)
    parser.add_argument(
        "--training",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", ""),
        help="Path to the training data channel.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/opt/ml/checkpoints",
        help="Directory for checkpoints (SageMaker syncs to S3).",
    )

    # Hyperparameters
    parser.add_argument("--n-hidden-layers", type=int, default=3)
    parser.add_argument("--hidden-dim-1", type=int, default=256)
    parser.add_argument("--hidden-dim-2", type=int, default=128)
    parser.add_argument("--hidden-dim-3", type=int, default=64)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=0)

    return parser.parse_args()


def load_data(data_dir):
    """Load training data from the SageMaker data channel directory."""
    csv_path = os.path.join(data_dir, "all_specs_training_data.csv")
    if not os.path.exists(csv_path):
        # Fall back to finding any CSV in the directory
        csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        csv_path = os.path.join(data_dir, csvs[0])

    df = pd.read_csv(csv_path)
    required_cols = ["primary_stat", "crit", "haste", "mastery", "versatility", "dps", "spec"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  {df['spec'].nunique()} unique specs detected")
    return df


def prepare_features(df):
    """Extract feature matrix X and target vector y with one-hot encoded specs."""
    base_cols = ["primary_stat", "crit", "haste", "mastery", "versatility"]
    dummies = pd.get_dummies(df["spec"], prefix="spec", dtype=np.float32)
    feature_df = pd.concat(
        [df[base_cols].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1
    )
    feature_cols = list(feature_df.columns)
    X = feature_df.values.astype(np.float32)
    y = df["dps"].values
    return X, y, feature_cols


def build_dps_net(input_dim, hidden_dims, activation_name, dropout):
    """Build the feedforward DPS prediction network."""
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


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, x_scaler, y_scaler):
    """Save training state for spot interruption recovery."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_dir, model, optimizer, device):
    """Load training state from a checkpoint if it exists. Returns start epoch."""
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)
    if not os.path.exists(checkpoint_path):
        return 0, None, None

    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    x_scaler = checkpoint.get("x_scaler")
    y_scaler = checkpoint.get("y_scaler")
    print(f"  Resumed at epoch {start_epoch}")
    return start_epoch, x_scaler, y_scaler


def train(args):
    """Main training loop with checkpointing and metric reporting."""
    # Seed all RNGs for reproducibility
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    total_epochs = args.epochs
    print(f"\nFULL TRAINING MODE ({total_epochs} epochs)")

    # Load and split data
    df = load_data(args.training)
    X, y, feature_names = prepare_features(df)
    specs = df["spec"].values

    X_train, X_test, y_train, y_test, _, specs_test = train_test_split(
        X, y, specs, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}, Features: {X_train.shape[1]}")

    # Build model
    all_dims = [args.hidden_dim_1, args.hidden_dim_2, args.hidden_dim_3]
    hidden_dims = all_dims[:args.n_hidden_layers]
    model = build_dps_net(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        activation_name=args.activation,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Try to resume from checkpoint (spot interruption recovery)
    start_epoch, ckpt_x_scaler, ckpt_y_scaler = load_checkpoint(
        args.checkpoint_dir, model, optimizer, device
    )

    # Fit scalers (always refit on the full dataset since data may have changed)
    if ckpt_x_scaler is not None and ckpt_y_scaler is not None:
        x_scaler = ckpt_x_scaler
        y_scaler = ckpt_y_scaler
    else:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_scaler.fit(X_train)
        y_scaler.fit(y_train.reshape(-1, 1))

    X_train_scaled = x_scaler.transform(X_train).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop with checkpointing
    print(f"Training epochs {start_epoch}..{total_epochs - 1} (hidden_dims={hidden_dims})")
    train_start = time.time()

    model.train()
    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        # Checkpoint every 50 epochs (balance between safety and I/O overhead)
        if (epoch + 1) % 50 == 0 or epoch == total_epochs - 1:
            save_checkpoint(args.checkpoint_dir, epoch, model, optimizer, x_scaler, y_scaler)
            avg_loss = epoch_loss / max(batches, 1)
            print(f"  Epoch {epoch + 1}/{total_epochs} | loss={avg_loss:.6f} | checkpoint saved")

    train_elapsed = time.time() - train_start
    print(f"Training completed in {train_elapsed:.1f}s")

    # Evaluate on train and test sets
    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(torch.from_numpy(X_train_scaled).to(device)).cpu().numpy()
        test_pred_scaled = model(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()

    y_train_pred = y_scaler.inverse_transform(train_pred_scaled).reshape(-1)
    y_test_pred = y_scaler.inverse_transform(test_pred_scaled).reshape(-1)

    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    train_r2 = float(r2_score(y_train, y_train_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Print metrics for SageMaker AMT regex capture
    print(f"train_mae={train_mae:.4f};")
    print(f"test_mae={test_mae:.4f};")
    print(f"train_rmse={train_rmse:.4f};")
    print(f"test_rmse={test_rmse:.4f};")
    print(f"train_r2={train_r2:.6f};")
    print(f"test_r2={test_r2:.6f};")

    # Per-spec MAE
    specs_test_arr = np.asarray(specs_test)
    per_spec_mae = {}
    for spec_name in sorted(pd.unique(specs_test_arr)):
        mask = specs_test_arr == spec_name
        if np.any(mask):
            per_spec_mae[spec_name] = float(mean_absolute_error(y_test[mask], y_test_pred[mask]))

    # Cross-validation MAE
    cv_mae = None
    if args.cv_folds > 1:
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        fold_maes = []
        for train_idx, valid_idx in cv.split(X_train):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train_scaled[train_idx]
            X_fold_valid = X_train_scaled[valid_idx]
            y_fold_valid_orig = y_train[valid_idx]

            fold_model = build_dps_net(
                input_dim=X_train.shape[1],
                hidden_dims=hidden_dims,
                activation_name=args.activation,
                dropout=args.dropout,
            ).to(device)
            fold_optimizer = torch.optim.Adam(
                fold_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            fold_dataset = TensorDataset(
                torch.from_numpy(X_fold_train), torch.from_numpy(y_fold_train)
            )
            fold_loader = DataLoader(fold_dataset, batch_size=args.batch_size, shuffle=True)

            fold_model.train()
            for _ in range(total_epochs):
                for xb, yb in fold_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    fold_optimizer.zero_grad()
                    pred = fold_model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    fold_optimizer.step()

            fold_model.eval()
            with torch.no_grad():
                fold_pred_scaled = fold_model(
                    torch.from_numpy(X_fold_valid).to(device)
                ).cpu().numpy()
            fold_pred = y_scaler.inverse_transform(fold_pred_scaled).reshape(-1)
            fold_maes.append(float(mean_absolute_error(y_fold_valid_orig, fold_pred)))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cv_mae = float(np.mean(fold_maes))
        print(f"cv_mae={cv_mae:.4f};")

    # Save model artifacts to SM_MODEL_DIR
    os.makedirs(args.model_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, "dps_net.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "params": {
                "n_hidden_layers": args.n_hidden_layers,
                "hidden_dim_1": args.hidden_dim_1,
                "hidden_dim_2": args.hidden_dim_2,
                "hidden_dim_3": args.hidden_dim_3,
                "activation": args.activation,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "epochs": total_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "hidden_dims": hidden_dims,
        },
        model_path,
    )

    scaler_path = os.path.join(args.model_dir, "scalers.pkl")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, scaler_path)

    metadata = {
        "device": str(device),
        "used_gpu": torch.cuda.is_available(),
        "best_params": {
            "n_hidden_layers": args.n_hidden_layers,
            "hidden_dim_1": args.hidden_dim_1,
            "hidden_dim_2": args.hidden_dim_2,
            "hidden_dim_3": args.hidden_dim_3,
            "activation": args.activation,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": total_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "hidden_dims": hidden_dims,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_mae": cv_mae,
        "cv_folds": args.cv_folds,
        "per_spec_mae": per_spec_mae,
        "training_time_seconds": train_elapsed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": int(X_train.shape[1]),
        "n_specs": int(df["spec"].nunique()),
        "feature_names": feature_names,
    }

    meta_path = os.path.join(args.model_dir, "nn_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model artifacts to {args.model_dir}:")
    print(f"  dps_net.pt ({os.path.getsize(model_path) / 1024:.1f} KB)")
    print(f"  scalers.pkl ({os.path.getsize(scaler_path) / 1024:.1f} KB)")
    print(f"  nn_metadata.json ({os.path.getsize(meta_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    args = parse_args()
    train(args)
