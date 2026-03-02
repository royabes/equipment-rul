# train.py — Train LSTM on NASA CMAPSS FD001 (or synthetic data)
# Run once locally: python train.py
# Outputs: models/rul_model.pt + models/scaler.pkl
#
# NASA CMAPSS download:
#   https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
#   Extract to data/CMAPSSData/ (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from model import RUL_LSTM
from preprocess import (
    load_cmapss, add_rul_labels, fit_scaler, scale,
    make_windows, make_test_windows, generate_synthetic_cmapss,
    SENSOR_COLS, MAX_RUL,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("data/CMAPSSData")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS      = 60
BATCH_SIZE  = 64
LR          = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS  = 2


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def train():
    print(f"Device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────────
    if (DATA_DIR / "train_FD001.txt").exists():
        print("Loading NASA CMAPSS FD001...")
        train_df, test_df, test_rul = load_cmapss(DATA_DIR, "FD001")
        train_df = add_rul_labels(train_df)
    else:
        print("NASA data not found — using synthetic degradation data.")
        print(f"  (Download from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)")
        print(f"  (Extract to {DATA_DIR}/)")
        train_df = generate_synthetic_cmapss(n_units=80)
        test_df = generate_synthetic_cmapss(n_units=20, seed=99)
        test_rul = None

    # ── Scale ──────────────────────────────────────────────────────────────────
    scaler = fit_scaler(train_df)
    train_df = scale(train_df, scaler)

    # ── Sliding windows ────────────────────────────────────────────────────────
    X_train, y_train = make_windows(train_df)
    print(f"Training samples: {len(X_train)}")

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RUL_LSTM(input_size=len(SENSOR_COLS), hidden_size=HIDDEN_SIZE,
                     num_layers=NUM_LAYERS).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nEpoch | Train RMSE")
    print("------+-----------")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        train_rmse = np.sqrt(epoch_loss / len(X_train))
        if epoch % 10 == 0:
            print(f"  {epoch:3d}  |  {train_rmse:.2f}")

    # ── Evaluate on test set ───────────────────────────────────────────────────
    if test_rul is not None:
        model.eval()
        X_test = make_test_windows(test_df, scaler)
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy().flatten()
        test_rmse = rmse(preds, test_rul)
        print(f"\nTest RMSE (FD001): {test_rmse:.2f}")
    else:
        print("\nNo test RUL available (synthetic mode).")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_DIR / "rul_model.pt")
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved: {MODEL_DIR}/rul_model.pt + {MODEL_DIR}/scaler.pkl")


if __name__ == "__main__":
    train()
