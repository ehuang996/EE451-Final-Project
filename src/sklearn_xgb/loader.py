"""
Python port of the C++ load_dataset used by svm/knn/mlp/dt/nb.

Replicates the exact preprocessing pipeline:
  - Reads data/train_cleaned.csv and data/test_cleaned.csv from repo root
  - Finds label column by name ("round_winner"), labels stay in {+1, -1}
  - Z-score normalizes with train-set statistics only
  - Skips normalization for: columns with name prefix "map_", the exact name
    "bomb_planted", and any column whose training values are all in {0, 1}
  - Returns (X_train, y_train, X_test, y_test, feature_names, binary_mask)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# This file lives at src/sklearn_xgb/loader.py — three parents up is repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(path: str) -> str:
    # Try the path as given (relative to cwd), then under repo root, then
    # under repo_root/data/ so callers that pass just a bare filename work.
    p = Path(path)
    if p.exists():
        return str(p)
    for candidate in (REPO_ROOT / path, REPO_ROOT / "data" / path):
        if candidate.exists():
            return str(candidate)
    return str(p)


def _is_one_hot(name: str) -> bool:
    return name.startswith("map_")


def _is_explicit_binary(name: str) -> bool:
    return name == "bomb_planted"


def _is_binary_column(col: np.ndarray) -> bool:
    return bool(np.all((np.abs(col) < 1e-9) | (np.abs(col - 1.0) < 1e-9)))


def load_dataset(
    train_path: str = "data/train_cleaned.csv",
    test_path: str = "data/test_cleaned.csv",
):
    train_path = _resolve(train_path)
    test_path = _resolve(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "round_winner" not in train_df.columns:
        raise ValueError("column 'round_winner' not found in training header")

    feature_names = [c for c in train_df.columns if c != "round_winner"]

    y_train = train_df["round_winner"].to_numpy(dtype=np.int64)
    y_test = test_df["round_winner"].to_numpy(dtype=np.int64)
    X_train = train_df[feature_names].to_numpy(dtype=np.float32, copy=True)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32, copy=True)

    binary_mask = np.zeros(len(feature_names), dtype=bool)
    for j, name in enumerate(feature_names):
        if _is_one_hot(name) or _is_explicit_binary(name):
            binary_mask[j] = True
        elif _is_binary_column(X_train[:, j]):
            binary_mask[j] = True

    mean = np.zeros(len(feature_names), dtype=np.float64)
    sd = np.ones(len(feature_names), dtype=np.float64)
    cont = ~binary_mask
    if cont.any():
        mean[cont] = X_train[:, cont].mean(axis=0)
        sd[cont] = X_train[:, cont].std(axis=0)
        sd[cont] = np.where(sd[cont] < 1e-9, 1.0, sd[cont])

    X_train = ((X_train - mean) / sd).astype(np.float32)
    X_test = ((X_test - mean) / sd).astype(np.float32)

    n_skip = int(binary_mask.sum())
    print("Dataset Info")
    print(f"  X_train  : {X_train.shape[0]} x {X_train.shape[1]}")
    print(f"  X_test   : {X_test.shape[0]} x {X_test.shape[1]}")
    tr_pos = int((y_train == 1).sum())
    tr_neg = int((y_train == -1).sum())
    te_pos = int((y_test == 1).sum())
    te_neg = int((y_test == -1).sum())
    print(f"  y_train  : +1={tr_pos}  -1={tr_neg}")
    print(f"  y_test   : +1={te_pos}  -1={te_neg}")
    print(
        f"  Normalize: {len(feature_names) - n_skip} z-scored, "
        f"{n_skip} passthrough (binary/one-hot)"
    )
    print()

    return X_train, y_train, X_test, y_test, feature_names, binary_mask


def evaluate(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Identical to the Metrics evaluate(...) function in each C++ file."""
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == -1) & (pred == 1)).sum())
    tn = int(((truth == -1) & (pred == -1)).sum())
    fn = int(((truth == 1) & (pred == -1)).sum())
    n = len(truth)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}
