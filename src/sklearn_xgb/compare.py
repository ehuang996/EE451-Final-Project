"""
Run scikit-learn (and XGBoost where applicable) baselines with hyperparameters
matched to the C++ implementations, and write results.csv / results.json with
the unified schema consumed by parse_cpp_output.py.

Thread count is controlled by the N_THREADS environment variable (same one
the C++ binaries read), defaulting to 8. Set before invocation:

    N_THREADS=4 python compare.py

This pins OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS AND passes
n_jobs to sklearn KNN and XGBoost. Models sklearn runs serially by design
(SGDClassifier, DecisionTreeClassifier, GaussianNB, BernoulliNB) still report
n_threads=1 in the CSV regardless of the env var.
"""

from __future__ import annotations

# Pin BLAS threading BEFORE importing numpy / sklearn / xgboost. Read the
# unified N_THREADS env var that the C++ binaries also read; default 8.
import os
_N_THREADS = max(1, int(os.environ.get("N_THREADS", "8")))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(_N_THREADS)

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from loader import load_dataset, evaluate
from hybrid_nb import HybridGaussianBernoulliNB

RESULTS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = RESULTS_DIR / "results.csv"
RESULTS_JSON = RESULTS_DIR / "results.json"

CSV_COLUMNS = [
    "algorithm",
    "framework",
    "variant",
    "n_threads",
    "train_ms",
    "infer_ms",
    "acc",
    "prec",
    "rec",
    "f1",
    "notes",
]


def _time_ms(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


def _row(algo, framework, variant, n_threads, train_ms, infer_ms, metrics, notes):
    return {
        "algorithm": algo,
        "framework": framework,
        "variant": variant,
        "n_threads": n_threads,
        "train_ms": round(train_ms, 2),
        "infer_ms": round(infer_ms, 2),
        "acc": round(metrics["acc"], 4),
        "prec": round(metrics["prec"], 4),
        "rec": round(metrics["rec"], 4),
        "f1": round(metrics["f1"], 4),
        "notes": notes,
    }


def run_sklearn_svm(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import SGDClassifier
    # Matches svm/svm.cpp: hinge + L2, lr=0.02 constant, lambda=1e-4, 20 epochs.
    clf = SGDClassifier(
        loss="hinge",
        alpha=1e-4,
        learning_rate="constant",
        eta0=0.02,
        max_iter=20,
        tol=None,
        random_state=42,
    )
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("svm", "sklearn", "default", 1, train_ms, infer_ms,
                evaluate(y_test, pred), "SGDClassifier single-threaded by design")


def run_sklearn_knn(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    # Matches knn/knn.cpp: k=11, brute Euclidean, uniform vote.
    clf = KNeighborsClassifier(
        n_neighbors=11,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
        n_jobs=_N_THREADS,
    )
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("knn", "sklearn", "default", _N_THREADS, train_ms, infer_ms,
                evaluate(y_test, pred), f"brute-force, n_jobs={_N_THREADS}")


def run_sklearn_mlp(X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPClassifier
    # Matches mlp/mlp.cpp: 103->64->32->1 ReLU/sigmoid BCE, SGD+momentum=0.9,
    # batch=128, 30 epochs, lr=0.01 invscaling. Parallelism is implicit via
    # OMP_NUM_THREADS for the BLAS backend.
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        learning_rate="invscaling",
        power_t=0.5,
        momentum=0.9,
        nesterovs_momentum=False,
        alpha=1e-4,
        batch_size=128,
        max_iter=30,
        tol=0.0,
        n_iter_no_change=30,
        shuffle=True,
        random_state=42,
    )
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("mlp", "sklearn", "default", _N_THREADS, train_ms, infer_ms,
                evaluate(y_test, pred),
                f"BLAS-parallel via OMP_NUM_THREADS={_N_THREADS}")


def run_sklearn_dt(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    # Matches dt/dt.cpp: max_depth=12, min_split=20, min_leaf=5, Gini.
    # sklearn uses exact splits, not histogram.
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=5,
        random_state=42,
    )
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("dt", "sklearn", "default", 1, train_ms, infer_ms,
                evaluate(y_test, pred),
                "exact-split single-threaded; sklearn DT has no n_jobs")


def run_xgboost_dt(X_train, y_train, X_test, y_test):
    try:
        import xgboost as xgb
    except ImportError:
        return None
    # Apples-to-apples with dt/dt.cpp: single hist tree, 64 bins, n_jobs=_N_THREADS.
    # XGBoost expects {0, 1} labels.
    y_train_01 = (y_train == 1).astype(np.int32)
    y_test_01 = (y_test == 1).astype(np.int32)
    clf = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=12,
        tree_method="hist",
        max_bin=64,
        n_jobs=_N_THREADS,
        learning_rate=1.0,
        base_score=0.5,
        reg_alpha=0.0,
        reg_lambda=0.0,
        objective="binary:logistic",
        verbosity=0,
        random_state=42,
    )
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train_01))
    infer_ms, pred01 = _time_ms(lambda: clf.predict(X_test))
    pred = np.where(pred01 == 1, 1, -1)
    return _row("dt", "xgboost", "default", _N_THREADS, train_ms, infer_ms,
                evaluate(y_test_01 * 2 - 1, pred),
                f"hist single-tree, max_bin=64, n_jobs={_N_THREADS}")


def run_hybrid_nb(X_train, y_train, X_test, y_test, binary_mask):
    clf = HybridGaussianBernoulliNB(binary_mask=binary_mask)
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("nb", "sklearn", "hybrid", 1, train_ms, infer_ms,
                evaluate(y_test, pred),
                "GaussianNB+BernoulliNB composed; matches nb.cpp is_binary mask")


def run_gaussian_nb_only(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB(var_smoothing=1e-9)
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("nb", "sklearn", "gaussian-only", 1, train_ms, infer_ms,
                evaluate(y_test, pred),
                "naive-sklearn-user baseline; Gaussian applied to all features")


def run_bernoulli_nb_only(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha=1.0, binarize=0.5)
    train_ms, _ = _time_ms(lambda: clf.fit(X_train, y_train))
    infer_ms, pred = _time_ms(lambda: clf.predict(X_test))
    return _row("nb", "sklearn", "bernoulli-only", 1, train_ms, infer_ms,
                evaluate(y_test, pred),
                "naive-sklearn-user baseline; Bernoulli applied to all features")


def main():
    parser = argparse.ArgumentParser(description="sklearn/XGBoost baseline runner")
    parser.add_argument("--train", default="data/train_cleaned.csv")
    parser.add_argument("--test", default="data/test_cleaned.csv")
    parser.add_argument("--out-csv", default=str(RESULTS_CSV))
    parser.add_argument("--out-json", default=str(RESULTS_JSON))
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, _, binary_mask = load_dataset(
        args.train, args.test
    )

    rows = []

    print("Running sklearn SVM (SGDClassifier)...")
    rows.append(run_sklearn_svm(X_train, y_train, X_test, y_test))

    print("Running sklearn KNN (brute, n_jobs=8)...")
    rows.append(run_sklearn_knn(X_train, y_train, X_test, y_test))

    print("Running sklearn MLP (BLAS-parallel)...")
    rows.append(run_sklearn_mlp(X_train, y_train, X_test, y_test))

    print("Running sklearn DT (exact-split, serial)...")
    rows.append(run_sklearn_dt(X_train, y_train, X_test, y_test))

    print("Running XGBoost DT (hist, n_jobs=8)...")
    xgb_row = run_xgboost_dt(X_train, y_train, X_test, y_test)
    if xgb_row is None:
        print("  xgboost not installed - skipping.")
    else:
        rows.append(xgb_row)

    print("Running sklearn NB (hybrid Gaussian+Bernoulli)...")
    rows.append(run_hybrid_nb(X_train, y_train, X_test, y_test, binary_mask))

    print("Running sklearn NB (Gaussian-only, naive baseline)...")
    rows.append(run_gaussian_nb_only(X_train, y_train, X_test, y_test))

    print("Running sklearn NB (Bernoulli-only, naive baseline)...")
    rows.append(run_bernoulli_nb_only(X_train, y_train, X_test, y_test))

    sanity_check(rows)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_results(rows, out_csv, out_json)
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    print(f"Wrote {out_json}")


def sanity_check(rows):
    for r in rows:
        for k in ("train_ms", "infer_ms", "acc", "prec", "rec", "f1"):
            v = r[k]
            if not np.isfinite(v):
                raise ValueError(f"non-finite {k}={v} in row {r}")
        if r["train_ms"] <= 0:
            raise ValueError(f"train_ms not positive in row {r}")
        if not (0.0 <= r["acc"] <= 1.0):
            raise ValueError(f"acc out of [0,1] in row {r}")


def write_results(rows, out_csv: Path, out_json: Path):
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # JSON is a rewrite (not append) for readability; read-modify-write.
    existing = []
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    existing.extend(rows)
    out_json.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
