"""
Generate paper figures from the sweep CSVs in ../results/.

Outputs PNGs to analysis/figures/:
    speedup_curves.png              5-panel S(T) per algo, OMP & pthreads
    speedup_t8_bars.png             grouped bars: speedup at T=8 per algo
    cross_framework_runtime.png     log-scale bars: total_ms ours vs sklearn/XGBoost
    ours_vs_sklearn_scaling.png     absolute-time scaling curves for KNN, MLP, DT

Run from anywhere (paths are resolved relative to this file):
    python analysis/plot_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

ALGOS = ["svm", "knn", "mlp", "dt", "nb"]
THREADS = [1, 2, 4, 8]

COLOR_OMP = "#4c72b0"
COLOR_PTH = "#dd8452"
COLOR_SKL = "#8172b2"
COLOR_XGB = "#55a868"
COLOR_IDEAL = "#999999"


def load():
    cpp = pd.read_csv(RESULTS / "sweep_results_cpp.csv")
    skl = pd.read_csv(RESULTS / "sweep_results_sklearn.csv")
    return cpp, skl


def serial_baseline(cpp: pd.DataFrame, algo: str) -> float:
    """Mean total_ms across all serial rows for the algorithm."""
    return cpp[(cpp.algorithm == algo) & (cpp.variant == "serial")].total_ms.mean()


def cpp_total(cpp: pd.DataFrame, algo: str, variant: str, t: int) -> float:
    row = cpp[(cpp.algorithm == algo) & (cpp.variant == variant) & (cpp.n_threads == t)]
    return float(row.total_ms.iloc[0])


def skl_total(skl: pd.DataFrame, algo: str, framework: str, variant: str, t: int) -> float:
    mask = (skl.algorithm == algo) & (skl.framework == framework) & (skl.n_threads == t)
    if variant is not None:
        mask &= skl.variant == variant
    row = skl[mask]
    return float(row.train_ms.iloc[0] + row.infer_ms.iloc[0])


def plot_speedup_curves(cpp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        baseline = serial_baseline(cpp, algo)
        for variant, color, marker in [
            ("omp", COLOR_OMP, "o"),
            ("pthreads", COLOR_PTH, "s"),
        ]:
            ys = [baseline / cpp_total(cpp, algo, variant, t) for t in THREADS]
            ax.plot(THREADS, ys, marker=marker, color=color, label=variant, linewidth=2)
        ax.plot(THREADS, THREADS, "--", color=COLOR_IDEAL, alpha=0.6, label="ideal linear")
        ax.set_title(algo.upper(), fontsize=13)
        ax.set_xlabel("threads")
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.set_xticklabels(THREADS)
        if i == 0:
            ax.set_ylabel("speedup vs serial")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("C++ parallel speedup: S(T) = total_ms(serial) / total_ms(T)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGS / "speedup_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_t8_bars(cpp: pd.DataFrame) -> None:
    omp_sp, pth_sp = [], []
    for algo in ALGOS:
        b = serial_baseline(cpp, algo)
        omp_sp.append(b / cpp_total(cpp, algo, "omp", 8))
        pth_sp.append(b / cpp_total(cpp, algo, "pthreads", 8))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ALGOS))
    w = 0.35
    ax.bar(x - w / 2, omp_sp, w, label="OpenMP", color=COLOR_OMP)
    ax.bar(x + w / 2, pth_sp, w, label="pthreads", color=COLOR_PTH)
    ax.axhline(y=8, color="k", linestyle="--", alpha=0.4, label="ideal (8x)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ALGOS])
    ax.set_ylabel("speedup at T=8")
    ax.set_title("C++ parallel speedup at 8 threads")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(omp_sp, pth_sp)):
        ax.text(i - w / 2, a + 0.08, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.08, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(max(omp_sp), max(pth_sp), 8) + 1)
    fig.tight_layout()
    fig.savefig(FIGS / "speedup_t8_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_framework_runtime(cpp: pd.DataFrame, skl: pd.DataFrame) -> None:
    """Log-scale bar chart: total_ms at T=8 (or best-available) per framework."""
    ours, sklearn_v, xgb_v = [], [], []
    for algo in ALGOS:
        ours.append(cpp_total(cpp, algo, "pthreads", 8))
        # sklearn side: use T=8 where available, T=1 for genuinely-serial ones
        if algo == "nb":
            sklearn_v.append(skl_total(skl, "nb", "sklearn", "hybrid", 1))
        elif algo in ("svm", "dt"):
            sklearn_v.append(skl_total(skl, algo, "sklearn", "default", 1))
        else:
            sklearn_v.append(skl_total(skl, algo, "sklearn", "default", 8))
        # XGBoost only exists for DT
        if algo == "dt":
            xgb_v.append(skl_total(skl, "dt", "xgboost", "default", 8))
        else:
            xgb_v.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ALGOS))
    w = 0.27
    ax.bar(x - w, ours, w, label="ours (pthreads, T=8)", color=COLOR_PTH)
    ax.bar(x, sklearn_v, w, label="sklearn (T=8 where supported, else T=1)", color=COLOR_SKL)
    xgb_positions = [xi + w for xi, v in zip(x, xgb_v) if not np.isnan(v)]
    xgb_values = [v for v in xgb_v if not np.isnan(v)]
    ax.bar(xgb_positions, xgb_values, w, label="XGBoost (hist, T=8)", color=COLOR_XGB)

    for xi, v in zip(x - w, ours):
        ax.text(xi, v * 1.15, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x, sklearn_v):
        ax.text(xi, v * 1.15, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(xgb_positions, xgb_values):
        ax.text(xi, v * 1.15, f"{v:.1f}", ha="center", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ALGOS])
    ax.set_ylabel("total_ms (log scale)")
    ax.set_title("Cross-framework runtime")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # headroom so labels don't clip
    fig.tight_layout()
    fig.savefig(FIGS / "cross_framework_runtime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ours_vs_sklearn_scaling(cpp: pd.DataFrame, skl: pd.DataFrame) -> None:
    """
    Absolute-time scaling for the three algorithms where sklearn/XGBoost is
    actually multi-threaded: KNN (n_jobs), MLP (BLAS), DT (XGBoost hist n_jobs).
    """
    panels = [
        ("knn", "sklearn", "default"),
        ("mlp", "sklearn", "default"),
        ("dt",  "xgboost", "default"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (algo, fw, variant) in enumerate(panels):
        ax = axes[i]
        ours_ys = [cpp_total(cpp, algo, "pthreads", t) for t in THREADS]
        skl_ys = [skl_total(skl, algo, fw, variant, t) for t in THREADS]
        ax.plot(THREADS, ours_ys, "o-", color=COLOR_PTH, label="ours (pthreads)", linewidth=2)
        fw_label = "XGBoost" if fw == "xgboost" else "sklearn"
        ax.plot(THREADS, skl_ys, "s-", color=COLOR_SKL, label=fw_label, linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(THREADS)
        ax.set_xticklabels(THREADS)
        ax.set_xlabel("threads")
        ax.set_ylabel("total_ms (log scale)")
        title = f"{algo.upper()} — ours vs {fw_label}"
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=10)
    fig.suptitle("Ours vs library baseline (algorithms where the baseline is actually parallel)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "ours_vs_sklearn_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cpp, skl = load()
    plot_speedup_curves(cpp)
    plot_speedup_t8_bars(cpp)
    plot_cross_framework_runtime(cpp, skl)
    plot_ours_vs_sklearn_scaling(cpp, skl)

    print(f"Wrote 4 figures to {FIGS}/:")
    for f in sorted(FIGS.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
