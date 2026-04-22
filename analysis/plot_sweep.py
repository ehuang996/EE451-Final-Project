"""
Generate paper figures from the refreshed CARC sweep CSVs in ../results/run2/.

Outputs PNGs to both:
    analysis/run2/figures/
    EE451-Final-Paper/figures/

Generated figures:
    speedup_curves.png
    speedup_t8_bars.png
    cross_framework_runtime.png
    ours_vs_sklearn_scaling.png
    pure_cpp_vs_blas_knn_mlp.png

Run from anywhere:
    python analysis/plot_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "run2"
ANALYSIS_FIGS = ROOT / "analysis" / "run2" / "figures"
PAPER_FIGS = ROOT / "EE451-Final-Paper" / "figures"
OUT_DIRS = [ANALYSIS_FIGS, PAPER_FIGS]
for out in OUT_DIRS:
    out.mkdir(parents=True, exist_ok=True)

ALGOS = ["svm", "knn", "mlp", "dt", "nb"]
THREADS = [1, 2, 4, 8]

COLOR_OMP = "#4c72b0"
COLOR_PTH = "#dd8452"
COLOR_SKL = "#8172b2"
COLOR_XGB = "#55a868"
COLOR_IDEAL = "#999999"
COLOR_PURE = "#4c72b0"
COLOR_BLAS = "#dd8452"

# Archived pure-C++ sweep summary from results/run1/results.md.
OLD_PURE_CPP = {
    "knn": {
        "serial_total_ms": 323019.07,
        "pth_t8_total_ms": 51548.34,
    },
    "mlp": {
        "serial_total_ms": 42008.91,
        "pth_t8_total_ms": 9087.56,
    },
}


def savefig_all(fig: plt.Figure, filename: str) -> None:
    for out in OUT_DIRS:
        fig.savefig(out / filename, dpi=150, bbox_inches="tight")


def load():
    cpp = pd.read_csv(RESULTS / "sweep_results_cpp.csv")
    skl = pd.read_csv(RESULTS / "sweep_results_sklearn.csv")
    return cpp, skl


def serial_baseline(cpp: pd.DataFrame, algo: str) -> float:
    row = cpp[(cpp.algorithm == algo) & (cpp.variant == "serial")].iloc[0]
    return float(row.total_ms)


def cpp_total(cpp: pd.DataFrame, algo: str, variant: str, t: int) -> float:
    row = cpp[(cpp.algorithm == algo) & (cpp.variant == variant) & (cpp.n_threads == t)]
    return float(row.total_ms.iloc[0])


def skl_total(skl: pd.DataFrame, algo: str, framework: str, variant: str, t: int) -> float:
    mask = (skl.algorithm == algo) & (skl.framework == framework) & (skl.n_threads == t)
    if variant is not None:
        mask &= skl.variant == variant
    row = skl[mask]
    return float((row.train_ms + row.infer_ms).iloc[-1])


def cross_framework_sklearn_total(skl: pd.DataFrame, algo: str) -> float:
    if algo == "nb":
        return skl_total(skl, "nb", "sklearn", "hybrid", 1)
    if algo in ("svm", "dt"):
        return skl_total(skl, algo, "sklearn", "default", 1)
    return skl_total(skl, algo, "sklearn", "default", 8)


def plot_speedup_curves(cpp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
    for i, algo in enumerate(ALGOS):
        ax = axes[i]
        baseline = serial_baseline(cpp, algo)
        for variant, color, marker, label in [
            ("omp", COLOR_OMP, "o", "OpenMP"),
            ("pthreads", COLOR_PTH, "s", "pthreads"),
        ]:
            ys = [baseline / cpp_total(cpp, algo, variant, t) for t in THREADS]
            ax.plot(THREADS, ys, marker=marker, color=color, label=label, linewidth=2)
        ax.plot(THREADS, THREADS, "--", color=COLOR_IDEAL, alpha=0.6, label="ideal")
        ax.set_title(algo.upper(), fontsize=13)
        ax.set_xlabel("threads")
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.set_xticklabels(THREADS)
        if i == 0:
            ax.set_ylabel("speedup vs serial")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Refreshed CARC sweep: C++ speedup vs thread count", fontsize=14)
    fig.tight_layout()
    savefig_all(fig, "speedup_curves.png")
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
    ax.set_title("Refreshed CARC sweep: speedup at 8 threads")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(omp_sp, pth_sp)):
        ax.text(i - w / 2, a + 0.06, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.06, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(max(omp_sp), max(pth_sp), 8) + 1)
    fig.tight_layout()
    savefig_all(fig, "speedup_t8_bars.png")
    plt.close(fig)


def plot_cross_framework_runtime(cpp: pd.DataFrame, skl: pd.DataFrame) -> None:
    ours, sklearn_v, xgb_v = [], [], []
    for algo in ALGOS:
        ours.append(cpp_total(cpp, algo, "pthreads", 8))
        sklearn_v.append(cross_framework_sklearn_total(skl, algo))
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
        ax.text(xi, v * 1.12, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x, sklearn_v):
        ax.text(xi, v * 1.12, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(xgb_positions, xgb_values):
        ax.text(xi, v * 1.12, f"{v:.1f}", ha="center", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ALGOS])
    ax.set_ylabel("total_ms (log scale)")
    ax.set_title("Refreshed CARC sweep: cross-framework runtime")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    fig.tight_layout()
    savefig_all(fig, "cross_framework_runtime.png")
    plt.close(fig)


def plot_ours_vs_sklearn_scaling(cpp: pd.DataFrame, skl: pd.DataFrame) -> None:
    panels = [
        ("knn", "sklearn", "default"),
        ("mlp", "sklearn", "default"),
        ("dt", "xgboost", "default"),
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
        ax.set_title(f"{algo.upper()} -- ours vs {fw_label}", fontsize=12)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle("Refreshed CARC sweep: library baselines that are actually parallel", fontsize=13)
    fig.tight_layout()
    savefig_all(fig, "ours_vs_sklearn_scaling.png")
    plt.close(fig)


def plot_pure_cpp_vs_blas(cpp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, algo in zip(axes, ["knn", "mlp"]):
        new_serial = serial_baseline(cpp, algo)
        new_pth8 = cpp_total(cpp, algo, "pthreads", 8)
        old_serial = OLD_PURE_CPP[algo]["serial_total_ms"]
        old_pth8 = OLD_PURE_CPP[algo]["pth_t8_total_ms"]

        x = np.arange(2)
        w = 0.34
        pure_vals = [old_serial, old_pth8]
        blas_vals = [new_serial, new_pth8]
        ax.bar(x - w / 2, pure_vals, w, color=COLOR_PURE, label="pure C++")
        ax.bar(x + w / 2, blas_vals, w, color=COLOR_BLAS, label="BLAS-backed")
        ax.set_xticks(x)
        ax.set_xticklabels(["serial", "pthreads T=8"])
        ax.set_yscale("log")
        ax.set_ylabel("total_ms (log scale)")
        ax.set_title(algo.upper(), fontsize=13)
        ax.grid(True, axis="y", alpha=0.3, which="both")

        old_s8 = old_serial / old_pth8
        new_s8 = new_serial / new_pth8
        ax.text(
            0.02,
            0.98,
            f"pure C++ $S_8$ = {old_s8:.2f}x\nBLAS $S_8$ = {new_s8:.2f}x",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
        for xi, v in zip(x - w / 2, pure_vals):
            ax.text(xi, v * 1.08, f"{v:,.0f}", ha="center", fontsize=8)
        for xi, v in zip(x + w / 2, blas_vals):
            ax.text(xi, v * 1.08, f"{v:,.0f}", ha="center", fontsize=8)

    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("Pure C++ vs BLAS on CARC: KNN and MLP", fontsize=14)
    fig.tight_layout()
    savefig_all(fig, "pure_cpp_vs_blas_knn_mlp.png")
    plt.close(fig)


def main():
    cpp, skl = load()
    plot_speedup_curves(cpp)
    plot_speedup_t8_bars(cpp)
    plot_cross_framework_runtime(cpp, skl)
    plot_ours_vs_sklearn_scaling(cpp, skl)
    plot_pure_cpp_vs_blas(cpp)

    print("Wrote figures to:")
    for out in OUT_DIRS:
        print(f"  {out}")
        for f in sorted(out.glob("*.png")):
            print(f"    {f.name}")


if __name__ == "__main__":
    main()
