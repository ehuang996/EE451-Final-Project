# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

EE 451 (USC, Distributed & Parallel Computing) **final project**: a C++17 empirical study of five classical ML algorithms parallelized with OpenMP + pthreads, benchmarked on the **CS:GO Round Winner** dataset.

- Binary classification: `round_winner ∈ {+1, −1}` from 103 numeric features.
- Train: 97,928 data rows in [data/train_cleaned.csv](data/train_cleaned.csv) plus header. Test: 24,482 data rows in [data/test_cleaned.csv](data/test_cleaned.csv) plus header.
- Target per-model accuracy: **76–80%** (community benchmark).
- Parallelism target: **8 threads** (SLURM `--cpus-per-task=8`).
- Hardware: USC CARC cluster; local builds are a convenience, not required.

The latest report is [EE451-Final-Paper/main.pdf](EE451-Final-Paper/main.pdf). Treat that PDF as the source of truth for the final paper narrative and numbers. The report covers two CARC sweeps:

- **Run 1 / Sweep 1** (`results/run1/`, job `3272373`): archived pure-C++ sweep for all five algorithms.
- **Run 2 / Sweep 2** (`results/run2/`, job `3278100`): current code path. SVM, DT, and NB remain pure C++; KNN and MLP use vendor BLAS for dense inner kernels.

Unless a result is explicitly labeled **pure C++** or **Run 1**, assume it refers to Run 2. The central final result is that kernel formulation comes first: BLAS reformulation flips KNN from a 16x loss to a 2.2x win over scikit-learn, while MLP gets much faster in absolute time but loses 8-thread speedup because the GEMMs are too small for threaded BLAS to pay off.

The five algorithms: **KNN, SVM, Naive Bayes, Decision Tree, MLP**. Each is a single, self-contained `.cpp` file with a shared conceptual interface (`load_dataset`, serial and parallel train/predict entry points, `evaluate`, and standardized `[tag]` output). [analytics_engine.cpp](analytics_engine.cpp) drives all five as subprocesses and emits `analytics_results.csv` / sweep CSVs for the writeup's comparison tables.

Team: Eric Huang (MLP, DT, NB, analytics_engine, sklearn_xgb), Harry Yang (SVM), Jinglu Sun, Mo Jiang (KNN), Pinru Wang.

## Repo layout

```
├── 451 project proposal.pdf   ← original project proposal, useful historical context
├── README.md                  ← user-facing overview; read alongside this file
├── CLAUDE.md                  ← this file
├── analytics_engine.cpp       ← subprocess driver; parses each binary's stdout
├── slurm/
│   ├── job.sl                 ← SLURM: compiles + runs all five with run2 defaults
│   ├── job_analytics.sl       ← SLURM: runs analytics_engine across all five
│   └── job_sweep.sl           ← SLURM: {1,2,4,8}-thread C++ + sklearn/XGBoost sweep
├── split_data.sh              ← 80/20 shuffle-split of the raw Kaggle CSV
├── data/
│   ├── train_cleaned.csv      ← 97,928 rows + header, 103 features + label
│   ├── test_cleaned.csv       ← 24,482 rows + header, 103 features + label
│   └── cs_go_winner_data_{train,test}.csv   ← raw pre-cleaning splits
├── src/
│   ├── cpp/
│   │   ├── svm/  ← Harry. Linear SVM, hinge + L2. svm.cpp + job_svm.sl + Data.ipynb
│   │   ├── knn/  ← Mo.    BLAS-backed squared-L2 k-NN. knn.cpp + job_knn.sl
│   │   ├── mlp/  ← Eric.  BLAS-backed 103→64→32→1 MLP. mlp.cpp + job_mlp.sl
│   │   ├── dt/   ← Eric.  Histogram CART, Gini. dt.cpp + job_dt.sl
│   │   └── nb/   ← Eric.  Hybrid Gaussian + Bernoulli NB. nb.cpp + job_nb.sl
│   └── sklearn_xgb/          ← Python comparison harness (sklearn + XGBoost)
├── results/
│   ├── run1/                  ← archived pure-C++ sweep
│   └── run2/                  ← final BLAS-backed KNN/MLP sweep
├── analysis/                  ← plotting script and generated run figures
├── EE451-Final-Paper/
│   ├── main.pdf               ← latest report; source of truth for paper claims
│   └── *.tex                  ← report source files
├── misc/
│   ├── *.md                   ← design notes, historical planning, and run logs
│   └── diff.md                ← C++ vs sklearn/XGBoost comparison notes
```

## Invariants when editing an algorithm file

All five algorithm files follow a tightly aligned convention so analytics_engine, the shared data loader, and cross-algorithm comparisons Just Work. Don't drift from it without a cross-algorithm refactor.

- **Shared public interface** (identical signature shape across all 5):
  - `using Labels = std::vector<int>;`
  - `struct Dataset { int n_features; std::vector<std::string> feature_names; std::vector<float> X_train, X_test; Labels y_train, y_test; };`
  - `struct Metrics { double acc = 0.0, prec = 0.0, rec = 0.0, f1 = 0.0; };`
  - `static Dataset load_dataset(const std::string&, const std::string&);`
  - `static XModel train_serial(const std::vector<float>& X, const Labels& y, int n_rows, int n_features, ...);`
  - `static XModel train_parallel_omp(...)` and `static XModel train_parallel_pthreads(...)` where the algorithm has parallel training. KNN is lazy and parallelizes inference; its train step just stores data and row norms.
  - `static Labels predict_serial(const XModel&, const std::vector<float>& X, int n_rows, int n_features);`
  - `static Labels predict_parallel(...)` or explicit `predict_parallel_omp(...)` / `predict_parallel_pthreads(...)` variants.
  - `static Metrics evaluate(const Labels& truth, const Labels& pred);`
  - `static void print_results(const std::string& tag, double train_ms, double infer_ms, const Metrics&);`
  - `int main(int argc, char* argv[])` — takes optional `train_csv` and `test_csv` paths; binaries also accept algorithm-specific hyperparams as argv[3+] (see each file's header).

- **Shared load_dataset behavior**:
  - Find the label column by searching the header for `"round_winner"` (NOT a hardcoded column index).
  - `resolve_path(path)` searches `./`, `../`, `data/`, `../data/`, `../../data/`, `../../../data/` — so binaries work from the repo root or any src/cpp/<algo>/ subfolder.
  - Skip z-score normalization for: column names with prefix `"map_"`, exact name `"bomb_planted"`, and any column whose training values are all in `{0.0, 1.0}`. Passthrough columns keep their original 0/1 values (mean=0, sd=1 for them, so the uniform normalize pass leaves them alone).
  - Print `Dataset Info` block with `Normalize: N z-scored, M passthrough (binary/one-hot)` summary.

- **Self-contained files**: loader + metrics code is **duplicated** across all five algorithm files by design. Do NOT preemptively extract them into a shared header unless explicitly requested. Each new algorithm file should copy the same loader/metrics block verbatim.

- **Hyperparameters at top-of-file as `static constexpr`**: `N_THREADS = 8`, `SEED = 42` are common to all. Algorithm-specific constants (MAX_EPOCHS, MAX_DEPTH, K_NEIGHBORS, LAMBDA, etc.) live above the `Labels` alias.

- **Plain-text console output** — every algorithm prints a `Dataset Info` block, `[tag]` metrics blocks per trainer, and a final `Speedup Summary`. **No Unicode box-drawing** — that was dropped during the SVM/KNN → MLP/DT/NB alignment refactor. analytics_engine parses the `[tag]` blocks by substring matching, so don't change the spacing (e.g., `"  Training time  : X ms"` must keep two spaces before the colon).

- **Deterministic parallel output**: parallel variants must produce the same predictions as serial (up to floating-point reduction order), and each algorithm's `main()` prints an `Accuracy parity check` block at the end comparing `|serial − omp|` and `|serial − pthreads|`. Expect `< 0.01` (often 0.0000).

- **Run 2 backend policy**: KNN and MLP now require a CBLAS-compatible backend. Do not restore a pure-C++ fallback unless the user explicitly asks to resurrect Run 1. KNN uses BLAS only for inner dot-product panels and pins BLAS to one internal thread during outer OpenMP/pthreads query scheduling. MLP uses BLAS for minibatch GEMMs; its OpenMP and pthreads training entry points intentionally reuse the serial BLAS trainer.

## Build

Single-file compile per algorithm, flags fixed for the final Run 2 comparison:

```bash
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp -DKNN_USE_BLAS src/cpp/knn/knn.cpp -o knn -lpthread -lopenblas
g++ -std=c++17 -O3 -march=native -fopenmp -DMLP_USE_BLAS src/cpp/mlp/mlp.cpp -o mlp -lpthread -lopenblas
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp -o dt -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp -o nb -lpthread
```

Exactly what [slurm/job_sweep.sl](slurm/job_sweep.sl) runs on CARC, with `module load gcc/13.3.0` and OpenBLAS available. `-march=native` means builds are tied to the machine; don't move binaries between architectures.

Runtime thread policy matters:

- KNN: pin BLAS to one internal thread (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.) and let our OpenMP/pthreads code own query-block parallelism.
- MLP: hand the thread budget to BLAS (`OPENBLAS_NUM_THREADS=$N_THREADS`, etc.) because the dense minibatch kernel is the experiment.

## Running

On CARC (the correct path):

```bash
sbatch slurm/job.sl             # compiles + runs all five with run2 defaults; output to alljob.out
sbatch slurm/job_analytics.sl   # runs analytics_engine; emits analytics_results.csv
sbatch slurm/job_sweep.sl       # reproduces the {1,2,4,8}-thread C++ + sklearn/XGBoost sweep
```

Per-algo SLURM scripts live in each algo's folder (e.g., [src/cpp/svm/job_svm.sl](src/cpp/svm/job_svm.sl)) and are run from within that folder with `cd src/cpp/svm && sbatch job_svm.sl`.

Local dev on macOS is intentionally unsupported — Apple Clang ships without OpenMP, and `pthread_barrier_t` is not implemented on Darwin. If you need to iterate locally:
- `brew install libomp` and use `clang++ -Xpreprocessor -fopenmp -lomp ...`, OR
- `brew install gcc` and use `g++-14` directly.
- Know that local Apple Clang toolchains can ship broken against libc++ for months at a time (2026 Xcode 13/macOS 26 header mismatch, etc.). If `#include <iostream>` fails on a syntax-check, it's the environment, not the code.

## Parallelization conventions

- **OpenMP for loop-level parallelism and inference** (`#pragma omp parallel for schedule(static) num_threads(N_THREADS)` over independent samples).
- **pthreads for training or inference with shared state**. Reference pattern in [src/cpp/svm/svm.cpp](src/cpp/svm/svm.cpp) for epoch-level gradient reduction, [src/cpp/dt/dt.cpp](src/cpp/dt/dt.cpp) for per-node barriers, [src/cpp/nb/nb.cpp](src/cpp/nb/nb.cpp) for one final mutex-protected merge, and [src/cpp/knn/knn.cpp](src/cpp/knn/knn.cpp) for outer query-range scheduling around BLAS.
- **Thread-local buffers must be hoisted out of any inner loop** (allocate once per thread, `std::fill` to zero per iteration). Allocating `Vec(thousands)` per batch dominates runtime.
- **Only thread 0 may mutate shared scheduler state** (index shuffling, `batch_start`, epoch counters). Otherwise parallel-vs-serial comparisons become meaningless.
- **Never assert bitwise equality between serial and parallel outputs** — float reduction order is non-deterministic. Check `|acc_serial − acc_parallel| < 0.01` instead (this check is already in each algorithm's `main()`).

## Data

- Cleaned CSVs at `data/`, already one-hot-encoded by [src/cpp/svm/Data.ipynb](src/cpp/svm/Data.ipynb). Column `round_winner` is the label in `{+1, −1}`; all other 103 columns are numeric features. Column ORDER in the header is load-order-dependent but the loader finds the label by name so column-order changes won't break anything.
- To regenerate from the raw Kaggle dump: re-run [split_data.sh](split_data.sh) to produce the pre-cleaning splits into `data/`, then run [src/cpp/svm/Data.ipynb](src/cpp/svm/Data.ipynb) which writes the cleaned CSVs to `data/`.
- Never commit the raw Kaggle CSV — it's not tracked, and at ~50 MB+ it doesn't belong in git.

## Documentation

- [EE451-Final-Paper/main.pdf](EE451-Final-Paper/main.pdf) — latest report and source of truth for final claims, tables, and wording.
- [EE451-Final-Paper/*.tex](EE451-Final-Paper/) — LaTeX source for the report. It should match `main.pdf`, but if there is conflict, prefer the PDF unless the user asks to edit/recompile the paper.
- [results/run1/results.md](results/run1/results.md) — archived pure-C++ sweep.
- [results/run2/results.md](results/run2/results.md) — final BLAS-backed KNN/MLP sweep used by the paper unless explicitly labeled otherwise.
- [misc/*.md](misc/) — design docs, historical planning notes, and results logs. Some entries predate the final BLAS rerun; cross-check against `main.pdf` and `results/run2/` before relying on old claims.
- [src/sklearn_xgb/README.md](src/sklearn_xgb/README.md) — sklearn/XGBoost comparison harness overview.
- [451 project proposal.pdf](451%20project%20proposal.pdf) is historical context. The proposal's MLP-scaling hypothesis was falsified; do not treat it as the current thesis.

## What NOT to do

- Do **not** add scikit-learn, Eigen, or other ML frameworks to the C++ algorithm files. BLAS is intentionally allowed and required for KNN and MLP in the final Run 2 code path; SVM, DT, and NB should remain pure hand-written C++ unless the user explicitly changes the experiment.
- Do **not** refactor the duplicated `load_dataset` / `evaluate` into a shared header unless explicitly requested. The duplication is deliberate — it keeps each algorithm self-contained and easy to reason about.
- Do **not** change compile flags for a single algorithm. Apples-to-apples comparison across the five models requires identical flags.
- Do **not** add commits, push branches, or open PRs without being asked — this is a student project and git actions should be explicitly directed.
- Do **not** modify [data/train_cleaned.csv](data/train_cleaned.csv) or [data/test_cleaned.csv](data/test_cleaned.csv) — they are the fixed experiment input.
- Do **not** add `-ffast-math` to any compile line. Gini gain / cross-entropy tie-breaking can drift by 1e-15 under `-ffast-math` and break the serial-vs-parallel determinism check.
