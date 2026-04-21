# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

EE 451 (USC, Distributed & Parallel Computing) **final project**: a from-scratch, pure-C++17 empirical study of five classical ML algorithms parallelized with OpenMP + pthreads, benchmarked on the **CS:GO Round Winner** dataset.

- Binary classification: `round_winner ∈ {+1, −1}` from 103 numeric features.
- Train: ~97,929 rows in [data/train_cleaned.csv](data/train_cleaned.csv). Test: ~24,483 rows in [data/test_cleaned.csv](data/test_cleaned.csv).
- Target per-model accuracy: **76–80%** (community benchmark).
- Parallelism target: **8 threads** (SLURM `--cpus-per-task=8`).
- Hardware: USC CARC cluster; local builds are a convenience, not required.

The five algorithms: **KNN, SVM, Naive Bayes, Decision Tree, MLP**. Each is a single, self-contained `.cpp` file with a shared public interface (`load_dataset`, `train_serial`, `train_parallel_omp`, `train_parallel_pthreads`, `predict_serial`, `predict_parallel`, `evaluate`). [analytics_engine.cpp](analytics_engine.cpp) drives all five as subprocesses and emits `analytics_results.csv` for the writeup's comparison tables.

Team: Eric Huang (MLP, DT, NB, analytics_engine, sklearn_xgb), Harry Yang (SVM), Jinglu Sun, Mo Jiang (KNN), Pinru Wang.

## Repo layout

```
├── 451 project proposal.pdf   ← project proposal (read this for context)
├── proposal.tex               ← LaTeX source
├── README.md                  ← user-facing overview; read alongside this file
├── CLAUDE.md                  ← this file
├── analytics_engine.cpp       ← subprocess driver; parses each binary's stdout
├── job.sl                     ← SLURM: compiles + runs all five
├── job_analytics.sl           ← SLURM: runs analytics_engine across all five
├── split_data.sh              ← 80/20 shuffle-split of the raw Kaggle CSV
├── data/
│   ├── train_cleaned.csv      ← 97929 × 104
│   ├── test_cleaned.csv       ← 24483  × 104
│   └── cs_go_winner_data_{train,test}.csv   ← raw pre-cleaning splits
├── src/
│   ├── cpp/
│   │   ├── svm/  ← Harry. Linear SVM, hinge + L2. svm.cpp + job_svm.sl + Data.ipynb
│   │   ├── knn/  ← Mo.    Squared-L2 k-NN. knn.cpp + job_knn.sl
│   │   ├── mlp/  ← Eric.  103→64→32→1, BCE + momentum. mlp.cpp + job_mlp.sl
│   │   ├── dt/   ← Eric.  Histogram CART, Gini. dt.cpp + job_dt.sl
│   │   └── nb/   ← Eric.  Hybrid Gaussian + Bernoulli NB. nb.cpp + job_nb.sl
│   └── sklearn_xgb/          ← Python comparison harness (sklearn + XGBoost)
├── misc/
│   ├── mlp.md                 ← MLP design doc + results log
│   ├── dt.md                  ← DT  design doc + results log
│   └── nb.md                  ← NB  design doc + results log
└── EE451-Final-Paper/         ← empty submodule placeholder for the final paper
```

## Invariants when editing an algorithm file

All five algorithm files follow a tightly aligned convention so analytics_engine, the shared data loader, and cross-algorithm comparisons Just Work. Don't drift from it without a cross-algorithm refactor.

- **Shared public interface** (identical signature shape across all 5):
  - `using Labels = std::vector<int>;`
  - `struct Dataset { int n_features; std::vector<std::string> feature_names; std::vector<float> X_train, X_test; Labels y_train, y_test; };`
  - `struct Metrics { double acc = 0.0, prec = 0.0, rec = 0.0, f1 = 0.0; };`
  - `static Dataset load_dataset(const std::string&, const std::string&);`
  - `static XModel train_serial(const std::vector<float>& X, const Labels& y, int n_rows, int n_features, ...);`
  - `static XModel train_parallel_omp(...)` and `static XModel train_parallel_pthreads(...)` — same sig as train_serial.
  - `static Labels predict_serial(const XModel&, const std::vector<float>& X, int n_rows, int n_features);`
  - `static Labels predict_parallel(const XModel&, const std::vector<float>& X, int n_rows, int n_features);` — OpenMP over test rows.
  - `static Metrics evaluate(const Labels& truth, const Labels& pred);`
  - `static void print_results(const std::string& tag, double train_ms, double infer_ms, const Metrics&);`
  - `int main(int argc, char* argv[])` — takes optional `train_csv` and `test_csv` paths; binaries also accept algorithm-specific hyperparams as argv[3+] (see each file's header).

- **Shared load_dataset behavior**:
  - Find the label column by searching the header for `"round_winner"` (NOT a hardcoded column index).
  - `resolve_path(path)` searches `./`, `../`, `data/`, `../data/`, `../../data/`, `../../../data/` — so binaries work from the repo root or any src/cpp/<algo>/ subfolder.
  - Skip z-score normalization for: column names with prefix `"map_"`, exact name `"bomb_planted"`, and any column whose training values are all in `{0.0, 1.0}`. Passthrough columns keep their original 0/1 values (mean=0, sd=1 for them, so the uniform normalize pass leaves them alone).
  - Print `Dataset Info` block with `Normalize: N z-scored, M passthrough (binary/one-hot)` summary.

- **Self-contained files**: loader + metrics code is **duplicated** across all five algorithm files by design. Do NOT preemptively extract them into a shared header — deferred as a followup once the paper is done. Each new algorithm file should copy the same loader/metrics block verbatim.

- **Hyperparameters at top-of-file as `static constexpr`**: `N_THREADS = 8`, `SEED = 42` are common to all. Algorithm-specific constants (MAX_EPOCHS, MAX_DEPTH, K_NEIGHBORS, LAMBDA, etc.) live above the `Labels` alias.

- **Plain-text console output** — every algorithm prints a `Dataset Info` block, `[tag]` metrics blocks per trainer, and a final `Speedup Summary`. **No Unicode box-drawing** — that was dropped during the SVM/KNN → MLP/DT/NB alignment refactor. analytics_engine parses the `[tag]` blocks by substring matching, so don't change the spacing (e.g., `"  Training time  : X ms"` must keep two spaces before the colon).

- **Deterministic parallel output**: parallel variants must produce the same predictions as serial (up to floating-point reduction order), and each algorithm's `main()` prints an `Accuracy parity check` block at the end comparing `|serial − omp|` and `|serial − pthreads|`. Expect `< 0.01` (often 0.0000).

## Build

Single-file compile per algorithm, flags fixed for the paper's apples-to-apples comparison:

```bash
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/<algo>/<algo>.cpp -o <algo> -lpthread
```

Exactly what [job.sl](job.sl) runs on CARC. `-march=native` means builds are tied to the machine; don't move binaries between architectures.

## Running

On CARC (the correct path):

```bash
sbatch job.sl              # compiles + runs all five; output to alljob.out
sbatch job_analytics.sl    # runs analytics_engine; emits analytics_results.csv
```

Per-algo SLURM scripts live in each algo's folder (e.g., [src/cpp/svm/job_svm.sl](src/cpp/svm/job_svm.sl)) and are run from within that folder with `cd src/cpp/svm && sbatch job_svm.sl`.

Local dev on macOS is intentionally unsupported — Apple Clang ships without OpenMP, and `pthread_barrier_t` is not implemented on Darwin. If you need to iterate locally:
- `brew install libomp` and use `clang++ -Xpreprocessor -fopenmp -lomp ...`, OR
- `brew install gcc` and use `g++-14` directly.
- Know that local Apple Clang toolchains can ship broken against libc++ for months at a time (2026 Xcode 13/macOS 26 header mismatch, etc.). If `#include <iostream>` fails on a syntax-check, it's the environment, not the code.

## Parallelization conventions

- **OpenMP for loop-level parallelism and inference** (`#pragma omp parallel for schedule(static) num_threads(N_THREADS)` over independent samples).
- **pthreads for training with shared mutable state**. Reference pattern in [src/cpp/svm/svm.cpp](src/cpp/svm/svm.cpp) and [src/cpp/mlp/mlp.cpp](src/cpp/mlp/mlp.cpp): `ParState` + thread-local compute + `pthread_mutex_t` for reduction + `pthread_barrier_t`s for per-iteration sync (two barriers per epoch/batch/node; one mutex-acquisition per worker for non-iterative NB). Copy the pattern that fits the algorithm's granularity.
- **Thread-local buffers must be hoisted out of any inner loop** (allocate once per thread, `std::fill` to zero per iteration). Allocating `Vec(thousands)` per batch dominates runtime.
- **Only thread 0 may mutate shared scheduler state** (index shuffling, `batch_start`, epoch counters). Otherwise parallel-vs-serial comparisons become meaningless.
- **Never assert bitwise equality between serial and parallel outputs** — float reduction order is non-deterministic. Check `|acc_serial − acc_parallel| < 0.01` instead (this check is already in each algorithm's `main()`).

## Data

- Cleaned CSVs at `data/`, already one-hot-encoded by [src/cpp/svm/Data.ipynb](src/cpp/svm/Data.ipynb). Column `round_winner` is the label in `{+1, −1}`; all other 103 columns are numeric features. Column ORDER in the header is load-order-dependent but the loader finds the label by name so column-order changes won't break anything.
- To regenerate from the raw Kaggle dump: re-run [split_data.sh](split_data.sh) to produce the pre-cleaning splits into `data/`, then run [src/cpp/svm/Data.ipynb](src/cpp/svm/Data.ipynb) which writes the cleaned CSVs to `data/`.
- Never commit the raw Kaggle CSV — it's not tracked, and at ~50 MB+ it doesn't belong in git.

## Documentation

- [misc/mlp.md](misc/mlp.md), [misc/dt.md](misc/dt.md), [misc/nb.md](misc/nb.md) — per-algorithm design docs + results logs. Append a row per CARC run.
- [src/sklearn_xgb/README.md](src/sklearn_xgb/README.md) — sklearn/XGBoost comparison harness overview.
- The project proposal ([451 project proposal.pdf](451%20project%20proposal.pdf) / [proposal.tex](proposal.tex)) is the source of truth for the hypothesis and deliverables. The **MLP is the headline algorithm** — the proposal predicts it will scale best of all five due to its high arithmetic intensity, and the ratio `speedup_MLP / speedup_DT` is the central empirical claim.

## What NOT to do

- Do **not** use external ML libraries (scikit-learn, Eigen, BLAS, etc.) in the C++ files. Hand-rolled C++ parallel code is the point of the paper. `src/sklearn_xgb/` is the intentional exception — it's the baseline comparison harness, not part of the five algorithms.
- Do **not** refactor the duplicated `load_dataset` / `evaluate` into a shared header until the paper is done. The duplication is deliberate — it keeps each algorithm self-contained and easy to reason about.
- Do **not** change compile flags for a single algorithm. Apples-to-apples comparison across the five models requires identical flags.
- Do **not** add commits, push branches, or open PRs without being asked — this is a student project and git actions should be explicitly directed.
- Do **not** modify [data/train_cleaned.csv](data/train_cleaned.csv) or [data/test_cleaned.csv](data/test_cleaned.csv) — they are the fixed experiment input.
- Do **not** add `-ffast-math` to any compile line. Gini gain / cross-entropy tie-breaking can drift by 1e-15 under `-ffast-math` and break the serial-vs-parallel determinism check.
