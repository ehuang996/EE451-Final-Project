# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

EE 451 (USC, Distributed & Parallel Computing) **final project**: a from-scratch, pure-C++17 empirical study of five classical ML algorithms parallelized with OpenMP + pthreads, benchmarked on the **CS:GO Round Winner** dataset.

- Binary classification: `round_winner ∈ {+1, −1}` from 103 numeric features.
- Train: ~97,929 rows in [train_cleaned.csv](train_cleaned.csv). Test: ~24,483 rows in [test_cleaned.csv](test_cleaned.csv).
- Target per-model accuracy: **76–80%** (community benchmark).
- Parallelism target: **8 threads** (SLURM `--cpus-per-task=8`).
- Hardware: USC CARC cluster; local builds are a convenience, not required.

The five algorithms: **KNN, SVM, Naive Bayes, Decision Tree, MLP**. Each is a single, self-contained `.cpp` file with a shared public interface (`load_dataset`, `train_serial`, `train_parallel`, `predict_serial`, `predict_parallel`, `evaluate`). A future `analytics_engine.cpp` will drive all five through the same interface for the writeup's comparison tables.

Team: Eric Huang, Harry Yang, Jinglu Sun, Mo Jiang, Pinru Wang. Eric owns MLP.

## Repo layout

```
├── 451 project proposal.pdf   ← project proposal (read this for context)
├── proposal.tex               ← LaTeX source
├── train_cleaned.csv          ← 97929 × 104 (103 features + round_winner at col 95)
├── test_cleaned.csv           ← 24483  × 104
├── split_data.sh              ← 80/20 shuffle-split of raw Kaggle CSV
├── job.sl                     ← SLURM job script; compiles + runs each algorithm
├── svm/
│   ├── svm.cpp                ← DONE (teammate). Linear SVM, hinge loss + L2, pthreads training, OMP inference.
│   ├── Data.ipynb             ← preprocessing notebook (one-hot maps, label encoding)
│   └── cs_go_winner_data_{train,test}.csv  ← raw pre-cleaning splits
├── mlp/
│   └── mlp.cpp                ← DONE (Eric). 103→64→32→1 MLP with BCE loss, four trainers.
├── misc/
│   └── mlp.md                 ← MLP design doc + results log. UPDATE WHEN RESULTS COME IN.
└── EE451-Final-Paper/         ← empty submodule placeholder for the final paper
```

Not yet implemented: `knn.cpp`, `nb.cpp`, `dt.cpp`, `analytics_engine.cpp`.

## Invariants when editing an algorithm file

- **Public interface must match SVM's**. All five algorithms will be driven by a shared `analytics_engine.cpp`, so every file exposes the same symbol names:
  - `Dataset`, `Metrics` structs; `Vec`/`Matrix`/`IVec` typedefs (identical across files).
  - `Dataset load_dataset(const std::string& train_path, const std::string& test_path)` — reads CSV, skips header, parses column 95 as the label, z-score normalizes using **train-set statistics only**, prints a class-distribution summary box.
  - `Metrics evaluate(const IVec& truth, const IVec& pred)` — TP/FP/TN/FN → acc/prec/rec/F1.
  - `XModel train_serial(...)`, `XModel train_parallel(...)`, `IVec predict_serial(...)`, `IVec predict_parallel(...)` — X ∈ {SVM, MLP, KNN, ...}.
  - `main(argc, argv)` — takes optional `train_csv` and `test_csv` paths, defaults to `train_cleaned.csv` / `test_cleaned.csv`.
- **Self-contained files**: loader + metrics code is currently **duplicated** between `svm/svm.cpp` and `mlp/mlp.cpp` by design. Do NOT preemptively extract them into a shared header — the dedup will happen when `analytics_engine.cpp` is written. Each new algorithm file should copy the same loader/metrics block verbatim.
- **Hyperparameters at top-of-file as `static constexpr`** (see SVM's and MLP's header blocks). `N_FEATURES = 103`, `N_THREADS = 8`, `SEED = 42` are common across all files.
- **Box-drawing console output** — every algorithm prints a Unicode box-drawn summary matching SVM's style (see `print_results` and the final speedup box).

## Build

Single-file compile per algorithm, flags fixed for the paper's apples-to-apples comparison:

```bash
g++ -std=c++17 -O3 -march=native -fopenmp <file>.cpp -o <name> -lpthread
```

Exactly what [job.sl](job.sl) runs on CARC. `-march=native` means builds are tied to the machine; don't move binaries between architectures.

## Running

On CARC (the correct path):
```bash
sbatch job.sl
# outputs to gpujob.out; compiles svm + mlp, runs both
```

Local dev on macOS is intentionally unsupported — Apple Clang ships without OpenMP, and `pthread_barrier_t` is not implemented on Darwin (it's an optional POSIX feature Apple omits). If you need to iterate locally:
- `brew install libomp` and use `clang++ -Xpreprocessor -fopenmp -lomp ...`, OR
- `brew install gcc` and use `g++-14` directly.
- Know that local Apple Clang toolchains can ship broken against libc++ for months at a time (see the 2026 Xcode 13/macOS 26 header mismatch). If `#include <iostream>` fails, it's the environment, not the code.

## Parallelization conventions

- **OpenMP is for loop-level parallelism and inference** (`#pragma omp parallel for schedule(static) num_threads(N_THREADS)` over independent samples).
- **pthreads is for training with shared mutable state** (gradient accumulators). The reference pattern is in [svm/svm.cpp](svm/svm.cpp): `ParState` + thread-local compute + `pthread_mutex_t` for reduction + two `pthread_barrier_t`s per epoch (`barrier_acc`, `barrier_upd`). [mlp/mlp.cpp](mlp/mlp.cpp) extends this to per-mini-batch synchronization. Copy this pattern for KNN/NB/DT where applicable.
- **Thread-local buffers must be hoisted out of any inner loop** (allocate once per thread, `std::fill` to zero per iteration). Allocating `Vec(thousands)` per batch dominates runtime.
- **Only thread 0 may mutate shared scheduler state** (index shuffling, `batch_start`, epoch counters). Otherwise parallel-vs-serial comparisons become meaningless.
- **Never assert bitwise equality between serial and parallel outputs** — float reduction order is non-deterministic. Check `|acc_serial − acc_parallel| < 0.01` instead.

## Data

- Cleaned CSVs at repo root, already normalized for the loader's expectations (column 95 = label in {+1, −1}, all other columns numeric).
- To regenerate from the raw Kaggle dump: re-run [split_data.sh](split_data.sh) to produce the pre-cleaning splits into [svm/](svm/), then run [svm/Data.ipynb](svm/Data.ipynb) which writes the cleaned CSVs back to the root.
- Never commit the raw Kaggle CSV — it's not yet tracked, and at ~50 MB+ it doesn't belong in git.

## Documentation

- [misc/mlp.md](misc/mlp.md) — MLP design decisions + results log. **Every CARC run of the MLP binary should have its summary appended to the results table.**
- The project proposal ([451 project proposal.pdf](451%20project%20proposal.pdf) / [proposal.tex](proposal.tex)) is the source of truth for the hypothesis and deliverables. The **MLP is the headline algorithm** — the proposal hypothesizes it will scale best of all five due to its high arithmetic intensity, so the MLP speedup numbers are the central empirical claim of the paper.

## What NOT to do

- Do **not** use external ML libraries (scikit-learn, Eigen, BLAS, etc.). The entire point of the paper is hand-rolled C++ parallel code.
- Do **not** refactor the duplicated `load_dataset` / `evaluate` into a shared header until `analytics_engine.cpp` is being written. The duplication is deliberate.
- Do **not** change compile flags for a single algorithm. Apples-to-apples comparison across the five models requires identical flags.
- Do **not** add commits, push branches, or open PRs without being asked — this is a student project and git actions should be explicitly directed.
- Do **not** modify [train_cleaned.csv](train_cleaned.csv) or [test_cleaned.csv](test_cleaned.csv) — they are the fixed experiment input.
