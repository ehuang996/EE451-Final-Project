# Parallelization of Classical Machine Learning Methods

An empirical study on the parallelization of five classical machine learning
algorithms — **KNN, SVM, Naive Bayes, Decision Tree, and MLP** — implemented
from scratch in pure C++17 and parallelized with OpenMP and POSIX threads.
Final project for EE 451 (Distributed & Parallel Computing) at USC.

The central question: how does each algorithm's **arithmetic intensity** and
**data dependency structure** govern its parallel speedup on a multi-core CPU?
Full motivation and deliverables in
[451 project proposal.pdf](451%20project%20proposal.pdf) /
[proposal.tex](proposal.tex).

## Team

Eric Huang, Harry Yang, Jinglu Sun, Mo Jiang, Pinru Wang — University of
Southern California.

## Dataset

[CS:GO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification/data)
(Lillelund, 2020). 122,411 round snapshots from ~700 professional
matches (2019–2020). Binary target `round_winner ∈ {CT, T}`. After one-hot
encoding the map feature, each record has 103 numeric features.

- 80/20 train/test split: 97,929 training rows, 24,483 test rows.
- Cleaned CSVs at repo root: [train_cleaned.csv](train_cleaned.csv),
  [test_cleaned.csv](test_cleaned.csv).
- Community benchmark accuracy: 76–80%.

## Algorithms

| Algorithm              | File                       | Status       | Owner   |
|------------------------|----------------------------|--------------|---------|
| Support Vector Machine | [svm/svm.cpp](svm/svm.cpp) | Complete     | Harry   |
| Multilayer Perceptron  | [mlp/mlp.cpp](mlp/mlp.cpp) | Complete     | Eric    |
| K-Nearest Neighbors    | `knn/knn.cpp` (TBD)        | Not started  | —       |
| Naive Bayes            | `nb/nb.cpp`   (TBD)        | Not started  | —       |
| Decision Tree          | `dt/dt.cpp`   (TBD)        | Not started  | —       |
| Benchmark driver       | `analytics_engine.cpp`     | Not started  | —       |

All algorithm files expose the same public interface (`load_dataset`,
`train_serial`, `train_parallel`, `predict_serial`, `predict_parallel`,
`evaluate`) so the eventual `analytics_engine.cpp` can run all five through
one benchmark harness. See [CLAUDE.md](CLAUDE.md) for the shared conventions.

## Hypothesis

We hypothesize that KNN and Naive Bayes will achieve **high initial speedup**
before hitting memory bandwidth limits, that Decision Trees and SVM will be
constrained by **synchronization overhead and iterative convergence**, and that
MLP will scale **most effectively** of the five due to its high arithmetic
intensity and suitability for SIMD optimizations.

## Repository layout

```
├── 451 project proposal.pdf   Project proposal (PDF)
├── proposal.tex               Project proposal (LaTeX source)
├── README.md                  This file
├── CLAUDE.md                  Repo-level conventions for AI assistance
├── job.sl                     SLURM job script (USC CARC)
├── split_data.sh              80/20 shuffle-split of the raw Kaggle CSV
├── train_cleaned.csv          Cleaned training split (97929 × 104)
├── test_cleaned.csv           Cleaned test split     (24483 × 104)
├── svm/
│   ├── svm.cpp                Linear SVM, hinge loss + L2, pthreads trainer
│   ├── Data.ipynb             Data-cleaning notebook (one-hot maps, labels)
│   └── cs_go_winner_data_{train,test}.csv   Raw pre-cleaning splits
├── mlp/
│   └── mlp.cpp                MLP, BCE + L2, OMP + pthreads trainers
└── misc/
    └── mlp.md                 MLP design doc + per-run results log
```

## Build & run

### USC CARC (primary)

```bash
sbatch job.sl
```

Compiles and runs both `svm` and `mlp` against the cleaned CSVs. Output goes
to `gpujob.out`. SLURM config: 8 CPU cores, 16 GB memory, 1-hour wall time.

### Local compile

The build command for each algorithm is intentionally the same so comparisons
stay apples-to-apples:

```bash
g++ -std=c++17 -O3 -march=native -fopenmp svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp mlp/mlp.cpp -o mlp -lpthread
./svm train_cleaned.csv test_cleaned.csv
./mlp train_cleaned.csv test_cleaned.csv
```

**macOS note:** Apple Clang ships without OpenMP and the Darwin pthreads
library omits `pthread_barrier_t`. Use Homebrew GCC (`brew install gcc`) or
install `libomp` and pass `-Xpreprocessor -fopenmp -lomp` to Apple Clang.
Easier path: just run on CARC.

## Output

Each binary prints a box-drawn summary per trainer (serial and parallel
variants), then a final speedup table. Example MLP summary shape:

```
┌─────────────────────────────────────────────────┐
│  Speedup Analysis                               │
├─────────────────────────────────────────────────┤
│  Serial total           : <T_serial>  ms
│  Better Serial total    : <T_better>  ms
│  Parallel OMP total     : <T_omp>     ms
│  Parallel pthreads total: <T_pth>     ms
│  Speedup (OMP)          : <S_omp>     x
│  Speedup (pthreads)     : <S_pth>     x
│  Threads used           : 8
└─────────────────────────────────────────────────┘
```

Reported per-model quality metrics: accuracy, precision, recall, F1.
System metrics: training time, inference time, parallel speedup.
MLP-specific design & results log in [misc/mlp.md](misc/mlp.md).

## References

[1] C. Lillelund, "CS:GO Round Winner Classification,"
<https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification/data>, 2020.

See the proposal for the full reference list.
