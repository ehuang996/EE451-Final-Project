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
- Cleaned CSVs live under [data/](data/): [data/train_cleaned.csv](data/train_cleaned.csv),
  [data/test_cleaned.csv](data/test_cleaned.csv). Raw pre-cleaning splits
  also live under `data/` (`cs_go_winner_data_{train,test}.csv`).
- Community benchmark accuracy: 76–80%.

## Algorithms

| Algorithm              | File                                       | Status       | Owner   |
|------------------------|--------------------------------------------|--------------|---------|
| Support Vector Machine | [src/cpp/svm/svm.cpp](src/cpp/svm/svm.cpp) | Complete     | Harry   |
| K-Nearest Neighbors    | [src/cpp/knn/knn.cpp](src/cpp/knn/knn.cpp) | Complete     | Mo      |
| Multilayer Perceptron  | [src/cpp/mlp/mlp.cpp](src/cpp/mlp/mlp.cpp) | Complete     | Eric    |
| Decision Tree          | [src/cpp/dt/dt.cpp](src/cpp/dt/dt.cpp)     | Complete     | Eric    |
| Naive Bayes            | [src/cpp/nb/nb.cpp](src/cpp/nb/nb.cpp)     | Complete     | Eric    |
| Benchmark driver       | [analytics_engine.cpp](analytics_engine.cpp) | Complete   | Eric    |
| sklearn/XGBoost baselines | [src/sklearn_xgb/](src/sklearn_xgb/)    | Complete     | Eric    |

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
├── job.sl                     SLURM — compiles + runs all five algos
├── job_analytics.sl           SLURM — runs analytics_engine across all five
├── analytics_engine.cpp       Master benchmarking driver (subprocess + CSV out)
├── split_data.sh              80/20 shuffle-split of the raw Kaggle CSV
├── data/
│   ├── train_cleaned.csv      Cleaned training split (97929 × 104)
│   ├── test_cleaned.csv       Cleaned test split     (24483 × 104)
│   └── cs_go_winner_data_{train,test}.csv   Raw pre-cleaning splits
├── src/
│   ├── cpp/
│   │   ├── svm/
│   │   │   ├── svm.cpp        Linear SVM, hinge loss + L2, OMP + pthreads
│   │   │   ├── job_svm.sl     Per-algo SLURM script (outputs gpujob.out)
│   │   │   └── Data.ipynb     Data-cleaning notebook (one-hot maps, labels)
│   │   ├── knn/
│   │   │   ├── knn.cpp        K-Nearest Neighbors, OMP + pthreads inference
│   │   │   └── job_knn.sl     Per-algo SLURM script (outputs knnjob.out)
│   │   ├── mlp/
│   │   │   ├── mlp.cpp        MLP, BCE + L2, sample-loop + optional BLAS batch backend
│   │   │   └── job_mlp.sl     Per-algo SLURM script (outputs mlpjob.out)
│   │   ├── dt/
│   │   │   ├── dt.cpp         Decision Tree, histogram CART, OMP + pthreads
│   │   │   └── job_dt.sl      Per-algo SLURM script (outputs dtjob.out)
│   │   └── nb/
│   │       ├── nb.cpp         Naive Bayes, Gaussian+Bernoulli, OMP + pthreads
│   │       └── job_nb.sl      Per-algo SLURM script (outputs nbjob.out)
│   └── sklearn_xgb/           sklearn + XGBoost comparison harness (Python)
│       ├── compare.py
│       ├── loader.py
│       ├── hybrid_nb.py
│       ├── parse_cpp_output.py
│       ├── job_compare.sl     SLURM — builds C++ + runs Python baselines
│       ├── requirements.txt
│       └── README.md
└── misc/
    ├── mlp.md                 MLP design doc + per-run results log
    ├── dt.md                  DT design doc + per-run results log
    └── nb.md                  NB design doc + per-run results log
```

## Build & run

### USC CARC (primary)

Run all five algorithms in one job (from the repo root):

```bash
sbatch job.sl       # outputs combined run to alljob.out
```

Or run one algorithm at a time (useful when iterating on a single algo).
Run from that algo's own folder so output files land next to the source:

```bash
(cd src/cpp/svm && sbatch job_svm.sl)   # outputs gpujob.out in src/cpp/svm/
(cd src/cpp/knn && sbatch job_knn.sl)   # outputs knnjob.out in src/cpp/knn/
(cd src/cpp/mlp && sbatch job_mlp.sl)   # outputs mlpjob.out in src/cpp/mlp/
(cd src/cpp/dt  && sbatch job_dt.sl)    # outputs dtjob.out  in src/cpp/dt/
(cd src/cpp/nb  && sbatch job_nb.sl)    # outputs nbjob.out  in src/cpp/nb/
```

For the consolidated benchmark CSV used by the writeup, run the analytics
engine — it invokes each algorithm as a subprocess, parses their `[tag]`
output blocks, and emits `analytics_results.csv`:

```bash
sbatch job_analytics.sl     # outputs to analyticsjob.out
# produces analytics_results.csv with columns:
#   algorithm,variant,n_threads,train_ms,infer_ms,total_ms,accuracy,
#   precision,recall,f1
```

For the sklearn + XGBoost comparison harness:

```bash
sbatch src/sklearn_xgb/job_compare.sl   # outputs to src/sklearn_xgb/comparejob.out
# produces src/sklearn_xgb/results.csv
```

All SLURM configs: 8 CPU cores, 16 GB memory, 1-hour wall time.

Optional vendor-BLAS backends on CARC:

```bash
USE_KNN_BLAS=1 sbatch src/cpp/knn/job_knn.sl
USE_MLP_BLAS=1 sbatch src/cpp/mlp/job_mlp.sl
USE_KNN_BLAS=1 sbatch slurm/job_sweep.sl
USE_MLP_BLAS=1 sbatch slurm/job_sweep.sl
```

### Local compile

The build command for each algorithm is intentionally the same so comparisons
stay apples-to-apples. Run these from the repo root:

```bash
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/knn/knn.cpp -o knn -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/mlp/mlp.cpp -o mlp -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread
./svm data/train_cleaned.csv data/test_cleaned.csv
./knn data/train_cleaned.csv data/test_cleaned.csv
./mlp data/train_cleaned.csv data/test_cleaned.csv
./dt  data/train_cleaned.csv data/test_cleaned.csv
./nb  data/train_cleaned.csv data/test_cleaned.csv
```

KNN also has an optional vendor-BLAS backend for the counterfactual comparison
against sklearn's brute-force path:

```bash
# Linux/OpenBLAS
g++ -std=c++17 -O3 -march=native -fopenmp -DKNN_USE_BLAS \
  src/cpp/knn/knn.cpp -o knn -lpthread -lopenblas
OPENBLAS_NUM_THREADS=1 N_THREADS=8 ./knn data/train_cleaned.csv data/test_cleaned.csv

# macOS/Accelerate
clang++ -std=c++17 -O3 -DKNN_USE_BLAS \
  src/cpp/knn/knn.cpp -o knn -lpthread -framework Accelerate
VECLIB_MAXIMUM_THREADS=1 N_THREADS=8 ./knn data/train_cleaned.csv data/test_cleaned.csv
```

For Intel oneMKL, build with `-DKNN_USE_MKL` and the link line generated by
Intel's MKL link advisor for your compiler/runtime.

Use `KNN_BACKEND=blocked` with a BLAS-compiled binary to force the pure C++
blocked-dot kernel, and `KNN_BLAS_BLOCK=<queries>` to tune SGEMM query-block
size.

MLP also has an optional vendor-BLAS backend. Unlike KNN, the BLAS path owns
the dense minibatch kernels directly, so the recommended run mode is to hand
the thread budget to the BLAS library rather than pinning it to 1:

```bash
# Linux/OpenBLAS
g++ -std=c++17 -O3 -march=native -fopenmp -DMLP_USE_BLAS \
  src/cpp/mlp/mlp.cpp -o mlp -lpthread -lopenblas
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 N_THREADS=8 \
  ./mlp data/train_cleaned.csv data/test_cleaned.csv

# macOS/Accelerate
clang++ -std=c++17 -O3 -DMLP_USE_BLAS \
  src/cpp/mlp/mlp.cpp -o mlp -lpthread -framework Accelerate
VECLIB_MAXIMUM_THREADS=8 N_THREADS=8 ./mlp data/train_cleaned.csv data/test_cleaned.csv
```

Use `MLP_BACKEND=loop` with a BLAS-compiled binary to force the original pure
C++ sample-loop kernel, and `MLP_BLAS_BLOCK=<rows>` to tune batched BLAS
prediction blocking.

Each binary's `resolve_path` helper searches `./`, `../`, `data/`, `../data/`,
`../../data/`, and `../../../data/` — so you can also run them from
`src/cpp/<algo>/` with no CSV args and they'll still find the data.

CLI signatures (arg 3+ overrides hyperparameters, see each algo's header):

```bash
./svm train test [epochs] [lr] [lambda]
./knn train test [k]
./mlp train test [epochs] [lr] [lambda]
./dt  train test [max_depth]
./nb  train test       # no CLI hyperparameters — closed-form MLE
```

**macOS note:** Apple Clang ships without OpenMP and the Darwin pthreads
library omits `pthread_barrier_t`. Use Homebrew GCC (`brew install gcc`) or
install `libomp` and pass `-Xpreprocessor -fopenmp -lomp` to Apple Clang.
Easier path: just run on CARC.

## Output

Each binary prints a plain-text `Dataset Info` block, a `[tag]` block per
trainer with per-model quality metrics (accuracy, precision, recall, F1) and
system metrics (training time, inference time), and a `Speedup Summary` at
the end. Example MLP summary shape:

```
Speedup Summary
  Serial total       : <T_serial>  ms
  Parallel OMP total : <T_omp>     ms
  Parallel pth total : <T_pth>     ms
  Speedup (OMP)      : <S_omp>     x
  Speedup (pthreads) : <S_pth>     x
```

All five algorithms report **two** parallel variants — OpenMP and pthreads —
plus an accuracy parity check at the end so you can confirm the parallel
variants produce the same model as serial. The OMP-vs-pthreads comparison
lets the writeup isolate synchronization-primitive overhead from
parallelization-strategy overhead.

Algorithm-specific design & results logs:
- MLP: [misc/mlp.md](misc/mlp.md)
- DT:  [misc/dt.md](misc/dt.md)
- NB:  [misc/nb.md](misc/nb.md)

## References

[1] C. Lillelund, "CS:GO Round Winner Classification,"
<https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification/data>, 2020.

See the proposal for the full reference list.
