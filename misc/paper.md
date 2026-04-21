# Paper — Research Question, Thesis, and Evidence Plan

Working document for the EE 451 final paper. This is the living artifact that
captures what the paper is going to *argue*, with what *evidence*, and why
that argument is non-trivial. The paper itself will be drafted in
[EE451-Final-Paper/](../EE451-Final-Paper/) (currently empty).

---

## 1. Research question

> **At 1 thread are our hand-rolled C++17 implementations of five classical
> ML algorithms competitive with scikit-learn, and do they scale to 8 threads
> where scikit-learn can't — exposing a gap in the mainstream parallel-CPU-ML
> ecosystem?**

The question is framed at two thread counts on purpose: at `N_THREADS=1`
both our C++ and scikit-learn are running the same algorithm on one core, so
the comparison isolates **implementation quality** (sklearn is well-optimized
Cython; we are hand-rolled C++17 with `-O3 -march=native`) — either we are
competitive or we aren't, and either outcome is a real data point. At
`N_THREADS=8` the question shifts to **scaling**, and the punchline is that
for three of our five algorithms (SVM, Decision Tree, Naive Bayes) the
mainstream Python ML stack ships **no parallel CPU implementation at all**,
so scaling is free headroom we get and sklearn users do not. The
`N_THREADS ∈ {1, 2, 4, 8}` sweep connects the two endpoints with an
Amdahl-style curve per algorithm.

---

## 2. Thesis (the one sentence the paper defends)

> On a modern multi-core CPU, hand-rolled parallel C++17 implementations of
> classical ML algorithms are **competitive with scikit-learn at 1 thread**
> and pull ahead at 2+ threads because **scikit-learn's default is serial
> for 3 of our 5 algorithms** (SVM, DT, NB) and only implicitly parallel for
> the other 2 (MLP via BLAS, KNN via `n_jobs`); for the one algorithm where
> a SOTA parallel CPU baseline exists in wide use (Decision Tree via XGBoost
> histogram), our hand-rolled version is **competitive rather than
> categorically better**, and the arithmetic-intensity roofline correctly
> predicts which algorithms scale best across the thread sweep.

This is the paper's contribution. It is simultaneously an empirical study of
parallel ML algorithm design *and* a light commentary on the CPU ML library
landscape — "there is a parallel SVM shaped hole in the ecosystem" is itself
a finding, not just a data point. The thread sweep is what makes the
comparison fair: without a 1-thread data point we would be comparing 8-thread
C++ to 1-thread sklearn and claiming a speedup that is partly ours and partly
just arithmetic (8 cores > 1 core). With the sweep, the paper can report the
*per-thread* comparison alongside the *scaling* comparison, and reviewers
can't accuse us of a rigged match.

---

## 3. Hypothesis (from the proposal, refined)

The proposal ([proposal.tex:43](../proposal.tex#L43)) commits to a specific
scaling prediction, grounded in per-algorithm arithmetic intensity:

| Algorithm | Predicted behavior                                          | Predicted 8-core speedup |
|-----------|-------------------------------------------------------------|-------------------------:|
| KNN       | Memory-bound; high *initial* speedup then BW-limited        | 2–4×                     |
| NB        | Memory-bound; one-shot frequency counting                   | 2–4×                     |
| SVM       | Memory-bound + sync-limited (iterative GD)                  | 2–4×                     |
| DT        | Crossover regime (histogram fill + Gini sweep)              | 3–5×                     |
| **MLP**   | **Compute-bound; dense GEMM-like; SIMD-friendly**           | **5–7×** ← headline      |

The **MLP scaling** number is the central empirical claim: if the proposal's
hypothesis is right, `speedup_MLP / speedup_NB ≳ 2` on 8 cores. That ratio is
what the paper's main figure should show.

### Arithmetic-intensity comparison (cross-algorithm)

From the per-algorithm design docs' analysis sections:

| Algorithm | Inner kernel                  | FLOPs / byte | Regime          | Source |
|-----------|-------------------------------|-------------:|-----------------|--------|
| SVM       | `dot(w, x)` + margin check    | ~0.5         | Memory-bound    | [misc/mlp.md:306](mlp.md#L306) |
| NB        | freq count + Gaussian MLE     | ~0.75        | Memory-bound    | [misc/nb.md:288](nb.md#L288)   |
| KNN       | squared-L2 distance           | ~0.5         | Memory-bound    | [misc/nb.md:302](nb.md#L302)   |
| DT        | histogram fill + Gini sweep   | ~3           | Crossover       | [misc/nb.md:303](nb.md#L303)   |
| MLP       | dense GEMM forward/back       | ~42          | Compute-bound   | [misc/mlp.md:307](mlp.md#L307) |

AI across 5 algorithms spans **two orders of magnitude**, which is exactly
why a five-algorithm comparison on one dataset is more informative than any
single-algorithm deep dive: the *spread* of observed speedups, plotted
against the analytical AI, is the paper's roofline-adjacent headline figure.

---

## 4. Contributions (what's actually new)

The paper claims three contributions. Each corresponds to an artifact
already in the repo:

1. **Five from-scratch parallel C++17 implementations** with a common public
   interface and matched hyperparameters, so all five are measured on
   identical input under the same compiler flags. Implementations at
   [svm/svm.cpp](../src/cpp/svm/svm.cpp), [knn/knn.cpp](../src/cpp/knn/knn.cpp),
   [mlp/mlp.cpp](../src/cpp/mlp/mlp.cpp), [dt/dt.cpp](../src/cpp/dt/dt.cpp),
   [nb/nb.cpp](../src/cpp/nb/nb.cpp). Each exposes both an OpenMP parallel trainer
   and a pthreads parallel trainer for the same algorithm, so the paper can
   compare **parallelism-primitive overhead** (mutex+barrier vs. `#pragma
   omp parallel`) *independently* of the parallelization strategy itself.

2. **A framework-comparison harness** (at [sklearn_xgb/](../src/sklearn_xgb/))
   that runs each algorithm through scikit-learn and, for DT, XGBoost
   with hyperparameters pulled from the C++ source's `constexpr` block.
   Produces a single unified [results.csv](../src/sklearn_xgb/) (pending first
   CARC run) across all three frameworks. Crucially, the harness labels
   every sklearn row's parallelism mode in the `notes` column, exposing the
   "no parallel CPU baseline exists" fact per-algorithm rather than
   hiding it. Design rationale documented in [misc/sk_learn_xgb.md](sk_learn_xgb.md).

3. **A cross-algorithm arithmetic-intensity roofline analysis** that
   *predicts* which algorithms scale best before any measurement, then
   validates against the empirical speedup curves. This is the analytical
   substance of the paper and the part most likely to generalize beyond
   CS:GO. The per-algorithm AI derivations are written out in the design
   docs; the paper synthesizes them into one figure.

### What is *not* a contribution

We should not overclaim:

- **Parallelizing classical ML with OpenMP + pthreads** is a textbook
  exercise that has been done many times since ~2010. Our implementations
  are not algorithmically novel.
- **The dataset is small** (~122K rows × 103 features fits in L3). This is
  not a claim about HPC-scale ML.
- **We do not evaluate against GPU implementations** (PyTorch, RAPIDS,
  etc.) or distributed-memory systems (Spark MLlib). CPU-only, single-node
  is the scope.

The honest contribution is **the framing and the comparison methodology**,
not the raw fact that C++ can be parallelized.

---

## 5. Why the comparison is non-trivial (for the intro)

Three observations that a reviewer might not realize without looking into
the sklearn source, and which justify the paper's framing:

1. **`sklearn.linear_model.SGDClassifier` has no `n_jobs` parameter.** It
   uses single-threaded Cython. So does `sklearn.tree.DecisionTreeClassifier`.
   So do `sklearn.naive_bayes.GaussianNB` and `BernoulliNB`. Three of our
   five algorithm classes have **no parallel sklearn baseline at all**.

2. **`sklearn.neural_network.MLPClassifier` is "parallel" only through
   implicit BLAS threading.** There is no `n_jobs`; parallelism requires
   `OMP_NUM_THREADS=8` to be set before the first numpy import and depends
   on the BLAS implementation numpy is linked against (OpenBLAS, MKL,
   Accelerate). This is the sklearn user's normal path to parallel MLP
   training but is not obvious from the API.

3. **`sklearn.neighbors.KNeighborsClassifier(n_jobs=8)` is the only one of
   our five that has a first-class `n_jobs` knob.** Even there, the
   parallelism is joblib process-pool over query rows, not threads — a
   different cost model from our OpenMP implementation.

Conclusion: for a user coming from sklearn, "parallel classical ML on CPU"
has *exactly one well-supported path* (kNN), *one implicit path* (MLP via
BLAS), and *three unsupported paths* (SVM, DT, NB). The paper uses this
observation to motivate the contribution rather than to dunk on sklearn —
the library is well-engineered for correctness and API quality, not wall-
clock parallelism. The gap is real, and closing it for DT requires reaching
out to a whole separate library (XGBoost).

---

## 6. Background: dataset, preprocessing, and problem framing

### 6.1 Dataset

**CS:GO Round Winner Classification** (Lillelund 2020, Kaggle). 122,411
round snapshots from ~700 professional Counter-Strike matches, 2019–2020.
Binary target `round_winner ∈ {CT, T}`, encoded as `{+1, −1}` on disk.

- **80/20 train/test split:** 97,929 train, 24,483 test rows.
- **103 numeric features** after one-hot encoding the categorical `map`
  feature. 72 continuous (team health, money, weapon counts, etc.), 31
  binary (8 `map_*` one-hots + `bomb_planted` + 22 weapon-count columns
  that happen to be all-0/1 on the training set).
- **Community benchmark accuracy:** 76–80% on binary classification. All
  five of our algorithms target this range.

The dataset is small — fits in L3 cache. That is a **deliberate scope
choice**: with the whole training set cached, parallel overhead
(synchronization, cache-line bouncing, thread spin-up) is magnified
relative to memory-bandwidth effects, making the *differences between
algorithms' scaling behavior* more visible at 8 cores than they would be
on a dataset that dominates memory bandwidth regardless of algorithm.

### 6.2 Preprocessing

All five algorithms share one `load_dataset` function (duplicated verbatim
across [svm/svm.cpp:193-209](../src/cpp/svm/svm.cpp#L193-L209) and mirrors in KNN,
MLP, DT, NB — deliberate per [CLAUDE.md](../CLAUDE.md) pending a future
shared `common.hpp`). Normalization rules:

- **Z-score using train-set statistics only** — the de facto standard.
- **Skip list** — columns with prefix `map_`, the exact column
  `bomb_planted`, and any column whose training values are all in `{0, 1}`.
  These pass through unchanged (by setting `mean=0, sd=1` in the common
  loop). This matters for NB: the hybrid Gaussian/Bernoulli NB dispatches
  on the same `is_binary` mask.
- **Labels stay in `{+1, −1}`** throughout the C++ codebase. Only XGBoost
  (which demands `{0, 1}`) remaps at the framework boundary.

### 6.3 Problem framing

Binary classification. Evaluation metrics: accuracy, precision, recall, F1
(macro-style TP/FP/TN/FN — all five C++ files have a byte-identical
`Metrics evaluate(truth, pred)` function). System metrics: training wall
time (ms), inference wall time (ms), 8-core speedup `S = T_serial /
T_parallel`. Target accuracy range per algorithm:

| Algorithm | Target accuracy | Rationale                                   |
|-----------|-----------------|---------------------------------------------|
| SVM       | 0.76–0.80       | Community benchmark; linear model sufficient |
| KNN       | 0.76–0.80       | Community benchmark                         |
| MLP       | 0.76–0.80       | Community benchmark                         |
| DT        | 0.76–0.80       | Community benchmark                         |
| NB        | **0.70–0.76**   | Feature-independence assumption violated ([misc/nb.md:24-28](nb.md#L24-L28)) |

NB is the expected accuracy-laggard, which is itself a useful data point:
it lets the paper separate "speedup" from "practical utility" without
relying on accidentally-poor implementations.

---

## 7. Methodology

### 7.1 Parallelization strategy (common to all five)

Two coexisting strategies, applied per algorithm's data-dependency
structure:

- **OpenMP loop-level parallelism** for inference (`#pragma omp parallel
  for schedule(static) num_threads(N_THREADS)` over test samples) — always
  embarrassingly parallel, no reductions needed.
- **OpenMP arena-style + pthreads mutex+barrier** for training — both
  variants of the *same training algorithm* are implemented per file, so
  the speedup numerator (`T_serial`) is unambiguous and the OMP-vs-pthread
  comparison isolates synchronization-primitive overhead from
  parallelization-strategy overhead.

This is the convention encoded in [CLAUDE.md](../CLAUDE.md) and followed
identically by all five files.

### 7.2 Hyperparameter matching

sklearn and XGBoost hyperparameters are pulled directly from each C++
file's top-of-file `static constexpr` block. Every hyperparameter is kept
in sync manually (no config file). Correspondence table in
[misc/sk_learn_xgb.md:§3](sk_learn_xgb.md).

### 7.3 Thread count — sweep over {1, 2, 4, 8}

Every C++ binary reads `N_THREADS` from the environment (default 8) and
every sklearn/XGBoost runner respects the same env var — see the top of
[src/sklearn_xgb/compare.py](../src/sklearn_xgb/compare.py), where
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` and sklearn's
`n_jobs` / XGBoost's `n_jobs` are all derived from it. [job_sweep.sl](../job_sweep.sl)
is the SLURM wrapper that loops `N_THREADS ∈ {1, 2, 4, 8}` and concatenates
per-thread-count CSVs into `sweep_results_cpp.csv` and
`sweep_results_sklearn.csv`. SLURM allocation is `--cpus-per-task=8`, which
is the ceiling; lower thread counts are enforced at the env-var level.

sklearn estimators that have no `n_jobs` parameter (`SGDClassifier`,
`DecisionTreeClassifier`, `GaussianNB`, `BernoulliNB`) still report
`n_threads=1` regardless of the env var — the `notes` column labels them
"single-threaded by design", and the sweep exposes this as a flat horizontal
line while our C++ shows the scaling curve alongside.

### 7.4 Compilation

```
g++ -std=c++17 -O3 -march=native -fopenmp <file>.cpp -o <name> -lpthread
```

Identical across all five algorithms — enforced per
[CLAUDE.md](../CLAUDE.md). `-march=native` ties binaries to the compile
host; the writeup reports the CARC node architecture alongside results.

### 7.5 Measurement

- **Wall time** via `std::chrono::steady_clock` (C++) and
  `time.perf_counter()` (Python).
- **Parity check** — every C++ binary prints `|acc_serial - acc_omp|`
  and `|acc_serial - acc_pthreads|`; both must be `< 0.01`. Floating-point
  reductions are non-associative so *bitwise* equality is not expected;
  accuracy parity within 0.01 is.
- **Multi-run variance** — single-run per configuration for the first
  submission; multiple `sbatch` submissions produce variance estimates.

---

## 8. Expected empirical results (filled in as runs complete)

The sweep produces two CSVs — `sweep_results_cpp.csv` and
`sweep_results_sklearn.csv` — each with one row per (algorithm, variant,
n_threads). The paper extracts the following tables and figures from those.

### 8.1 Per-thread-count accuracy table (Table 1)

Accuracy should be invariant across thread counts for each algorithm (model
is the same, only the reduction order differs). This table lets the paper
assert that the parallel runs are producing the same model as the serial
reference.

| Algorithm | ours @1 | ours @2 | ours @4 | ours @8 | sklearn | XGBoost @8 | Target |
|-----------|---------|---------|---------|---------|---------|------------|--------|
| SVM       |         |         |         |         |         | N/A        | 0.76–0.80 |
| KNN       |         |         |         |         |         | N/A        | 0.76–0.80 |
| MLP       |         |         |         |         |         | N/A        | 0.76–0.80 |
| DT        |         |         |         |         |         |            | 0.76–0.80 |
| NB        |         |         |         |         | hybrid / gauss / bern | N/A | 0.70–0.76 |

### 8.2 Training-time sweep table (Table 2)

The core data. Each cell is `train_ms` (or `train_ms ± σ` when variance runs
are available). Shows per-thread-count timing side-by-side for C++ and
sklearn, plus XGBoost for DT.

| Algorithm | ours @1 | ours @2 | ours @4 | ours @8 | sklearn @1 | sklearn @2/4/8 | XGBoost @1 / @8 |
|-----------|---------|---------|---------|---------|------------|----------------|-----------------|
| SVM (hinge+L2) |    |         |         |         |            | serial @ all threads | N/A  |
| KNN (brute)    |    |         |         |         |            |                | N/A             |
| MLP (SGD+mom)  |    |         |         |         |            |                | N/A             |
| DT (hist)      |    |         |         |         |            | serial @ all threads |       |
| NB (hybrid)    |    |         |         |         |            | serial @ all threads | N/A  |

The sklearn column collapses for SVM/DT/NB since those estimators are serial
by design — one cell covers all four thread counts.

### 8.3 Scaling curve (Figure 1 — headline)

One line per algorithm on one plot. X-axis: `N_THREADS ∈ {1, 2, 4, 8}`.
Y-axis: `T_serial / T(N_THREADS)` (speedup relative to the 1-thread run of
the *same* implementation). Expected shape:

- **MLP**: steepest slope, approaching 5–7× at 8 threads (compute-bound).
- **DT**: moderate slope, 3–5× at 8 threads (crossover regime).
- **SVM / KNN / NB**: shallow slopes, 2–4× at 8 threads, flattening early
  (memory-bandwidth-limited).
- **Ideal linear speedup** (diagonal line `y = x`) drawn for reference.

This is the Amdahl-style figure the proposal's hypothesis lives or dies on.

### 8.4 Per-thread framework comparison (Figure 2)

Grouped bar chart per algorithm: `ours @1`, `sklearn @1` (i.e. the
implementation-quality comparison) side-by-side, then `ours @8`, `sklearn
@8` (the parallel-vs-serial-sklearn comparison). Log-scaled training time.
Makes both the **1-thread fairness** claim and the **8-thread scaling win**
claim visible in one plot.

### 8.5 Arithmetic intensity vs. observed speedup (Figure 3)

Scatter plot: X = analytical AI (from §3), Y = observed 8-core speedup
(`T_1 / T_8`). If the hypothesis is right, the 5 points roughly follow a
saturating curve (low AI → low speedup, high AI → high speedup), with the
memory-bandwidth-bounded values (SVM/KNN/NB) clustering and MLP pulling
away. This is the **roofline-adjacent validation** of the paper's
theoretical prediction.

---

## 9. Related work (what to cite)

The proposal's bibliography is currently sparse with placeholder citation
keys — [proposal.tex:41](../proposal.tex#L41) uses
`phani2026stratuminfrastructuremassiveagentcentric` and
`prevparallelMLwork` which are not real references. The paper needs to
fill these in with:

- **Foundational parallel ML on multi-core CPUs** — a survey or seminal
  paper on OpenMP/pthreads parallelization of classical ML. Candidates:
  Zaharia et al. on Spark MLlib; cuML's CPU-fallback paths; the LightGBM
  and XGBoost technical papers (the latter is directly cited by our DT
  comparison).
- **sklearn's architecture paper** (Buitinck et al. 2013) — the
  proposal already cites this; keep. Useful for contextualizing sklearn's
  API-first-not-speed-first design philosophy.
- **Histogram-based CART / gradient boosting** — LightGBM (Ke et al. 2017),
  XGBoost (Chen and Guestrin 2016). Both directly relevant to our DT
  choice of `N_BINS=64` histogram CART.
- **Roofline model** — Williams, Waterman, Patterson (2009). The
  arithmetic-intensity framing is borrowed directly; the paper's Figure
  3 is a roofline-adjacent plot.
- **CS:GO dataset** — Lillelund 2020 (Kaggle). Already cited correctly.

The previously placeholder `[svmsharedstate]`, `[MLPbottleneck]` citations
in [proposal.tex:41](../proposal.tex#L41) need real references or
removal.

---

## 10. Paper outline (proposed section structure)

1. **Abstract** — thesis in 150 words. Lead with "3 of 5 classical ML
   algorithms lack a parallel sklearn baseline; we provide one and
   characterize their scaling."
2. **Introduction** (1.5 pages) — research question, thesis, contributions.
   Includes the "parallel SVM-shaped hole" observation from §5.
3. **Background** (1 page) — CS:GO dataset, preprocessing, evaluation
   metrics, hardware (CARC node spec).
4. **Methodology** (2 pages) — shared interface, per-algorithm parallel-
   ization strategy (one paragraph each), hyperparameter matching, sklearn
   / XGBoost harness.
5. **Arithmetic intensity analysis** (1 page) — the predictive AI table
   and derivations. Feeds Figure 3.
6. **Empirical results** (2–3 pages) — Tables 1 & 2, Figures 1–3. Explain
   where the observed speedups match the AI prediction and where they
   don't.
7. **Framework comparison** (1 page) — the "ours vs. sklearn vs. XGBoost"
   comparison with the "3 of 5 sklearn rows are serial" caveats made
   foreground, not footnotes.
8. **Discussion** (1 page) — what the observations say about the parallel
   CPU ML ecosystem. "Why doesn't sklearn ship parallel SVM/NB/DT?" is a
   legitimate subsection.
9. **Threats to validity** (0.5 page) — see §12.
10. **Conclusion & future work** (0.5 page) — GPU comparison, larger
    dataset, distributed-memory comparison.

Target length: **6–8 pages** (IEEE conference format). Suitable for a
course-adjacent workshop submission; not a top-tier venue paper (see §12).

---

## 11. Current status of deliverables

As of **2026-04-20**:

| Deliverable                                  | Status       | Location                                        |
|----------------------------------------------|--------------|-------------------------------------------------|
| SVM implementation                           | ✅ Done      | [svm/svm.cpp](../src/cpp/svm/svm.cpp) (645 lines)       |
| KNN implementation                           | ✅ Done      | [knn/knn.cpp](../src/cpp/knn/knn.cpp) (530 lines)       |
| MLP implementation                           | ✅ Done      | [mlp/mlp.cpp](../src/cpp/mlp/mlp.cpp) (941 lines)       |
| DT implementation                            | ✅ Done      | [dt/dt.cpp](../src/cpp/dt/dt.cpp) (1050 lines)          |
| NB implementation                            | ✅ Done      | [nb/nb.cpp](../src/cpp/nb/nb.cpp) (hybrid G+B)          |
| Analytics engine (unified C++ driver)        | ✅ Done      | [analytics_engine.cpp](../analytics_engine.cpp) (379 lines) |
| sklearn + XGBoost harness                    | ✅ Done      | [sklearn_xgb/](../src/sklearn_xgb/)                 |
| MLP design doc                               | ✅ Done      | [misc/mlp.md](mlp.md)                           |
| DT design doc                                | ✅ Done      | [misc/dt.md](dt.md)                             |
| NB design doc                                | ✅ Done      | [misc/nb.md](nb.md)                             |
| SVM design doc                               | ❌ Missing   | —                                               |
| KNN design doc                               | ❌ Missing   | —                                               |
| sklearn/XGBoost design doc                   | ✅ Done      | [misc/sk_learn_xgb.md](sk_learn_xgb.md)         |
| **Thread-sweep refactor (N_THREADS runtime)** | ✅ Done     | env var read in each `main()`; see [svm.cpp](../src/cpp/svm/svm.cpp), [knn.cpp](../src/cpp/knn/knn.cpp), [mlp.cpp](../src/cpp/mlp/mlp.cpp), [dt.cpp](../src/cpp/dt/dt.cpp), [nb.cpp](../src/cpp/nb/nb.cpp) |
| **Sweep SLURM script**                        | ✅ Done     | [job_sweep.sl](../job_sweep.sl) loops {1, 2, 4, 8} over both C++ + sklearn sides |
| **First CARC run of job_sweep.sl**            | ⚠ **Pending**| Would produce `sweep_results_cpp.csv` + `sweep_results_sklearn.csv` |
| **Multi-run variance data**                  | ❌ Not started | Requires 3+ submissions of `job_sweep.sl`    |
| **Figures 1, 2, 3**                          | ❌ Not started | Need results first                            |
| **Paper draft**                              | ❌ Not started | [EE451-Final-Paper/](../EE451-Final-Paper/) empty |
| **Bibliography (real citations)**            | ❌ Not started | [proposal.tex:41](../proposal.tex#L41) has placeholders |

The critical path to a submittable paper is: one CARC run of
[job_sweep.sl](../job_sweep.sl) → Tables 1 & 2 populated from the two
sweep CSVs → Figures 1–3 plotted → write 6-page draft. That is ~1 week of
focused work given the code is done.

---

## 12. Threats to validity (for honest review)

The paper must address these up front or a reviewer will. Each is a real
concern — not a rhetorical one.

1. **Single-run timings will have 10–30% variance** on CARC shared nodes
   (other users, thermal throttling, NUMA effects). The writeup needs 3+
   runs per config and error bars, or a clear disclaimer. Currently
   neither is in place.

2. **Hyperparameter matching is not mathematical equivalence.** sklearn's
   `SGDClassifier` with `max_iter=20` is not equivalent to our full-batch
   20-epoch gradient descent — they explore different optimization
   trajectories and converge differently. Same for sklearn's MLP using
   softmax+log-loss vs. our sigmoid+BCE (analytically equivalent for 2
   classes, different implementation). The paper should say "we matched
   hyperparameters" rather than "we matched algorithms" — a reviewer
   will catch the distinction.

3. **XGBoost single-tree is not CART.** `XGBClassifier(n_estimators=1,
   learning_rate=1.0, base_score=0.5)` optimizes log-loss gradient-
   boosted, not unweighted Gini impurity. The trees look similar but are
   not the same algorithm. Documented in [misc/sk_learn_xgb.md:§7](sk_learn_xgb.md).

4. **Dataset is in-cache.** 122K × 103 × 4 B ≈ 50 MB. CARC nodes have
   ≥25 MB L3 per socket. After the first epoch everything is resident;
   observed "memory-bound" speedup ceilings may reflect cache-coherence
   overhead rather than DRAM bandwidth. This complicates the roofline
   interpretation — the paper should say so.

5. **No GPU baseline.** For classical ML, GPUs (via cuML, XGBoost-GPU,
   PyTorch) are what practitioners actually reach for at scale. Our
   CPU-only comparison is deliberately scoped but weakens the claim's
   generality. The paper frames this as scope, not oversight.

6. **N = 1 dataset.** All results are on one dataset. A second dataset
   (even a synthetic one) would strengthen the AI-vs-speedup correlation
   claim. Deferred — would require re-running the full harness.

7. **"Parallel SVM-shaped hole" observation is anecdotal.** The paper's
   framing leans on the claim that sklearn / "typical stack" lacks
   parallel SVM/NB/DT. Making this rigorous requires surveying sklearn +
   alternatives (liblinear, sklearnex, Intel's DAAL) and reporting where
   each lands. [misc/sk_learn_xgb.md:§7](sk_learn_xgb.md) notes 5–6 libs
   but doesn't systematically compare them. A short "survey of parallel
   CPU ML libraries" subsection in the related-work section would close
   this gap.

---

## 13. Gap list for submission

Ordered by blocking-ness. Each item maps to a specific artifact.

1. **Run [job_sweep.sl](../job_sweep.sl) on CARC** (~2 hour wall time,
   covers the full {1, 2, 4, 8} sweep for both C++ and sklearn/XGBoost).
   Produces `sweep_results_cpp.csv` and `sweep_results_sklearn.csv`.
   Blocks Tables 1 & 2 and Figures 1–3.

2. **Submit job_sweep.sl 3× for variance.** No code changes needed; just
   resubmit. Blocks error bars. ~6 hours wall time in aggregate.

3. **Write SVM and KNN design docs** at `misc/svm.md` and `misc/knn.md`,
   matching the structure of the existing three. Needed for the paper's
   per-algorithm subsection paragraphs. Each should be ~2 hours of
   writeup time.

4. **Fill in the bibliography** — replace the three placeholder keys in
   [proposal.tex](../proposal.tex) with real references. ~1 hour.

5. **Draft the paper** in [EE451-Final-Paper/](../EE451-Final-Paper/).
   Outline is in §10. ~2–3 days of focused work assuming results are in
   hand.

6. **Generate plots** — a `scripts/plot_results.py` that reads
   `sweep_results_cpp.csv` + `sweep_results_sklearn.csv` and produces
   Figures 1–3 as PDF. ~4 hours.

**Critical path total:** ~1 week, assuming the CARC runs succeed on first
attempt and no implementation bugs surface.

### Non-blocking but high-value followups

- **LightGBM row** in the sklearn/XGBoost harness — second SOTA parallel
  tree library strengthens the DT comparison. ~30 minutes to add.
- **Parallel CPU ML library survey subsection** — addresses threat #7.
  ~half a day of literature search.
- **Dedup loader/metrics into `common.hpp`** — mechanical refactor per
  the deferred work in each algorithm's design doc. Not blocking but
  cleans up the codebase for the reproducibility artifact.

---

## 14. Reproducibility artifacts (for submission)

A reviewer needs to be able to reproduce every number in the paper. The
reproducibility package is:

- [`train_cleaned.csv`](../data/train_cleaned.csv), [`test_cleaned.csv`](../data/test_cleaned.csv) — fixed inputs.
- [`svm/svm.cpp`](../src/cpp/svm/svm.cpp), [`knn/knn.cpp`](../src/cpp/knn/knn.cpp), [`mlp/mlp.cpp`](../src/cpp/mlp/mlp.cpp), [`dt/dt.cpp`](../src/cpp/dt/dt.cpp), [`nb/nb.cpp`](../src/cpp/nb/nb.cpp) — algorithm implementations.
- [`analytics_engine.cpp`](../analytics_engine.cpp) — unified C++ driver.
- [`sklearn_xgb/`](../src/sklearn_xgb/) — Python comparison harness.
- [`job.sl`](../job.sl), [`job_analytics.sl`](../job_analytics.sl), [`sklearn_xgb/job_compare.sl`](../src/sklearn_xgb/job_compare.sl) — SLURM scripts.
- [`CLAUDE.md`](../CLAUDE.md) — conventions so the reader can extend it.

Compiler flags, hyperparameters, and preprocessing rules are documented
both in-source and in this repo's design docs. Random seeds are fixed at
`SEED=42` for trainers that use randomness. Resulting tables and figures
are reproducible to within FP-reduction-order noise (the ±0.01 parity
bound documented in [misc/nb.md:336-342](nb.md#L336-L342)).

---

## 15. Summary for the impatient reader

**One-sentence thesis:** At 1 thread our hand-rolled C++17 implementations
are competitive with scikit-learn on the same algorithms, and they scale to
8 threads where scikit-learn can't — because 3 of our 5 algorithm classes
(SVM, DT, NB) have no parallel CPU baseline in the mainstream Python ML
stack; the one that does (DT via XGBoost histogram) is where we are
competitive rather than categorically better.

**Blocker:** Run [job_sweep.sl](../job_sweep.sl) three times on CARC to fill
the {1, 2, 4, 8} × 3-repeat sweep. Everything else is paper-writing from
artifacts that already exist.

**Honest scope:** CPU-only, single-node, one dataset, thread sweep over
{1, 2, 4, 8}. Workshop paper, not a top-tier conference. Genuine
contribution is the comparison methodology — specifically the per-thread
comparison that makes the "sklearn can't scale" claim rigorous — and the
"no parallel CPU baseline" observation, not algorithmic novelty.
