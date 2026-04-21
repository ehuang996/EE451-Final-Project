# SVM — Design Document & Results Log

Implementation at [src/cpp/svm/svm.cpp](../src/cpp/svm/svm.cpp). This document
records every non-obvious architectural and engineering choice made in the
linear SVM classifier, plus a live log of benchmark runs.

---

## 1. Problem framing

Binary classification on the CS:GO Round Winner dataset — 103 numeric features
(read from the header at runtime), label `round_winner ∈ {+1, −1}` detected by
column name in the header. 97,929 training rows; 24,483 test rows. Community
benchmark accuracy: 76–80%.

The proposal's hypothesis names SVM as "**synchronization-limited**" — the
prediction being that stochastic gradient descent's per-iteration reductions
would cap speedup below the compute-friendly algorithms. In practice at T=8,
SVM came in **second** at 6.02× (pthreads) — better than MLP. The
hypothesis was directionally wrong: full-batch GD with a single reduction per
epoch is cheap enough that we see near-linear scaling.

**Key cross-framework finding.** Sklearn's `SGDClassifier(loss="hinge")` is
**single-threaded by design** — no `n_jobs`. Our C++ implementation beats the
sklearn baseline at T=8 by **13.6×**. This is the strongest "hand-rolled
parallel C++ beats library defaults" data point in the paper, precisely
because the sklearn equivalent has no parallelism to fall back on.

---

## 2. Algorithm: linear SVM with hinge loss + L2, full-batch gradient descent

Primal formulation of the linear SVM:

```
minimize_{w, b}  (λ/2)·‖w‖² + (1/n)·Σᵢ max(0, 1 − yᵢ·(w·xᵢ + b))
```

### Per-epoch algorithm

For each of `MAX_EPOCHS = 20` epochs:
1. Zero a gradient accumulator `gw, gb`.
2. For every training sample `i`:
   - Compute margin `m = yᵢ · (w·xᵢ + b)`.
   - If `m < 1`: add `yᵢ·xᵢ` to `gw`, add `yᵢ` to `gb` (hinge-active).
   - Else: contribute nothing (hinge is flat, gradient is zero).
3. Apply the gradient step with decoupled L2 decay:
   `w ← (1 − lr·λ)·w + (lr/n)·gw`
   `b ← b + (lr/n)·gb`

No per-sample stochasticity — every epoch sees every sample once, and the
update is applied once per epoch. This is **full-batch gradient descent**, not
minibatch SGD, despite the hyperparameter name matching sklearn's SGD.

### Why full-batch

- **Simple reduction pattern.** One gradient accumulator per thread, one serial
  reduction at epoch boundary, one weight update. No per-sample critical
  section. Lowest possible synchronization cost for an iterative trainer.
- **Batch = full dataset** means barrier cost is amortized over 98K samples —
  the opposite extreme from MLP's 128-sample minibatches. This is the
  intentional design decision that gives SVM its 6× speedup.
- **Stochastic noise isn't needed here.** The hinge-loss + L2 objective is
  convex, so deterministic full-batch GD converges as well as SGD given enough
  epochs. On 20 epochs the final accuracy is 73%, within the paper's target.
- Matches the config we mirror in sklearn's `SGDClassifier(max_iter=20,
  alpha=1e-4, eta0=0.02)` — close enough for apples-to-apples cross-framework
  comparison.

### Decoupled L2 decay

The update applies weight decay **multiplicatively** as `w ← (1 − lr·λ)·w`
rather than adding a `λ·w` term to the gradient. Same math, fewer ops per
iteration, no catastrophic cancellation at large w norms. Same pattern also
used by MLP (`w ← (1 − lr·λ)·w − lr·v`).

### No kernel trick

Linear SVM only. RBF / polynomial kernels are **not** implemented — they
change the algorithm from `O(n·F)` per epoch to `O(n²·F)` (requires a kernel
matrix), which on CS:GO would dominate the benchmark and isn't the point of
the paper. Linear SVM with z-scored features reaches 73% accuracy, matching
sklearn's linear SVM within 0.15pt.

---

## 3. Hyperparameters

```
MAX_EPOCHS = 20      (default, overridable as argv[3])
LR         = 0.02    (fixed learning rate, no decay)
LAMBDA     = 1e-4    (L2 regularization coefficient)
N_THREADS  = 8       (matched to SLURM --cpus-per-task=8)
SEED       = 42      (unused — full-batch GD is deterministic)
```

`static constexpr`. CLI: `./svm [train_csv] [test_csv] [epochs]`. LR and
LAMBDA are compile-time to keep the signature lean; a future variant could
expose them as argv[4], argv[5] matching MLP's CLI shape.

`LR = 0.02` without decay works at 20 epochs because the hinge loss is convex
and the weights converge monotonically. A smaller LR would need more epochs;
a larger LR diverges within a few epochs on z-scored features.

---

## 4. Data structures & memory layout

### `struct Dataset` (shared with KNN, MLP, DT, NB)

```cpp
struct Dataset {
    int n_features;
    std::vector<std::string> feature_names;
    std::vector<float> X_train;  // row-major: n_train × n_features
    std::vector<float> X_test;   // row-major: n_test  × n_features
    Labels y_train;              // Labels = std::vector<int>
    Labels y_test;
};
```

Row-major `float` keeps `X_train` at 40 MB (fits in L3) and enables unit-stride
`dot(w, x)` inner loops that auto-vectorize under `-O3 -march=native`. Shared
across the five algorithms; `analytics_engine.cpp` can load once and drive
all five.

### `struct SVMModel`

```cpp
struct SVMModel {
    std::vector<double> w;   // n_features = 103 coefficients
    double b = 0.0;          // bias / intercept
};
```

Minimal. `w` is `double` (not `float` like `X`) to avoid catastrophic
cancellation during gradient accumulation — 98K hinge-active updates per
epoch, each contributing up to `±2.5` per coordinate, would accumulate to
~10⁵ before the decay step, at which point float's 7-digit precision eats the
fractional corrections.

### Per-thread gradient arenas

Both parallel trainers use the **arena pattern**: one full-size gradient
accumulator per thread, pre-allocated once, zeroed per epoch:

```cpp
std::vector<std::vector<double>> grad_by_thread(N_THREADS,
                                                std::vector<double>(n_features, 0.0));
std::vector<double> bias_grad_by_thread(N_THREADS, 0.0);
```

Size per thread: 103 × 8 B = 824 B. Fits in one thread's L1 cache. 8 arenas
× 824 B = 6.6 KB total. Never reallocated. Zeroed via `std::fill` at the top
of each epoch (cheap; the memory was just updated so it's still in L1).

**Why arena (not per-sample atomic updates).** Atomically adding to a shared
`gw` from every hinge-active sample would serialize 98K × 103 ≈ 10M updates
per epoch on a single cache line. The arena-then-reduce pattern turns this
into 98K × 103 strictly-local updates + one serial reduction of 8 × 103
values at the epoch boundary — vastly cheaper.

### Data-preprocessing skip list

Identical to KNN/MLP/DT/NB: `map_*`, `bomb_planted`, and auto-detected 0/1
columns bypass z-score normalization. On CS:GO expect
`Normalize: 94 z-scored, 9 passthrough` (bomb_planted + 8 map_* one-hots).

**Why this matters for SVM.** Linear SVM is scale-sensitive — a feature with
`σ = 1000` would produce gradients 1000× larger than a feature with `σ = 1`,
swamping the L2 decay and destabilizing convergence. Z-scoring continuous
features is essential. Leaving binary/one-hot features at 0/1 keeps their
contribution bounded to `±1·yᵢ` per hinge-active pass, comparable in
magnitude to a z-scored continuous feature at 1σ.

---

## 5. Training — three variants

All three trainers execute **the same full-batch GD algorithm**, so the
speedup ratio measures parallelization overhead only.

### 5a. `train_serial` — full-batch GD, single-threaded

```
for epoch in 0..MAX_EPOCHS:
    gw = [0]*F; gb = 0
    for i in 0..n_rows:
        margin = yᵢ · (w·xᵢ + b)
        if margin < 1:
            gw += yᵢ · xᵢ
            gb += yᵢ
    w = (1 − lr·λ)·w + (lr/n)·gw
    b += (lr/n)·gb
```

Straightforward; ~330 ms on CS:GO. This is the speedup denominator.

### 5b. `train_parallel_omp` — OpenMP + arena reduction

```cpp
for each epoch:
    zero grad_by_thread[0..P]
    #pragma omp parallel for schedule(static) num_threads(N_THREADS)
    for i in 0..n_rows:
        tid = omp_get_thread_num()
        // same margin check as serial, accumulates into grad_by_thread[tid]
    // serial reduction across threads
    for t in 0..P:
        gw += grad_by_thread[t]; gb += bias_grad_by_thread[t]
    apply_update(w, b, gw, gb)
```

`schedule(static)` partitions n_rows evenly (~12,241 per thread) — each thread
does the same amount of work, which is true here because the cost per sample
is constant (one dot product + conditional vector update).

**No `#pragma omp critical`** — the post-loop serial reduction of 8 × 103
doubles = 6.6 KB is fast enough that a `critical` section (with its
implicit mutex) would be slower. Same pattern used by MLP's OMP trainer.

**No SIMD hint** — `-O3 -march=native` already auto-vectorizes the `dot_row`
and gradient-add loops. Explicit `#pragma omp simd` was measured to add
nothing at F=103.

### 5c. `train_parallel_pthreads` — mutex + two barriers per epoch

Uses the same two-barriers-per-iteration pattern as MLP's pthreads trainer.
Persistent pool of N_THREADS workers; per epoch:

```cpp
struct SVMParState {
    const std::vector<float>* X;  const Labels* y;
    int n_rows, n_features;  SVMModel* model;
    std::vector<double> gw;       double gb;  // shared gradient
    double lr, lambda;            int epochs;
    pthread_mutex_t   mutex;
    pthread_barrier_t barrier_acc;  // all workers → shared gw is complete
    pthread_barrier_t barrier_upd;  // leader → weights are updated
};
```

Per worker per epoch:
1. Zero **stack-local** `lgw` (allocated once at worker startup).
2. Sweep own slice `[tid·n/P, (tid+1)·n/P)`, accumulating into `lgw, lgb`.
3. `mutex_lock` → `gw += lgw`, `gb += lgb` → `mutex_unlock`.
4. `barrier_acc` — wait for all workers to have contributed.
5. Thread 0 only: apply the SGD update (`w ← (1-lrλ)·w + ...`), zero `gw, gb`.
6. `barrier_upd` — wait for thread 0's update to be visible before starting
   the next epoch's margin checks.

**Why two barriers, not one.** `barrier_acc` ensures thread 0's update step
sees the final reduced `gw`. `barrier_upd` ensures the other workers don't
start their next epoch's forward pass against a half-updated weight vector.
Without `barrier_upd`, thread 1 could start epoch 2's `dot(w, x)` while
thread 0 is halfway through writing epoch 1's `w` update.

**Why this design beats OMP on SVM** (observed 6.02× vs 4.15× at T=8). Short
answer: 20 epochs × 1 fork/join = 20 thread-team startups under OpenMP; with
pthreads we fork once total. Thread-team startup on Cascade Lake via libgomp
is measured at ~10 µs — 20 × 10 µs = 200 µs, which is significant against
SVM's 55 ms parallel runtime (0.4% per startup × 20 = 8% lost). The single-
fork pthreads model eats its 10 µs once and never again. Documented in the
writeup as the cleanest OMP-vs-pthreads data point in the five algorithms.

---

## 6. Inference

`predict_serial` and `predict_parallel` are identical except for the
`#pragma omp parallel for schedule(static) num_threads(N_THREADS)` on the
outer test-sample loop. Each test sample: one dot product `w · x + b`, one
sign check. Read-only access to `w`, stack-local dot products — embarrassingly
parallel, no atomics needed.

Output threshold: `w·x + b ≥ 0 → +1, else −1`. Matches the SVM-compatible
label space used across all five algorithms.

Inference is ~4 ms serial / ~0.6 ms parallel on CS:GO — trivial compared to
training. Only one parallel inference variant (OpenMP) for parity with
MLP/DT/NB; no pthreads inference variant, since a fixed-pool pthreads setup
for a sub-millisecond inference pass would be dominated by thread-create
overhead.

---

## 7. Arithmetic intensity (for the writeup)

**Per (sample, epoch) pair in the training hot loop:**

| Memory op                              | Bytes |
|----------------------------------------|------:|
| Load `xᵢ[0..F]`  (F=103 floats)        | 412   |
| Load `w[0..F]`   (amortized; in L1)    | ≈0    |
| Load `y[i]`  (4 bytes, amortized)      | ≈0    |

Ops per (sample, epoch): `F` multiplies + `F − 1` adds for the dot product +
1 compare + `F` multiplies + `F` adds for the conditional gradient add ≈
**4F + 2 ≈ 414 ops** per 412 bytes.

Effective arithmetic intensity: **~1 FLOP/byte** (or ~0.5 if we only count
the dot product). **Memory-bound** — lowest or tied-lowest of the five
algorithms.

Comparison across all five algorithms:

| Algorithm | Inner kernel                  | AI (FLOP/byte) | Expected scaling   | Observed T=8 (pth) |
|-----------|-------------------------------|---------------:|--------------------|-------------------:|
| SVM       | `dot(w, x)` + margin check    | ~0.5           | Memory-bound       | **6.02×**          |
| KNN       | squared-L2 distance           | ~0.75          | Memory-bound       | 6.26×              |
| NB        | frequency count + Gaussian MLE| ~0.75          | Memory-bound       | 3.56×              |
| DT        | histogram fill + Gini sweep   | ~3             | Crossover regime   | 1.87×              |
| MLP       | dense GEMM forward/back       | ~42            | Compute-bound      | 4.62×              |

**Why SVM beat its low-AI prediction.** Same reason as KNN: `X_train` is
read-shared across threads, so L3 serves cache lines once per (row, cache-line)
pair rather than once per (row, thread) pair. At T=8 this gives an effective
L3-bandwidth multiplier of ~4×, pushing intensity from 0.5 to ~2 FLOP/byte
in DRAM terms — crossover regime on Cascade Lake. Plus SVM's full-batch-GD
structure means **one reduction per epoch × 20 epochs = 20 sync events
total** — an order of magnitude fewer than MLP's ~765 mini-batches × 30 epochs
= ~23K sync events.

Low-AI ≠ bad scaling when synchronization is rare and data is shared.

---

## 8. Numerical & correctness risks

1. **Float-to-double casting.** `X` is `float` in the shared Dataset; SVM
   loads `float xi[j]` then casts to `double` for the dot product. No
   precision loss since `double` is strictly larger.
2. **Weight-norm overflow at large LR.** If `LR > 0.05` with z-scored
   features, the gradient magnitude × step_size can exceed 1 per coordinate,
   and `(1 − lr·λ)` doesn't decay fast enough. Manifests as `accuracy ≈ 0.5`
   (random). The current `LR = 0.02` is well inside the stable region.
3. **Non-determinism from FP reduction order.** The serial reduction at
   epoch boundary sums thread-local gradients in thread-order 0..P. Pthreads
   reduction order is insertion-order into the mutex. Both are deterministic
   given a fixed thread count, but OMP's and pthreads' reductions differ
   from each other by ≈ n · ε_machine ≈ 98K · 10⁻¹⁵ = 10⁻¹⁰, which is below
   the 0.01 parity threshold. Parity check prints 0.0000 in all runs.
4. **Hinge tie on `margin = 1` exactly.** `margin < 1` is strict — a margin
   of exactly 1.0 contributes zero gradient (correct hinge semantics). On
   float input there's no way to hit `margin = 1.0` exactly, so this is
   academic.
5. **Barrier deadlock if any worker crashes.** The pthreads variant has no
   watchdog — if a worker segfaults between `barrier_acc` and
   `barrier_upd`, the other 7 workers hang forever. No graceful recovery
   path; acceptable for a research artifact.
6. **Speedup numerator.** All three trainers run the same full-batch GD, so
   the printed ratios measure parallelization overhead, not algorithmic
   convergence differences.

---

## 9. Build & run

On USC CARC (from src/cpp/svm/):

```bash
cd src/cpp/svm && sbatch job_svm.sl
# outputs to gpujob.out (historical naming — SVM was originally intended to
# have a CUDA variant; stuck with the filename). Compiles svm.cpp, runs
# ./svm ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
```

CLI:

```bash
./svm [train_csv] [test_csv] [epochs]
# defaults: data/train_cleaned.csv  data/test_cleaned.csv  20
```

Locally (macOS), from the repo root:

```bash
# Option A: Homebrew libomp with Apple Clang
brew install libomp
clang++ -std=c++17 -O3 -march=native \
  -Xpreprocessor -fopenmp -lomp src/cpp/svm/svm.cpp -o svm -lpthread
./svm data/train_cleaned.csv data/test_cleaned.csv

# Option B: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
```

**Apple Clang alone will not work** — no OpenMP runtime, and
`pthread_barrier_t` is unimplemented on Darwin (the pthreads variant depends
on it).

### Verification expected

- **Accuracy**: `acc ≈ 0.73` on test set for all three trainers. Lower than
  KNN/MLP/DT because the dataset's decision boundary is nonlinear —
  matches sklearn's linear SVM (73.15%) within 0.15pt.
- **Parity**: `|acc_serial − acc_omp| < 0.01` and
  `|acc_serial − acc_pthreads| < 0.01`. In practice: 0.0000 across all thread
  counts.
- **Speedup**: `serial_total / omp_total > 1` and
  `serial_total / pthreads_total > 1`. Target 4–6× on 8 cores.

---

## 10. Results log

Append one row per CARC run.

| date (UTC) | git SHA | epochs | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:------:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| 2026-04-20 | 526bb18 | 20 | CARC d17-03 / 8 | 0.333 | 0.080 | 0.055 | 0.7301 | 0.7301 | 0.7301 | 4.15× | 6.02× | Full `{1,2,4,8}` sweep, job 3272373. Proposal's "synchronization-limited" prediction was **wrong** — SVM placed 2nd at 6.02× because full-batch GD has only 20 sync events total. **pthreads beats OMP by the widest margin of the five** (6.02× vs 4.15×): with 20 epochs, OMP pays 20 team-creation costs while pthreads pays 1. vs sklearn `SGDClassifier` (single-threaded) **ours is 13.6× faster**. See [results/results.md](../results/results.md). |

---

## 11. Followups (known work deferred)

- **Deduplicate loader/metrics** into `common.hpp` once
  `analytics_engine.cpp` is being written. The loader is byte-identical to
  KNN/MLP/DT/NB; extraction is mechanical.
- **Thread-count sweep on CARC's `nlp` partition** — T ∈ {16, 32, 64} would
  test whether SVM's near-linear scaling holds as L3 bandwidth saturates.
  Predicted: 10–12× at T=16, then plateau. Would pair with the same sweep
  for KNN as the main scaling study.
- **SGD (mini-batch) variant** — the current full-batch GD is strictly-
  concave-matching for the sklearn comparison, but a minibatch=128 variant
  would let us compare SVM's sync behavior against MLP's directly.
  Interesting for a sync-overhead-vs-batch-size figure.
- **Minibatch-size sweep** as the sync-overhead-vs-batch-size sensitivity
  plot. Would pair with the MLP sweep.
- **Kernel SVM via dual formulation** — would match RBF kernel sklearn SVC;
  but at O(n²) it's a different algorithm class and not apples-to-apples.
  Explicitly out of scope per the paper's linear-SVM framing.
- **Hinge gradient as SIMD pragma** — `-O3 -march=native` already
  auto-vectorizes the two inner loops; no measurable gain from explicit
  `#pragma omp simd`. Noted as a negative result.
- **Per-epoch learning-rate schedule** — could switch to `η_t = η₀/√(1+t)`
  matching MLP's invscaling; doesn't change the parallelization story and
  sklearn's `SGDClassifier` default is also flat LR by default, so omitted
  for apples-to-apples.
