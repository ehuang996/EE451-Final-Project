# MLP — Design Document & Results Log

Implementation at [src/cpp/mlp/mlp.cpp](../src/cpp/mlp/mlp.cpp). This document records every
non-obvious architectural and engineering choice made in the MLP, plus a live
log of benchmark runs.

---

## 1. Problem framing

Binary classification on the CS:GO Round Winner dataset — 103 numeric features
(read from the header at runtime), label `round_winner ∈ {+1, −1}` detected by
column name in the header. 97,929 training rows; 24,483 test rows. Community
benchmark accuracy: 76–80%.

The proposal's hypothesis names MLP as the algorithm expected to scale best of
the five: dense matrix operations give high arithmetic intensity (FLOPs / byte
of memory traffic), so the parallel work isn't memory-bandwidth-starved.
The MLP's 8-core speedup number is therefore the **headline empirical claim**
of the paper.

---

## 2. Architecture

### Topology: `n_features → 64 → 32 → 1` (103 → 64 → 32 → 1 on this dataset)

| Layer       | In  | Out | Weights | Activation |
|-------------|----:|----:|--------:|------------|
| Hidden 1    | 103 |  64 |   6,592 | ReLU       |
| Hidden 2    |  64 |  32 |   2,048 | ReLU       |
| Output      |  32 |   1 |      32 | Sigmoid    |
| **Total**   |     |     | **8,672 + 97 biases = 8,769 params** | |

**Why this size.** Large enough to hit the 76–80% target (CS:GO data is close
to linearly separable, most signal is captured by the first layer). Small
enough that the entire model fits in L2 cache (`W1`≈51 KB, `W2`≈16 KB,
`W3`≈256 B ≈ 67 KB), so every sample's forward pass reuses fully-cached
weights. 128 / 64 would slightly improve accuracy but would not change the
parallel scaling story — and scaling is the point.

### Activation choices
- **ReLU** on hidden layers: cheap derivative (`x > 0 ? 1 : 0`), no vanishing
  gradient, pairs naturally with He initialization.
- **Sigmoid** on output: needed for BCE. Computed as `1 / (1 + exp(-z))` in the
  forward path; in the backward path the sigmoid + BCE derivative collapses to
  `dL/dz = ŷ − t`, avoiding a redundant `sigmoid'` multiply.

### Loss: fused sigmoid + binary cross-entropy

Computed in the **logit form** for numerical stability:

```
L(z, t) = max(z, 0) − z·t + log1p(exp(−|z|))
```

This avoids `log(0)` when sigmoid saturates (|z| > 36). Never evaluate
`log(sigmoid(z))` directly.

### Regularization: L2 weight decay on weights only

Coefficient `LAMBDA = 1e-4` (matched to SVM). Biases are **not** regularized
(standard practice — bias shrinkage would bias the decision boundary toward
the origin, which is undesirable after z-score normalization centers features
at 0). Implemented multiplicatively as `W ← (1 − lr·λ)·W − lr·gW/B` rather
than adding a `λ·W` term to the gradient — same math, fewer operations,
same pattern as SVM's `scale = 1 − lr·LAMBDA`.

### Weight initialization
- **He** for ReLU layers: `N(0, √(2 / fan_in))`. `fan_in = n_features` for
  `W1`, 64 for `W2`. Keeps pre-activation variance ≈ 1 at initialization,
  avoiding dead neurons and saturated gradients in the first backward pass.
- **Small uniform** for the output layer: `U(−0.01, 0.01)`. Prevents the
  sigmoid from saturating at init.
- All biases: zero.
- Seeded by `std::mt19937(SEED=42)` for reproducibility across the three
  training variants.

### Labels

The dataset on disk stores `y ∈ {+1, −1}` — matches the SVM convention, so
the three models share one CSV loader. Internally the MLP converts to
`t ∈ {0, 1}` at each point of use (`int t = (y[s] == 1) ? 1 : 0`). The
`Labels y_train` vector is never mutated, so a future `analytics_engine.cpp`
can load once and pass the same `Dataset` to SVM, KNN, and MLP.

---

## 3. Hyperparameters

```
MAX_EPOCHS = 30    (default, overridable as argv[3])
BATCH      = 128   (mini-batch SGD; divides N_THREADS=8 evenly)
LR         = 0.01  (default, overridable as argv[4]; decayed as LR/√(epoch+1))
LAMBDA     = 1e-4  (default, overridable as argv[5])
MOMENTUM   = 0.9   (Polyak)
N_THREADS  = 8     (matched to SLURM --cpus-per-task=8)
SEED       = 42
```

`static_assert(BATCH % N_THREADS == 0)` enforces the even-slicing invariant
used by the pthreads trainer. `N_THREADS` and the algorithm-internal
hyperparameters (`H1`, `H2`, `BATCH`, `MOMENTUM`, `SEED`) stay `constexpr`;
`epochs`, `lr`, and `lambda` are runtime parameters so the CLI signature
(`./mlp train test [epochs] [lr] [lambda]`) matches SVM.

---

## 4. Data structures & memory layout

### `struct Dataset` (shared with SVM and KNN)

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

Flat `std::vector<float>` row-major with explicit stride (`&X[i*n_features]`)
beats a nested `vector<vector<double>>` for three reasons: single allocation
(no per-row heap headers, no pointer-chase), half the memory footprint
(float vs. double on the input side — weights stay `double`), and unit-stride
inner loops that auto-vectorize under `-O3 -march=native`. SVM, KNN, and MLP
all use this shape so a future `analytics_engine.cpp` can share one loader.

### `struct MLPModel`
Six weight/bias pairs plus six matching momentum buffers, all flat
`std::vector<double>` in row-major order:

```
W1[i * n_features + j]   = edge from input j → hidden-1 unit i
W2[i * H1         + j]   = edge from hidden-1 unit j → hidden-2 unit i
W3[i * H2         + j]   = edge from hidden-2 unit j → output unit i
```

Sized at `init_weights(model, n_features, seed)` — the model picks up the
runtime `n_features` from the dataset, so the MLP will work unchanged if a
teammate regenerates `train_cleaned.csv` with a different feature count.

### Activation buffers on the stack

`forward_sample` and `predict_*` declare `double a1[H1], a2[H2]` as local
fixed-size arrays. `H1`/`H2` are `constexpr`, so these are true stack
allocations — no heap traffic in the inference hot loop. Same for `dz1[H1]`,
`dz2[H2]` in `backward_sample`.

### Thread-local gradient buffers

For both parallel trainers, each thread owns its own gradient accumulator
(same shapes as the weight matrices). **Allocated once at the top of the
trainer function / pthread worker**, then `std::fill`ed to zero per
mini-batch — never reallocated. If allocated inside the batch loop,
`vector<double>(~6.6K)` × 8 threads × ~765 batches × 30 epochs ≈ 9 GB of
churn would dominate runtime.

In the OMP trainer this is a `std::vector<std::vector<double>>` indexed by
`tid` (arena style). In the pthreads trainer each worker holds its own
stack-local buffers — no shared container, zero false-sharing risk.

### Data-preprocessing skip list

The cleaned CSVs now store raw (un-z-scored) values, and each `.cpp` file
normalizes itself on load. The MLP's `load_dataset` matches SVM/KNN
**exactly** on which columns to normalize:

- Skip `map_*` (one-hot encoded map indicator).
- Skip `bomb_planted` (explicit binary flag).
- Auto-detect any column whose training values are all in `{0.0, 1.0}`.
- All other columns get z-scored using training-set statistics only.

Skipped columns pass through normalization unchanged (implemented by setting
`mean = 0, sd = 1` for those columns, so the common normalize loop works
uniformly). The loader prints a one-line summary:
`Normalize: <N> z-scored, <K> passthrough (binary/one-hot)` — at 103
features on the CS:GO dataset, expect `94 z-scored, 9 passthrough`
(bomb_planted + 8 map_* one-hots).

---

## 5. Training — three variants

All three trainers share the same forward/backward kernel (`forward_sample`,
`backward_sample`) and differ only in the outer structure. Serial and
parallel run the **same algorithm** (mini-batch SGD + momentum + decaying LR),
so the speedup ratio measures parallelization overhead, not algorithm
improvement.

### 5a. `train_serial` — mini-batch SGD + Polyak momentum + decaying LR

`η_t = LR / √(epoch + 1)` + momentum update `v ← β·v + g/B`,
`w ← (1 − η·λ)·w − η·v`. The single serial baseline; the `train_better_serial`
variant from an earlier draft was removed — the "better" algorithm is now
the sole algorithm.

### 5b. `train_parallel_omp` — OpenMP data-parallel over the mini-batch

Matches SVM's `train_parallel` pattern exactly (no `critical`, no mutex):

```
for each mini-batch:
  zero per-thread gradient buffers (N_THREADS slots, allocated once)
  #pragma omp parallel for schedule(static) num_threads(N_THREADS)
  for i in [0, BATCH):
      tid = omp_get_thread_num()
      forward + backward into lgW*[tid], lgb*[tid]
      lloss[tid] += bce(...)
  # serial reduction
  for t in [0, N_THREADS):
      gW* += lgW*[t]; gb* += lgb*[t]; epoch_loss += lloss[t]
  sgd_momentum_update(model, gW*, gb*, lambda, lr_t, BATCH)
```

Why no `#pragma omp critical`? The post-loop serial reduction over 8 threads
and ~9K doubles is ~70 KB to sum — fast enough that a single-threaded pass
beats spin-waiting on a critical section. Matches the SVM convention
directly.

### 5c. `train_parallel_pthreads` — MLP-specific, mutex + two barriers

Kept alongside the OMP variant so the writeup has **two parallel
implementations of the same algorithm** to compare — a direct result of the
earlier design decision. Uses the classic mutex + barrier pattern:

```cpp
struct MLPParState {
    const std::vector<float>* X;  const Labels* y;
    int n_rows, n_features;       MLPModel* model;
    std::vector<int> idx;         int batch_start;  double epoch_loss;
    std::vector<double> gW1, gb1, gW2, gb2, gW3, gb3;  double shared_loss;
    double lr, lambda;
    pthread_mutex_t   mutex;
    pthread_barrier_t barrier_acc;
    pthread_barrier_t barrier_upd;
    unsigned seed;  int epochs;
};
```

Per epoch:
- **(Thread 0 only)** reshuffle `idx` with `mt19937(SEED + epoch)`;
  reset `batch_start`, `epoch_loss`.
- `barrier_upd` — all threads wait for the shuffle before they start.

Per mini-batch:
1. Each thread computes forward + backward on its fixed slice
   `[tid · 16, (tid+1) · 16)` of the batch into stack-local gradient buffers.
2. `mutex_lock` → add locals to shared `st->gW*` → `mutex_unlock`.
3. `barrier_acc` — all threads have contributed their gradient.
4. **(Thread 0 only)** `sgd_momentum_update`; zero shared grads;
   advance `batch_start`.
5. `barrier_upd` — no thread starts next batch's forward pass before
   weights are written.

**Why two barriers?** `barrier_acc` protects against "thread 0 reads the
shared gradient before thread N has locked-and-added its contribution".
`barrier_upd` protects against "thread N starts the next batch's forward
pass against a half-updated weight vector".

### Why mini-batch, not full-batch or per-sample?

- **Full-batch** on 97,929 samples: one update per epoch — poor convergence,
  and only `MAX_EPOCHS` parallel regions per run is not enough to amortize
  thread spin-up.
- **Per-sample (batch=1)**: every sample requires synchronization →
  sync dominates compute, speedup collapses.
- **batch=128** gives ~765 parallel sections per epoch × 30 epochs ≈ 23K
  chances to amortize, and each section does ~128 × 17K FLOPs ≈ 2.2M FLOPs
  of compute — way more than barrier cost.

Note: SVM uses **full-batch** gradient descent (`train_parallel` is one update
per epoch over the full 97,929 rows). KNN doesn't train at all — it's a lazy
learner that stores the training set and does all work at inference time.
The MLP using mini-batch is an intentional algorithm-level difference, not a
parallelization difference — non-convex BCE loss needs the stochastic noise
to escape saddle points.

---

## 6. Inference

`predict_serial` and `predict_parallel` are identical except for
`#pragma omp parallel for schedule(static) num_threads(N_THREADS)` on the
outer sample loop. Each iteration's forward pass is independent (read-only
access to weights, stack-local activations) → embarrassingly parallel,
no atomics needed.

Static scheduling chosen because every sample does the same amount of work
(no branching in the forward pass), so dynamic scheduling would just add
atomic-counter overhead.

Output threshold: `ŷ ≥ 0.5 → +1, else −1` — converts sigmoid output back to
the SVM-compatible label space, so the shared `evaluate()` works unchanged.

---

## 7. Why MLP should out-scale SVM (for the writeup)

**Arithmetic intensity comparison** (FLOPs per byte of input data loaded):

| Algorithm | Inner kernel             | FLOPs / sample | Bytes / sample | AI (FLOP/byte) |
|-----------|--------------------------|---------------:|---------------:|---------------:|
| SVM       | `dot(w, x)` + margin check | ~210          | 412            | ~0.5           |
| MLP       | forward + backward         | ~17,400       | 412            | ~42            |

(`412 = 103 × 4` bytes, X is stored as `float` post-refactor.)

~80× higher arithmetic intensity. The MLP does far more compute per byte of
memory traffic, so eight threads each running GEMM-style inner loops aren't
competing for memory bandwidth. SVM is closer to memory-bound — its observed
speedup ceiling should sit below MLP's on the same 8-core machine.

---

## 8. Numerical & correctness risks (documented for debugging)

1. **Sigmoid saturation** — `log(sigmoid(z))` overflows at |z| > 36. Use the
   logit-form BCE (already done).
2. **Exploding gradients** at LR=0.01 — if epoch 0 prints `loss=nan`, drop LR
   to 0.005 via `./mlp ... 30 0.005`. Weight init should prevent this but
   verify.
3. **Non-deterministic reductions** — OpenMP's serial post-loop reduction and
   the pthreads mutex-protected reduction both accumulate in the order
   threads arrive, which differs per run. The final accuracy is stable
   (to ±0.005), but printed per-epoch loss will differ in the ~5th decimal
   place. Do **not** diff loss lines of two parallel runs — they will never
   match bitwise.
4. **Shuffle reproducibility** — only thread 0 reshuffles the index array,
   and does so with `mt19937(SEED + epoch)` so the shuffle is deterministic
   across runs. If another thread also shuffles, parallel-vs-serial
   comparison becomes meaningless.
5. **Speedup numerator** — the summary compares `train_serial` against the
   two parallel variants. All three use the same algorithm (mini-batch SGD +
   momentum + decaying LR), so the ratio measures parallelization only —
   not algorithm-level convergence differences.

---

## 9. Build & run

On USC CARC (from src/cpp/mlp/):

```bash
cd src/cpp/mlp && sbatch job_mlp.sl
# outputs to mlpjob.out (separate from SVM's gpujob.out, KNN's knnjob.out,
# DT's dtjob.out)
# compiles mlp.cpp, runs ./mlp ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
```

CLI:

```bash
./mlp [train_csv] [test_csv] [epochs] [lr] [lambda]
# defaults: data/train_cleaned.csv  data/test_cleaned.csv  30  0.01  1e-4
```

Locally (macOS), from the repo root:

```bash
# Option A: Homebrew libomp with Apple Clang
brew install libomp
clang++ -std=c++17 -O3 -march=native \
  -Xpreprocessor -fopenmp -lomp src/cpp/mlp/mlp.cpp -o mlp -lpthread
./mlp data/train_cleaned.csv data/test_cleaned.csv

# Option B: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp src/cpp/mlp/mlp.cpp -o mlp -lpthread
```

**Apple Clang alone will not work** — it lacks an OpenMP runtime, and
`pthread_barrier_t` is an optional POSIX feature that macOS does not
implement. On a mismatched-Xcode-CLT Mac (e.g. Xcode 13 against macOS 26)
even simple `<iostream>` includes fail to compile, at which point the only
option is CARC.

### Verification expected

- **Accuracy**: `acc ∈ [0.76, 0.80]` on test set for all three trainers.
- **Loss**: monotone-ish decrease per epoch (some oscillation from mini-batch
  noise is normal).
- **Parity**: `|acc_serial − acc_omp| < 0.01` and
  `|acc_serial − acc_pthreads| < 0.01`. `main()` prints these deltas at the
  end of every run.
- **Speedup**: `serial_total / omp_total > 1` and
  `serial_total / pthreads_total > 1`. Target 4–6× on 8 cores.

---

## 10. Results log

Append one row per CARC run. Format:

| date (UTC) | git SHA | epochs | batch | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:------:|:-----:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| (pending first CARC run) | | 30 | 128 | CARC / 8 | | | | | | | | | expected ~0.78 acc, 4–6× speedup |

---

## 11. Followups (known work deferred)

- **Deduplicate loader/metrics** into `common.hpp` once `analytics_engine.cpp`
  is being written. The refactor just done aligned MLP's `Dataset`,
  `load_dataset`, `evaluate`, and `print_results` to match SVM/KNN byte-for-
  byte, so the eventual extraction is mechanical.
- **Mini-batch size sweep** — the writeup may want a BATCH ∈ {32, 64, 128,
  256, 512} scan to show how batch size trades off synchronization overhead
  against gradient noise. Cheap to run, adds a figure. Needs `BATCH` moved
  from `constexpr` to a CLI arg.
- **Thread count sweep** — the full speedup curve S(P) for P ∈ {1, 2, 4, 8}
  is the most informative figure for the paper. Requires parameterizing
  `N_THREADS` at the CLI rather than hardcoding it at top-of-file.
- **Cache-blocked GEMM** — not needed at current layer widths
  (W1 fits in L1d = 64 KB on CARC's Skylake/Cascade Lake nodes). Would only
  matter if H1 ≥ 512.
- **SIMD hinting** — `-O3 -march=native` already auto-vectorizes the inner
  dot products; explicit `#pragma omp simd` on the `j` loops inside
  `forward_sample` might squeeze another 10–20% but is not required.
