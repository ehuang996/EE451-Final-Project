# MLP — Design Document & Results Log

Implementation at [mlp/mlp.cpp](../mlp/mlp.cpp). This document records every
non-obvious architectural and engineering choice made in the MLP, plus a live
log of benchmark runs.

---

## 1. Problem framing

Binary classification on the CS:GO Round Winner dataset — 103 numeric features,
label `round_winner ∈ {+1, −1}` at CSV column 95. 97,929 training rows;
24,483 test rows. Community benchmark accuracy: 76–80%.

The proposal's hypothesis names MLP as the algorithm expected to scale best of
the five: dense matrix operations give high arithmetic intensity (FLOPs / byte
of memory traffic), so the parallel work isn't memory-bandwidth-starved.
The MLP's 8-core speedup number is therefore the **headline empirical claim**
of the paper.

---

## 2. Architecture

### Topology: `103 → 64 → 32 → 1`

| Layer       | In  | Out | Weights | Activation |
|-------------|----:|----:|--------:|------------|
| Hidden 1    | 103 |  64 |   6,592 | ReLU       |
| Hidden 2    |  64 |  32 |   2,048 | ReLU       |
| Output      |  32 |   1 |      32 | Sigmoid    |
| **Total**   |     |     | **8,672 + 97 biases = 8,769 params** | |

**Why this size.** Large enough to hit the 76–80% target (CS:GO data is close
to linearly separable, most signal is captured by the first layer). Small
enough that the entire model fits in L2 cache (`W1`≈51 KB, `W2`≈16 KB,
`W3`≈256 B = ~67 KB), so every sample's forward pass reuses fully-cached
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

Coefficient `LAMBDA = 1e-4` (matched to SVM for apples-to-apples comparison).
Biases are **not** regularized (standard practice — bias shrinkage would bias
the decision boundary toward the origin, which is undesirable after z-score
normalization centers features at 0). Implemented multiplicatively as
`W ← (1 − lr·λ)·W − lr·gW/B` rather than adding a `λ·W` term to the gradient —
same math, fewer operations, same pattern as SVM's `scale = 1 − lr·LAMBDA`.

### Weight initialization
- **He** for ReLU layers: `N(0, √(2 / fan_in))`. fan_in = 103 for `W1`, 64 for
  `W2`. Keeps pre-activation variance ≈ 1 at initialization, avoiding dead
  neurons and saturated gradients in the first backward pass.
- **Small uniform** for the output layer: `U(−0.01, 0.01)`. Prevents the
  sigmoid from saturating at init.
- All biases: zero.
- Seeded by `std::mt19937(SEED=42)` for reproducibility across the four
  training variants.

### Labels
Dataset stores `y ∈ {+1, −1}` to stay compatible with the SVM loader.
Converted to `t ∈ {0, 1}` at each point of use inside the MLP
(`int t = (y[s] == 1) ? 1 : 0`). `ds.y_train` is never mutated, so the same
`Dataset` struct can be passed to any model.

---

## 3. Hyperparameters

```
MAX_EPOCHS = 30       (BCE + momentum converges far faster than hinge;
                       SVM's 200 epochs would be wasted)
BATCH      = 128      (mini-batch SGD; divides N_THREADS=8 evenly)
LR         = 0.01     (base; decayed as LR/√(epoch+1) in the momentum trainers)
LAMBDA     = 1e-4     (matched to SVM)
MOMENTUM   = 0.9      (Polyak; used in better_serial, OMP, pthreads)
N_THREADS  = 8        (matched to SLURM --cpus-per-task=8)
SEED       = 42
```

`static_assert(BATCH % N_THREADS == 0)` enforces the even-slicing invariant.

---

## 4. Data structures & memory layout

### `struct MLPModel`
Six weight/bias pairs plus six matching momentum buffers. All stored as flat
`std::vector<double>` in **row-major** order:

```
W1[i * N_FEATURES + j]   = edge from input  j → hidden-1 unit i
W2[i * H1         + j]   = edge from hidden-1 unit j → hidden-2 unit i
W3[i * H2         + j]   = edge from hidden-2 unit j → output unit i
```

Row-major means the `j` loop (over fan-in) is unit-stride and auto-vectorizable
under `-O3 -march=native`. Nested `vector<vector<double>>` would add a pointer
chase per row, kill vectorization, and scatter the weights across heap
allocations — all bad for a cache-resident model.

### Activation buffers on the stack

`forward_sample` and `predict_*` declare `double a1[H1], a2[H2]` as local
fixed-size arrays. Since `H1`/`H2` are `constexpr`, these are true stack
allocations — no heap traffic in the inference hot loop.

### Thread-local gradient buffers

For both parallel trainers, each thread owns its own gradient accumulator
(same shapes as the weight matrices). **Allocated once at the top of
the trainer function / pthread worker**, then `std::fill`ed to zero per
mini-batch — never reallocated. If allocated inside the batch loop,
`Vec(6592)` × 8 threads × 765 batches × 30 epochs ≈ 9 GB of churn would
dominate runtime.

In the OMP trainer this is a `std::vector<Vec>` indexed by `tid` (arena style).
In the pthreads trainer each worker holds its own stack-local `Vec`s — no
shared container, zero false-sharing risk.

---

## 5. Training — four variants

All four trainers share the same forward/backward kernel (`forward_sample`,
`backward_sample`) and differ only in the outer structure.

### 5a. `train_serial` — plain mini-batch SGD
Fixed LR, no momentum. Reference implementation. Not used as the speedup
baseline (momentum converges much faster, so comparing parallel-with-momentum
against serial-without-momentum would conflate two effects).

### 5b. `train_better_serial` — decaying LR + momentum
`η_t = LR / √(epoch + 1)` + Polyak momentum (`v ← β·v + g/B`, `w ← (1 − ηλ)w − η·v`).
**This is the speedup baseline.** Mirrors SVM's `train_better_serial` in role.

### 5c. `train_parallel_omp` — OpenMP data-parallel over the mini-batch
Per mini-batch:
1. Zero the shared gradient accumulator.
2. Open `#pragma omp parallel num_threads(8)`. Inside:
   - Zero thread-local grad + loss (using pre-allocated arena).
   - `#pragma omp for schedule(static) nowait` over the BATCH samples —
     each sample's forward + backward accumulates into thread-local buffers.
   - `#pragma omp critical` merges thread-local grads into the shared
     accumulator. (OpenMP `reduction(+:...)` does not work on `std::vector`.)
3. SGD+momentum update runs single-threaded after the parallel region
   (~9K params × 6 updates = fast enough that threading the update is not
   worth the barrier overhead).

Why a `critical` section and not finer-grained? Eight threads, six vectors
totaling ~9K doubles → ~72KB of data to reduce per batch. Serializing this is
cheap compared to the forward+backward compute, and the code is straightforward
to reason about. If profiling shows the critical section dominates, fan out
into per-vector atomics or tree reductions.

### 5d. `train_parallel_pthreads` — SVM-style mirror
Mirrors the SVM pattern exactly for writeup symmetry:

```
struct MLPParState {
  const Matrix* X;  const IVec* y;   MLPModel* model;
  IVec idx;          int batch_start;  double epoch_loss;
  Vec gW1, gb1, gW2, gb2, gW3, gb3;   double shared_loss;
  pthread_mutex_t   mutex;
  pthread_barrier_t barrier_acc;
  pthread_barrier_t barrier_upd;
  unsigned seed;
};
```

Worker loop per epoch:
- **(Thread 0 only)** reshuffle `idx` using `mt19937(SEED + epoch)`; reset
  `batch_start`, `epoch_loss`.
- `barrier_upd` ← all threads wait for the shuffle.

Per mini-batch:
1. Each thread computes forward + backward on its slice
   `[tid · 16, (tid+1) · 16)` of the current batch, into its stack-local
   gradient buffers.
2. `mutex_lock` → add locals to shared `st->gW*` → `mutex_unlock`.
3. `barrier_acc` ← all threads have contributed.
4. **(Thread 0 only)** `sgd_momentum_update`; advance `batch_start`; zero the
   shared grad accumulator.
5. `barrier_upd` ← none may proceed until the weights are written.

**Why two barriers?** `barrier_acc` protects against "thread 0 reads the
shared gradient before thread N has locked-and-added its contribution".
`barrier_upd` protects against "thread N starts the next batch's forward
pass against a half-updated weight vector".

### Why mini-batch, not full-batch or per-sample?

- **Full-batch** on 97,929 samples: one update per epoch, poor convergence with
  the given LR budget, also means only 30 parallel regions per full training
  run — not enough to amortize thread spin-up.
- **Per-sample (batch=1)**: every sample requires a barrier → synchronization
  dominates compute, speedup collapses.
- **batch=128** gives ~765 parallel regions per epoch × 30 epochs ≈ 23K
  chances to amortize, and each region has ~128 × 8.7K FLOPs ≈ 1.1M FLOPs of
  compute — way more than the barrier cost.

---

## 6. Inference

`predict_serial` and `predict_parallel` are identical except for the
`#pragma omp parallel for schedule(static) num_threads(8)` on the outer sample
loop. Each iteration's forward pass is independent (read-only access to
weights, stack-local activations) → embarrassingly parallel, no atomics needed.

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
| SVM       | `dot(w, x)` + margin check | ~210          | 824            | ~0.25          |
| MLP       | forward + backward         | ~17,400       | 824            | ~21            |

~80× higher arithmetic intensity. The MLP does far more compute per byte of
memory traffic, so eight threads each running GEMM-style inner loops aren't
competing for memory bandwidth. SVM is closer to memory-bound — its observed
speedup ceiling should sit below MLP's on the same 8-core machine.

---

## 8. Numerical & correctness risks (documented for debugging)

1. **Sigmoid saturation** — `log(sigmoid(z))` overflows at |z| > 36. Use the
   logit-form BCE (already done).
2. **Exploding gradients** at LR=0.01 — if epoch 0 prints `loss=nan`, drop LR
   to 0.005 first; weight init should prevent this but verify.
3. **Non-deterministic reductions** — OpenMP `critical` and pthread mutex
   reductions accumulate in thread-arrival order, which differs per run. The
   final accuracy is stable (to ±0.005) but the printed per-epoch loss will
   differ in the 5th decimal place. Do **not** diff the loss lines of two
   parallel runs — they will never match bitwise.
4. **Shuffle reproducibility** — only thread 0 reshuffles the index array, and
   does so with `mt19937(SEED + epoch)` so the shuffle is deterministic
   across runs. If another thread also shuffles, parallel-vs-serial comparison
   becomes meaningless.
5. **Speedup numerator** — the speedup summary compares `train_better_serial`
   (momentum baseline) against the two parallel variants (also momentum), so
   the numerator and denominator are the same algorithm. If you compared
   `train_serial` (no momentum) against `train_parallel_omp` (with momentum),
   you'd be measuring "algorithm improvement + parallelization" together.

---

## 9. Build & run

On USC CARC:

```bash
sbatch job.sl
# outputs to gpujob.out
# compiles svm + mlp, runs both against train_cleaned.csv / test_cleaned.csv
```

Locally (macOS):

```bash
# Option A: Homebrew libomp with Apple Clang
brew install libomp
clang++ -std=c++17 -O3 -march=native \
  -Xpreprocessor -fopenmp -lomp mlp/mlp.cpp -o mlp -lpthread
./mlp train_cleaned.csv test_cleaned.csv

# Option B: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp mlp/mlp.cpp -o mlp -lpthread
```

**Apple Clang alone will not work** — it lacks an OpenMP runtime, and
`pthread_barrier_t` is an optional POSIX feature that macOS does not
implement. On a mismatched-Xcode-CLT Mac (e.g. Xcode 13 against macOS 26)
even simple `<iostream>` includes fail to compile, at which point the only
option is CARC.

### Verification expected

- **Accuracy**: `acc ∈ [0.76, 0.80]` on test set for all four trainers.
- **Loss**: monotone-ish decrease per epoch (some oscillation from mini-batch
  noise is normal).
- **Parity**: `|acc_better_serial − acc_omp| < 0.01` and
  `|acc_better_serial − acc_pthreads| < 0.01`. The `main()` prints these
  deltas at the end of every run.
- **Speedup**: `better_serial_total / omp_total > 1` and
  `better_serial_total / pthreads_total > 1`. Target 4–6× on 8 cores.

---

## 10. Results log

Append one row per CARC run. Format:

```
| date (UTC) | git SHA | epochs | batch | machine/cores | serial (s) | better (s) | omp (s) | pth (s) | acc_better | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
```

| date (UTC) | git SHA  | epochs | batch | machine/cores | serial (s) | better (s) | omp (s) | pth (s) | acc_better | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:------:|:-----:|---------------|-----------:|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| (pending first CARC run) | | 30 | 128 | CARC / 8 | | | | | | | | | | expected ~0.78 acc, 4–6× speedup |

---

## 11. Followups (known work deferred)

- **Deduplicate loader/metrics** into `common.hpp` once `analytics_engine.cpp`
  is being written. Don't do it sooner — keeping every algorithm self-contained
  matches the project's current style.
- **Mini-batch size sweep** — the writeup may want a BATCH ∈ {32, 64, 128,
  256, 512} scan to show how batch size trades off synchronization overhead
  against gradient noise. Cheap to run, adds a figure.
- **Thread count sweep** — the full speedup curve S(P) for P ∈ {1, 2, 4, 8}
  is the most informative figure for the paper. Requires parameterizing
  `N_THREADS` at the CLI rather than hardcoding it at top-of-file.
- **Cache-blocked GEMM** — not needed at current layer widths
  (W1 fits in L1d = 64 KB on CARC's Skylake/Cascade Lake nodes). Would only
  matter if H1 ≥ 512.
- **SIMD hinting** — `-O3 -march=native` already auto-vectorizes the inner
  dot products; explicit `#pragma omp simd` on the `j` loops inside
  `forward_sample` might squeeze another 10–20% but is not required.
