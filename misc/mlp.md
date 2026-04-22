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

That statement originally described the sample-loop backend. The current source
tree no longer ships that path: MLP is now BLAS-only, with each minibatch
rewritten as dense matrix multiplies. Historical sample-loop benchmark rows
remain below for comparison, but the supported implementation on CARC is the
vendor-BLAS backend and the right counterfactual against sklearn's own
BLAS-backed `MLPClassifier`.

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

`N_THREADS` and the algorithm-internal hyperparameters (`H1`, `H2`, `BATCH`,
`MOMENTUM`, `SEED`) stay `constexpr`; `epochs`, `lr`, and `lambda` are runtime
parameters so the CLI signature
(`./mlp train test [epochs] [lr] [lambda]`) matches SVM.

Runtime backend controls:

- `N_THREADS=<n>`: thread budget used by the BLAS-backed execution path.
- `MLP_BLAS_BLOCK=<rows>`: prediction block size for the BLAS inference path
  (`512` by default).

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
features on the current cleaned CSVs, expect `72 z-scored, 31 passthrough`.

---

## 5. Two execution backends

The file now exposes the same public entry points (`train_serial`,
`train_parallel_omp`, `train_parallel_pthreads`, `predict_serial`,
`predict_parallel`) over two inner-kernel eras:

1. **Sample-loop backend**: the original pure C++ implementation, now removed
   from the current tree but kept here as historical context for earlier
   benchmark rows.
2. **BLAS batch backend**: the current implementation, compiled with
   `MLP_USE_BLAS` / `MLP_USE_MKL`.

The model, loss, optimizer, normalization, and metrics are unchanged across
the two backends. Only the dense linear algebra implementation changes.

### 5a. Historical sample-loop backend (removed)

This is the original design:

- `forward_sample(...)` computes one sample at a time into stack-local
  `a1[H1]`, `a2[H2]`.
- `backward_sample(...)` accumulates one sample's contribution into flat
  gradient buffers.
- `train_parallel_omp(...)` uses per-thread gradient buffers plus one serial
  reduction over `N_THREADS`.
- `train_parallel_pthreads(...)` uses mutex + two barriers per minibatch.

This path remains relevant only as historical baseline context for the writeup.
It is no longer present in the current source tree.

### 5b. BLAS batch backend

The BLAS path stops thinking in terms of one sample at a time and instead
treats each minibatch as dense row-major matrices:

```text
Xb : [B x F]   minibatch features  (B=128, F=103)
A1 : [B x H1]  hidden layer 1 activations
A2 : [B x H2]  hidden layer 2 activations
Z3 : [B x 1]   output logits
```

Forward pass:

```text
A1 = ReLU(Xb * W1^T + b1)
A2 = ReLU(A1 * W2^T + b2)
Z3 = A2 * W3 + b3
```

Backward pass:

```text
DZ3 = sigmoid(Z3) - T
gW3 = A2^T * DZ3
D2  = (DZ3 * W3^T) o 1[A2 > 0]
gW2 = D2^T * A1
D1  = (D2  * W2   ) o 1[A1 > 0]
gW1 = D1^T * Xb
```

Implementation details:

- Inputs remain `float` in `Dataset`, but each minibatch is packed once into a
  contiguous `double` buffer (`pack_indexed_batch_double`) so the weight
  matrices and gradients stay in `double`.
- `forward_batch_blas(...)` uses `cblas_dgemm` for `Xb * W1^T` and
  `A1 * W2^T`, then `cblas_dgemv` for `A2 * W3`.
- `backward_batch_blas(...)` uses one `dgemv` and three `dgemm` calls plus
  cheap pointwise ReLU masks and column sums.
- `MLPBlasWorkspace` owns reusable `A1`, `A2`, `Z3`, `DZ3`, `D2`, and `D1`
  buffers so the batch path allocates once and reuses memory every minibatch.

### Why the BLAS path does not slice the minibatch across outer threads

The first implementation attempt kept the old outer-threading structure and
gave each worker `BATCH / N_THREADS` rows. On this network that means
`16 x 103`, `16 x 64`, and `16 x 32` panels at `N_THREADS=8`, which is too
small to amortize BLAS-call overhead or extra barriers. The result was slower
than just handing the full 128-row minibatch to the vendor kernel.

So the current dispatch is deliberate:

- `train_serial(...)` calls `train_serial_blas(...)` when BLAS is enabled.
- `train_parallel_omp(...)` and `train_parallel_pthreads(...)` also call
  `train_serial_blas(...)` when BLAS is enabled.

That means the BLAS backend keeps the three public result blocks for CSV /
parser compatibility, but all three training variants intentionally reuse the
same dense batch kernel.

### Why mini-batch, not full-batch or per-sample?

- **Full-batch** on 97,929 samples: one update per epoch is too coarse for a
  non-convex problem and doesn't fit the existing SGD+momentum comparison.
- **Per-sample**: no dense kernel to hand to BLAS and synchronization would
  dominate.
- **Batch=128** is large enough to form useful GEMM panels while still
  matching the tuned hyperparameters already used by the paper and sklearn
  baseline.

---

## 6. Inference

The pure C++ inference path is unchanged: one forward pass per sample, with
optional OpenMP over the outer test-row loop.

The BLAS inference path mirrors training:

- `predict_blas_range(...)` packs a contiguous block of test rows into `double`
  and runs the same batched forward pass used by training.
- `MLP_BLAS_BLOCK` controls how many test rows are grouped into one BLAS call.
  Default is `512`.
- `predict_parallel_blas(...)` parallelizes over prediction blocks with OpenMP
  when available; otherwise it falls back to the serial batched path.

Prediction threshold stays `logit >= 0 -> +1 else -1`, which is equivalent to
`sigmoid >= 0.5` and avoids one redundant sigmoid during inference.

---

## 7. Why the BLAS backend helps

The original sample-loop MLP already had high arithmetic intensity compared to
SVM:

| Algorithm | Inner kernel                | FLOPs / sample | Bytes / sample | AI (FLOP/byte) |
|-----------|-----------------------------|---------------:|---------------:|---------------:|
| SVM       | `dot(w, x)` + margin check  | ~210           | 412            | ~0.5           |
| MLP       | forward + backward          | ~17,400        | 412            | ~42            |

But arithmetic intensity alone is not enough. The pure backend still executes
those FLOPs through many small scalar-ish loops and reductions. sklearn's
`MLPClassifier`, by contrast, already routes its dense layers through BLAS via
NumPy / `safe_sparse_dot`.

The new backend closes that implementation gap:

- our model math is unchanged;
- the dense work is now presented to the CPU as GEMM/GEMV panels;
- the vendor library handles register blocking, cache tiling, and SIMD;
- we avoid the failed design of wrapping tiny outer slices around already-fast
  BLAS kernels.

### Local head-to-head (same machine)

On the local macOS Accelerate build:

- **our BLAS backend**: `2.236 s` train, `4.89 ms` infer, `0.7773` accuracy
  (`VECLIB_MAXIMUM_THREADS=8`, `N_THREADS=8`);
- **sklearn MLP**: `5.182 s` train, `5.22 ms` infer, `0.7546` accuracy
  (`N_THREADS=8`, same dataset / matched hyperparameters).

So the improvement is not "our hand-written scalar loops beat sklearn." The
actual claim is narrower and more defensible:

> once the same vendor-BLAS inner-kernel quality is available to our C++
> implementation, the remaining estimator overhead is small enough that our
> MLP can beat sklearn locally on this benchmark.

One more nuance: for MLP, unlike KNN, the best local configuration was **not**
"outer threads + BLAS threads pinned to 1." The training backend wants the
vendor library to own the dense kernel. That is why the MLP SLURM script now
hands the CPU budget to BLAS instead of forcing single-threaded BLAS.

---

## 8. Numerical and runtime risks

1. **Sigmoid saturation**: handled by the logit-form BCE already in the code.
2. **Exploding gradients**: still possible if `lr` is pushed well above `0.01`.
3. **Nested parallelism / oversubscription**: the failed outer-sliced BLAS
   path was removed for exactly this reason. Do not reintroduce per-thread
   minibatch slices on top of threaded BLAS unless layer sizes change enough
   to justify it.
4. **Historical parity expectations**: the removed sample-loop backend and the
   current BLAS backend should stay within the same accuracy band. The exact
   per-epoch loss text can still differ
   slightly because the reduction order changes.

---

## 9. Build and run

On USC CARC (from `src/cpp/mlp/`):

```bash
cd src/cpp/mlp && sbatch job_mlp.sl
```

The BLAS job path accepts:

- `MLP_BLAS_CFLAGS` (for example `-DMLP_USE_MKL`)
- `MLP_BLAS_LIBS`   (for example the oneMKL or OpenBLAS link flags)
- `MLP_THREADS`     (defaults to `N_THREADS` or `SLURM_CPUS_PER_TASK`)

The consolidated sweep script also supports:

```bash
sbatch slurm/job_sweep.sl
```

CLI:

```bash
./mlp [train_csv] [test_csv] [epochs] [lr] [lambda]
# defaults: data/train_cleaned.csv  data/test_cleaned.csv  30  0.01  1e-4
```

Local build examples from the repo root:

```bash
# Linux / OpenBLAS
g++ -std=c++17 -O3 -march=native -fopenmp -DMLP_USE_BLAS \
  src/cpp/mlp/mlp.cpp -o mlp -lpthread -lopenblas

# macOS / Accelerate
clang++ -std=c++17 -O3 -DMLP_USE_BLAS \
  src/cpp/mlp/mlp.cpp -o mlp -lpthread -framework Accelerate
```

For oneMKL, compile with `-DMLP_USE_MKL` and the link line from Intel's MKL
Link Advisor.

Runtime examples:

```bash
# Recommended BLAS-backed run on macOS
VECLIB_MAXIMUM_THREADS=8 N_THREADS=8 ./mlp data/train_cleaned.csv data/test_cleaned.csv

# Tune BLAS prediction blocking
MLP_BLAS_BLOCK=1024 ./mlp data/train_cleaned.csv data/test_cleaned.csv
```

Verification targets:

- **Accuracy**: `acc` stays in the `0.76-0.80` band on the full test split.
- **Parity**: `|acc_serial - acc_omp| < 0.01` and
  `|acc_serial - acc_pthreads| < 0.01`.
- **Parity**: serial / OMP / pthread wrappers should stay numerically aligned,
  because they all call the same BLAS-backed trainer.

---

## 10. Results log

Append one row per CARC run. Format:

| date (UTC) | git SHA | epochs | batch | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:------:|:-----:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| 2026-04-20 | 526bb18 | 30 | 128 | CARC d17-03 / 8 | 42.01 | 9.24 | 9.09 | 0.7748 | 0.7748 | 0.7727 | 4.55× | 4.62× | Full `{1,2,4,8}` sweep, job 3272373. This is the original pure sample-loop backend. Accuracy matches target, but KNN outscaled it. See [results/run1/results.md](../results/run1/results.md). |

Latest local BLAS notes (same-machine checks after the vendor-BLAS rewrite):

- `MLP_USE_BLAS` + Accelerate, `VECLIB_MAXIMUM_THREADS=8`, `N_THREADS=8`:
  serial `2.241 s`, OMP `2.268 s`, pthreads `2.242 s`, all at `0.7773`
  accuracy with exact parity.
- Historical pure sample-loop build from the earlier local run:
  serial `22.323 s`, pthreads `6.341 s` on the full dataset.
- sklearn `MLPClassifier` with matched hyperparameters on the same machine:
  train `5.182 s`, infer `5.22 ms`, accuracy `0.7546`.

---

## 11. Followups

- **Rerun CARC sweep** so the paper has cluster numbers for the new backend
  instead of only the original sample-loop row.
- **Per-algorithm BLAS env in `analytics_engine.cpp`** if we want one sweep to
  optimize KNN (`BLAS threads = 1`) and MLP (`BLAS threads = N_THREADS`)
  simultaneously.
- **Float / SGEMM experiment**: the current BLAS path keeps weights and
  gradients in `double`. A float-weight path may be faster if accuracy stays
  stable.
- **Shared loader/metrics extraction** into a common header remains
  mechanical; the MLP file already follows the same `Dataset` / `Metrics` /
  `print_results` conventions as SVM, KNN, DT, and NB.
