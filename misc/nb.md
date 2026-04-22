# NB — Design Document & Results Log

Implementation at [src/cpp/nb/nb.cpp](../src/cpp/nb/nb.cpp). This document records every
non-obvious architectural and engineering choice made in the Naive Bayes
classifier, plus a live log of benchmark runs.

---

## 1. Problem framing

Binary classification on the CS:GO Round Winner dataset — 103 numeric features
(read from the header at runtime), label `round_winner ∈ {+1, −1}` detected by
column name in the header. 97,929 training rows; 24,483 test rows. Community
benchmark accuracy: 76–80%.

The proposal's hypothesis names Naive Bayes alongside KNN as algorithms that
will "achieve **high initial speedup** before hitting memory bandwidth limits"
(§ I.A). § II.A also calls NB out explicitly as a data-parallel workload:
"Naïve Bayes class conditional frequency counting, where **no inter-thread
communication is required** during the main computation phase." NB's scaling
curve is therefore a direct test of the memory-bandwidth-ceiling prediction,
paired with KNN for comparison.

NB is expected to be the **weakest classifier of the five** on accuracy: CS:GO
features are strongly correlated (team health ↔ money ↔ weapon counts), which
directly violates NB's feature-independence assumption. Target accuracy range
is **0.70–0.76** — a few points below the 76–80% community benchmark, which
is itself the ceiling for classifiers that exploit feature correlations.

---

## 2. Algorithm: hybrid Gaussian + Bernoulli

Pure Gaussian NB breaks on the dataset's binary features (`map_*` one-hots
and `bomb_planted`): those columns are point masses at 0.0 and 1.0, giving
near-zero variance that requires arbitrary smoothing to prevent divide-by-
zero. Pure Bernoulli NB throws away the continuous signal (health, money,
weapon counts take many values).

**Solution:** dispatch per-feature on an `is_binary` mask. This is the same
set of columns that [`load_dataset`](../src/cpp/nb/nb.cpp) skips during z-score
normalization, so the mask is derived from the existing convention without
any new bookkeeping.

| `is_binary[j]` | Feature examples                         | Likelihood                | Parameters per class |
|:--------------:|------------------------------------------|---------------------------|----------------------|
| `true`         | `map_*`, `bomb_planted`, auto-detected 0/1 | **Bernoulli** (Laplace α=1) | `P(x = 1 \| y)`      |
| `false`        | continuous (z-scored)                     | **Gaussian**               | `μ`, `σ²`            |

Predict `argmax_y log P(y | x) = argmax_y [log P(y) + Σ_j log P(x_j | y)]`.
Log-space accumulation avoids underflow (103 features × tiny probabilities
easily underflows double if done in probability space).

### Why the mask is valid post-normalization

`load_dataset` sets `mean[j] = 0, sd[j] = 1` for binary columns, so the
normalize-in-place pass leaves those columns' values literally unchanged —
`bomb_planted` stays `{0.0, 1.0}`, each `map_*` stays `{0.0, 1.0}`. After
normalization the `BIN_THRESHOLD = 0.5` dichotomy in the NB code cleanly
separates on-values from off-values. Z-scored columns, by contrast, are
roughly N(0, 1) and have no meaningful threshold at 0.5 — but for those
columns the Gaussian branch fires instead.

---

## 3. Hyperparameters

```
LAPLACE_ALPHA = 1.0    (Bernoulli add-α smoothing; also applied to class priors)
VAR_SMOOTHING = 1e-9   (floor added to every Gaussian variance)
BIN_THRESHOLD = 0.5    (x > 0.5 → Bernoulli "on")
N_THREADS     = 8      (matched to SLURM --cpus-per-task=8)
SEED          = 42     (unused — NB is deterministic; kept for convention)
```

All `constexpr`. NB has no iterative training, no learning rate, no batch
size — the CLI is simply `./nb [train_csv] [test_csv]` with no
hyperparameters, deliberately simpler than MLP/SVM/DT/KNN.

---

## 4. Data structures & memory layout

### `struct Dataset` (shared with SVM, KNN, MLP, DT)

Identical 5-way shared struct:

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

### `struct NBModel`

```cpp
struct NBModel {
    int n_features = 0;
    std::vector<unsigned char> is_binary;   // n_features; dispatch mask

    double log_prior[2];                    // [0] = y=-1, [1] = y=+1

    // Gaussian (used when !is_binary[j]); layout j*2 + class_idx:
    std::vector<double> mean;               // 2 * n_features
    std::vector<double> var_inv_half;       // 2 * n_features; 1 / (2σ²)
    std::vector<double> log_var_term;       // 2 * n_features; -0.5·log(2π σ²)

    // Bernoulli (used when is_binary[j]); same j*2 + class_idx layout:
    std::vector<double> log_p_on;           // 2 * n_features; log P(x=1 | y)
    std::vector<double> log_p_off;          // 2 * n_features; log P(x=0 | y)
};
```

**Class-index convention.** `c = 0` for `y == −1`, `c = 1` for `y == +1`.
Array lookups use `j * 2 + c`, which keeps both classes' parameters for a
feature on the same cache line (16 bytes apart) — important for the
inference hot loop that scores both classes in tandem.

**Precomputed Gaussian terms.** `var_inv_half` and `log_var_term` store the
constants that appear in `log N(x; μ, σ²) = log_var_term − (x − μ)²
· var_inv_half`. Training finalization computes these once per feature per
class; inference does only two multiplies and two adds per (sample, feature)
rather than reconstructing from `μ, σ²` every time.

### Per-thread accumulators

Training reduces four quantities per (feature, class):
`n_class`, `count_on`, `sum`, `sumsq`. Both parallel trainers allocate
per-thread arenas for these — same arena-style pattern used by
SVM/MLP train_parallel_omp. In the pthreads variant the arenas are
stack-local to each worker; in the OMP variant they're indexed by
`omp_get_thread_num()` into a pre-allocated `std::vector<std::vector<...>>`.

Arena size per thread: `2 · n_features` `long long` (count_on, n_class) +
`2 · n_features` `double` (sum) + `2 · n_features` `double` (sumsq)
≈ 2 · 103 · (8 + 8 + 8) = ~5 KB per thread. Fits in L1 trivially.

### Data-preprocessing skip list

Identical to SVM/KNN/MLP/DT: `map_*`, `bomb_planted`, and auto-detected 0/1
columns bypass z-score normalization. NB **reuses** this same logic in a
helper `detect_binary()` that rebuilds the mask from the Dataset (the
Dataset struct itself doesn't carry the mask, to avoid breaking the 5-way
shared shape). On CS:GO expect `NB features: 94 Gaussian, 9 Bernoulli`
(8 map_* + bomb_planted).

---

## 5. Training — three variants

All three trainers execute the **same** closed-form MLE: count per-class
frequencies, compute Gaussian `(μ, σ²)` and Bernoulli `P(x=1 | y)`, apply
Laplace/variance smoothing, precompute log terms. They differ only in how
the single pass over training rows is parallelized. Serial and parallel
should produce **byte-identical** models up to FP reduction order; the
accuracy parity check at end of `main()` should show `|Δacc| = 0.0` in
practice.

### 5a. `train_serial` — single-pass accumulation

Iterate all `n_rows` samples, for each compute its class index `c` and
update accumulators:

```
n_class[c] += 1
for j in 0..n_features:
    v = X[i*F + j]
    if is_binary[j]:
        count_on[j*2 + c] += (v > BIN_THRESHOLD)
    else:
        sum[j*2 + c]   += v
        sumsq[j*2 + c] += v·v
```

Then `finalize_model()` computes all the derived quantities in one pass over
the `(j, c)` grid. Single pass → `O(n · F)` time, `O(F)` space.

### 5b. `train_parallel_omp` — OpenMP + per-thread arena + serial reduction

Mirrors the SVM/MLP `train_parallel_omp` pattern exactly. No `#pragma omp
critical`, no atomics:

```cpp
std::vector<std::vector<long long>> l_n_class (N_THREADS, ...);
std::vector<std::vector<long long>> l_count_on(N_THREADS, ...);
std::vector<std::vector<double>>    l_sum     (N_THREADS, ...);
std::vector<std::vector<double>>    l_sumsq   (N_THREADS, ...);

#pragma omp parallel for schedule(static) num_threads(N_THREADS)
for (int i = 0; i < n_rows; ++i) {
    int tid = omp_get_thread_num();
    // ...accumulate into l_*[tid]
}

// Serial reduction across tid
// finalize_model(...)
```

`schedule(static)` partitions n_rows evenly — ~12,241 rows per thread on
CS:GO — giving each thread identical work.

### 5c. `train_parallel_pthreads` — fixed pool + mutex-protected merge

Pthread variant. Fixed pool of `N_THREADS` workers, each takes a static
slice `[tid·n/P, (tid+1)·n/P)`. The worker:

1. Allocates its **own stack-local** accumulators (no sharing).
2. Sweeps its slice, filling locals.
3. Acquires `pthread_mutex_t` **once**, merges locals into shared counters,
   releases the mutex.
4. Returns. `pthread_join` on the main thread supplies the final barrier.

```cpp
struct NBParState {
    // ...inputs + is_binary mask...
    long long           n_class[2];
    std::vector<long long> count_on;
    std::vector<double>    sum, sumsq;
    pthread_mutex_t        mutex;
};
```

**Why no barriers?** NB is one-shot — no iteration, no per-iteration
synchronization. A single mutex-protected merge at the end of each worker
(N_THREADS mutex acquisitions total, serialized naturally) is all the
synchronization needed. This is the simplest and most faithful
implementation of the proposal's claim: "no inter-thread communication
is required during the main computation phase."

**Why mutex at all (vs. arena-style)?** Could also be done arena-style with
no mutex (per-thread slots + serial reduction after `pthread_join`). The
mutex variant was chosen for stylistic parity with SVM/MLP/DT pthreads
trainers — keeps the OMP-vs-pthreads A/B test about "OpenMP runtime +
implicit sync" vs "pthread_create + explicit mutex", not about arena
layout.

---

## 6. Inference

`predict_serial` and `predict_parallel` compute log-posteriors for both
classes in tandem, argmax:

```cpp
for each test sample i:
    score[0] = log_prior[0]
    score[1] = log_prior[1]
    for each feature j:
        v = X[i*F + j]
        if is_binary[j]:
            on = (v > 0.5)
            score[0] += on ? log_p_on[j*2+0] : log_p_off[j*2+0]
            score[1] += on ? log_p_on[j*2+1] : log_p_off[j*2+1]
        else:
            d0 = v - mean[j*2+0]
            d1 = v - mean[j*2+1]
            score[0] += log_var_term[j*2+0] - d0*d0 * var_inv_half[j*2+0]
            score[1] += log_var_term[j*2+1] - d1*d1 * var_inv_half[j*2+1]
    pred[i] = (score[1] > score[0]) ? +1 : -1
```

`predict_parallel` uses `#pragma omp parallel for schedule(static)
num_threads(N_THREADS)`. Each test sample is independent — no reduction,
no atomics. Only one parallel inference variant (OpenMP) for parity with
SVM/MLP/DT; the full inference kernel is so cheap (`O(n_test · F)` with no
branches besides the Gaussian/Bernoulli dispatch) that a second pthreads
variant isn't necessary.

---

## 7. Arithmetic intensity (for the writeup)

**Per (sample, feature) pair in the training hot loop:**

| Memory op                                     | Bytes |
|-----------------------------------------------|------:|
| Load `X[i*F+j]` (float, unit-stride in j)     | 4     |
| Load `is_binary[j]` (1 byte, amortized)       | ≈0    |
| Load `y[i]` (4 bytes, amortized across j)     | ≈0    |

Ops per (sample, feature): ~3 (compare, conditional add, optional square).

**~4 bytes / 3 ops → ~0.75 FLOP/byte.** **Memory-bound**, in the SVM range.

With 8 threads contending for ~25 GB/s DRAM bandwidth: BW-limited throughput
≈ 25 GB/s / 4 B = 6.25 G iter/s × 3 ops ≈ 19 Gops/s total. Single-core
compute peak ≈ 12 Gops/s (AVX2 on Cascade Lake). Expected speedup **2–4× on
8 cores**, plateauing early due to memory bandwidth saturation. This
matches the proposal's hypothesis precisely.

Comparison across all five algorithms (updated with the new data point):

| Algorithm | Inner kernel                  | AI (FLOP/byte) | Expected scaling   |
|-----------|-------------------------------|---------------:|--------------------|
| SVM       | `dot(w, x)` + margin check    | ~0.5           | Memory-bound       |
| NB        | frequency count + Gaussian MLE| ~0.75          | Memory-bound       |
| KNN       | squared-L2 distance           | ~0.5           | Memory-bound       |
| DT        | histogram fill + Gini sweep   | ~3             | Crossover regime   |
| MLP       | dense GEMM forward/back       | ~42            | Compute-bound      |

The SVM / NB / KNN trio should cluster tightly at ~2–4× speedup. DT should
sit above them. MLP should be the outlier at 5–7× (the proposal's headline
claim).

---

## 8. Numerical & correctness risks

1. **Class imbalance collapsing priors.** If one class has zero samples in a
   slice handed to a worker, the worker's local `n_class[c]` is 0 for that
   class. That's fine for the per-thread reduction (it just contributes
   nothing). The shared `n_class[c]` across all workers is 0 only if the
   full training set has zero samples of class c — in which case we
   apply Laplace smoothing to the priors (`(n_class + α) / (n + 2α)`) to
   avoid `log(0)`. Without this, underflow produces `−inf` log-priors and
   breaks inference.

2. **Degenerate variance on continuous features.** If a feature is constant
   within a class (e.g., `time_left = 0` for every +1 sample in a small
   partition), `var = sumsq/n − mean² = 0` exactly. The `VAR_SMOOTHING =
   1e-9` floor prevents `log(0)` and `1/0` in the precomputed Gaussian
   terms. scikit-learn uses a similar `1e-9 · max_var` default; we use a
   fixed `1e-9` since features are already z-scored.

3. **Float-to-double casting of `X`.** `X` is `float` (4-byte) in the shared
   Dataset shape, but NB's accumulators are `double` to avoid catastrophic
   cancellation in `sumsq − mean²`. The inner loop loads `float X[i·F+j]`
   then casts to `double v = static_cast<double>(xi[j])`. No loss of
   precision since `double` is strictly larger.

4. **Non-determinism from FP reduction order.** The three training variants
   execute the same algorithm but in different reduction orders. Floating-
   point summation is not associative, so `sum` and `sumsq` values will
   differ by ≈ `n · ε_machine ≈ 98000 · 10^−15 = 10^−11` across variants.
   This propagates to `mean` and `var` at the same magnitude, well below
   `VAR_SMOOTHING`. The model's prediction vector should be byte-identical
   across all three trainers — verified by the parity check.

5. **Binary detection divergence.** `detect_binary()` re-runs the same
   logic used internally by `load_dataset`'s `skip_normalize` computation
   rather than storing the mask in the Dataset struct. If someone later
   modifies `load_dataset`'s detection logic without updating
   `detect_binary()`, the NB model will train on a different feature-
   partition than the normalizer assumed. Safer long-term: thread the mask
   through the Dataset — but that's a cross-algorithm change, so deferred.

6. **Class-index convention.** `c = 0 ↔ y = −1, c = 1 ↔ y = +1`. Any code
   that indexes `log_prior[c]` etc. must respect this. The classify function
   and finalize function are the two places to verify if making changes.

---

## 9. Build & run

On USC CARC (from src/cpp/nb/):

```bash
cd src/cpp/nb && sbatch job_nb.sl
# outputs to nbjob.out (separate from other per-algo output files)
# compiles nb.cpp, runs ./nb ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
```

CLI:

```bash
./nb [train_csv] [test_csv]
# defaults: data/train_cleaned.csv  data/test_cleaned.csv
```

Locally (macOS), from the repo root:

```bash
# Option A: Homebrew libomp with Apple Clang
brew install libomp
clang++ -std=c++17 -O3 -march=native \
  -Xpreprocessor -fopenmp -lomp src/cpp/nb/nb.cpp -o nb -lpthread
./nb data/train_cleaned.csv data/test_cleaned.csv

# Option B: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp -o nb -lpthread
```

**Apple Clang alone will not work** — no OpenMP runtime, and `pthread_barrier_t`
is an optional POSIX feature that macOS does not implement (even though NB
itself doesn't use barriers, the shared build conventions include other
algorithms that do).

### Verification expected

- **Accuracy**: `acc ∈ [0.70, 0.76]` on test set for all three trainers.
  The 76–80% community benchmark applies to classifiers that exploit
  feature correlations; NB assumes feature independence, so it undershoots
  by a few points. If `acc < 0.65`, there's likely a class-index flip or
  the `is_binary` mask is wrong.
- **Parity**: `|acc_serial − acc_omp| = 0.0` and
  `|acc_serial − acc_pthreads| = 0.0` — all three trainers run the same
  closed-form MLE. Parity check printed at end of `main()`.
- **Speedup**: `serial / omp > 1` and `serial / pthreads > 1`.
  Target 2–4× on 8 cores (memory-bound plateau).

---

## 10. Results log

Append one row per CARC run.

| date (UTC) | git SHA | features Gauss/Bern | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:-------------------:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| 2026-04-20 | 526bb18 | 94/9 | CARC d17-03 / 8 | 0.0166 | 0.0061 | 0.0047 | 0.6946 | 0.6946 | 0.6946 | 2.72× | 3.56× | Full `{1,2,4,8}` sweep, job 3272373. Falls within the predicted 2–4× memory-bound range; pthreads edges OMP (3.56× vs 2.72×) because the one-shot mutex-per-worker merge is cheaper than OMP's team startup at 16 ms serial. Accuracy matches hybrid sklearn exactly (0.6946 in all three variants, 0.0000 parity). Below the 0.70–0.76 target — feature-independence assumption is badly violated on CS:GO (health↔money↔weapon counts correlated). A sklearn Bernoulli-only variant beats hybrid by 0.8pt; noted in [results/run1/results.md](../results/run1/results.md). |

---

## 11. Followups (known work deferred)

- **Deduplicate loader / mask detection** into `common.hpp` once
  `analytics_engine.cpp` is being written. `detect_binary()` currently
  re-implements `load_dataset`'s skip-normalize logic; both should pull
  from a shared helper.
- **Thread count sweep** — full speedup curve `S(P)` for `P ∈ {1, 2, 4, 8}`
  pairs with the same sweep for the other four algorithms to produce the
  paper's headline figure.
- **Arena-style pthreads variant** (no mutex) — the current pthreads
  trainer uses one mutex acquisition per worker. An arena variant would
  eliminate the mutex entirely and isolate the comparison to "pthread
  team creation vs OpenMP team creation". Interesting only if the mutex
  is measurably a bottleneck, which on NB's one-shot pattern it shouldn't
  be.
- **Log-sum-exp for calibrated probabilities** — current code returns a
  hard `argmax`, which is all the evaluator needs. If downstream code
  wants `P(y | x)` with `sum = 1`, add a log-sum-exp normalization at the
  end of inference. Trivial if needed.
- **Per-class conditional feature importance** — NB's `log_p_on /
  log_p_off` (Bernoulli) and `mean / var` ratios (Gaussian) give a
  natural per-feature contribution score, which would be a nice
  cross-comparison to DT's feature-importance ranking if space permits
  in the writeup.
