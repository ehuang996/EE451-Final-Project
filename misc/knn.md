# KNN — Design Document & Results Log

Implementation at [src/cpp/knn/knn.cpp](../src/cpp/knn/knn.cpp). This document
records every non-obvious architectural and engineering choice made in the
k-Nearest-Neighbors classifier, plus a live log of benchmark runs.

---

## 1. Problem framing

Binary classification on the CS:GO Round Winner dataset — 103 numeric features
(read from the header at runtime), label `round_winner ∈ {+1, −1}` detected by
column name in the header. 97,929 training rows; 24,483 test rows. Community
benchmark accuracy: 76–80%.

The proposal's hypothesis pairs KNN with NB as algorithms that will hit
**memory-bandwidth limits early** because their inner kernels are low
arithmetic-intensity. In practice KNN turned out to scale **best of the five**
at 8 threads, because at our problem size the triple-nested distance loop is
actually compute-bound — bandwidth would only bite at 32+ threads. Details in
[results/results.md §"Hypothesis vs reality"](../results/results.md).

**Key cross-framework finding.** Sklearn's `algorithm="brute"` KNN beats our
implementation by 16× at T=8 — but *not* because of better parallelism. Sklearn
reformulates the distance matrix as a single dense GEMM
(`‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·b`) and delegates to OpenBLAS/MKL. Both sides
scale ~6.2× from T=1 to T=8; sklearn just starts 16× ahead at T=1. This is the
paper's central "algorithm vs parallelism" narrative thread.

---

## 2. Algorithm: brute-force k-NN with squared-L2 distance

For each test sample `xq`:
1. Compute squared-L2 distance to every training sample
   `d²(xq, xi) = Σⱼ (xq[j] − xi[j])²`.
2. Track the k smallest distances via an O(k) linear-scan heap kept in
   `best_dist[0..k]` / `best_label[0..k]`. On each new candidate distance less
   than the current worst, replace that slot and rescan for the new max.
3. Majority-vote among the k labels; on tie, pick the label of the nearest
   neighbor.

Stopping conditions: none — single pass over the full training set per query.

### Why brute-force (no KD-tree / ball-tree)

At 103 features, KD-trees degenerate to near-linear scan (curse of
dimensionality). Ball-trees help for low-k on clustered data but add
construction time and complicate parallelization. For CS:GO at `n_train ≈ 98K`,
`n_test ≈ 24K`, `F = 103`, brute-force is simpler, parallelizes trivially
(every query is independent), and matches sklearn's `algorithm="brute"`
baseline exactly for an apples-to-apples comparison.

### Why squared-L2 (not plain L2)

Skipping the per-distance `sqrt()` saves `n_test × n_train = 2.4B` sqrt ops.
Rank-ordering by `d²` is identical to ordering by `d` (both are monotone), so
top-k selection is correct. Sklearn's `euclidean` metric does the same
optimization internally.

### Tie-breaking on majority vote

When `pos == neg` among the k neighbors (possible only for even k or odd k
with a ghost-tie from a numerical coincidence), the nearest single neighbor's
label wins. Deterministic and matches sklearn's behavior.

---

## 3. Hyperparameters

```
K_NEIGHBORS = 11     (default, overridable as argv[3])
N_THREADS   = 8      (default; overridable via N_THREADS env var — used by the
                     thread-sweep harness to run {1,2,4,8} with the same binary)
```

`K_NEIGHBORS = 11` is odd (avoids ties in the common case) and matches the
sklearn comparison config in [src/sklearn_xgb/compare.py](../src/sklearn_xgb/compare.py).
KNN has no learning rate, no regularization, no iteration — the CLI is
`./knn [train_csv] [test_csv] [k]`, deliberately simpler than MLP/SVM/DT.

`N_THREADS` is `static int` (not `constexpr`) so the environment variable can
override it at startup for the sweep. Threading during inference reads the
then-current value.

---

## 4. Data structures & memory layout

### `struct Dataset` (shared with SVM, MLP, DT, NB)

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

Identical to the four other algorithms — `analytics_engine.cpp` can share one
loader. `float` (not `double`) halves the memory footprint of `X_train`:
97,929 × 103 × 4 = **40 MB** fits in L3 on Cascade Lake (35 MB+ per socket).
`double` would be 80 MB and spill to DRAM every query. Accuracy is unaffected
— the downstream `d²` is accumulated in `double`.

### `struct KNNModel`

```cpp
struct KNNModel {
    std::vector<float> X;   // copy of X_train — see "Why a copy" below
    Labels y;                // copy of y_train
    int n_rows;
    int n_features;
    int k = K_NEIGHBORS;
};
```

KNN is a **lazy learner** — the "model" is just a reference to the training
data plus `k`. `train_serial` / `train_parallel` are trivial aliases that
populate this struct; **all actual work happens at inference**.

**Why a copy (not a pointer) into the model.** The three inference variants
(serial / OMP / pthreads) each get their own `KNNModel` in `main()`:
`serial_model`, `omp_model`, `pth_model`. Copies keep the three variants
independent — a future refactor could share one underlying pointer, but the
~40 MB × 3 = 120 MB memory footprint is negligible, and owning the data inside
the model cleans up the interface.

### Stack-local top-k heap (the hot data structure)

```cpp
std::vector<double> best_dist(k, +inf);
std::vector<int>    best_label(k, 0);
int    max_idx  = 0;
double max_dist = best_dist[0];
```

For each test query, allocate a tiny k-slot array (k=11, so ~176 B of data +
vector overhead). Track the current-worst distance (`max_dist` at `max_idx`).
On each training-row candidate `d² < max_dist`: overwrite the worst slot, then
scan `best_dist[0..k]` to find the new worst.

This is **O(k)-insert, O(1)-compare-to-worst**, and is faster than
`std::priority_queue` at k=11 because the O(log k) heap-update has higher
constant factor than a linear 11-element scan with auto-vectorization.
A full sort after the fact is also slower (O(n log k) vs O(n + n·P) for the
P=fraction-of-rows that pass the threshold, which is small after the first
few hundred candidates).

### Data-preprocessing skip list

Identical to SVM/MLP/DT/NB: `map_*`, `bomb_planted`, and auto-detected 0/1
columns bypass z-score normalization. On CS:GO expect
`Normalize: 94 z-scored, 9 passthrough` (bomb_planted + 8 map_* one-hots).

**Why this matters for KNN specifically.** If map one-hots were z-scored they
would contribute distance proportional to their class frequency (a rare map
would dominate nearest-neighbor distance). Leaving them as raw 0/1 makes their
contribution ≤ 1.0 per feature-pair, comparable to a z-scored continuous
feature at ~1σ. Cleaner physical interpretation of distance.

---

## 5. Training — three variants

KNN is a **lazy learner** — there is no training phase that touches the data
beyond copying it into the model. All three `train_*` functions are O(n_rows)
memcpy-equivalents that exist only to keep the API signature consistent across
the five algorithms.

### 5a. `train_serial` — copy

Populates the model's `X`, `y`, `n_rows`, `n_features`, `k`. Runs in ~13 ms
on CS:GO (mostly `std::vector::operator=`). This is the "serial baseline"
for speedup, but is essentially a constant — the actual speedup numerator is
**inference time**.

### 5b. `train_parallel` — same as train_serial

Literally identical code — `train_parallel` is retained for naming parity
with the other algorithms, so `analytics_engine.cpp` can time a uniform
"train then predict" block without special-casing KNN. **No OMP or pthreads
is used at train time** (would be a waste).

Consequence: **KNN's speedup is driven entirely by parallel inference.** The
serial model and the parallel model contain bit-identical data.

### 5c. No third trainer variant

Unlike SVM/MLP/DT/NB, which have separate `train_parallel_omp` and
`train_parallel_pthreads`, KNN has one `train_parallel` + two **inference**
variants (`predict_parallel_omp`, `predict_parallel_pthreads`). The speedup
story is moved to inference, where it actually happens.

---

## 6. Inference — three variants (this is the hot loop)

All three variants call the same `predict_one(model, xq)` kernel (single-query
top-k scan). They differ only in how `n_test = 24,483` queries are distributed
across threads.

### 6a. `predict_serial`

Trivial `for i in [0, n_test)` loop. Single baseline.

### 6b. `predict_parallel_omp`

```cpp
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
for (int i = 0; i < n_test; ++i)
    pred[i] = predict_one(model, &X_test[i * n_features]);
```

Static scheduling is correct because every query does the same amount of work
(one full sweep of the training set). Each thread holds its own stack-local
top-k heap — no shared mutable state, no critical section, no atomics.

### 6c. `predict_parallel_pthreads`

Fixed pool of `N_THREADS` workers, each takes a static slice
`[t · n_test / P, (t+1) · n_test / P)` of the test set. Worker just runs the
same `predict_one` over its slice. `pthread_join` at the end supplies the
only barrier — no mutex, no intermediate sync.

```cpp
struct KNNPredArg {
    const KNNModel* model;
    const std::vector<float>* X_test;
    Labels* pred;
    int s_off, e_off;
};
```

**Why no mutex?** Each `pred[i]` is written by exactly one thread (its slice).
Writes to disjoint indices in a `std::vector<int>` with `sizeof(int) = 4` on
consecutive slots could theoretically false-share on a 64-byte cache line at
slice boundaries, but since each thread writes its slice *sequentially*
(no random access), the cache-line-level pingpong is only at the boundary —
negligible over 24K+ writes per thread.

**Why pthreads when OMP is simpler?** Stylistic parity — the proposal says
each algorithm has OMP + pthreads. KNN's pthreads variant is the simplest
form of that pattern (no reduction, no barrier, no mutex), and exists as a
clean baseline for what OMP-vs-pthreads overhead looks like on an
embarrassingly-parallel workload.

### Cost model per query

`predict_one` touches:
- `n_rows × n_features × 4 B = 98K × 103 × 4 ≈ 40 MB` of X_train (reread once
  per query, streams through L3)
- `n_rows × 4 B = 400 KB` of y_train (stays in L2)
- per-candidate k-slot scan: at worst `n_rows` passes, each touching ~88 B of
  the top-k array (fits in L1)

Per-test-row: ~40 MB streamed + trivial compute. **At T=1 this is bandwidth-
bound.** At T=8, eight threads split the queries — *each* thread still
streams ~40 MB of X per query, but they're all reading the same X (shared
read-only) which the L3 serves once per cache line, not once per thread.
Net: compute and L3-bandwidth are both used efficiently, hence the 6.26×
observed speedup.

---

## 7. Arithmetic intensity (for the writeup)

**Per (test-sample, train-sample) pair in the inner distance loop:**

| Memory op                              | Bytes   |
|----------------------------------------|--------:|
| Load `xi[0..F]`  (F=103 floats)        | 412     |
| Load `xq[0..F]`  (amortized; in L1)    | ≈0      |
| Load `y[i]`     (1 byte, amortized)    | ≈0      |

Ops per (test, train) pair: `F` subtracts + `F` multiplies + `F` adds = 3F
≈ **309 FLOPs** per 412 bytes.

Effective arithmetic intensity: **~0.75 FLOP/byte.** **Memory-bound** in the
same class as SVM and NB.

**Why we scaled despite low AI.** The 0.75 FLOP/byte number is the *per-pair*
intensity. What actually matters is the **DRAM** intensity, because the train
matrix is read once per query and **shared across threads**. At T=8 with 24K
queries split 3K each, each thread streams 40 MB from L3 (not DRAM, since the
second thread's read hits the cache lines loaded by the first). DRAM bandwidth
divided by L3 bandwidth ≈ 4×, so effective intensity climbs 4× under sharing,
leaving ~3 FLOP/byte in DRAM terms — comfortably in the compute-bound regime
at 8 cores.

This is why the proposal's "KNN is memory-bound, will plateau early" was
wrong: it measured the per-pair AI without accounting for inter-thread L3
sharing. Would likely bite at T=32+ when L3 bandwidth becomes the bottleneck.

Comparison across all five algorithms:

| Algorithm | Inner kernel                  | AI (FLOP/byte) | Expected scaling   | Observed T=8 (pth) |
|-----------|-------------------------------|---------------:|--------------------|-------------------:|
| SVM       | `dot(w, x)` + margin check    | ~0.5           | Memory-bound       | 6.02×              |
| KNN       | squared-L2 distance           | ~0.75          | Memory-bound       | **6.26×**          |
| NB        | frequency count + Gaussian MLE| ~0.75          | Memory-bound       | 3.56×              |
| DT        | histogram fill + Gini sweep   | ~3             | Crossover regime   | 1.87×              |
| MLP       | dense GEMM forward/back       | ~42            | Compute-bound      | 4.62×              |

Low-AI algorithms (SVM/KNN/NB) *did* cluster together in the hypothesis but
KNN landed at the **top** of the speedup ranking, not the bottom. The
difference vs. SVM and NB: KNN has the highest absolute FLOP count per query
(~3 G FLOPs × 24K queries ≈ 75 TFLOPs), so thread startup and reduction
overhead are a negligible fraction of runtime. SVM and NB each have ~10 ms
total serial time, so *any* per-epoch/per-worker overhead is visible at T=8.

---

## 8. Numerical & correctness risks

1. **`double` accumulation of `d²`.** `squared_l2` accumulates into `double`
   even though `xq`, `xi` are `float` — prevents catastrophic cancellation
   at 103-term dot products. Changing the accumulator to `float` would break
   tie-breaking ordering on near-equal distances.
2. **Unstable majority-vote on even k.** `K_NEIGHBORS = 11` is odd, so
   strict majority vote always breaks ties. If a future run uses even k, the
   current code falls back to the nearest-neighbor's label on `pos == neg`
   — still deterministic. Documented.
3. **Non-determinism from parallel inference.** Each query is independent,
   so OMP and pthreads produce byte-identical predictions to serial — the
   top-k heap fills deterministically given a fixed scan order (single pass
   over `i = 0..n_rows`). Parity check prints `|acc_serial − acc_par| =
   0.0000` in all runs.
4. **No training-set sharing between variants.** `serial_model`,
   `omp_model`, `pth_model` each copy the full 40 MB `X_train`. Cheap
   compared to inference runtime, but a memory-pressure issue if a future
   refactor adds more variants. Fix is trivially to share via
   `std::shared_ptr<KNNModel>`.
5. **Thread count at runtime, not compile time.** `N_THREADS` is `static int`
   and read from the env var at startup. A subsequent call to the inference
   function uses the current value. If `N_THREADS = 0` gets set (bad env
   var), `predict_parallel_pthreads` divides by zero on the slice
   computation — protected by the `std::atoi` floor of 1.
6. **Speedup numerator is inference-only.** The sweep CSV uses
   `total_ms = train_ms + infer_ms` to compare against the other
   algorithms; train_ms is ~13 ms and infer_ms is ~323 s, so this is
   essentially inference speedup. Noted in the paper.

---

## 9. Build & run

On USC CARC (from src/cpp/knn/):

```bash
cd src/cpp/knn && sbatch job_knn.sl
# outputs to knnjob.out
# compiles knn.cpp, runs ./knn ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
```

CLI:

```bash
./knn [train_csv] [test_csv] [k]
# defaults: data/train_cleaned.csv  data/test_cleaned.csv  11
# env var N_THREADS overrides the default 8 (used by the sweep harness)
```

Locally (macOS), from the repo root:

```bash
# Option A: Homebrew libomp with Apple Clang
brew install libomp
clang++ -std=c++17 -O3 -march=native \
  -Xpreprocessor -fopenmp -lomp src/cpp/knn/knn.cpp -o knn -lpthread
./knn data/train_cleaned.csv data/test_cleaned.csv

# Option B: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp src/cpp/knn/knn.cpp -o knn -lpthread
```

**Apple Clang alone will not work** — no OpenMP runtime. `pthread_barrier_t`
isn't used by KNN itself (no sync needed) but the build flags are shared
across the five algorithms.

### Verification expected

- **Accuracy**: `acc ≈ 0.81` (KNN is the highest-accuracy of the five on
  CS:GO). Target range 0.78–0.82.
- **Parity**: `|acc_serial − acc_omp| = 0.0000` and
  `|acc_serial − acc_pth|  = 0.0000` — KNN is deterministic at every
  scheduling order because each query independently fills a fixed-order
  top-k heap.
- **Speedup**: inference-only. Target 5–7× at 8 cores.

---

## 10. Results log

Append one row per CARC run.

| date (UTC) | git SHA | k | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:-:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| 2026-04-20 | 526bb18 | 11 | CARC d17-03 / 8 | 323.02 | 50.52 | 51.55 | 0.8104 | 0.8104 | 0.8104 | 6.39× | 6.26× | Full `{1,2,4,8}` sweep, job 3272373. **Highest speedup of the five** — proposal's "memory-bound, plateaus early" hypothesis was wrong at T=8. Accuracy is the highest of the five. vs sklearn brute (T=8) **ours is 16× slower** because sklearn reformulates the distance matrix as a GEMM (`X_test @ X_train.T`) through OpenBLAS; at T=1 sklearn already beats us 16.4×. Scaling is identical on both sides (~6.2× per thread octave). See [results/results.md §"Key insight"](../results/results.md). |

---

## 11. Followups (known work deferred)

- **Deduplicate loader/metrics** into `common.hpp` once
  `analytics_engine.cpp` is being written. The loader is byte-identical to
  SVM/MLP/DT/NB; extraction is mechanical.
- **Scale past 8 threads** — `shared` partition caps at 20 cores/node; `nlp`
  has 96–128. Running T ∈ {16, 32, 64} would confirm whether memory
  bandwidth bites at 16 or 32 threads (predicted: 15–25× at T=64, not
  linear 50×). Publish the full Amdahl/Gustafson curve.
- **GEMM-based KNN** — close the 16× gap vs sklearn by implementing a
  cache-tiled `X_test @ X_train.T` by hand. Explicitly *against the spirit*
  of the no-external-libraries rule, but the data point would strengthen
  the "algorithmic reformulation dominates parallelization" argument.
  Deferred unless the paper needs a counterfactual.
- **Partial sort / approximate KNN** — for k=11 out of n_rows=98K, a
  priority-queue-based selection (Dutch-national-flag / introselect) would
  shave the per-query constant but doesn't change complexity.
  Low-priority.
- **`K_NEIGHBORS` sweep** — k ∈ {1, 3, 5, 11, 25, 51} would show the
  accuracy vs. runtime trade-off. Cheap to run (inference-only). Adds a
  figure to the writeup if needed.
- **Thread-count sweep on CARC's `nlp` partition** — already scheduled as
  followup in the results doc. KNN is the natural candidate because it's
  the best-scaling algorithm and bandwidth-limited behavior would be
  easiest to see at T=32+.
