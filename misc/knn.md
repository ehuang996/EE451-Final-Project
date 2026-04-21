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

**Key cross-framework finding.** Sklearn's `algorithm="brute"` KNN beat the
original pure C++ implementation by 16× at T=8 — but *not* because sklearn was
serial and we were parallel. Our comparison harness passes `n_jobs=N_THREADS`
to `KNeighborsClassifier`, and sklearn's brute Euclidean path uses optimized
pairwise-distance reductions before falling back to chunked `n_jobs` work when
needed. The bigger difference was kernel formulation: sklearn evaluates the
distance matrix through a BLAS-style dot product
(`‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·b`) instead of a scalar `(xq[j] - xi[j])²`
inner loop.

**Implementation update (2026-04-21 / 2026-04-21 follow-on cleanup).** The
current tree no longer ships the old pure C++ KNN path. KNN is now BLAS-only:
precompute `‖xi‖²`, process query blocks, and evaluate distances as
`‖xq‖² + ‖xi‖² − 2·xq·xi` via CBLAS `sgemm`. Historical pure-C++ benchmark rows
remain below for comparison, but the supported implementation on CARC is the
vendor-BLAS backend. With Apple Accelerate capped to one internal BLAS thread
and our pthreads owning the outer query parallelism, local full-split
inference improved to 0.592 s vs. 1.586 s for the matching local sklearn brute
KNN run.

---

## 2. Algorithm: brute-force k-NN with squared-L2 distance

For each test sample `xq`:
1. Compute squared-L2 distance to every training sample using the equivalent
   dot-product identity `d²(xq, xi) = ‖xq‖² + ‖xi‖² − 2·xq·xi`.
2. Process `KNN_BLAS_BLOCK` queries at a time and use CBLAS `sgemm` for
   `X_test_block @ X_train.T`.
3. Track the k smallest distances via an O(k) linear-scan heap kept in
   `best_dist[0..k]` / `best_label[0..k]`. On each new candidate distance less
   than the current worst, replace that slot and rescan for the new max.
4. Majority-vote among the k labels; on tie, pick the label of the nearest
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

### Why the dot-product identity

The original kernel computed `(xq[j] - xi[j])²` directly inside a
query-major triple loop. That is simple and exact, but it reloads each training
row separately for every test query. The current kernel precomputes training
row norms in the model and test row norms per prediction call, transposes a
small block of 16 test queries into `qbuf`, then streams each training row once
across all queries in the block:

```cpp
dist[q] = train_norm[i] + test_norm[q];
for (int j = 0; j < n_features; ++j)
    dist[q] += -2.0f * xi[j] * qbuf[j][q];
```

This is the same formulation sklearn uses before dispatching to optimized
pairwise-distance kernels. The default backend keeps it as a hand-written C++
loop. When compiled with `KNN_USE_BLAS`, the implementation calls vendor CBLAS
`sgemm` for the dot matrix and keeps the top-k scan, voting, and outer
query-block scheduling in our code. That path uses OpenBLAS/MKL/Accelerate's
register blocking and assembly microkernels without giving up control over the
outer pthread/OpenMP schedule.

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
KNN_BLAS_BLOCK = 256 (default BLAS query block)
```

`K_NEIGHBORS = 11` is odd (avoids ties in the common case) and matches the
sklearn comparison config in [src/sklearn_xgb/compare.py](../src/sklearn_xgb/compare.py).
KNN has no learning rate, no regularization, no iteration — the CLI is
`./knn [train_csv] [test_csv] [k]`, deliberately simpler than MLP/SVM/DT.

`N_THREADS` is `static int` (not `constexpr`) so the environment variable can
override it at startup for the sweep. Threading during inference reads the
then-current value.

For outer-parallel BLAS runs, cap the vendor library to one internal thread
(`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, or `VECLIB_MAXIMUM_THREADS=1`).
Otherwise each of our worker threads can spawn its own BLAS thread team and
oversubscribe the node.

### Backend selection model

The current source tree supports one exact KNN backend:

| Build mode | Runtime backend | Inner kernel | Dependency | Intended use |
|------------|-----------------|--------------|------------|--------------|
| `-DKNN_USE_BLAS` | `blas-sgemm` | CBLAS `sgemm` for dot-product panels | OpenBLAS, Accelerate, or generic CBLAS | default build |
| `-DKNN_USE_MKL` | `blas-sgemm` | CBLAS `sgemm` via oneMKL | Intel oneMKL | cluster/vendor-BLAS run |

Compile-time macros only decide whether the binary has access to CBLAS:

```cpp
#if defined(KNN_USE_MKL)
    #include <mkl.h>
#elif defined(KNN_USE_ACCELERATE) || (defined(KNN_USE_BLAS) && defined(__APPLE__))
    #include <Accelerate/Accelerate.h>
#elif defined(KNN_USE_BLAS) || defined(KNN_USE_OPENBLAS) || defined(KNN_USE_CBLAS)
    #include <cblas.h>
#endif
```

There is no runtime fallback to the removed blocked-dot path in the current
tree. Historical notes below still reference it because the earlier benchmark
rows were collected before this cleanup.

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
`double` would be 80 MB and spill to DRAM every query. The current blocked
dot-product kernel keeps distances in `float`, matching the storage type and
the BLAS-style formulation we are trying to approximate.

### `struct KNNModel`

```cpp
struct KNNModel {
    std::vector<float> X;       // copy of X_train — see "Why a copy" below
    std::vector<float> norm2;   // precomputed ‖xi‖² for every training row
    Labels y;                   // copy of y_train
    int n_rows;
    int n_features;
    int k = K_NEIGHBORS;
};
```

KNN is still a **lazy learner** in the ML sense: no parameters are fitted.
The model stores a copy of the training matrix, labels, `k`, and one additional
row-norm vector used by the blocked dot-product inference kernel.

**Why a copy (not a pointer) into the model.** The three inference variants
(serial / OMP / pthreads) each get their own `KNNModel` in `main()`:
`serial_model`, `omp_model`, `pth_model`. Copies keep the three variants
independent — a future refactor could share one underlying pointer, but the
~40 MB × 3 = 120 MB memory footprint is negligible, and owning the data inside
the model cleans up the interface.

### Block-local top-k heap (the hot data structure)

```cpp
std::vector<float> best_dist(QUERY_BLOCK * k, +inf);
std::vector<int>    best_label(QUERY_BLOCK * k, 0);
std::vector<int>    max_idx(QUERY_BLOCK, 0);
std::vector<float>  max_dist(QUERY_BLOCK, +inf);
```

For each block of up to 16 test queries, allocate one contiguous k-slot array
per query. Track each query's current-worst distance (`max_dist[q]` at
`max_idx[q]`). On each training-row candidate `d² < max_dist[q]`: overwrite
the worst slot, then scan that query's `best_dist[q*k..q*k+k]` to find the new
worst.

This is **O(k)-insert, O(1)-compare-to-worst**, and is faster than
`std::priority_queue` at k=11 because the O(log k) heap-update has higher
constant factor than a linear 11-element scan with auto-vectorization.
A full sort after the fact is also slower (O(n log k) vs O(n + n·P) for the
P=fraction-of-rows that pass the threshold, which is small after the first
few hundred candidates).

### Data-preprocessing skip list

Identical to SVM/MLP/DT/NB: `map_*`, `bomb_planted`, and auto-detected 0/1
columns bypass z-score normalization. On CS:GO expect
roughly 72 z-scored columns and 31 passthrough binary/one-hot columns.

**Why this matters for KNN specifically.** If map one-hots were z-scored they
would contribute distance proportional to their class frequency (a rare map
would dominate nearest-neighbor distance). Leaving them as raw 0/1 makes their
contribution ≤ 1.0 per feature-pair, comparable to a z-scored continuous
feature at ~1σ. Cleaner physical interpretation of distance.

---

## 5. Training — three variants

KNN is a **lazy learner** — there is no fitted weight vector or tree. Training
copies data into the model and precomputes one `float` row norm per training
sample. All `train_*` functions remain O(n_rows · n_features) setup work and
exist mostly to keep the API signature consistent across the five algorithms.

### 5a. `train_serial` — copy

Populates the model's `X`, `norm2`, `y`, `n_rows`, `n_features`, `k`. The row
norm pass is small relative to inference, but it is now part of `train_ms` so
the benchmark accounts for the optimization's setup cost.

### 5b. `train_parallel` — same as train_serial

Same setup code as `train_serial` — `train_parallel` is retained for naming
parity with the other algorithms, so `analytics_engine.cpp` can time a uniform
"train then predict" block without special-casing KNN. **No OMP or pthreads
is used at train time** (would be a waste).

Consequence: **KNN's speedup is driven almost entirely by parallel inference.**
The serial model and the parallel model contain bit-identical data and norms.

### 5c. No third trainer variant

Unlike SVM/MLP/DT/NB, which have separate `train_parallel_omp` and
`train_parallel_pthreads`, KNN has one `train_parallel` + two **inference**
variants (`predict_parallel_omp`, `predict_parallel_pthreads`). The speedup
story is moved to inference, where it actually happens.

---

## 6. Inference — three variants (this is the hot loop)

All three variants call the same BLAS-backed SGEMM path. They differ only in
how query blocks are distributed across threads.

### 6a. `predict_serial`

Computes `‖xq‖²` for the test matrix, then runs `predict_blas_range` over the
full test range in one thread. This is the serial baseline.

### 6b. `predict_parallel_omp`

```cpp
#pragma omp parallel num_threads(N_THREADS)
{
    #pragma omp for schedule(static)
    for (int b = 0; b < n_blocks; ++b)
        predict_blas_block(... block b ...);
}
```

Static scheduling is correct because every block does the same training-set
sweep. Each thread holds its own block-local buffers (`qbuf`, `dist`,
`best_dist`, `best_label`, `max_idx`, `max_dist`) — no shared mutable state,
no critical section, no atomics.

### 6c. `predict_parallel_pthreads`

Fixed pool of `N_THREADS` workers, each takes a static slice
`[t · n_test / P, (t+1) · n_test / P)` of the test set. Worker just runs the
same blocked predictor over its slice. `pthread_join` at the end supplies the
only barrier — no mutex, no intermediate sync.

```cpp
struct KNNPredArg {
    const KNNModel* model;
    const std::vector<float>* X_test;
    const std::vector<float>* test_norm2;
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

### 6d. Optional BLAS backend

The BLAS backend is still exact brute-force KNN. It does **not** change the
model, the distance metric, the k-neighbor voting rule, or the output labels.
It only replaces the hottest arithmetic subproblem:

```text
for each query q:
    for each train row i:
        dot[q, i] = sum_j X_test[q, j] * X_train[i, j]
```

with one dense matrix multiply over a query panel:

```text
dots = X_test[q_begin:q_end, :] @ X_train.T
```

Then the code converts those dot products into squared distances and runs the
same k-selection/voting logic as the pure C++ backend.

#### Dataflow

Training stores the normalized training matrix row-major in `model.X` and
precomputes one squared norm per training row in `model.norm2`:

```text
model.X      shape: n_train × n_features
model.norm2  shape: n_train
model.y      shape: n_train
```

Prediction first computes `test_norm2` for all test rows. The BLAS path then
walks the query range in panels of `KNN_BLAS_BLOCK` rows. For each panel:

1. Call `sgemm` to compute all query/train dot products for the panel.
2. For each query row in the panel, scan the corresponding `dots[q, :]` row.
3. Convert dot products to squared distances:
   `d² = test_norm2[q] + model.norm2[i] - 2 * dots[q, i]`.
4. Maintain the k best candidates with the existing O(k) replacement scan.
5. Finalize the majority vote for each query and write directly into `pred`.

No full `n_test × n_train` distance matrix is materialized. Only one panel of
dot products exists per worker.

#### SGEMM layout

The C++ arrays are row-major, so the call is:

```cpp
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            q_count, n_train, n_features,
            1.0f, X_test_block, n_features,
            X_train, n_features,
            0.0f, dots, n_train);
```

That means:

| SGEMM operand | Logical matrix | Shape | Transpose flag | Leading dimension |
|---------------|----------------|------:|----------------|------------------:|
| A | `X_test_block` | `q_count × n_features` | `NoTrans` | `n_features` |
| B | `model.X` | `n_train × n_features` | `Trans` | `n_features` |
| C | `dots` | `q_count × n_train` | output | `n_train` |

The output row `dots[q, :]` contains all training-set dot products for one test
query, which makes the subsequent top-k scan linear and cache-friendly.

#### What stays outside BLAS

BLAS only handles the dot-product panel. The following work remains in our C++
implementation:

- CSV loading, feature normalization, and binary/one-hot skip logic.
- Training-set copy and `norm2` precomputation.
- Test-row norm computation.
- Conversion from dot product to squared-L2 distance.
- Top-k maintenance for `k=11`.
- Majority vote and nearest-neighbor tie break.
- Serial, OpenMP, and pthread query scheduling.

This split is intentional. BLAS is excellent at dense multiply-add panels, but
not at irregular top-k selection with labels. Keeping top-k outside BLAS avoids
materializing or sorting a giant full distance matrix.

#### Threading strategy

There are two possible places to parallelize a BLAS-backed KNN:

1. Inside BLAS, where `sgemm` itself uses multiple threads.
2. Outside BLAS, where our KNN code splits query ranges across workers and each
   worker calls single-threaded `sgemm`.

The implemented benchmark path uses option 2. For pthreads, each worker gets a
static slice of test queries, allocates its own dot buffer, and repeatedly calls
`predict_blas_range` on that slice. The BLAS library is pinned to one internal
thread:

```bash
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

This prevents nested oversubscription. Without those caps, an 8-thread KNN run
could become 8 outer workers × 8 BLAS workers = 64 runnable compute threads,
which usually loses to scheduler overhead and cache pressure.

#### Why this can beat sklearn locally

Sklearn is not serial in our comparison. The Python harness passes
`n_jobs=N_THREADS`, and sklearn's brute Euclidean path uses optimized compiled
distance reductions. The reason the new path can still win locally is that it
is narrower and more explicit for this exact benchmark:

- The inner dot-product kernel is now vendor SGEMM, so the old hand-loop
  disadvantage is gone.
- The outer query partitioning is fixed and simple: one pthread worker owns a
  contiguous query slice and writes disjoint prediction indices.
- BLAS internal threading is pinned to 1, so the only parallel layer is the one
  we control.
- The code immediately reduces each dot panel into k nearest labels instead of
  routing through a more general estimator API.

The correct claim is therefore not "raw C++ loops beat sklearn." The accurate
claim is: **pure C++ remains the from-scratch baseline; the optional BLAS path
uses the same vendor-grade inner-kernel idea as sklearn while keeping a lean,
explicit outer schedule for this workload.**

#### Memory and block-size tradeoff

`KNN_BLAS_BLOCK` controls the number of test queries per SGEMM panel. The dot
buffer size per worker is:

```text
KNN_BLAS_BLOCK × n_train × sizeof(float)
```

At the default block size:

```text
256 × 97,929 × 4 B ≈ 96 MB per worker
```

With 8 pthread workers, the transient dot buffers are roughly 770 MB total.
Larger blocks usually improve SGEMM efficiency because the matrix multiply has
more work per call, but they also increase memory pressure. Smaller blocks
reduce memory footprint but may underfeed the vendor kernel. The current
default is a pragmatic local choice, not a universal optimum; CARC/OpenBLAS or
MKL should be swept before claiming a cluster-wide result.

### Cost model per query

For each pure-C++ query block, `predict_query_block` touches:
- `n_rows × n_features × 4 B = 98K × 103 × 4 ≈ 40 MB` of `X_train`, streamed
  once and reused across up to 16 test queries.
- `QUERY_BLOCK × n_features × 4 B ≈ 6.6 KB` of transposed query data in `qbuf`
  (fits in L1).
- `n_rows × 4 B = 400 KB` of `y_train`, plus `n_rows × 4 B` of precomputed
  train norms.
- per-candidate k-slot scans only when a candidate beats the current worst
  distance; the k arrays fit in L1.

Compared with the old query-major loop, the current block-major loop trades a
little more per-model setup (`norm2`) for much better training-row reuse. The
local pthread targeted benchmark improved from 24.35 s to 7.83 s on the full
split. The BLAS backend adds a `KNN_BLAS_BLOCK × n_train` dot buffer per worker
(default `256 × 97,929 × 4 B ≈ 96 MB`), so an 8-thread run uses roughly
770 MB of transient dot buffers. At T=8, the high-level scaling story still
applies: threads split query blocks, share read-only training data through L3,
and synchronize only at the final join/barrier.

---

## 7. Arithmetic intensity (for the writeup)

**Per (test-sample, train-sample) pair in the current dot-product inner loop:**

| Memory op                              | Bytes   |
|----------------------------------------|--------:|
| Load `xi[0..F]`  (F=103 floats, reused across query block) | 412 |
| Load `qbuf[0..F]` (amortized; in L1)                       | ≈0  |
| Load norms / label (amortized)                             | ≈0  |

Ops per (test, train) pair are approximately `F` multiply-adds for `-2·xq·xi`
plus two scalar norm operations, roughly **2F ≈ 206 FLOPs** per 412 training
bytes if counted as separate multiply and add operations.

Effective per-pair arithmetic intensity is therefore **~0.5 FLOP/byte** before
accounting for query-block reuse and L3 sharing. It remains in the same
low-AI class as SVM and NB, but the blocked formulation substantially improves
constant factors.

The BLAS backend does not change the exact algorithm or big-O cost. It improves
the constant by handing the dot-product panel to a tuned SGEMM microkernel, then
returns to our scalar top-k reduction.

**Why we scaled despite low AI.** The per-pair AI number is not the full
story. What actually matters is the **DRAM** intensity, because the train
matrix is read repeatedly and **shared across threads**. At T=8 with 24K
queries split 3K each, each thread streams training rows mostly from L3 (not
DRAM, since nearby reads hit cache lines loaded by other workers). DRAM
bandwidth divided by L3 bandwidth is roughly 4×, so effective DRAM-side
intensity is several times higher than the naive per-pair estimate.

This is why the proposal's "KNN is memory-bound, will plateau early" was
wrong on the original CARC sweep: it measured the per-pair AI without
accounting for inter-thread L3 sharing and low synchronization frequency.
Memory bandwidth would likely bite at T=32+ when L3 bandwidth becomes the
bottleneck.

Comparison across all five algorithms:

| Algorithm | Inner kernel                  | AI (FLOP/byte) | Expected scaling   | Observed T=8 (pth) |
|-----------|-------------------------------|---------------:|--------------------|-------------------:|
| SVM       | `dot(w, x)` + margin check    | ~0.5           | Memory-bound       | 6.02×              |
| KNN       | blocked squared-L2 / dot product | ~0.5–0.75   | Memory-bound       | **6.26×**          |
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

1. **Float dot-product distance ordering.** Both active kernels compute
   `‖xq‖² + ‖xi‖² − 2·xq·xi` in `float`. This is the right formulation for
   BLAS, but it can perturb near-tie distances compared with the old direct
   `double` squared-L2 loop. The observed accuracy remains in the same band
   (`0.8101` to `0.8104` locally), and serial/OMP/pthreads parity is still
   enforced within each binary run.
2. **Unstable majority-vote on even k.** `K_NEIGHBORS = 11` is odd, so
   strict majority vote always breaks ties. If a future run uses even k, the
   current code falls back to the nearest-neighbor's label on `pos == neg`
   — still deterministic. Documented.
3. **Non-determinism from parallel inference.** Each query block is independent,
   so OMP and pthreads produce the same predictions as serial for a fixed
   kernel and scan order. Parity check prints `|acc_serial − acc_par| =
   0.0000` in smoke tests.
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
   algorithms; train_ms is still tiny compared with inference, so this is
   essentially inference speedup. Noted in the paper.
7. **Nested BLAS threads.** If a BLAS-compiled binary runs with
   `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, or `VECLIB_MAXIMUM_THREADS`
   greater than 1, every outer KNN worker can create its own inner BLAS team.
   The documented benchmark path pins BLAS to one internal thread and lets our
   OpenMP/pthread layer own query-level parallelism.

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
# env var KNN_BLAS_BLOCK overrides the default 256-query SGEMM panel
```

Locally (macOS), from the repo root:

```bash
# Option A: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp -DKNN_USE_BLAS \
  src/cpp/knn/knn.cpp -o knn -lpthread -lopenblas

# Option B: macOS Accelerate BLAS backend
clang++ -std=c++17 -O3 -DKNN_USE_BLAS \
  src/cpp/knn/knn.cpp -o knn -lpthread -framework Accelerate
VECLIB_MAXIMUM_THREADS=1 N_THREADS=8 ./knn data/train_cleaned.csv data/test_cleaned.csv

# Option C: Linux/OpenBLAS backend
g++ -std=c++17 -O3 -march=native -fopenmp -DKNN_USE_BLAS \
  src/cpp/knn/knn.cpp -o knn -lpthread -lopenblas
OPENBLAS_NUM_THREADS=1 N_THREADS=8 ./knn data/train_cleaned.csv data/test_cleaned.csv
```

For Intel oneMKL, compile with `-DKNN_USE_MKL` and the oneMKL link line for the
active compiler/runtime. The source includes `mkl.h` under that macro and uses
the same CBLAS `sgemm` call.

Apple Clang without `libomp` still builds the serial and pthread paths, but the
OpenMP predictor falls back to the serial range path because `_OPENMP` is not
defined. Use Homebrew `libomp` when the OMP number matters.

### Verification expected

- **Accuracy**: `acc ≈ 0.81` (KNN is the highest-accuracy of the five on
  CS:GO). Target range 0.78–0.82.
- **Parity**: `|acc_serial − acc_omp| = 0.0000` and
  `|acc_serial − acc_pth|  = 0.0000` — KNN is deterministic at every
  scheduling order because each query block scans training rows in a fixed
  order and writes a disjoint output slice.
- **Speedup**: inference-only. Target 5–7× at 8 cores.
- **BLAS counterfactual**: with BLAS internal threads pinned to 1, the pthread
  path should be substantially faster than the blocked-dot backend. Local
  Accelerate timing is 0.592 s for pthread inference on the full test split.

---

## 10. Results log

Append one row per CARC run.

| date (UTC) | git SHA | k | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:-:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| 2026-04-20 | 526bb18 | 11 | CARC d17-03 / 8 | 323.02 | 50.52 | 51.55 | 0.8104 | 0.8104 | 0.8104 | 6.39× | 6.26× | Full `{1,2,4,8}` sweep, job 3272373. **Highest speedup of the five** — proposal's "memory-bound, plateaus early" hypothesis was wrong at T=8. Accuracy is the highest of the five. vs sklearn brute (T=8) **ours is 16× slower** because sklearn reformulates the distance matrix as a GEMM (`X_test @ X_train.T`) through OpenBLAS; at T=1 sklearn already beats us 16.4×. Scaling is identical on both sides (~6.2× per thread octave). See [results/results.md §"Key insight"](../results/results.md). |

### Local optimization checks (not CARC-comparable)

| date | implementation | machine | infer (s) | accuracy | notes |
|------|----------------|---------|----------:|:--------:|-------|
| 2026-04-21 | old direct loop | Eric's MacBook Pro / 8 pthreads | 24.35 | 0.8104 | Targeted harness calling only `predict_parallel_pthreads`; Apple Clang/Xcode SDK build. |
| 2026-04-21 | blocked dot-product loop | Eric's MacBook Pro / 8 pthreads | 7.83 | 0.8104 | Pure C++ backend; ~3.1× faster than old direct loop locally, still behind local sklearn brute (~1.64 s). |
| 2026-04-21 | committed BLAS SGEMM backend | Eric's MacBook Pro / 8 pthreads + Accelerate internal threads pinned to 1 | 0.592 | 0.8101 | `KNN_USE_BLAS`, `backend=blas-sgemm`, `KNN_BLAS_BLOCK=256`; serial/OMP/pthreads accuracy parity all 0.0000. |
| 2026-04-21 | sklearn brute reference | Eric's MacBook Pro / `n_jobs=8`, Accelerate internal threads pinned to 1 | 1.586 | 0.8106 | Same local data loader and `KNeighborsClassifier(algorithm="brute", metric="euclidean")`; included only as a same-machine reference. |

---

## 11. Followups (known work deferred)

- **Deduplicate loader/metrics** into `common.hpp` once
  `analytics_engine.cpp` is being written. The loader is byte-identical to
  SVM/MLP/DT/NB; extraction is mechanical.
- **Scale past 8 threads** — `shared` partition caps at 20 cores/node; `nlp`
  has 96–128. Running T ∈ {16, 32, 64} would confirm whether memory
  bandwidth bites at 16 or 32 threads (predicted: 15–25× at T=64, not
  linear 50×). Publish the full Amdahl/Gustafson curve.
- **CARC BLAS sweep** — the portable `KNN_USE_BLAS` path is now the supported
  implementation, but the reported paper row should only change after a CARC
  OpenBLAS/MKL sweep. Use `sbatch slurm/job_sweep.sl`, with BLAS internal
  threads pinned to 1, to measure whether the local Accelerate win holds on the
  cluster.
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
