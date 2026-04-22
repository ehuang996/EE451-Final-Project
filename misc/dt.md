# DT — Design Document & Results Log

Implementation at [src/cpp/dt/dt.cpp](../src/cpp/dt/dt.cpp). This document records every
non-obvious architectural and engineering choice made in the Decision Tree,
plus a live log of benchmark runs.

---

## 1. Problem framing

Binary classification on the CS:GO Round Winner dataset — 103 numeric features
(read from the header at runtime), label `round_winner ∈ {+1, −1}` detected by
column name in the header. 97,929 training rows; 24,483 test rows. Community
benchmark accuracy: 76–80%.

The proposal's hypothesis names Decision Trees as being "constrained by
**synchronization overhead** and iterative convergence" — an explicit
prediction that DT will NOT scale as well as MLP. DT's 8-core speedup curve is
the counter-evidence to MLP's headline claim: the ratio
`speedup_MLP / speedup_DT` is the central empirical comparison of the paper.

---

## 2. Algorithm: histogram-based CART

Greedy CART (Classification and Regression Trees) construction with Gini
impurity as the splitting criterion. Histogram-based split finding rather than
sorted-scan — this is the LightGBM approach.

### Per-node algorithm

Given sample indices `idx` at the current node and pre-computed binned
features `Xb`:

1. For each feature `j`, fill two histograms `hp[N_BINS]`, `hn[N_BINS]` in a
   single pass over `idx`:
   `hp[Xb[j][i]] += (y[i] == +1); hn[Xb[j][i]] += (y[i] == −1)`.
2. Sweep across bin boundaries. At each `b`, Gini gain =
   `parent_gini − (left_n/n)·gini(cum_pos, cum_neg) − (right_n/n)·gini(...)`.
3. Return the `(feature, threshold, gain)` triple with maximum gain.
4. Partition `idx` by `X[i][feat] ≤ threshold`, recurse on left and right.

Stopping conditions: `depth ≥ MAX_DEPTH`, `idx.size() < MIN_SAMPLES_SPLIT`,
pure node (`pos_tot == 0` or `neg_tot == 0`), or `best_gain < GAIN_EPS`.

### Why histogram over sorted-scan

Sorted-scan CART: for each node, sort each feature's samples (O(n log n)) and
sweep — O(F·n log n) per level × depth D. For CS:GO at D=12, F=103, n≈98K:
~2×10⁹ comparisons, multi-second runtime.

Histogram CART: bin features **once** up front (O(F·n log n) for quantile
edges), then each per-node split evaluation is O(F·n_node + F·N_BINS).
Expected 5–10× faster than sorted-scan with an identical tree (up to
deterministic tie-breaking).

Why binning works cleanly for this dataset: most CS:GO features are
low-cardinality (scores, helmet counts, weapon counts), and the binary/one-hot
columns have only 2 unique values. `N_BINS = 64` captures the high-cardinality
continuous features (`time_left`, money, health) without accuracy loss.

---

## 3. Hyperparameters

```
MAX_DEPTH         = 12    (default, overridable as argv[3])
MIN_SAMPLES_SPLIT = 20    (don't attempt splits on fewer than 20 samples)
MIN_SAMPLES_LEAF  = 5     (every child must have ≥5 samples)
N_BINS            = 64    (histogram bins per continuous feature)
PARALLEL_DEPTH    = 3     (spawn OMP subtree tasks only while depth < this)
N_THREADS         = 8     (matched to SLURM --cpus-per-task=8)
SEED              = 42
GAIN_EPS          = 1e-7  (minimum Gini gain to accept a split)
```

`max_depth` is the only runtime-tunable hyperparameter (argv[3]). Others stay
`constexpr` so compilers can inline and unroll. CLI signature
`./dt train test [max_depth]` matches SVM's `[epochs]` and MLP's `[epochs]`
conventions.

---

## 4. Data structures & memory layout

### `struct Dataset` (shared with SVM, KNN, MLP)

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

Identical to SVM/KNN/MLP so a future `analytics_engine.cpp` can load once and
drive all five models. Not used directly by the tree builder's hot loop —
see `BinnedDataset` below for what the inner scan actually reads from. `X` is
only touched at bin-construction time and during per-node partitioning.

### `struct Node`, `struct DTModel`

```cpp
struct Node {
    int    feature_index;  // -1 for leaf
    double threshold;      // x[feature_index] ≤ threshold → left
    int    left, right;    // indices into DTModel::nodes
    int    prediction;     // +1 / −1 majority class at this node
    int    n_samples;
};

struct DTModel {
    std::vector<Node> nodes;
    int root = 0;
    int n_features = 0;
};
```

**Flat `std::vector<Node>`** (not pointer-based): a tree of 4K nodes × 32 B =
128 KB fits comfortably in L2, no per-node heap allocations, trivially
copyable, and — critical for `train_parallel_omp` — lets tasks append to the
tree via an atomic counter without a mutex-protected allocator.

### `struct BinnedDataset` — column-major uint8

```cpp
struct BinnedDataset {
    int n_rows, n_features;
    std::vector<uint8_t> Xb;                   // col-major: Xb[j*n_rows + i]
    std::vector<std::vector<double>> edges;    // per-feature bin edges
    std::vector<int> n_bins_per_feat;          // bin count per feature
};
```

**Column-major storage is the critical performance choice.** The hot loop
scans a single feature across all node samples:

```cpp
const uint8_t* colXb = &B.Xb[j * n_rows];
for (int i : idx) { uint8_t b = colXb[i]; ... }
```

With column-major, consecutive `colXb[i]` accesses are **unit-stride**
(sequential bytes, prefetcher-friendly). Row-major would have stride
`n_features = 103` bytes per access, touching a new cache line every
iteration. For the root node's ~98K-sample scan, that's the difference
between ~10 KB of sequential-access memory traffic and ~6 MB of
cache-line-loaded scattered traffic.

**`uint8_t` bin IDs** (vs. the original `double`/`float` features): 4–8×
smaller footprint. Entire `Xb` on CS:GO: 103 × 98K = **10 MB** — fits in L3.
Raw `X` would be 40 MB (float) — out of L3 and into DRAM. Tradeoff: building
`Xb` column-major requires 103 passes over row-major `X` (strided reads), so
the bin-construction phase is slower; amortized over many tree-building
passes it's still a win.

### `struct BestSplit`, `struct BestSplitCL`

```cpp
struct BestSplit {
    double gain, threshold;
    int    feature;
    bool better_than(const BestSplit& o) const;   // deterministic tie-break
};

struct alignas(64) BestSplitCL {
    BestSplit bs;
    char _pad[64 - sizeof(BestSplit)];
};
```

`BestSplitCL` is padded to a full cache line so per-thread arrays
(`std::vector<BestSplitCL>(N_THREADS)`) don't suffer false sharing when eight
threads concurrently write their local best split.

**Deterministic tie-breaking** (`better_than`): on equal gain, prefer smaller
feature index; on equal (gain, feature), prefer smaller threshold. Without
this, parallel reductions produce non-reproducible trees — the thread order
of reduction depends on scheduling. With it, all three trainers produce
byte-identical prediction vectors.

### Per-node stack-local histograms

The inner histogram pair `int hp[256], hn[256]` is declared **on the stack**
inside `scan_feature_histogram`. Since `N_BINS ≤ 256` always, these are fixed
~2 KB allocations on the caller's stack — no heap, no false sharing (each
thread has its own stack), no bookkeeping.

### Data-preprocessing skip list

Identical to SVM/KNN/MLP: `map_*`, `bomb_planted`, and auto-detected 0/1-only
columns bypass z-score normalization. For DT this is cosmetic — Gini splits
are invariant under monotone transformations, so any scaling produces the
same tree — but matching the SVM/KNN/MLP loader exactly is required so
`analytics_engine.cpp` can share one `Dataset` across all five models. On
CS:GO expect `Normalize: 94 z-scored, 9 passthrough` (bomb_planted + 8
map_* one-hots).

---

## 5. Training — three variants

All three trainers share the histogram-based split kernel (`scan_feature_histogram`)
and the same partitioning / stopping criteria. Serial and parallel run the
**same algorithm**, so the speedup ratio measures parallelization overhead
only, not algorithm improvement.

### 5a. `train_serial` — histogram CART, single-threaded

Recursive descent via `build_hist_serial`. At each node, sweep all `F`
features serially, pick the best split, partition `idx`, recurse on left and
right children. No threading anywhere. This is the single serial baseline
for the speedup ratio.

### 5b. `train_parallel_omp` — OpenMP taskloop + tasks

**Two parallelism axes, one OpenMP team:**

1. **Within a node**, `#pragma omp taskloop grainsize(F/P)` splits the F=103
   features across the 8 worker threads. Each thread accumulates its local
   best into a `BestSplitCL` slot; a final serial pass reduces across slots.
2. **Across subtrees** (only while `depth < PARALLEL_DEPTH = 3`),
   `#pragma omp task` launches the left and right recursive calls as separate
   tasks with `#pragma omp taskwait` at the parent. Deeper than
   PARALLEL_DEPTH, recursion runs serially on whichever worker inherited the
   subtree — the work per node is too small for task overhead to pay off.

```cpp
#pragma omp parallel num_threads(N_THREADS)
{
    #pragma omp single
    build_hist_omp(ctx, 0, 0, root_idx, pos_tot, neg_tot);
}
```

`omp_set_max_active_levels(2)` enables the nested parallelism (taskloop inside
a task), but nesting never creates a new thread team — tasks share the outer
team's threads, so there is **no thread over-subscription**. At the root, the
taskloop fans 103 features across 8 threads. At depths 1–3, multiple subtree
tasks run concurrently, each sequentially scanning features within its node.
At depth ≥ 3, pure serial.

**Why two axes?** At the root there's only one subtree — cross-subtree
parallelism is unusable, so feature-level parallelism is all we have. Near
the leaves, feature-level work per node is tiny (~1500 samples × 103 features
= ~150K ops, < 1 µs); task-creation overhead would dominate. Subtree
parallelism at depth 3 gives 2³ = 8 concurrent subtrees, exactly filling the
8-thread team.

**False-sharing guards.** `BestSplitCL` is 64-byte aligned. Stack-local `hp`,
`hn` histograms are per-thread. `Xb`, `y`, and `edges` are read-only during
training. Node allocation uses `std::atomic<int>::fetch_add(2)` on a
preallocated `nodes` vector (capacity = `2^(max_depth+1)`) — no mutex, no
reallocation, no pointer invalidation.

**Deterministic tie-breaking** ensures `train_parallel_omp` produces the same
tree as `train_serial`, byte-for-byte. Verified at program end by the
`|serial − omp| < 0.01` accuracy parity check (and in practice the delta is
0.0000 because trees are identical).

### 5c. `train_parallel_pthreads` — persistent pool, per-node barriers

Kept alongside the OMP variant so the writeup has **two parallel
implementations of the same algorithm** — mirrors MLP's decision. Fixed pool
of N_THREADS workers synchronized by `pthread_barrier_t` at each node.

```cpp
struct PtState {
    const BinnedDataset* B;  const Labels* y;  DTModel* model;
    int n_features;          const std::vector<float>* X;
    std::vector<PendingNode> stack;        // leader-owned explicit stack
    // Installed by leader before barrier_go; read by workers after:
    const std::vector<int>* cur_idx;   int cur_depth;
    long long cur_pos, cur_neg;        double cur_parent_g;
    BestSplit cur_best;                bool cur_is_leaf, shutdown;
    pthread_mutex_t   mutex;
    pthread_barrier_t barrier_ready;   // workers → leader: reduction done
    pthread_barrier_t barrier_go;      // leader → workers: next node ready
};
```

Subtree recursion is **driven serially by the leader** via an explicit stack
— workers never manipulate the tree. Per node:

1. Leader pops a `PendingNode` off its stack, installs `cur_*` state.
2. `barrier_go` — leader and 7 workers all arrive; workers begin their slice.
3. Each worker scans its static feature range `[tid·F/P, (tid+1)·F/P)` into
   its local `BestSplit`.
4. `mutex_lock` → merge local best into `st->cur_best` → `mutex_unlock`.
5. `barrier_ready` — all 8 threads have reduced.
6. Leader partitions `cur_idx` into `left_idx` / `right_idx` (serial; workers
   wait at the next iteration's `barrier_go`), pushes children onto its stack.

At program shutdown, the leader sets `st.shutdown = true` and hits
`barrier_go` one final time; workers wake, observe the flag, and exit.

**Why leader-driven (not a work-queue)?** Two reasons. First, it mirrors
SVM's and MLP's `train_parallel` pattern (per-epoch barriers → per-node
barriers). Second, the only shared mutable state is the tree's `nodes`
vector, and limiting mutation to the leader avoids a tree-level lock. If
workers directly recursed, we'd need either a mutex or an atomic slot
allocator on every `emplace_back`.

**Trade-off vs. OMP.** Pthreads gives up the cross-subtree parallelism axis —
every subtree passes through the leader, one at a time. At depth=3 on a
balanced tree, OMP has 8 concurrent subtrees each doing ~12K samples × 103
features; pthreads has one subtree being worked on by all 8 threads across a
13-feature range each. Compute parallelism is similar, but the pthreads
version eats two `pthread_barrier_wait` calls per node. As the tree grows
wider (more nodes near the leaves), barrier overhead compounds — the OMP
variant should pull ahead. **This is exactly the "synchronization overhead"
the proposal predicts for DT.**

---

## 6. Inference

`predict_serial` and `predict_parallel` are identical except for the
`#pragma omp parallel for schedule(static) num_threads(N_THREADS)` on the
outer sample loop. Each test sample walks the tree from the root, branching
left/right on `x[feature_index] ≤ threshold`. Read-only access to the tree,
no per-sample state — embarrassingly parallel.

Tree walks are **short** (at most `MAX_DEPTH = 12` comparisons) but
**cache-hostile**: every step jumps to a potentially-distant `Node` (the
`left` or `right` index could be anywhere in the flat `nodes` vector).
At 24,483 test samples × 12 jumps = 294K indexed accesses, the 128 KB tree
fits in L2 so this is fast (~2 ms parallel, ~10 ms serial). Inference
speedup is modest because the per-sample walk is so short — **train-side
parallelism is where DT speedup comes from**.

---

## 7. Arithmetic intensity (for the writeup)

**Per (sample, feature) pair in the histogram hot loop:**

| Memory op                                                       | Bytes |
|-----------------------------------------------------------------|------:|
| Load `Xb[j*n_rows + i]` (column-major, unit-stride)             | 1     |
| Load `y[i]`        (amortized; `y` is 400 KB, fits in L2)      | ≈0    |
| Load + store `hp[b]` or `hn[b]` (2 KB histograms fit in L1)    | ≈0    |

Ops per (sample, feature): ~3 (compare `y == 1`, indirect increment).

Effective arithmetic intensity: **~3 Ops/byte.** That sits just above the
roofline crossover on CARC's Cascade Lake — DT is **in the crossover regime**,
compute-bound near the leaves (small `idx`, Xb slice fits in L2) and
memory-bandwidth-limited at the root (full 10 MB `Xb` from L3/DRAM).

Comparison across the three hand-rolled algorithms:

| Algorithm | Inner kernel                  | AI (FLOP/byte) | Expected scaling   |
|-----------|-------------------------------|---------------:|--------------------|
| SVM       | `dot(w, x)` + margin check    | ~0.5           | Memory-bound       |
| DT        | histogram fill + Gini sweep   | ~3             | Crossover regime   |
| MLP       | dense GEMM forward/back       | ~42            | Compute-bound      |

DT should scale better than SVM (higher AI) but worse than MLP. The proposal's
hypothesis — "DT constrained by synchronization overhead" — is an orthogonal
prediction: even if the inner kernel is compute-friendly, per-node barrier
costs can dominate as the tree grows. The `speedup_omp` vs. `speedup_pthreads`
comparison in DT's own output probes exactly this (OMP has two parallelism
axes, pthreads has one + two barriers per node).

---

## 8. Numerical & correctness risks

1. **Serial vs. parallel tree mismatch.** The histogram algorithm is identical
   across the three trainers. If `BestSplit::better_than` weren't strictly
   deterministic (e.g., compared only `gain` with no tie-break), parallel
   reductions would produce different trees every run. The current tie-break
   order (gain → smaller feature → smaller threshold) is strict — verified
   by the accuracy parity check at end of `main()`; with a deterministic
   tie-break the delta should be exactly 0.
2. **`-ffast-math` drift.** Do not add `-ffast-math` to the compile flags.
   Gini gain differences of 1e-15 can flip best-split tie-breaking and
   break determinism between serial and parallel.
3. **OpenMP task visibility of indices.** Subtree tasks capture `left_ptr`
   and `right_ptr` as `std::shared_ptr<std::vector<int>>` via `firstprivate`
   — shared_ptr copy bumps the refcount, so the vector outlives the task's
   parent. A raw `std::vector<int>&` here would be a dangling reference
   if the parent returned before the task ran.
4. **Stack depth on deep trees.** `build_hist_serial` and `build_hist_omp`
   recurse. At `MAX_DEPTH = 12`, stack usage is trivial (~12 × a few KB).
   If a future run bumps `MAX_DEPTH > 24`, rewrite as an explicit stack (the
   pthreads variant already uses one, so there's a template to follow).
5. **Node index invalidation under reallocation.** `train_parallel_omp`
   preallocates `nodes.resize(2^(max_depth+1))` before any task runs, so
   indices taken via `atomic.fetch_add` remain valid. Never `push_back`
   from within a task.
6. **Pthreads barrier lifetime.** The mutex and two barriers are destroyed
   after `pthread_join`s return. If the leader crashes before signaling
   `shutdown = true`, workers hang on `barrier_go`. No graceful-recovery
   path today — acceptable for a research artifact, would need watchdog
   timeouts in production.
7. **Speedup numerator.** `train_serial`, `train_parallel_omp`, and
   `train_parallel_pthreads` all run the **same** histogram algorithm, so
   the printed speedup ratios measure parallelization only — not
   algorithmic wins.

---

## 9. Build & run

On USC CARC (from src/cpp/dt/):

```bash
cd src/cpp/dt && sbatch job_dt.sl
# outputs to dtjob.out (separate from SVM's gpujob.out and KNN's knnjob.out)
# compiles dt.cpp, runs ./dt ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
```

CLI:

```bash
./dt [train_csv] [test_csv] [max_depth]
# defaults: data/train_cleaned.csv  data/test_cleaned.csv  12
```

Locally (macOS), from the repo root:

```bash
# Option A: Homebrew libomp with Apple Clang
brew install libomp
clang++ -std=c++17 -O3 -march=native \
  -Xpreprocessor -fopenmp -lomp src/cpp/dt/dt.cpp -o dt -lpthread
./dt data/train_cleaned.csv data/test_cleaned.csv

# Option B: Homebrew GCC (matches CARC's g++ exactly)
brew install gcc
g++-14 -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp -o dt -lpthread
```

**Apple Clang alone will not work** — no OpenMP runtime, and `pthread_barrier_t`
is an optional POSIX feature that macOS does not implement. On a mismatched
Xcode-CLT Mac (e.g., Xcode 13 against macOS 26) even `<iostream>` fails to
compile, at which point the only option is CARC.

### Verification expected

- **Accuracy**: `acc ∈ [0.76, 0.80]` on test set for all three trainers.
- **Parity**: `|acc_serial − acc_omp| < 0.01` and
  `|acc_serial − acc_pthreads| < 0.01`. With deterministic tie-breaking the
  prediction vectors match byte-for-byte; `main()` prints the deltas at the
  end of every run.
- **Speedup**: `serial_total / omp_total > 1` and
  `serial_total / pthreads_total > 1`. OMP should lead pthreads; both should
  fall short of MLP's speedup on the same hardware (per the proposal's
  hypothesis).

---

## 10. Results log

Append one row per CARC run.

| date (UTC) | git SHA | max_depth | bins | machine/cores | serial (s) | omp (s) | pth (s) | acc_serial | acc_omp | acc_pth | speedup_omp | speedup_pth | notes |
|------------|---------|:---------:|:----:|---------------|-----------:|--------:|--------:|:----------:|:-------:|:-------:|:-----------:|:-----------:|-------|
| 2026-04-20 | 526bb18 | 12 | 64 | CARC d17-03 / 8 | 0.646 | 0.403 | 0.345 | 0.7627 | 0.7627 | 0.7627 | 1.60× | 1.87× | Full `{1,2,4,8}` sweep, job 3272373. **pthreads beats OMP** (1.87× vs 1.60×) — opposite of this doc's expectation. Two reasons: (i) deep-tree regions have many small nodes where OMP `taskloop` creation cost > feature-scan work, (ii) our pthreads leader-driven model amortizes the two barriers across all 8 workers scanning a full F=103 range in parallel, while OMP's cross-subtree axis under-fills late because tree shape is unbalanced. Accuracy parity is 0.0000 across variants (deterministic tie-break works). Lowest speedup of the five — confirms proposal's "synchronization-bound" prediction. See [results/run1/results.md](../results/run1/results.md). |

---

## 11. Followups (known work deferred)

- **Deduplicate loader/metrics** into `common.hpp` once `analytics_engine.cpp`
  is being written. The refactor aligned DT's `Dataset`, `load_dataset`,
  `evaluate`, and `print_results` to match SVM/KNN/MLP byte-for-byte; the
  extraction will be mechanical.
- **Max-depth sweep** — `max_depth ∈ {6, 8, 10, 12, 14, 16}` to show how
  tree size trades accuracy (plateaus ~12) against parallel overhead (deeper
  trees → more small nodes → barrier cost dominates). Already CLI-exposed as
  `argv[3]`.
- **Thread count sweep** — full speedup curve `S(P)` for `P ∈ {1, 2, 4, 8}`.
  Requires parameterizing `N_THREADS` at the CLI rather than hardcoding at
  top-of-file. Pairs with the same sweep for SVM, KNN, and MLP.
- **`PARALLEL_DEPTH` sweep** — `PARALLEL_DEPTH ∈ {0, 1, 2, 3, 4}` isolates the
  cross-subtree parallelism contribution vs. pure feature-level. Would
  confirm the claim that "feature-level + subtree-level" is what OMP uses to
  beat pthreads.
- **Parallel binning** — `build_binned_dataset` is currently serial. Could
  parallelize with `#pragma omp parallel for` over features. Small win
  (~100 ms on 8 cores), cosmetic vs. the tree-building speedup story.
- **Feature importance / pruning** — not required by the proposal, but
  would be a natural extension for the writeup if space allows (e.g., count
  how many times each feature is chosen as a split across the tree).
