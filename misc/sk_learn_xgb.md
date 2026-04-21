# sklearn + XGBoost — Framework Comparison Harness

Implementation at [src/sklearn_xgb/](../src/sklearn_xgb/). This document records every
non-obvious design decision made in the Python comparison harness that runs
each of the five algorithms through scikit-learn (and XGBoost for the decision
tree) at `n_threads=8`, plus a live log of benchmark runs.

---

## 1. Problem framing

The five C++ implementations ([svm/svm.cpp](../src/cpp/svm/svm.cpp),
[knn/knn.cpp](../src/cpp/knn/knn.cpp), [mlp/mlp.cpp](../src/cpp/mlp/mlp.cpp),
[dt/dt.cpp](../src/cpp/dt/dt.cpp), [nb/nb.cpp](../src/cpp/nb/nb.cpp)) are self-contained
pure-C++17 + OpenMP + pthreads training/inference pipelines. They already
report serial-vs-parallel speedups within their own framework. What they do
**not** answer is the question any HPC/ML reviewer asks first:

> "How does your hand-rolled parallel C++ compare to the standard CPU ML
> libraries people actually use?"

The comparison harness answers that question with a single unified
`results.csv` row per (algorithm, framework, variant). It runs **scikit-learn**
(the de facto CPU ML baseline) and **XGBoost CPU with `tree_method='hist'`**
(the SOTA histogram-tree implementation we can honestly compare our
histogram CART against), with hyperparameters matched exactly to what the
C++ files compile in. The empirical output is the apples-to-apples
comparison table that the writeup's framework-comparison section is built
around.

### Scope (deliberately narrow)

- **`n_threads = 8` only.** No thread-scaling sweep across {1, 2, 4, 8}.
  The C++ side already produces its own speedup curves; adding sklearn's
  thread sweep doubles the complexity of the harness for marginal gain on
  the first-pass comparison table.
- **CSV/JSON output only.** No auto-generated matplotlib plots — the
  `results.csv` schema feeds a downstream plot script that lives outside
  this harness.
- **Three repeats are not collected here.** The harness runs each configuration
  once per `sbatch` submission; variance is obtained by submitting the job
  multiple times. This keeps each row unambiguous about which `n_threads=8`
  slot on which CARC node produced which number.

---

## 2. Architecture

All code lives under [src/sklearn_xgb/](../src/sklearn_xgb/) (new directory, matches
the existing per-algorithm subdirectory convention).

```
src/sklearn_xgb/
├── loader.py              # Python port of C++ load_dataset
├── hybrid_nb.py           # HybridGaussianBernoulliNB wrapper
├── compare.py             # sklearn + XGBoost runner (writes results.csv/json)
├── parse_cpp_output.py    # Parses C++ stdout "Speedup Summary" blocks
├── job_compare.sl         # SLURM job: build+run C++ → run Python → merge
├── requirements.txt       # numpy, pandas, scikit-learn, xgboost
├── README.md              # Short usage doc
└── logs/                  # Per-binary stdout captures, gitignored
```

### End-to-end flow on CARC

```
sbatch src/sklearn_xgb/job_compare.sl
 ├─ g++ -std=c++17 -O3 -march=native -fopenmp ... (×5 binaries)
 ├─ ./svm … > logs/svm.out
 ├─ ./knn … > logs/knn.out       ← captures the Unicode "Speedup Summary"
 ├─ ./mlp … > logs/mlp.out         blocks for parse_cpp_output.py
 ├─ ./dt  … > logs/dt.out
 ├─ ./nb  … > logs/nb.out
 ├─ python compare.py               ← writes 8 rows (sklearn + XGB)
 └─ python parse_cpp_output.py      ← appends 15 rows (5 algos × 3 variants)
                                   → total 23 rows in results.csv
```

Why two separate Python scripts instead of one monolith? `compare.py` pins
`OMP_NUM_THREADS=8` *before* importing numpy/sklearn — mixing that with the
stdout parser (which doesn't touch any linked BLAS) keeps the parser
trivially re-runnable on old `logs/*.out` files without the whole import
chain.

---

## 3. Framework ↔ algorithm mapping (all at n_threads=8)

| Algo | C++ (ours) | scikit-learn equivalent | XGBoost | Parallelism in sklearn |
|------|------------|-------------------------|---------|------------------------|
| SVM  | hinge + L2, `LR=0.02`, `LAMBDA=1e-4`, 20 epochs (pthreads + OMP) | `SGDClassifier(loss='hinge', alpha=1e-4, learning_rate='constant', eta0=0.02, max_iter=20)` | — | **Serial** — sklearn's SGDClassifier has no `n_jobs` |
| KNN  | brute-force squared Euclidean, k=11, uniform vote (OMP) | `KNeighborsClassifier(n_neighbors=11, weights='uniform', algorithm='brute', metric='euclidean', n_jobs=8)` | — | `n_jobs=8` (joblib process-pool parallelism over query rows) |
| MLP  | 103→64→32→1, ReLU/sigmoid, BCE, SGD+momentum=0.9, batch=128, 30 epochs, `lr=0.01 / √(epoch+1)` (pure sample-loop backend plus optional vendor-BLAS batch backend) | `MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='sgd', learning_rate_init=0.01, learning_rate='invscaling', momentum=0.9, alpha=1e-4, batch_size=128, max_iter=30, nesterovs_momentum=False)` | — | **BLAS-implicit** — sklearn parallelism lives inside the BLAS GEMM calls; our optional BLAS backend now uses the same style of inner kernel on the C++ side |
| DT   | histogram CART, `max_depth=12`, `min_split=20`, `min_leaf=5`, 64 bins, Gini (pthreads + OMP) | `DecisionTreeClassifier(criterion='gini', max_depth=12, min_samples_split=20, min_samples_leaf=5)` | `XGBClassifier(n_estimators=1, max_depth=12, tree_method='hist', max_bin=64, n_jobs=8, learning_rate=1.0, base_score=0.5, objective='binary:logistic')` | sklearn DT is **serial and exact-split**, not histogram; XGBoost DT with 1 tree + `hist` + `max_bin=64` is the apples-to-apples baseline |
| NB   | **hybrid** Gaussian (continuous) + Bernoulli (binary), Laplace α=1.0, variance smoothing 1e-9, BIN_THRESHOLD=0.5 (pthreads + OMP) | **Primary**: `HybridGaussianBernoulliNB` wrapper (see §4). **Secondary (naive baselines)**: `GaussianNB(var_smoothing=1e-9)` and `BernoulliNB(alpha=1.0, binarize=0.5)` applied to all 103 features | — | All three sklearn NB rows are serial |

### Hyperparameter sourcing

All sklearn hyperparameter blocks in [src/sklearn_xgb/compare.py](../src/sklearn_xgb/compare.py)
are pulled directly from the `static constexpr` declarations at the top of
each C++ file, not re-tuned. Specifically:

- SVM: [svm/svm.cpp:26-29](../src/cpp/svm/svm.cpp#L26-L29) → `LR=0.02`, `LAMBDA=1e-4`, `MAX_EPOCHS=20`.
- KNN: [knn/knn.cpp:28-29](../src/cpp/knn/knn.cpp#L28-L29) → `K=11`, squared Euclidean.
- MLP: [mlp/mlp.cpp:36-45](../src/cpp/mlp/mlp.cpp#L36-L45) → `(64, 32)`, SGD+momentum 0.9, `lr=0.01` invscaling, batch=128, 30 epochs, L2 α=1e-4.
- DT: [dt/dt.cpp:39-46](../src/cpp/dt/dt.cpp#L39-L46) → `MAX_DEPTH=12`, `MIN_SAMPLES_SPLIT=20`, `MIN_SAMPLES_LEAF=5`, `N_BINS=64`, Gini.
- NB: [nb/nb.cpp:37-43](../src/cpp/nb/nb.cpp#L37-L43) → α=1.0, `VAR_SMOOTHING=1e-9`, `BIN_THRESHOLD=0.5`.

When anyone edits a C++ hyperparameter, the corresponding `compare.py` line
has to be kept in sync. There is no config file linking the two — the
duplication is deliberate, same philosophy as the duplicated C++ loaders
per [CLAUDE.md](../CLAUDE.md).

---

## 4. Hybrid NB wrapper (the non-trivial piece)

sklearn ships no native mixed-likelihood Naive Bayes. `GaussianNB` assumes
every feature is continuous; `BernoulliNB` binarizes every feature at a
global threshold. Applying either to all 103 CS:GO features produces the
two "naive-sklearn-user" baselines the harness records — but neither is a
fair comparison to our hybrid [nb/nb.cpp](../src/cpp/nb/nb.cpp), which dispatches
per-feature on an `is_binary` mask (matching the same
`map_* | bomb_planted | auto-{0,1}` rules the loader uses for
normalization).

The fair comparison is a wrapper class, [src/sklearn_xgb/hybrid_nb.py](../src/sklearn_xgb/hybrid_nb.py):

```python
class HybridGaussianBernoulliNB:
    def __init__(self, binary_mask):
        self.binary_mask = binary_mask                     # from loader
        self.gaussian  = GaussianNB(var_smoothing=1e-9)    # matches nb.cpp
        self.bernoulli = BernoulliNB(alpha=1.0, binarize=0.5)

    def fit(self, X, y):
        self.gaussian.fit(X[:, ~self.binary_mask], y)
        self.bernoulli.fit(X[:, self.binary_mask], y)
        self.classes_ = self.gaussian.classes_

    def _joint_log_likelihood(self, X):
        jll_g = self.gaussian._joint_log_likelihood(X[:, ~self.binary_mask])
        jll_b = self.bernoulli._joint_log_likelihood(X[:, self.binary_mask])
        # Subtract one copy of the class prior: both components include it,
        # and the total log-posterior should count it exactly once.
        return jll_g + (jll_b - self.bernoulli.class_log_prior_)

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
```

### Why this composes cleanly

Naive Bayes factorizes:
`log P(y | x) = log P(y) + Σ_j log P(x_j | y)`.

Splitting the feature sum across two independent likelihood families is
mathematically trivial — the continuous-feature and binary-feature log-
likelihood contributions live in completely separate features, so
`Σ_j log P(x_j | y) = Σ_{j ∈ cont} log p_g(x_j | y) + Σ_{j ∈ bin} log p_b(x_j | y)`.
What needs care is that *each* sklearn NB component adds its own prior
contribution at `fit` time. Summing the two `_joint_log_likelihood` outputs
naively double-counts the prior. Subtracting `bernoulli.class_log_prior_`
from the Bernoulli output (before adding) zeroes out one copy.

### Why the same `binary_mask` that the C++ code uses

The `binary_mask` array returned by [src/sklearn_xgb/loader.py](../src/sklearn_xgb/loader.py)
implements the exact same predicate as
[nb/nb.cpp:298-316](../src/cpp/nb/nb.cpp#L298-L316) `detect_binary()`:
`map_* prefix OR bomb_planted OR all-values-in-{0,1}`. Local smoke test
confirmed 31 binary / 72 continuous — larger than the 9/94 naive count
because many weapon-count columns are literally 0/1 on the CS:GO training
set. If the C++ NB ever changes its binary-detection predicate, loader.py
needs to match.

### Why not use `_joint_log_likelihood` directly — isn't it private?

`_joint_log_likelihood` is underscore-prefixed sklearn-internal API, but it
is stable across sklearn 0.24 → 1.x: it is the method every sklearn NB
subclass overrides to define its likelihood. Calling it from outside is
the canonical way to stack log-posteriors without re-implementing NB.
Alternative: reach into `gaussian.theta_` / `gaussian.var_` directly and
compute log-likelihoods by hand, which duplicates known-working code for
no benefit. If `_joint_log_likelihood` is ever removed, the test row
`sklearn/nb/hybrid` will error visibly on the next run — the harness
`sanity_check` will catch it — and the wrapper can be rewritten at that
point.

---

## 5. Data preprocessing parity

[src/sklearn_xgb/loader.py](../src/sklearn_xgb/loader.py) reproduces the C++
`load_dataset` byte-for-byte in observable behavior:

- Reads `data/train_cleaned.csv` / `data/test_cleaned.csv` (via a
  `_resolve` helper that searches cwd, repo-root, and repo-root/data/).
- Finds the label column by name `round_winner` in the header.
- Labels stay in {+1, −1} (matches the C++ convention).
- Z-scores continuous columns with **train-set statistics only**.
- Skips: `map_*` prefix, `bomb_planted`, auto-detected 0/1 columns.
  Skipped columns are passed through with `mean=0, sd=1`, so the common
  in-place normalization is still uniform.
- Prints the same `Dataset Info` box that the C++ binaries print, for
  visual diff against captured C++ logs.

Local smoke test (python3 without CARC):

```
Dataset Info
  X_train  : 97928 x 103
  X_test   : 24482 x 103
  y_train  : +1=47982  -1=49946
  y_test   : +1=12022  -1=12460
  Normalize: 72 z-scored, 31 passthrough (binary/one-hot)
```

Non-binary column post-normalization: `mean=-0.0000, std=1.0000`. Binary
column post-normalization: `min=0.0, max=1.0` (unchanged, as the passthrough
is designed to do).

**Note**: the 72 / 31 split is larger than the MLP doc's "94 z-scored /
9 passthrough (bomb_planted + 8 map_* one-hots)" claim. The extra 22
columns are weapon-count features that happen to be all-0/1 on the training
set (a player either has or doesn't have each niche weapon). The C++ auto-
detection catches these; the Python loader mirrors the same predicate, and
both passes skip those columns during normalization so the hybrid NB sees
them as Bernoulli features. This is correct behavior — not a discrepancy.

### Label handling for XGBoost

XGBoost requires labels in {0, 1} for binary classification. The harness
converts `y == 1 → 1, y == -1 → 0` inside `run_xgboost_dt`, runs `fit` /
`predict`, then maps predictions back to {+1, −1} so the shared `evaluate()`
sees the same label space as all other rows.

---

## 6. Output schema

`src/sklearn_xgb/results.csv`:

```
algorithm,framework,variant,n_threads,train_ms,infer_ms,acc,prec,rec,f1,notes
```

| Column      | Values / meaning                                               |
|-------------|----------------------------------------------------------------|
| `algorithm` | `svm` `knn` `mlp` `dt` `nb`                                    |
| `framework` | `ours-cpp` (from stdout parser) / `sklearn` / `xgboost`        |
| `variant`   | `serial` `omp` `pthreads` (C++) · `default` (sklearn/XGB one-flavor rows) · `hybrid` `gaussian-only` `bernoulli-only` (NB rows) |
| `n_threads` | `1` for known-serial sklearn models; `8` otherwise             |
| `train_ms`  | Wall time — `clf.fit()` for Python, `train_*_ms` from C++ stdout |
| `infer_ms`  | Wall time — `clf.predict(X_test)` for Python, `inference_*_ms` from C++ |
| `acc` `prec` `rec` `f1` | Same TP/FP/TN/FN formulation as the shared C++ `evaluate()` — [loader.py](../src/sklearn_xgb/loader.py) reimplements the identical math |
| `notes`     | Key metadata — e.g. "SGDClassifier single-threaded by design", "hist single-tree, max_bin=64", "naive-sklearn-user baseline" |

`results.json` is the same data as a list of dicts. Kept in lockstep with
`results.csv` by both writer scripts.

### Expected row count: 23

- **15 C++ rows** — 5 algorithms × {serial, omp, pthreads} from the
  `parse_cpp_output.py` pass over `logs/*.out`.
- **4 sklearn single-algorithm rows** — SVM, KNN, MLP, DT.
- **1 XGBoost row** — DT.
- **3 NB rows** — `hybrid` (primary), `gaussian-only` + `bernoulli-only`
  (naive-sklearn-user baselines).

Any count materially below 23 means the SLURM job had a silent failure and
needs investigating (most likely: Python module didn't load, or a C++
binary crashed mid-run so its `logs/*.out` is truncated).

---

## 7. Fairness / methodology notes

The framework-comparison table is the centerpiece of the writeup's
framework-comparison section, so the annotations that go alongside each
number matter at least as much as the numbers themselves:

1. **`SGDClassifier` is serial by design.** sklearn's SGDClassifier uses
   cython-level single-threaded SGD. Its `n_threads=1` column is not a
   bug in our harness — it's the library's architecture. The
   `notes` column labels this explicitly. A fair writeup says "our
   parallel SVM beats sklearn's serial SGDClassifier"; it does not say
   "our SVM is N× faster than sklearn at 8 threads".

2. **sklearn's `DecisionTreeClassifier` is also single-threaded**, and uses
   **exact splits**, not histogram. The fair apples-to-apples comparison for
   our histogram CART is the XGBoost row, not the sklearn DT row. The
   sklearn DT row is kept for context: it answers "what does the naive
   sklearn user get out of the box?", not "what's the SOTA CPU tree
   implementation?".

3. **sklearn's `MLPClassifier` parallelism is implicit through BLAS.**
   There is no `n_jobs` knob — the parallelism comes from whatever BLAS
   implementation numpy is linked against (OpenBLAS, MKL, etc.) and respects
   `OMP_NUM_THREADS` / `MKL_NUM_THREADS`. The harness sets both to 8
   before importing sklearn/numpy so the BLAS thread count matches our
   C++ MLP's 8-thread budget. This is the sklearn user's normal way to
   scale MLP training; the comparison is fair.

4. **XGBoost uses {0, 1} labels and our code uses {+1, −1}.** Handled in
   `run_xgboost_dt`; every evaluation metric is computed against the same
   ground-truth label space.

5. **Mini-batch vs. full-batch convergence differences.** Our SVM uses
   full-batch GD ([svm/svm.cpp:276-311](../src/cpp/svm/svm.cpp#L276-L311)), while
   sklearn's `SGDClassifier` with `max_iter=20` runs 20 passes over the data
   with per-sample updates. Different optimization regimes, likely
   different final accuracies. This is a known wart — the harness records
   both and the writeup should note it. No way to flag `max_iter=1` on
   `SGDClassifier` because that doesn't even complete one pass. Deferred to
   the writeup to reconcile.

6. **Identical hyperparameters ≠ identical math.** sklearn's
   `MLPClassifier` uses softmax output for 2-class problems (not sigmoid),
   and uses log-loss (not BCE — mathematically equivalent for 2 classes but
   computed differently). sklearn's DT uses exact splits (not histogram).
   These are documented in the `notes` column rather than hidden. The
   purpose of the comparison is not "sklearn running our algorithm" —
   it is "sklearn's standard implementation of the same problem class".

---

## 8. Risks & known limitations

1. **Single-run timings — no variance.** Each `results.csv` represents one
   SLURM submission, one run per config. Multi-run variance requires
   repeated `sbatch` submissions and a separate aggregation script.
   Deferred per user scoping decision.

2. **CARC module name drift.** [src/sklearn_xgb/job_compare.sl](../src/sklearn_xgb/job_compare.sl)
   calls `module load python/3.11 2>/dev/null || true`. If CARC drops or
   renames that module, the pip install still tries to run under the
   system Python at whatever version happens to be default. Installs will
   still succeed (wheels are pure-Python for numpy/pandas/sklearn/xgboost),
   but the `module load` line should be updated when the site's Python
   offering changes.

3. **`pip install --user` in a SLURM job.** The installs write to
   `~/.local/lib/pythonX.Y/site-packages/`. If a teammate has already
   installed incompatible versions there, pip will not uninstall them
   without `--force-reinstall`. Cleanest fix: the writeup's final
   reproducibility submission should pin a `venv` and activate it in the
   SLURM script instead of `--user`. Deferred.

4. **Hybrid NB depends on sklearn-internal API.**
   `_joint_log_likelihood` is underscore-prefixed. Stable for the sklearn
   releases listed in [requirements.txt](../src/sklearn_xgb/requirements.txt)
   (`>=1.3`), but may disappear in a future major version. `compare.py`'s
   `sanity_check` catches it — `acc` out of `[0, 1]` or non-finite values
   → raise.

5. **XGBoost single-tree boost ≠ CART.** XGBoost with `n_estimators=1,
   learning_rate=1.0, base_score=0.5` fits a single shallow gradient-
   boosted tree optimizing log-loss, not an unweighted Gini-impurity CART.
   Gini and log-loss have different split preferences; trees from this
   XGBoost config will not be identical to a Gini CART of the same
   depth. For a strict apples-to-apples tree construction there is no CPU
   baseline library in wide use — everybody uses gradient-boosted
   wrappers. Documented in `notes`; the writeup should call this out.

6. **Stdout parser is tag-specific.** [src/sklearn_xgb/parse_cpp_output.py](../src/sklearn_xgb/parse_cpp_output.py)
   uses the regex `^\[(Serial|Parallel)\s+(\w+)` to identify blocks.
   If a future algorithm prints a block with a different prefix (e.g. a
   distilled variant), parsing will silently drop those rows. Smoke-tested
   with a synthetic SVM log during implementation — all three variants
   parsed correctly.

---

## 9. Build & run

On USC CARC:

```bash
sbatch src/sklearn_xgb/job_compare.sl
# outputs to src/sklearn_xgb/comparejob.out
# builds all 5 C++ binaries, captures each stdout to logs/<algo>.out,
# runs compare.py, then parse_cpp_output.py — results land in
# src/sklearn_xgb/results.csv and src/sklearn_xgb/results.json
```

CLI (Python only, no C++ rebuild):

```bash
cd src/sklearn_xgb
pip install --user -r requirements.txt
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python compare.py \
    --train ../data/train_cleaned.csv \
    --test  ../data/test_cleaned.csv
```

Locally (macOS): only the loader and parser are exercisable locally.
Apple's Python toolchain frequently ships with a numpy / sklearn ABI
mismatch (e.g. sklearn compiled against numpy 1.x, runtime numpy 2.x →
`numpy.dtype size changed` at `import sklearn`). Per
[CLAUDE.md](../CLAUDE.md), local dev is intentionally unsupported.
Parser and loader do not depend on sklearn and can still be smoke-tested
in isolation via `python3 -c "from loader import load_dataset; ..."`.

### Verification expected

- **Row count**: `wc -l src/sklearn_xgb/results.csv` returns 24 (1 header +
  23 rows). Any count materially lower = silent failure; inspect
  `src/sklearn_xgb/logs/*.out` and `src/sklearn_xgb/comparejob.out`.
- **Accuracy range**: every `acc` in `[0.5, 1.0]`. Anything below 0.5 =
  label-flip bug somewhere; anything above 1.0 = parse bug. `compare.py`'s
  `sanity_check` enforces this on the Python side; for the C++ rows this
  must be eyeballed.
- **C++ parity**: for each of the 5 algorithms, the `|acc_serial -
  acc_omp|` and `|acc_serial - acc_pthreads|` deltas printed by the C++
  binary itself must be `< 0.01`. This is the same invariant the C++
  harness always enforces; the Python harness inherits it by passively
  parsing the values the C++ code already computed.
- **Hybrid NB ~= ours-cpp NB**: `sklearn/nb/hybrid` accuracy should be
  within ~2% of `ours-cpp/nb/serial` accuracy. Wider gaps indicate the
  wrapper's `binary_mask` disagrees with the C++ `is_binary` mask —
  same-predicate-different-data bug.

---

## 10. Results log

Append one row per SLURM submission of `job_compare.sl`. Format:

| date (UTC) | git SHA | machine/cores | SVM acc (ours-pth / sklearn) | KNN acc (ours-omp / sklearn) | MLP acc (ours-pth / sklearn) | DT acc (ours-pth / sklearn / xgboost) | NB acc (ours-pth / hybrid / gauss-only / bern-only) | SVM train ms (ours-pth / sklearn) | MLP train ms (ours-pth / sklearn) | DT train ms (ours-pth / xgboost) | notes |
|------------|---------|---------------|:----------------------------:|:----------------------------:|:----------------------------:|:-------------------------------------:|:---------------------------------------------------:|:---------------------------------:|:---------------------------------:|:--------------------------------:|-------|
| (pending first CARC run) | | CARC / 8 | | | | | | | | | expected: ours-cpp within ±2% of sklearn; hybrid NB matches ours-cpp NB exactly; XGBoost DT train_ms competitive with ours-pth |

Per-run raw data lives in `src/sklearn_xgb/results.csv` (unified schema for all
23 rows); this table is a human-readable summary that picks out the most
interesting cells per algorithm for quick at-a-glance comparison.

---

## 11. Followups (known work deferred)

- **Multi-run variance.** Submit `job_compare.sl` three times (or parameterize
  the job with `--array=1-3`), then aggregate mean ± std into a second
  results table. Blocks reporting confidence intervals in the writeup.
- **Thread-scaling sweep on the sklearn side.** Currently `OMP_NUM_THREADS=8`
  is hardcoded. Parameterizing would add columns to `results.csv` for
  `n_threads ∈ {1, 2, 4, 8}` and let the writeup compare scaling curves
  side-by-side. The C++ side would also need `N_THREADS` at runtime rather
  than `constexpr` — that's a cross-algorithm change deferred per the
  existing per-algo followups.
- **Full XGBoost baseline (100 trees).** Currently only `n_estimators=1` is
  recorded, for apples-to-apples against our single-CART tree. A second
  row with `n_estimators=100, tree_method='hist', n_jobs=8` would
  contextualize where our single-tree DT lands against "what a real
  XGBoost user runs". One line to add; useful writeup context.
- **Strict-CART CPU baseline.** `XGBClassifier(n_estimators=1)` is
  gradient-boosted, not Gini-CART. LightGBM's `LGBMClassifier(n_estimators=1,
  objective='binary', max_bin=64)` is the same class of thing. `xgb-vs-lgb`
  disagreement would expose whether `n_estimators=1` is a meaningful
  baseline or a library-specific quirk.
- **Plot script.** `plot_results.py` that reads `results.csv` and emits
  bar charts (accuracy by framework, train_ms by framework) + a latency
  scatter. Deferred per user scoping decision ("CSV/JSON only" for the
  first pass).
- **Fast-iteration SLURM variant (no C++ rebuild).** `compare.py` itself
  never invokes the C++ binaries — the rebuild + stdout capture only happens
  in [job_compare.sl](../src/sklearn_xgb/job_compare.sl). A companion SLURM
  script that skips the `g++` + `./algo` steps (or a flag on the main one)
  would cut the job to ~2 minutes for iterating on the Python side alone.
