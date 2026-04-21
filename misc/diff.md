# C++ vs sklearn/XGBoost: per-algorithm differences

How each sklearn/XGBoost baseline in [src/sklearn_xgb/compare.py](../src/sklearn_xgb/compare.py) differs from our C++ implementation, and what the `n_threads` story is for each.

## SVM

- **Ours**: linear SVM, hinge + L2, custom SGD with `lr=0.02`, `λ=1e-4`, 20 epochs. OpenMP + pthreads parallelized.
- **sklearn**: `SGDClassifier(loss="hinge", alpha=1e-4, eta0=0.02, max_iter=20)`. Same loss, same regularization, same LR schedule — as close a match as sklearn allows.
- **Key difference**: `SGDClassifier` is **single-threaded by design** (no `n_jobs`). So the baseline is "our serial vs their serial vs our parallel" — the parallelism-vs-sklearn story here is about what we gain by implementing SVM ourselves *with threading*.

## KNN

- **Ours**: k=11, brute-force squared Euclidean, uniform vote.
- **sklearn**: `KNeighborsClassifier(n_neighbors=11, algorithm="brute", metric="euclidean", weights="uniform", n_jobs=N_THREADS)`. Everything matched.
- **Key difference**: **sklearn's `n_jobs` parallelizes KNN prediction** (joblib over test chunks), so at N_THREADS=8 it *is* parallel. This is the fairest head-to-head of the five.

## MLP

- **Ours**: 103→64→32→1, ReLU/sigmoid BCE, SGD+momentum 0.9, batch=128, 30 epochs, `lr=0.01`, invscaling.
- **sklearn**: `MLPClassifier(hidden_layer_sizes=(64,32), solver="sgd", momentum=0.9, nesterovs_momentum=False, batch_size=128, max_iter=30, learning_rate="invscaling", power_t=0.5)`. Architecturally and hyperparametrically aligned.
- **Key difference**: **sklearn's parallelism is implicit via BLAS**, not explicit threading — `OMP_NUM_THREADS=N_THREADS` is pinned before `numpy` is imported so the BLAS matmuls use the same thread budget as our explicit OpenMP. That's the apples-to-apples move.

## DT (two baselines)

- **Ours**: histogram CART, Gini, `max_depth=12`, `min_split=20`, `min_leaf=5`.
- **sklearn** `DecisionTreeClassifier`: same hyperparameters but uses **exact splits** (sorts each feature per node), **single-threaded** (no `n_jobs`). Serves as a "naive sklearn user" baseline.
- **XGBoost** (the truer apples-to-apples): `n_estimators=1, tree_method="hist", max_bin=64, max_depth=12, n_jobs=N_THREADS, learning_rate=1.0, reg_alpha=0, reg_lambda=0`. A single histogram tree with regularization disabled — matches our algorithm class *and* is threaded. This is the row that belongs in the paper's DT comparison.

## NB (three baselines because sklearn has no mixed likelihood)

- **Ours**: hybrid Gaussian (continuous features) + Bernoulli (binary features), per-feature likelihood choice via an `is_binary` mask.
- **sklearn hybrid**: [hybrid_nb.py](../src/sklearn_xgb/hybrid_nb.py) composes `GaussianNB` + `BernoulliNB` over the same `binary_mask` — a faithful reimplementation.
- **sklearn Gaussian-only**: `GaussianNB` applied to all 103 features. "Naive sklearn user who doesn't notice half their features are binary."
- **sklearn Bernoulli-only**: `BernoulliNB(alpha=1.0, binarize=0.5)`. Same, other way round.
- **Key difference**: all three are **single-threaded** (`var_smoothing=1e-9` on the Gaussian side). The gaussian-only and bernoulli-only rows exist specifically to quantify the accuracy cost of *not* doing the hybrid — which is the argument for our custom C++ impl being worth it.

## Cross-cutting differences

| Concern | Ours (C++) | sklearn/XGBoost |
|---|---|---|
| SVM threading | OMP + pthreads | Single-threaded only |
| KNN threading | OMP + pthreads | `n_jobs` (joblib) |
| MLP threading | OMP + pthreads (explicit) | Implicit via BLAS (`OMP_NUM_THREADS`) |
| DT threading | OMP + pthreads | sklearn single-threaded; XGBoost `n_jobs` |
| DT algorithm | Histogram CART | sklearn=exact, XGBoost=hist |
| NB | Hybrid mixed-likelihood | No native equivalent — `hybrid_nb.py` composes it |

The sklearn baselines are labeled `n_threads=1` in the CSV for the genuinely serial ones (SVM, sklearn-DT, all NB variants) regardless of `N_THREADS`, so the paper can't accidentally claim them as parallel.
