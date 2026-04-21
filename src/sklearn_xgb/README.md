# sklearn + XGBoost comparison harness

Runs the five algorithms through scikit-learn (and XGBoost for decision trees) at 8 threads with hyperparameters matched to the C++ implementations, then merges with the existing C++ binary outputs into one `results.csv`.

## Files

- [loader.py](loader.py) — Python port of the shared C++ `load_dataset`. Replicates the normalization skip rules from [svm/svm.cpp](../cpp/svm/svm.cpp).
- [hybrid_nb.py](hybrid_nb.py) — `HybridGaussianBernoulliNB` wrapper that mirrors [nb/nb.cpp](../cpp/nb/nb.cpp)'s per-feature likelihood selection by composing `GaussianNB` + `BernoulliNB`.
- [compare.py](compare.py) — Main driver. Runs all sklearn/XGBoost baselines and writes `results.csv` + `results.json`.
- [parse_cpp_output.py](parse_cpp_output.py) — Parses the "Speedup Summary" blocks from each C++ binary's stdout into the same schema.
- [job_compare.sl](job_compare.sl) — SLURM job that builds the C++ binaries, captures their stdout, runs the Python harness, and merges everything.
- [requirements.txt](requirements.txt) — Python deps.

## Run on CARC

Run from the repo root:

```bash
sbatch src/sklearn_xgb/job_compare.sl
# wait, then:
cat src/sklearn_xgb/results.csv
```

## Local smoke test (Python only, no C++ build)

Run from the repo root:

```bash
pip install -r src/sklearn_xgb/requirements.txt
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/sklearn_xgb/compare.py
```

## Results schema

`results.csv` columns:

```
algorithm,framework,variant,n_threads,train_ms,infer_ms,acc,prec,rec,f1,notes
```

- `algorithm` ∈ {svm, knn, mlp, dt, nb}
- `framework` ∈ {ours-cpp, sklearn, xgboost}
- `variant` distinguishes ours (serial/omp/pthreads), sklearn default, XGBoost default, and the three NB flavors (hybrid / gaussian-only / bernoulli-only)

## Notes on fairness

- **`SGDClassifier` and sklearn `DecisionTreeClassifier` are single-threaded by design** — their `n_threads=1` is not a regression, it's the library's choice. Labeled in the `notes` column so paper tables don't accidentally claim them as parallel.
- **sklearn's `MLPClassifier` parallelism is implicit** through BLAS (NumPy/OpenBLAS/MKL). The harness pins `OMP_NUM_THREADS=8` before importing so the comparison is at the same thread budget.
- **sklearn's `DecisionTreeClassifier` uses exact splits**, while our C++ implementation and XGBoost use histogram-based CART. The XGBoost `tree_method='hist', max_bin=64` row is the true apples-to-apples baseline for our DT.
- **sklearn has no native mixed-likelihood NB.** The hybrid row composes `GaussianNB` (for continuous features) and `BernoulliNB` (for binary features) on the same feature split [nb/nb.cpp](../cpp/nb/nb.cpp) uses. The gaussian-only and bernoulli-only rows document the accuracy loss from skipping that wrapper.
