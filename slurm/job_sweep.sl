#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=sweepjob.out

set -e

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0

# ----------------------------------------------------------------------------
# Thread-scaling sweep: runs every algorithm + framework at
# N_THREADS ∈ {1, 2, 4, 8}. Produces:
#   - sweep_results_cpp.csv     (C++ side, from analytics_engine)
#   - sweep_results_sklearn.csv (sklearn + XGBoost side, from compare.py)
# one row per (algorithm, variant, n_threads).
#
# Run from the repo root: `sbatch slurm/job_sweep.sl`.
# ----------------------------------------------------------------------------

THREAD_COUNTS=(1 2 4 8)

mkdir -p src/sklearn_xgb/logs

# Compile all five C++ binaries + analytics_engine once (same flags everywhere).
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
if [[ "${USE_KNN_BLAS:-0}" == "1" ]]; then
    module load openblas 2>/dev/null || true
    : "${KNN_BLAS_CFLAGS:=-DKNN_USE_BLAS}"
    : "${KNN_BLAS_LIBS:=-lopenblas}"
    g++ -std=c++17 -O3 -march=native -fopenmp $KNN_BLAS_CFLAGS src/cpp/knn/knn.cpp -o knn -lpthread $KNN_BLAS_LIBS
else
    g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/knn/knn.cpp -o knn -lpthread
fi
if [[ "${USE_MLP_BLAS:-0}" == "1" ]]; then
    module load openblas 2>/dev/null || true
    : "${MLP_BLAS_CFLAGS:=-DMLP_USE_BLAS}"
    : "${MLP_BLAS_LIBS:=-lopenblas}"
    g++ -std=c++17 -O3 -march=native -fopenmp $MLP_BLAS_CFLAGS src/cpp/mlp/mlp.cpp -o mlp -lpthread $MLP_BLAS_LIBS
else
    g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/mlp/mlp.cpp -o mlp -lpthread
fi
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread
g++ -std=c++17 -O3 -march=native analytics_engine.cpp -o analytics_engine

# Clean prior sweep outputs.
rm -f sweep_results_cpp.csv sweep_results_sklearn.csv
rm -f src/sklearn_xgb/results.csv src/sklearn_xgb/results.json

# ----------------------------------------------------------------------------
# C++ side: analytics_engine runs per thread count, emitting its own CSV.
# Concatenate into one sweep CSV at the end (header from first file only).
# ----------------------------------------------------------------------------
for T in "${THREAD_COUNTS[@]}"; do
    echo "=== C++ sweep: N_THREADS=$T ==="
    # KNN's BLAS backend wants outer query parallelism with BLAS pinned to 1.
    # MLP's BLAS backend wants the opposite: give the full thread budget to
    # the vendor library. If both are enabled together, prioritize KNN's
    # setting here and rerun an MLP-only sweep when you want best MLP timing.
    if [[ "${USE_KNN_BLAS:-0}" == "1" ]]; then
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
            N_THREADS=$T ./analytics_engine \
            data/train_cleaned.csv \
            data/test_cleaned.csv \
            "analytics_results_t${T}.csv"
    elif [[ "${USE_MLP_BLAS:-0}" == "1" ]]; then
        OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
        BLIS_NUM_THREADS=$T VECLIB_MAXIMUM_THREADS=$T \
            N_THREADS=$T ./analytics_engine \
            data/train_cleaned.csv \
            data/test_cleaned.csv \
            "analytics_results_t${T}.csv"
    else
        N_THREADS=$T ./analytics_engine \
            data/train_cleaned.csv \
            data/test_cleaned.csv \
            "analytics_results_t${T}.csv"
    fi
done

# Merge: keep header from the first file, skip headers from the rest.
head -1 "analytics_results_t1.csv" > sweep_results_cpp.csv
for T in "${THREAD_COUNTS[@]}"; do
    tail -n +2 "analytics_results_t${T}.csv" >> sweep_results_cpp.csv
done

# ----------------------------------------------------------------------------
# sklearn / XGBoost side: compare.py reads N_THREADS env var. Run once per
# thread count, writing to a per-thread-count CSV, then merge.
# ----------------------------------------------------------------------------
module load python/3.11 2>/dev/null || true
python -m pip install --user -r src/sklearn_xgb/requirements.txt

for T in "${THREAD_COUNTS[@]}"; do
    echo "=== sklearn sweep: N_THREADS=$T ==="
    rm -f "src/sklearn_xgb/results_t${T}.csv" "src/sklearn_xgb/results_t${T}.json"
    N_THREADS=$T python src/sklearn_xgb/compare.py \
        --train data/train_cleaned.csv \
        --test  data/test_cleaned.csv \
        --out-csv  "src/sklearn_xgb/results_t${T}.csv" \
        --out-json "src/sklearn_xgb/results_t${T}.json"
done

# Merge sklearn per-thread-count CSVs.
head -1 "src/sklearn_xgb/results_t1.csv" > sweep_results_sklearn.csv
for T in "${THREAD_COUNTS[@]}"; do
    tail -n +2 "src/sklearn_xgb/results_t${T}.csv" >> sweep_results_sklearn.csv
done

echo ""
echo "=== sweep_results_cpp.csv (C++ side) ==="
cat sweep_results_cpp.csv
echo ""
echo "=== sweep_results_sklearn.csv (sklearn + XGBoost side) ==="
cat sweep_results_sklearn.csv
