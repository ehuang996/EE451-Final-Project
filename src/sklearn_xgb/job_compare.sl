#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=src/sklearn_xgb/comparejob.out

set -e

# ----------------------------------------------------------------------------
# Build and run the five C++ binaries (same flags as job.sl), capturing each
# binary's stdout into src/sklearn_xgb/logs/<algo>.out for the parser.
# Run from the repo root: `sbatch src/sklearn_xgb/job_compare.sl`.
# ----------------------------------------------------------------------------
module load gcc/13.3.0 2>/dev/null || true
module load openblas 2>/dev/null || true
mkdir -p src/sklearn_xgb/logs

: "${KNN_BLAS_CFLAGS:=-DKNN_USE_BLAS}"
: "${KNN_BLAS_LIBS:=-lopenblas}"
: "${MLP_BLAS_CFLAGS:=-DMLP_USE_BLAS}"
: "${MLP_BLAS_LIBS:=-lopenblas}"
: "${THREADS:=${N_THREADS:-${SLURM_CPUS_PER_TASK:-8}}}"

g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp $KNN_BLAS_CFLAGS src/cpp/knn/knn.cpp -o knn -lpthread $KNN_BLAS_LIBS
g++ -std=c++17 -O3 -march=native -fopenmp $MLP_BLAS_CFLAGS src/cpp/mlp/mlp.cpp -o mlp -lpthread $MLP_BLAS_LIBS
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread

./svm data/train_cleaned.csv data/test_cleaned.csv > src/sklearn_xgb/logs/svm.out
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 N_THREADS=$THREADS \
    ./knn data/train_cleaned.csv data/test_cleaned.csv > src/sklearn_xgb/logs/knn.out
OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
BLIS_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS N_THREADS=$THREADS \
    ./mlp data/train_cleaned.csv data/test_cleaned.csv > src/sklearn_xgb/logs/mlp.out
./dt  data/train_cleaned.csv data/test_cleaned.csv > src/sklearn_xgb/logs/dt.out
./nb  data/train_cleaned.csv data/test_cleaned.csv > src/sklearn_xgb/logs/nb.out

# ----------------------------------------------------------------------------
# Python harness: sklearn + XGBoost baselines at 8 threads.
# CARC typically requires `module load python/3.x` — adjust to match the
# site's available module. pip installs into a user-local prefix to avoid
# needing write access to system site-packages.
# ----------------------------------------------------------------------------
module load python/3.11 2>/dev/null || true

python -m pip install --user -r src/sklearn_xgb/requirements.txt

# Remove any prior run's CSV/JSON so the final file reflects this job only.
rm -f src/sklearn_xgb/results.csv src/sklearn_xgb/results.json

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python src/sklearn_xgb/compare.py \
    --train data/train_cleaned.csv \
    --test  data/test_cleaned.csv \
    --out-csv  src/sklearn_xgb/results.csv \
    --out-json src/sklearn_xgb/results.json

# Append the C++ rows parsed from the captured stdout logs.
python src/sklearn_xgb/parse_cpp_output.py \
    src/sklearn_xgb/logs/svm.out \
    src/sklearn_xgb/logs/knn.out \
    src/sklearn_xgb/logs/mlp.out \
    src/sklearn_xgb/logs/dt.out  \
    src/sklearn_xgb/logs/nb.out  \
    --append src/sklearn_xgb/results.csv \
    --json   src/sklearn_xgb/results.json

echo ""
echo "=== Unified results.csv ==="
cat src/sklearn_xgb/results.csv
