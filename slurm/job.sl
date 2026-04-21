#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=alljob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0
module load openblas 2>/dev/null || true

# Compile all five algorithms from src/cpp/<algo>/<algo>.cpp, output binaries
# to the repo root so all ./algo invocations use the same cwd.
: "${KNN_BLAS_CFLAGS:=-DKNN_USE_BLAS}"
: "${KNN_BLAS_LIBS:=-lopenblas}"
: "${MLP_BLAS_CFLAGS:=-DMLP_USE_BLAS}"
: "${MLP_BLAS_LIBS:=-lopenblas}"
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp $KNN_BLAS_CFLAGS src/cpp/knn/knn.cpp -o knn -lpthread $KNN_BLAS_LIBS
g++ -std=c++17 -O3 -march=native -fopenmp $MLP_BLAS_CFLAGS src/cpp/mlp/mlp.cpp -o mlp -lpthread $MLP_BLAS_LIBS
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread

# Run each against the shared cleaned CSVs at data/.
: "${THREADS:=${N_THREADS:-${SLURM_CPUS_PER_TASK:-8}}}"
./svm data/train_cleaned.csv data/test_cleaned.csv
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 N_THREADS=$THREADS \
    ./knn data/train_cleaned.csv data/test_cleaned.csv
OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
BLIS_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS N_THREADS=$THREADS \
    ./mlp data/train_cleaned.csv data/test_cleaned.csv
./dt  data/train_cleaned.csv data/test_cleaned.csv
./nb  data/train_cleaned.csv data/test_cleaned.csv
