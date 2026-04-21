#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=knnjob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0

# Compile. Default stays pure C++ for apples-to-apples project results.
# For the vendor-BLAS counterfactual:
#   USE_KNN_BLAS=1 sbatch job_knn.sl
# Override KNN_BLAS_CFLAGS / KNN_BLAS_LIBS if the cluster uses MKL or a
# non-default OpenBLAS install.
if [[ "${USE_KNN_BLAS:-0}" == "1" ]]; then
    module load openblas 2>/dev/null || true
    : "${KNN_BLAS_CFLAGS:=-DKNN_USE_BLAS}"
    : "${KNN_BLAS_LIBS:=-lopenblas}"
    g++ -std=c++17 -O3 -march=native -fopenmp $KNN_BLAS_CFLAGS knn.cpp -o knn -lpthread $KNN_BLAS_LIBS
else
    g++ -std=c++17 -O3 -march=native -fopenmp knn.cpp -o knn -lpthread
fi

# Run
if [[ "${USE_KNN_BLAS:-0}" == "1" ]]; then
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        ./knn ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
else
    ./knn ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
fi
