#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=knnjob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0
module load openblas 2>/dev/null || true

: "${KNN_BLAS_CFLAGS:=-DKNN_USE_BLAS}"
: "${KNN_BLAS_LIBS:=-lopenblas}"
g++ -std=c++17 -O3 -march=native -fopenmp $KNN_BLAS_CFLAGS knn.cpp -o knn -lpthread $KNN_BLAS_LIBS

# Run: keep outer query parallelism in our code and pin BLAS to one thread.
: "${KNN_THREADS:=${N_THREADS:-${SLURM_CPUS_PER_TASK:-8}}}"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 N_THREADS=$KNN_THREADS \
    ./knn ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
