#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=mlpjob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0

# Compile. Default stays pure C++ for apples-to-apples project results.
# For the vendor-BLAS backend:
#   USE_MLP_BLAS=1 sbatch job_mlp.sl
# Override MLP_BLAS_CFLAGS / MLP_BLAS_LIBS if the cluster uses MKL or a
# non-default OpenBLAS install.
if [[ "${USE_MLP_BLAS:-0}" == "1" ]]; then
    module load openblas 2>/dev/null || true
    : "${MLP_BLAS_CFLAGS:=-DMLP_USE_BLAS}"
    : "${MLP_BLAS_LIBS:=-lopenblas}"
    g++ -std=c++17 -O3 -march=native -fopenmp $MLP_BLAS_CFLAGS mlp.cpp -o mlp -lpthread $MLP_BLAS_LIBS
else
    g++ -std=c++17 -O3 -march=native -fopenmp mlp.cpp -o mlp -lpthread
fi

# Run
if [[ "${USE_MLP_BLAS:-0}" == "1" ]]; then
    : "${MLP_THREADS:=${N_THREADS:-${SLURM_CPUS_PER_TASK:-8}}}"
    OMP_NUM_THREADS=$MLP_THREADS OPENBLAS_NUM_THREADS=$MLP_THREADS \
    MKL_NUM_THREADS=$MLP_THREADS BLIS_NUM_THREADS=$MLP_THREADS \
    VECLIB_MAXIMUM_THREADS=$MLP_THREADS N_THREADS=$MLP_THREADS \
        ./mlp ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
else
    ./mlp ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
fi
