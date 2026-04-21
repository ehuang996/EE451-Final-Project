#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=analyticsjob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0
module load openblas 2>/dev/null || true

# Compile all five algorithm binaries (analytics_engine invokes them as subprocesses).
# Sources live at src/cpp/<algo>/<algo>.cpp; binaries land in the repo root so
# the driver can invoke them as ./<algo>.
: "${KNN_BLAS_CFLAGS:=-DKNN_USE_BLAS}"
: "${KNN_BLAS_LIBS:=-lopenblas}"
: "${MLP_BLAS_CFLAGS:=-DMLP_USE_BLAS}"
: "${MLP_BLAS_LIBS:=-lopenblas}"
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp $KNN_BLAS_CFLAGS src/cpp/knn/knn.cpp -o knn -lpthread $KNN_BLAS_LIBS
g++ -std=c++17 -O3 -march=native -fopenmp $MLP_BLAS_CFLAGS src/cpp/mlp/mlp.cpp -o mlp -lpthread $MLP_BLAS_LIBS
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread

# Compile the driver. It does no math itself — just popen, parse, CSV write —
# so no OpenMP/pthread flags needed.
g++ -std=c++17 -O3 analytics_engine.cpp -o analytics_engine

# Run: invokes each binary in sequence, parses [tag] blocks, writes
# analytics_results.csv, and prints a summary table.
N_THREADS=${N_THREADS:-${SLURM_CPUS_PER_TASK:-8}} \
    ./analytics_engine data/train_cleaned.csv data/test_cleaned.csv
