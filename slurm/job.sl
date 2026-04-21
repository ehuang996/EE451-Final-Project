#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=alljob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0

# Compile all five algorithms from src/cpp/<algo>/<algo>.cpp, output binaries
# to the repo root so all ./algo invocations use the same cwd.
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/knn/knn.cpp -o knn -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/mlp/mlp.cpp -o mlp -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread

# Run each against the shared cleaned CSVs at data/.
./svm data/train_cleaned.csv data/test_cleaned.csv
./knn data/train_cleaned.csv data/test_cleaned.csv
./mlp data/train_cleaned.csv data/test_cleaned.csv
./dt  data/train_cleaned.csv data/test_cleaned.csv
./nb  data/train_cleaned.csv data/test_cleaned.csv
