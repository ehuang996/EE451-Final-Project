#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=36:00:00
#SBATCH --output=gpujob.out

# Compute nodes don't inherit the login-node default toolchain; without this
# `g++` resolves to system GCC 8 with missing headers on some partitions.
module load gcc/13.3.0

# Compile
g++ -std=c++17 -O3 -march=native -fopenmp svm.cpp -o svm -lpthread

# Run
./svm ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv