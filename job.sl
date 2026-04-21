#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=gpujob.out

# Compile
g++ -std=c++17 -O3 -march=native -fopenmp svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp mlp/mlp.cpp -o mlp -lpthread

# Run
./svm train_cleaned.csv test_cleaned.csv
./mlp train_cleaned.csv test_cleaned.csv
