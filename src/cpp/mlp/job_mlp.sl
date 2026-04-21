#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=mlpjob.out

# Compile
g++ -std=c++17 -O3 -march=native -fopenmp mlp.cpp -o mlp -lpthread

# Run
./mlp ../../../data/train_cleaned.csv ../../../data/test_cleaned.csv
