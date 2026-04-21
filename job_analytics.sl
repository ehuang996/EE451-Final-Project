#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=analyticsjob.out

# Compile all five algorithm binaries (analytics_engine invokes them as subprocesses).
# Sources live at src/cpp/<algo>/<algo>.cpp; binaries land in the repo root so
# the driver can invoke them as ./<algo>.
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/knn/knn.cpp -o knn -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/mlp/mlp.cpp -o mlp -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread
g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread

# Compile the driver. It does no math itself — just popen, parse, CSV write —
# so no OpenMP/pthread flags needed.
g++ -std=c++17 -O3 analytics_engine.cpp -o analytics_engine

# Run: invokes each binary in sequence, parses [tag] blocks, writes
# analytics_results.csv, and prints a summary table.
./analytics_engine data/train_cleaned.csv data/test_cleaned.csv
