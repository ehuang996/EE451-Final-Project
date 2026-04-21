// =============================================================================
// mlp.cpp — Serial and Parallel Multilayer Perceptron (BCE + L2 regularisation)
// Topology : 103 -> 64 -> 32 -> 1  (ReLU, ReLU, sigmoid)
// Dataset  : train_cleaned.csv / test_cleaned.csv
//            (103 features, label: round_winner in {+1, -1})
// Language : C++17
//
// NOTE: CSV loader, Metrics, evaluate(), now_ms(), and print_results() are
// duplicated from svm/svm.cpp to keep this translation unit self-contained
// (matching the style of svm.cpp). When analytics_engine.cpp is written,
// lift these into a shared common.hpp.
// =============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <cassert>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Hyperparameters
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int    N_FEATURES = 103;    // input dimension
static constexpr int    H1         = 64;     // hidden layer 1 width
static constexpr int    H2         = 32;     // hidden layer 2 width
static constexpr int    OUT        = 1;      // single logit -> sigmoid
static constexpr int    MAX_EPOCHS = 30;     // BCE converges fast vs. hinge
static constexpr int    BATCH      = 128;    // mini-batch size
static constexpr double LR         = 0.01;   // base learning rate
static constexpr double LAMBDA     = 1e-4;   // L2 weight decay (matches SVM)
static constexpr double MOMENTUM   = 0.9;    // Polyak momentum (better_serial + parallel)
static constexpr int    N_THREADS  = 8;      // worker threads
static constexpr int    SEED       = 42;     // RNG seed

static_assert(BATCH % N_THREADS == 0, "BATCH must be divisible by N_THREADS");

using Vec    = std::vector<double>;
using Matrix = std::vector<Vec>;
using IVec   = std::vector<int>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

static inline double sigmoid(double z) noexcept {
    return 1.0 / (1.0 + std::exp(-z));
}

// Numerically stable BCE from raw logit z and target t in {0,1}:
//   -t*log(sigmoid(z)) - (1-t)*log(1-sigmoid(z))
// = max(z,0) - z*t + log1p(exp(-|z|))
static inline double bce_loss(double z, int t) noexcept {
    double zmax  = (z > 0.0) ? z : 0.0;
    double abs_z = std::fabs(z);
    return zmax - z * static_cast<double>(t) + std::log1p(std::exp(-abs_z));
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset Loading and Preprocessing (copied from svm.cpp)
// ─────────────────────────────────────────────────────────────────────────────
struct Dataset {
    Matrix X_train, X_test;
    IVec   y_train, y_test; // labels in {+1, -1}
};

Dataset load_dataset(const std::string& train_path, const std::string& test_path) {
    auto parse_row = [&](const std::string& line) -> std::pair<Vec, int> {
        Vec row;
        row.reserve(N_FEATURES);
        std::istringstream ss(line);
        std::string tok;
        int col = 0, label = 0;
        while (std::getline(ss, tok, ',')) {
            if (col == 95) {
                label = std::stoi(tok);
            } else {
                row.push_back(std::stod(tok));
            }
            ++col;
        }
        return {std::move(row), label};
    };

    auto load_file = [&](const std::string& path, Matrix& X, IVec& y) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error: cannot open file: " << path << "\n";
            std::exit(1);
        }
        std::string line;
        std::getline(file, line); // skip header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto [row, lbl] = parse_row(line);
            X.push_back(std::move(row));
            y.push_back(lbl);
        }
    };

    Dataset ds;
    load_file(train_path, ds.X_train, ds.y_train);
    load_file(test_path,  ds.X_test,  ds.y_test);

    int n_train = static_cast<int>(ds.X_train.size());
    int n_test  = static_cast<int>(ds.X_test.size());

    Vec mean(N_FEATURES, 0.0);
    Vec sd(N_FEATURES, 0.0);

    for (int i = 0; i < n_train; ++i)
        for (int j = 0; j < N_FEATURES; ++j)
            mean[j] += ds.X_train[i][j];
    for (int j = 0; j < N_FEATURES; ++j)
        mean[j] /= n_train;

    for (int i = 0; i < n_train; ++i)
        for (int j = 0; j < N_FEATURES; ++j) {
            double d = ds.X_train[i][j] - mean[j];
            sd[j] += d * d;
        }
    for (int j = 0; j < N_FEATURES; ++j) {
        sd[j] = std::sqrt(sd[j] / n_train);
        if (sd[j] < 1e-9) sd[j] = 1.0;
    }

    auto norm_inplace = [&](Matrix& X) {
        for (auto& row : X)
            for (int j = 0; j < N_FEATURES; ++j)
                row[j] = (row[j] - mean[j]) / sd[j];
    };
    norm_inplace(ds.X_train);
    norm_inplace(ds.X_test);

    int tr_pos = 0, tr_neg = 0, te_pos = 0, te_neg = 0;
    for (int l : ds.y_train) (l == 1 ? tr_pos : tr_neg)++;
    for (int l : ds.y_test)  (l == 1 ? te_pos : te_neg)++;

    std::cout << "┌─────────────────────────────────────────────────┐\n";
    std::cout << "│  Dataset Info                                   │\n";
    std::cout << "├─────────────────────────────────────────────────┤\n";
    std::cout << "│  X_train : " << n_train << " x " << N_FEATURES << "\n";
    std::cout << "│  X_test  : " << n_test  << " x " << N_FEATURES << "\n";
    std::cout << "│  y_train : +1=" << tr_pos << "  -1=" << tr_neg << "\n";
    std::cout << "│  y_test  : +1=" << te_pos << "  -1=" << te_neg << "\n";
    std::cout << "└─────────────────────────────────────────────────┘\n\n";
    return ds;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Structure
// ─────────────────────────────────────────────────────────────────────────────
// All weight matrices are row-major flat vectors:
//   W1[i*N_FEATURES + j] = edge from input j to hidden-1 unit i
//   W2[i*H1 + j]         = edge from hidden-1 unit j to hidden-2 unit i
//   W3[i*H2 + j]         = edge from hidden-2 unit j to output unit i
// Row-major means the inner loop over j is contiguous -> SIMD-friendly.
struct MLPModel {
    Vec W1, b1;
    Vec W2, b2;
    Vec W3, b3;
    // Momentum buffers (used by momentum-based trainers)
    Vec vW1, vb1, vW2, vb2, vW3, vb3;
};

static void init_weights(MLPModel& m, unsigned seed) {
    m.W1.assign(H1 * N_FEATURES, 0.0);  m.b1.assign(H1, 0.0);
    m.W2.assign(H2 * H1,         0.0);  m.b2.assign(H2, 0.0);
    m.W3.assign(OUT * H2,        0.0);  m.b3.assign(OUT, 0.0);
    m.vW1.assign(H1 * N_FEATURES, 0.0); m.vb1.assign(H1, 0.0);
    m.vW2.assign(H2 * H1,         0.0); m.vb2.assign(H2, 0.0);
    m.vW3.assign(OUT * H2,        0.0); m.vb3.assign(OUT, 0.0);

    std::mt19937 rng(seed);
    // He init: N(0, sqrt(2/fan_in)) for ReLU layers
    std::normal_distribution<double> he1(0.0, std::sqrt(2.0 / N_FEATURES));
    std::normal_distribution<double> he2(0.0, std::sqrt(2.0 / H1));
    // Small uniform for output layer (sigmoid)
    std::uniform_real_distribution<double> out_u(-0.01, 0.01);

    for (auto& w : m.W1) w = he1(rng);
    for (auto& w : m.W2) w = he2(rng);
    for (auto& w : m.W3) w = out_u(rng);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward and Backward (single sample, stack-allocated activations)
// ─────────────────────────────────────────────────────────────────────────────
static inline void forward_sample(const MLPModel& m, const Vec& x,
                                  double* a1, double* a2,
                                  double& z_out, double& yhat) {
    for (int i = 0; i < H1; ++i) {
        double s = m.b1[i];
        const double* wr = m.W1.data() + i * N_FEATURES;
        for (int j = 0; j < N_FEATURES; ++j) s += wr[j] * x[j];
        a1[i] = (s > 0.0) ? s : 0.0;       // ReLU
    }
    for (int i = 0; i < H2; ++i) {
        double s = m.b2[i];
        const double* wr = m.W2.data() + i * H1;
        for (int j = 0; j < H1; ++j) s += wr[j] * a1[j];
        a2[i] = (s > 0.0) ? s : 0.0;       // ReLU
    }
    double z = m.b3[0];
    const double* wr = m.W3.data();        // OUT == 1
    for (int j = 0; j < H2; ++j) z += wr[j] * a2[j];
    z_out = z;
    yhat  = sigmoid(z);
}

// Accumulates gradients into the caller-owned gW*/gb* buffers.
// Uses the fused sigmoid+BCE derivative: dL/dz = yhat - t.
static inline void backward_sample(const MLPModel& m, const Vec& x,
                                   const double* a1, const double* a2,
                                   double yhat, int t,
                                   double* gW1, double* gb1,
                                   double* gW2, double* gb2,
                                   double* gW3, double* gb3) {
    double dz3 = yhat - static_cast<double>(t);
    for (int j = 0; j < H2; ++j) gW3[j] += dz3 * a2[j];
    gb3[0] += dz3;

    double dz2[H2];
    for (int j = 0; j < H2; ++j) {
        double da = m.W3[j] * dz3;
        dz2[j] = (a2[j] > 0.0) ? da : 0.0;   // ReLU'
    }
    for (int i = 0; i < H2; ++i) {
        double* gw_row = gW2 + i * H1;
        double v = dz2[i];
        for (int j = 0; j < H1; ++j) gw_row[j] += v * a1[j];
        gb2[i] += v;
    }

    double dz1[H1];
    for (int j = 0; j < H1; ++j) {
        double da = 0.0;
        for (int i = 0; i < H2; ++i) da += m.W2[i * H1 + j] * dz2[i];
        dz1[j] = (a1[j] > 0.0) ? da : 0.0;
    }
    for (int i = 0; i < H1; ++i) {
        double* gw_row = gW1 + i * N_FEATURES;
        double v = dz1[i];
        for (int j = 0; j < N_FEATURES; ++j) gw_row[j] += v * x[j];
        gb1[i] += v;
    }
}

// Plain SGD with L2 decay on weights (not biases)
static void sgd_update(MLPModel& m,
                       const Vec& gW1, const Vec& gb1,
                       const Vec& gW2, const Vec& gb2,
                       const Vec& gW3, const Vec& gb3,
                       double lr, int batch_size) {
    double inv_b = 1.0 / static_cast<double>(batch_size);
    double scale = 1.0 - lr * LAMBDA;
    for (size_t k = 0; k < gW1.size(); ++k) m.W1[k] = scale * m.W1[k] - lr * inv_b * gW1[k];
    for (size_t k = 0; k < gb1.size(); ++k) m.b1[k] -=                  lr * inv_b * gb1[k];
    for (size_t k = 0; k < gW2.size(); ++k) m.W2[k] = scale * m.W2[k] - lr * inv_b * gW2[k];
    for (size_t k = 0; k < gb2.size(); ++k) m.b2[k] -=                  lr * inv_b * gb2[k];
    for (size_t k = 0; k < gW3.size(); ++k) m.W3[k] = scale * m.W3[k] - lr * inv_b * gW3[k];
    for (size_t k = 0; k < gb3.size(); ++k) m.b3[k] -=                  lr * inv_b * gb3[k];
}

// SGD + Polyak momentum + L2 decay on weights
static void sgd_momentum_update(MLPModel& m,
                                const Vec& gW1, const Vec& gb1,
                                const Vec& gW2, const Vec& gb2,
                                const Vec& gW3, const Vec& gb3,
                                double lr, int batch_size) {
    double inv_b = 1.0 / static_cast<double>(batch_size);
    auto step_w = [&](Vec& w, Vec& v, const Vec& g) {
        double scale = 1.0 - lr * LAMBDA;
        for (size_t k = 0; k < w.size(); ++k) {
            v[k] = MOMENTUM * v[k] + inv_b * g[k];
            w[k] = scale * w[k] - lr * v[k];
        }
    };
    auto step_b = [&](Vec& w, Vec& v, const Vec& g) {
        for (size_t k = 0; k < w.size(); ++k) {
            v[k] = MOMENTUM * v[k] + inv_b * g[k];
            w[k] -= lr * v[k];
        }
    };
    step_w(m.W1, m.vW1, gW1);  step_b(m.b1, m.vb1, gb1);
    step_w(m.W2, m.vW2, gW2);  step_b(m.b2, m.vb2, gb2);
    step_w(m.W3, m.vW3, gW3);  step_b(m.b3, m.vb3, gb3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────────────
IVec predict_serial(const MLPModel& m, const Matrix& X) {
    int n = static_cast<int>(X.size());
    IVec out(n);
    for (int i = 0; i < n; ++i) {
        double a1[H1], a2[H2], z, yhat;
        forward_sample(m, X[i], a1, a2, z, yhat);
        out[i] = (yhat >= 0.5) ? 1 : -1;
    }
    return out;
}

IVec predict_parallel(const MLPModel& m, const Matrix& X) {
    int n = static_cast<int>(X.size());
    IVec out(n);
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(N_THREADS)
    #endif
    for (int i = 0; i < n; ++i) {
        double a1[H1], a2[H2], z, yhat;
        forward_sample(m, X[i], a1, a2, z, yhat);
        out[i] = (yhat >= 0.5) ? 1 : -1;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation Metrics (copied from svm.cpp)
// ─────────────────────────────────────────────────────────────────────────────
struct Metrics { double acc, prec, rec, f1; };

Metrics evaluate(const IVec& truth, const IVec& pred) {
    int TP = 0, FP = 0, TN = 0, FN = 0;
    for (int i = 0; i < static_cast<int>(truth.size()); ++i) {
        if      (truth[i] ==  1 && pred[i] ==  1) ++TP;
        else if (truth[i] == -1 && pred[i] ==  1) ++FP;
        else if (truth[i] == -1 && pred[i] == -1) ++TN;
        else                                       ++FN;
    }
    double acc  = static_cast<double>(TP + TN) / static_cast<double>(truth.size());
    double prec = (TP + FP) > 0 ? static_cast<double>(TP) / (TP + FP) : 0.0;
    double rec  = (TP + FN) > 0 ? static_cast<double>(TP) / (TP + FN) : 0.0;
    double f1   = (prec + rec) > 0.0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
    return {acc, prec, rec, f1};
}

// ─────────────────────────────────────────────────────────────────────────────
// Training — Serial (plain SGD, constant LR)
// ─────────────────────────────────────────────────────────────────────────────
MLPModel train_serial(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    assert(n > 0 && "train_serial: empty training set");

    MLPModel model;
    init_weights(model, SEED);

    IVec idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(SEED);

    Vec gW1(H1 * N_FEATURES), gb1(H1);
    Vec gW2(H2 * H1),         gb2(H2);
    Vec gW3(OUT * H2),        gb3(OUT);

    int n_batches = n / BATCH;

    for (int e = 0; e < MAX_EPOCHS; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double epoch_loss = 0.0;

        for (int bi = 0; bi < n_batches; ++bi) {
            int bs = bi * BATCH;
            std::fill(gW1.begin(), gW1.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gW2.begin(), gW2.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);
            std::fill(gW3.begin(), gW3.end(), 0.0);
            std::fill(gb3.begin(), gb3.end(), 0.0);

            for (int i = 0; i < BATCH; ++i) {
                int s = idx[bs + i];
                int t = (y[s] == 1) ? 1 : 0;
                double a1[H1], a2[H2], z, yhat;
                forward_sample(model, X[s], a1, a2, z, yhat);
                epoch_loss += bce_loss(z, t);
                backward_sample(model, X[s], a1, a2, yhat, t,
                                gW1.data(), gb1.data(),
                                gW2.data(), gb2.data(),
                                gW3.data(), gb3.data());
            }

            sgd_update(model, gW1, gb1, gW2, gb2, gW3, gb3, LR, BATCH);
        }

        std::cout << "epoch " << e << "  loss="
                  << (epoch_loss / static_cast<double>(n_batches * BATCH)) << "\n";
    }
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training — Better Serial (decaying LR + momentum)
// ─────────────────────────────────────────────────────────────────────────────
MLPModel train_better_serial(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    assert(n > 0 && "train_better_serial: empty training set");

    MLPModel model;
    init_weights(model, SEED);

    IVec idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(SEED);

    Vec gW1(H1 * N_FEATURES), gb1(H1);
    Vec gW2(H2 * H1),         gb2(H2);
    Vec gW3(OUT * H2),        gb3(OUT);

    int n_batches = n / BATCH;

    for (int e = 0; e < MAX_EPOCHS; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double lr_t = LR / std::sqrt(static_cast<double>(e + 1));
        double epoch_loss = 0.0;

        for (int bi = 0; bi < n_batches; ++bi) {
            int bs = bi * BATCH;
            std::fill(gW1.begin(), gW1.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gW2.begin(), gW2.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);
            std::fill(gW3.begin(), gW3.end(), 0.0);
            std::fill(gb3.begin(), gb3.end(), 0.0);

            for (int i = 0; i < BATCH; ++i) {
                int s = idx[bs + i];
                int t = (y[s] == 1) ? 1 : 0;
                double a1[H1], a2[H2], z, yhat;
                forward_sample(model, X[s], a1, a2, z, yhat);
                epoch_loss += bce_loss(z, t);
                backward_sample(model, X[s], a1, a2, yhat, t,
                                gW1.data(), gb1.data(),
                                gW2.data(), gb2.data(),
                                gW3.data(), gb3.data());
            }

            sgd_momentum_update(model, gW1, gb1, gW2, gb2, gW3, gb3, lr_t, BATCH);
        }

        std::cout << "epoch " << e << "  lr=" << lr_t << "  loss="
                  << (epoch_loss / static_cast<double>(n_batches * BATCH)) << "\n";
    }
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training — Parallel (OpenMP, data-parallel over mini-batch samples)
// ─────────────────────────────────────────────────────────────────────────────
// Strategy:
//   • Thread-local gradient buffers (one arena slot per thread) are hoisted
//     outside the batch loop — zeroed per batch, never reallocated.
//   • Per mini-batch, a parallel region splits the batch across threads with
//     #pragma omp for. Each thread accumulates into its local buffer.
//   • A critical section merges local buffers into the shared gradient.
//   • SGD update is run single-threaded by the primary thread after the
//     parallel region (fast: ~9K params).
MLPModel train_parallel_omp(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    assert(n > 0 && "train_parallel_omp: empty training set");

    MLPModel model;
    init_weights(model, SEED);

    IVec idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(SEED);

    // Shared gradient accumulators
    Vec gW1(H1 * N_FEATURES), gb1(H1);
    Vec gW2(H2 * H1),         gb2(H2);
    Vec gW3(OUT * H2),        gb3(OUT);

    // Per-thread (arena-indexed by tid) gradient buffers, allocated once
    std::vector<Vec> lgW1(N_THREADS, Vec(H1 * N_FEATURES, 0.0));
    std::vector<Vec> lgb1(N_THREADS, Vec(H1, 0.0));
    std::vector<Vec> lgW2(N_THREADS, Vec(H2 * H1,         0.0));
    std::vector<Vec> lgb2(N_THREADS, Vec(H2, 0.0));
    std::vector<Vec> lgW3(N_THREADS, Vec(OUT * H2,        0.0));
    std::vector<Vec> lgb3(N_THREADS, Vec(OUT, 0.0));
    std::vector<double> lloss(N_THREADS, 0.0);

    int n_batches = n / BATCH;

    for (int e = 0; e < MAX_EPOCHS; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        double lr_t = LR / std::sqrt(static_cast<double>(e + 1));
        double epoch_loss = 0.0;

        for (int bi = 0; bi < n_batches; ++bi) {
            int bs = bi * BATCH;
            std::fill(gW1.begin(), gW1.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gW2.begin(), gW2.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);
            std::fill(gW3.begin(), gW3.end(), 0.0);
            std::fill(gb3.begin(), gb3.end(), 0.0);

            #ifdef _OPENMP
                #pragma omp parallel num_threads(N_THREADS)
            #endif
            {
                #ifdef _OPENMP
                    int tid = omp_get_thread_num();
                #else
                    int tid = 0;
                #endif

                std::fill(lgW1[tid].begin(), lgW1[tid].end(), 0.0);
                std::fill(lgb1[tid].begin(), lgb1[tid].end(), 0.0);
                std::fill(lgW2[tid].begin(), lgW2[tid].end(), 0.0);
                std::fill(lgb2[tid].begin(), lgb2[tid].end(), 0.0);
                std::fill(lgW3[tid].begin(), lgW3[tid].end(), 0.0);
                std::fill(lgb3[tid].begin(), lgb3[tid].end(), 0.0);
                lloss[tid] = 0.0;

                #ifdef _OPENMP
                    #pragma omp for schedule(static) nowait
                #endif
                for (int i = 0; i < BATCH; ++i) {
                    int s = idx[bs + i];
                    int t = (y[s] == 1) ? 1 : 0;
                    double a1[H1], a2[H2], z, yhat;
                    forward_sample(model, X[s], a1, a2, z, yhat);
                    lloss[tid] += bce_loss(z, t);
                    backward_sample(model, X[s], a1, a2, yhat, t,
                                    lgW1[tid].data(), lgb1[tid].data(),
                                    lgW2[tid].data(), lgb2[tid].data(),
                                    lgW3[tid].data(), lgb3[tid].data());
                }

                #ifdef _OPENMP
                    #pragma omp critical
                #endif
                {
                    for (size_t k = 0; k < gW1.size(); ++k) gW1[k] += lgW1[tid][k];
                    for (size_t k = 0; k < gb1.size(); ++k) gb1[k] += lgb1[tid][k];
                    for (size_t k = 0; k < gW2.size(); ++k) gW2[k] += lgW2[tid][k];
                    for (size_t k = 0; k < gb2.size(); ++k) gb2[k] += lgb2[tid][k];
                    for (size_t k = 0; k < gW3.size(); ++k) gW3[k] += lgW3[tid][k];
                    for (size_t k = 0; k < gb3.size(); ++k) gb3[k] += lgb3[tid][k];
                }
            }

            for (int t = 0; t < N_THREADS; ++t) epoch_loss += lloss[t];
            sgd_momentum_update(model, gW1, gb1, gW2, gb2, gW3, gb3, lr_t, BATCH);
        }

        std::cout << "epoch " << e << "  lr=" << lr_t << "  loss="
                  << (epoch_loss / static_cast<double>(n_batches * BATCH)) << "\n";
    }
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training — Parallel (pthreads, mirrors SVM's train_parallel pattern)
// ─────────────────────────────────────────────────────────────────────────────
// Strategy (per mini-batch):
//   1. Each thread computes its slice of the batch into thread-local grads
//      (no synchronisation; reads stable model weights).
//   2. Mutex-protected reduction of local grads into shared accumulators.
//   3. barrier_acc: wait for all threads to finish accumulation.
//   4. Thread 0 applies SGD+momentum update, advances batch_start,
//      zeroes accumulators.
//   5. barrier_upd: no thread reads the updated weights for the next batch
//      before thread 0 finished writing them.
// Per-epoch housekeeping (shuffle, batch_start reset) also gated by barriers.

struct MLPParState {
    const Matrix* X;
    const IVec*   y;
    int           n;
    MLPModel*     model;
    IVec          idx;           // owned; thread 0 reshuffles per epoch
    int           batch_start;   // advanced by thread 0 each batch
    double        epoch_loss;    // accumulated over batches; printed by thread 0
    // Shared gradient accumulators + loss
    Vec           gW1, gb1, gW2, gb2, gW3, gb3;
    double        shared_loss;
    pthread_mutex_t   mutex;
    pthread_barrier_t barrier_acc;
    pthread_barrier_t barrier_upd;
    unsigned      seed;
};

struct MLPTArg {
    MLPParState* st;
    int          tid;
};

static void* mlp_pthread_worker(void* raw) {
    auto*   a  = static_cast<MLPTArg*>(raw);
    auto*   st = a->st;
    const Matrix& X = *st->X;
    const IVec&   y = *st->y;
    int n = st->n;

    // Thread-local grad buffers (allocated once per thread)
    Vec lgW1(H1 * N_FEATURES), lgb1(H1);
    Vec lgW2(H2 * H1),         lgb2(H2);
    Vec lgW3(OUT * H2),        lgb3(OUT);

    // Fixed slice of every mini-batch this thread handles
    const int slice = BATCH / N_THREADS;
    const int s_off = a->tid * slice;
    const int e_off = (a->tid == N_THREADS - 1) ? BATCH : (a->tid + 1) * slice;

    const int n_batches = n / BATCH;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {

        // ── Epoch housekeeping ── thread 0 reshuffles + resets batch_start ──
        if (a->tid == 0) {
            std::mt19937 rng(st->seed + static_cast<unsigned>(epoch));
            std::shuffle(st->idx.begin(), st->idx.end(), rng);
            st->batch_start = 0;
            st->epoch_loss  = 0.0;
        }
        pthread_barrier_wait(&st->barrier_upd);

        const double lr_t = LR / std::sqrt(static_cast<double>(epoch + 1));

        for (int bi = 0; bi < n_batches; ++bi) {

            // ── Step 1: local forward+backward on this thread's slice ──
            std::fill(lgW1.begin(), lgW1.end(), 0.0);
            std::fill(lgb1.begin(), lgb1.end(), 0.0);
            std::fill(lgW2.begin(), lgW2.end(), 0.0);
            std::fill(lgb2.begin(), lgb2.end(), 0.0);
            std::fill(lgW3.begin(), lgW3.end(), 0.0);
            std::fill(lgb3.begin(), lgb3.end(), 0.0);
            double lloss = 0.0;

            const int bs = st->batch_start;
            for (int i = s_off; i < e_off; ++i) {
                int sample = st->idx[bs + i];
                int t = (y[sample] == 1) ? 1 : 0;
                double a1[H1], a2[H2], z, yhat;
                forward_sample(*st->model, X[sample], a1, a2, z, yhat);
                lloss += bce_loss(z, t);
                backward_sample(*st->model, X[sample], a1, a2, yhat, t,
                                lgW1.data(), lgb1.data(),
                                lgW2.data(), lgb2.data(),
                                lgW3.data(), lgb3.data());
            }

            // ── Step 2: mutex-protected reduction ──
            pthread_mutex_lock(&st->mutex);
            for (size_t k = 0; k < st->gW1.size(); ++k) st->gW1[k] += lgW1[k];
            for (size_t k = 0; k < st->gb1.size(); ++k) st->gb1[k] += lgb1[k];
            for (size_t k = 0; k < st->gW2.size(); ++k) st->gW2[k] += lgW2[k];
            for (size_t k = 0; k < st->gb2.size(); ++k) st->gb2[k] += lgb2[k];
            for (size_t k = 0; k < st->gW3.size(); ++k) st->gW3[k] += lgW3[k];
            for (size_t k = 0; k < st->gb3.size(); ++k) st->gb3[k] += lgb3[k];
            st->shared_loss += lloss;
            pthread_mutex_unlock(&st->mutex);

            // ── Step 3: all threads have contributed ──
            pthread_barrier_wait(&st->barrier_acc);

            // ── Step 4: thread 0 applies update ──
            if (a->tid == 0) {
                sgd_momentum_update(*st->model,
                                    st->gW1, st->gb1,
                                    st->gW2, st->gb2,
                                    st->gW3, st->gb3,
                                    lr_t, BATCH);
                st->epoch_loss += st->shared_loss;
                std::fill(st->gW1.begin(), st->gW1.end(), 0.0);
                std::fill(st->gb1.begin(), st->gb1.end(), 0.0);
                std::fill(st->gW2.begin(), st->gW2.end(), 0.0);
                std::fill(st->gb2.begin(), st->gb2.end(), 0.0);
                std::fill(st->gW3.begin(), st->gW3.end(), 0.0);
                std::fill(st->gb3.begin(), st->gb3.end(), 0.0);
                st->shared_loss  = 0.0;
                st->batch_start += BATCH;
            }

            // ── Step 5: weights written; safe to proceed to next batch ──
            pthread_barrier_wait(&st->barrier_upd);
        }

        if (a->tid == 0) {
            std::cout << "epoch " << epoch << "  lr=" << lr_t << "  loss="
                      << (st->epoch_loss / static_cast<double>(n_batches * BATCH))
                      << "\n";
        }
    }

    return nullptr;
}

MLPModel train_parallel_pthreads(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    assert(n > 0 && "train_parallel_pthreads: empty training set");

    MLPModel model;
    init_weights(model, SEED);

    MLPParState st;
    st.X = &X;
    st.y = &y;
    st.n = n;
    st.model = &model;
    st.idx.resize(n);
    std::iota(st.idx.begin(), st.idx.end(), 0);
    st.batch_start = 0;
    st.epoch_loss  = 0.0;
    st.gW1.assign(H1 * N_FEATURES, 0.0); st.gb1.assign(H1, 0.0);
    st.gW2.assign(H2 * H1,         0.0); st.gb2.assign(H2, 0.0);
    st.gW3.assign(OUT * H2,        0.0); st.gb3.assign(OUT, 0.0);
    st.shared_loss = 0.0;
    st.seed = SEED;

    pthread_mutex_init(&st.mutex, nullptr);
    pthread_barrier_init(&st.barrier_acc, nullptr, static_cast<unsigned>(N_THREADS));
    pthread_barrier_init(&st.barrier_upd, nullptr, static_cast<unsigned>(N_THREADS));

    std::vector<MLPTArg>   args(N_THREADS);
    std::vector<pthread_t> threads(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        args[t].st  = &st;
        args[t].tid = t;
        pthread_create(&threads[t], nullptr, mlp_pthread_worker, &args[t]);
    }
    for (int t = 0; t < N_THREADS; ++t) pthread_join(threads[t], nullptr);

    pthread_mutex_destroy(&st.mutex);
    pthread_barrier_destroy(&st.barrier_acc);
    pthread_barrier_destroy(&st.barrier_upd);

    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
// Printing Results
// ─────────────────────────────────────────────────────────────────────────────
static void print_results(const std::string& tag,
                          double train_ms, double infer_ms,
                          const Metrics& m) {
    std::cout << std::left << std::setw(40) << ("[" + tag + "]") << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Training time    : " << std::setw(10) << train_ms << " ms\n";
    std::cout << "  Inference time   : " << std::setw(10) << infer_ms << " ms\n";
    std::cout << "  Total time       : " << std::setw(10) << (train_ms + infer_ms) << " ms\n";
    std::cout << std::setprecision(4);
    std::cout << "  Accuracy         : " << m.acc  << "\n";
    std::cout << "  Precision        : " << m.prec << "\n";
    std::cout << "  Recall           : " << m.rec  << "\n";
    std::cout << "  F1 Score         : " << m.f1   << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string train_csv = "train_cleaned.csv";
    std::string test_csv  = "test_cleaned.csv";
    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv  = argv[2];

    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  MLP — CS:GO Round Winner Classification         ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "Topology: " << N_FEATURES << " -> " << H1 << " -> " << H2
              << " -> " << OUT << " (ReLU, ReLU, sigmoid)\n";
    std::cout << "Config:  epochs=" << MAX_EPOCHS
              << "  batch=" << BATCH
              << "  lr="     << LR
              << "  lambda=" << LAMBDA
              << "  momentum=" << MOMENTUM
              << "  threads=" << N_THREADS << "\n\n";

    // ── Load data ────────────────────────────────────────────────────────────
    double t_load_start = now_ms();
    Dataset ds = load_dataset(train_csv, test_csv);
    double t_load_end = now_ms();
    std::cout << std::fixed << std::setprecision(2)
              << "Data loading: " << (t_load_end - t_load_start) << " ms\n\n";

    // ── Serial MLP (plain SGD) ───────────────────────────────────────────────
    std::cout << "── Training: Serial MLP ──\n";
    double ts0 = now_ms();
    MLPModel serial_model = train_serial(ds.X_train, ds.y_train);
    double ts1 = now_ms();
    IVec serial_preds = predict_serial(serial_model, ds.X_test);
    double ts2 = now_ms();
    Metrics sm = evaluate(ds.y_test, serial_preds);
    print_results("Serial MLP", ts1 - ts0, ts2 - ts1, sm);

    // ── Better Serial MLP (decaying LR + momentum) ───────────────────────────
    std::cout << "── Training: Better Serial MLP ──\n";
    double tb0 = now_ms();
    MLPModel better_model = train_better_serial(ds.X_train, ds.y_train);
    double tb1 = now_ms();
    IVec better_preds = predict_serial(better_model, ds.X_test);
    double tb2 = now_ms();
    Metrics bm = evaluate(ds.y_test, better_preds);
    print_results("Better Serial MLP (decaying LR + momentum)", tb1 - tb0, tb2 - tb1, bm);

    // ── Parallel MLP (OpenMP) ────────────────────────────────────────────────
    std::cout << "── Training: Parallel MLP (OpenMP) ──\n";
    double to0 = now_ms();
    MLPModel omp_model = train_parallel_omp(ds.X_train, ds.y_train);
    double to1 = now_ms();
    IVec omp_preds = predict_parallel(omp_model, ds.X_test);
    double to2 = now_ms();
    Metrics om = evaluate(ds.y_test, omp_preds);
    print_results("Parallel MLP (" + std::to_string(N_THREADS) + " threads, OpenMP)",
                  to1 - to0, to2 - to1, om);

    // ── Parallel MLP (pthreads) ──────────────────────────────────────────────
    std::cout << "── Training: Parallel MLP (pthreads) ──\n";
    double tp0 = now_ms();
    MLPModel pth_model = train_parallel_pthreads(ds.X_train, ds.y_train);
    double tp1 = now_ms();
    IVec pth_preds = predict_parallel(pth_model, ds.X_test);
    double tp2 = now_ms();
    Metrics pm = evaluate(ds.y_test, pth_preds);
    print_results("Parallel MLP (" + std::to_string(N_THREADS) + " threads, pthreads)",
                  tp1 - tp0, tp2 - tp1, pm);

    // ── Speedup summary ──────────────────────────────────────────────────────
    double serial_total        = (ts1 - ts0) + (ts2 - ts1);
    double better_serial_total = (tb1 - tb0) + (tb2 - tb1);
    double omp_total           = (to1 - to0) + (to2 - to1);
    double pth_total           = (tp1 - tp0) + (tp2 - tp1);

    std::cout << "┌─────────────────────────────────────────────────┐\n";
    std::cout << "│  Speedup Analysis                               │\n";
    std::cout << "├─────────────────────────────────────────────────┤\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "│  Serial total           : " << std::setw(10) << serial_total        << " ms\n";
    std::cout << "│  Better Serial total    : " << std::setw(10) << better_serial_total << " ms\n";
    std::cout << "│  Parallel OMP total     : " << std::setw(10) << omp_total           << " ms\n";
    std::cout << "│  Parallel pthreads total: " << std::setw(10) << pth_total           << " ms\n";
    std::cout << "│  Speedup (OMP)          : " << std::setw(10)
              << (better_serial_total / omp_total) << " x\n";
    std::cout << "│  Speedup (pthreads)     : " << std::setw(10)
              << (better_serial_total / pth_total) << " x\n";
    std::cout << "│  Threads used           : " << std::setw(10) << N_THREADS << " \n";
    std::cout << "└─────────────────────────────────────────────────┘\n";
    std::cout << std::setprecision(4);
    std::cout << "Accuracy parity check (|Δacc| should be < 0.01):\n";
    std::cout << "  |better - omp|      = " << std::fabs(bm.acc - om.acc) << "\n";
    std::cout << "  |better - pthreads| = " << std::fabs(bm.acc - pm.acc) << "\n";

    return 0;
}
