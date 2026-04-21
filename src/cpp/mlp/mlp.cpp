// =============================================================================
// mlp.cpp - Serial and Parallel Multilayer Perceptron (BCE + L2 regularization)
// Topology : n_features -> 64 -> 32 -> 1  (ReLU, ReLU, sigmoid)
// Dataset  : train_cleaned.csv / test_cleaned.csv
// Label    : round_winner in {+1, -1}
// =============================================================================
// Pure C++:
//   g++ -std=c++17 -O3 -march=native -fopenmp mlp.cpp -o mlp -lpthread
// OpenBLAS:
//   g++ -std=c++17 -O3 -march=native -fopenmp -DMLP_USE_BLAS mlp.cpp -o mlp -lpthread -lopenblas
// MKL:
//   define MLP_USE_MKL and link with the oneMKL flags for your compiler/runtime
// macOS Accelerate:
//   clang++ -std=c++17 -O3 -DMLP_USE_BLAS mlp.cpp -o mlp -lpthread -framework Accelerate
// ./mlp train_cleaned.csv test_cleaned.csv [epochs] [lr] [lambda]
//
// Mirrors the shared Dataset / load_dataset / Metrics / evaluate / print_results
// conventions in svm/svm.cpp and knn/knn.cpp so a future analytics_engine.cpp
// can drive all five algorithms through one loader.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD 1
#endif
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    unsigned        count;
    unsigned        waiting;
    unsigned        generation;
} pthread_barrier_t;

static int pthread_barrier_init(pthread_barrier_t* barrier,
                                const void*,
                                unsigned count) {
    if (!barrier || count == 0) return -1;
    if (pthread_mutex_init(&barrier->mutex, nullptr) != 0) return -1;
    if (pthread_cond_init(&barrier->cond, nullptr) != 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    barrier->count = count;
    barrier->waiting = 0;
    barrier->generation = 0;
    return 0;
}

static int pthread_barrier_destroy(pthread_barrier_t* barrier) {
    if (!barrier) return -1;
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}

static int pthread_barrier_wait(pthread_barrier_t* barrier) {
    if (!barrier) return -1;
    pthread_mutex_lock(&barrier->mutex);
    const unsigned generation = barrier->generation;
    barrier->waiting++;
    if (barrier->waiting == barrier->count) {
        barrier->waiting = 0;
        barrier->generation++;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    while (generation == barrier->generation) {
        pthread_cond_wait(&barrier->cond, &barrier->mutex);
    }
    pthread_mutex_unlock(&barrier->mutex);
    return 0;
}
#endif

#if defined(MLP_USE_MKL)
#include <mkl.h>
#define MLP_HAS_CBLAS 1
#elif defined(MLP_USE_ACCELERATE) || (defined(MLP_USE_BLAS) && defined(__APPLE__))
#include <Accelerate/Accelerate.h>
#define MLP_HAS_CBLAS 1
#elif defined(MLP_USE_BLAS) || defined(MLP_USE_OPENBLAS) || defined(MLP_USE_CBLAS)
#include <cblas.h>
#define MLP_HAS_CBLAS 1
#else
#define MLP_HAS_CBLAS 0
#endif

// -----------------------------------------------------------------------------
// Hyperparameters (MLP-specific; epochs/lr/lambda also overridable via argv)
// -----------------------------------------------------------------------------
static constexpr int    H1         = 64;    // hidden layer 1 width
static constexpr int    H2         = 32;    // hidden layer 2 width
static constexpr int    OUT        = 1;     // single logit -> sigmoid
static constexpr int    MAX_EPOCHS = 30;    // BCE converges fast vs. hinge
static constexpr int    BATCH      = 128;   // mini-batch size
static constexpr double LR         = 0.01;  // base learning rate
static constexpr double LAMBDA     = 1e-4;  // L2 weight decay
static constexpr double MOMENTUM   = 0.9;   // Polyak momentum
static int              N_THREADS  = 8;     // overridable via N_THREADS env var (read in main)
static constexpr int    SEED       = 42;    // RNG seed
static int              BLAS_PRED_BLOCK = 512;  // overridable via MLP_BLAS_BLOCK env var
#if MLP_HAS_CBLAS
static bool             USE_BLAS_BACKEND = true;   // overridable via MLP_BACKEND=loop
#else
static bool             USE_BLAS_BACKEND = false;
#endif

// BATCH % N_THREADS == 0 is required only by the sample-loop pthreads trainer.
// The BLAS backend routes training through the batch-GEMM path instead.

using Labels = std::vector<int>;

// -----------------------------------------------------------------------------
// Dataset / Metrics (layout identical to svm.cpp and knn.cpp)
// -----------------------------------------------------------------------------
struct Dataset {
    int n_features = 0;
    std::vector<std::string> feature_names;
    std::vector<float> X_train;  // row-major: n_train x n_features
    std::vector<float> X_test;   // row-major: n_test  x n_features
    Labels y_train;
    Labels y_test;
};

struct Metrics {
    double acc  = 0.0;
    double prec = 0.0;
    double rec  = 0.0;
    double f1   = 0.0;
};

struct MLPModel {
    std::vector<double> W1, b1;    // H1 * n_features, H1
    std::vector<double> W2, b2;    // H2 * H1,         H2
    std::vector<double> W3, b3;    // OUT * H2,        OUT
    // Polyak momentum buffers (same shapes as W*/b*)
    std::vector<double> vW1, vb1, vW2, vb2, vW3, vb3;
    int n_features = 0;
};

struct MLPBlasWorkspace {
    std::vector<double> A1;
    std::vector<double> A2;
    std::vector<double> Z3;
    std::vector<double> DZ3;
    std::vector<double> D2;
    std::vector<double> D1;
};

static const char* blas_provider_name() {
#if defined(MLP_USE_MKL)
    return "mkl";
#elif defined(MLP_USE_ACCELERATE) || (defined(MLP_USE_BLAS) && defined(__APPLE__))
    return "accelerate";
#elif defined(MLP_USE_OPENBLAS)
    return "openblas";
#elif defined(MLP_USE_CBLAS) || defined(MLP_USE_BLAS)
    return "cblas";
#else
    return "none";
#endif
}

static const char* active_backend_name() {
#if MLP_HAS_CBLAS
    return USE_BLAS_BACKEND ? "blas-batch" : "sample-loop";
#else
    return "sample-loop";
#endif
}

// -----------------------------------------------------------------------------
// Helpers (resolve_path / split_csv / is_* / parse_* match svm.cpp + knn.cpp)
// -----------------------------------------------------------------------------
static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

static std::string resolve_path(const std::string& path) {
    // Search the path as given, then a handful of common relative locations
    // so binaries work whether run from the repo root, src/, src/cpp/, or
    // src/cpp/<algo>/. Repo layout keeps CSVs at data/ under the repo root.
    static const char* const prefixes[] = {
        "", "../", "data/", "../data/", "../../data/", "../../../data/",
    };
    for (const char* p : prefixes) {
        const std::string candidate = std::string(p) + path;
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return path;
}

static bool has_prefix(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool is_one_hot_feature(const std::string& feature_name) {
    return has_prefix(feature_name, "map_");
}

static bool is_explicit_binary_indicator(const std::string& feature_name) {
    return feature_name == "bomb_planted";
}

static bool is_binary_01_value(double v) {
    return std::abs(v) < 1e-9 || std::abs(v - 1.0) < 1e-9;
}

static int parse_int_or_default(const char* raw, int fallback, const char* name) {
    try {
        return std::stoi(raw);
    } catch (...) {
        std::cerr << "Warning: invalid " << name << "='" << raw << "', using " << fallback << ".\n";
        return fallback;
    }
}

static double parse_double_or_default(const char* raw, double fallback, const char* name) {
    try {
        return std::stod(raw);
    } catch (...) {
        std::cerr << "Warning: invalid " << name << "='" << raw << "', using " << fallback << ".\n";
        return fallback;
    }
}

static inline double sigmoid(double z) noexcept {
    return 1.0 / (1.0 + std::exp(-z));
}

// Numerically stable BCE from logit z and target t in {0,1}:
//   max(z,0) - z*t + log1p(exp(-|z|))
static inline double bce_loss(double z, int t) noexcept {
    double zmax  = (z > 0.0) ? z : 0.0;
    double abs_z = std::fabs(z);
    return zmax - z * static_cast<double>(t) + std::log1p(std::exp(-abs_z));
}

// -----------------------------------------------------------------------------
// Dataset loading and preprocessing
//   - label column found by header name "round_winner"
//   - z-score normalization computed on training set only
//   - one-hot ("map_*"), explicit binary ("bomb_planted"), and auto-detected
//     0/1-only columns are NOT normalized (passthrough)
// -----------------------------------------------------------------------------
static void load_csv_rows(const std::string& path,
                          int label_col,
                          int n_features,
                          std::vector<float>& X,
                          Labels& y) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Error: cannot open file: " << path << "\n";
        std::exit(1);
    }

    std::string line;
    std::getline(file, line);  // skip header
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        int col = 0;
        int feat_count = 0;
        int label = 0;

        while (std::getline(ss, tok, ',')) {
            if (col == label_col) {
                label = std::stoi(tok);
            } else {
                X.push_back(static_cast<float>(std::stod(tok)));
                ++feat_count;
            }
            ++col;
        }

        if (feat_count != n_features) {
            std::cerr << "Error: row has " << feat_count
                      << " features, expected " << n_features << "\n";
            std::exit(1);
        }
        y.push_back(label);
    }
}

static Dataset load_dataset(const std::string& train_path_in,
                            const std::string& test_path_in) {
    Dataset ds;
    const std::string train_path = resolve_path(train_path_in);
    const std::string test_path  = resolve_path(test_path_in);

    std::ifstream train_file(train_path);
    if (!train_file) {
        std::cerr << "Error: cannot open training file: " << train_path << "\n";
        std::exit(1);
    }

    std::string train_header;
    if (!std::getline(train_file, train_header)) {
        std::cerr << "Error: training file is empty: " << train_path << "\n";
        std::exit(1);
    }

    auto cols = split_csv(train_header);
    int label_col = -1;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
        if (cols[i] == "round_winner") {
            label_col = i;
            break;
        }
    }
    if (label_col < 0) {
        std::cerr << "Error: column 'round_winner' not found in training header.\n";
        std::exit(1);
    }

    ds.n_features = static_cast<int>(cols.size()) - 1;
    ds.feature_names.reserve(ds.n_features);
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
        if (i == label_col) continue;
        ds.feature_names.push_back(cols[i]);
    }
    train_file.close();

    load_csv_rows(train_path, label_col, ds.n_features, ds.X_train, ds.y_train);
    load_csv_rows(test_path,  label_col, ds.n_features, ds.X_test,  ds.y_test);

    const int n_train = static_cast<int>(ds.y_train.size());
    const int n_test  = static_cast<int>(ds.y_test.size());

    // Flag columns to skip during z-score normalization
    std::vector<unsigned char> skip_normalize(ds.n_features, 0);
    for (int j = 0; j < ds.n_features; ++j) {
        bool skip = is_one_hot_feature(ds.feature_names[j]) ||
                    is_explicit_binary_indicator(ds.feature_names[j]);
        if (!skip) {
            bool binary_01 = true;
            for (int i = 0; i < n_train; ++i) {
                const float* row = &ds.X_train[static_cast<size_t>(i) * ds.n_features];
                if (!is_binary_01_value(static_cast<double>(row[j]))) {
                    binary_01 = false;
                    break;
                }
            }
            if (binary_01) skip = true;
        }
        skip_normalize[j] = static_cast<unsigned char>(skip);
    }

    std::vector<double> mean(ds.n_features, 0.0);
    std::vector<double> sd(ds.n_features, 0.0);

    for (int i = 0; i < n_train; ++i) {
        const float* row = &ds.X_train[static_cast<size_t>(i) * ds.n_features];
        for (int j = 0; j < ds.n_features; ++j) {
            if (skip_normalize[j]) continue;
            mean[j] += row[j];
        }
    }
    for (int j = 0; j < ds.n_features; ++j) {
        if (skip_normalize[j]) continue;
        mean[j] /= n_train;
    }

    for (int i = 0; i < n_train; ++i) {
        const float* row = &ds.X_train[static_cast<size_t>(i) * ds.n_features];
        for (int j = 0; j < ds.n_features; ++j) {
            if (skip_normalize[j]) continue;
            const double d = static_cast<double>(row[j]) - mean[j];
            sd[j] += d * d;
        }
    }
    for (int j = 0; j < ds.n_features; ++j) {
        if (skip_normalize[j]) {
            mean[j] = 0.0;
            sd[j]   = 1.0;
            continue;
        }
        sd[j] = std::sqrt(sd[j] / n_train);
        if (sd[j] < 1e-9) sd[j] = 1.0;
    }

    auto normalize_inplace = [&](std::vector<float>& X, int n_rows) {
        for (int i = 0; i < n_rows; ++i) {
            float* row = &X[static_cast<size_t>(i) * ds.n_features];
            for (int j = 0; j < ds.n_features; ++j) {
                row[j] = static_cast<float>((row[j] - mean[j]) / sd[j]);
            }
        }
    };
    normalize_inplace(ds.X_train, n_train);
    normalize_inplace(ds.X_test,  n_test);

    int n_skipped = 0;
    for (int j = 0; j < ds.n_features; ++j) n_skipped += skip_normalize[j];
    int tr_pos = 0, tr_neg = 0, te_pos = 0, te_neg = 0;
    for (int l : ds.y_train) (l == 1 ? tr_pos : tr_neg)++;
    for (int l : ds.y_test)  (l == 1 ? te_pos : te_neg)++;

    std::cout << "Dataset Info\n";
    std::cout << "  X_train  : " << n_train << " x " << ds.n_features << "\n";
    std::cout << "  X_test   : " << n_test  << " x " << ds.n_features << "\n";
    std::cout << "  y_train  : +1=" << tr_pos << "  -1=" << tr_neg << "\n";
    std::cout << "  y_test   : +1=" << te_pos << "  -1=" << te_neg << "\n";
    std::cout << "  Normalize: " << (ds.n_features - n_skipped)
              << " z-scored, " << n_skipped << " passthrough (binary/one-hot)\n\n";

    return ds;
}

// -----------------------------------------------------------------------------
// Model initialization
// -----------------------------------------------------------------------------
static void init_weights(MLPModel& m, int n_features, unsigned seed) {
    m.n_features = n_features;
    m.W1.assign(H1  * n_features, 0.0); m.b1.assign(H1,  0.0);
    m.W2.assign(H2  * H1,         0.0); m.b2.assign(H2,  0.0);
    m.W3.assign(OUT * H2,         0.0); m.b3.assign(OUT, 0.0);
    m.vW1.assign(H1  * n_features, 0.0); m.vb1.assign(H1,  0.0);
    m.vW2.assign(H2  * H1,         0.0); m.vb2.assign(H2,  0.0);
    m.vW3.assign(OUT * H2,         0.0); m.vb3.assign(OUT, 0.0);

    std::mt19937 rng(seed);
    // He init (ReLU): N(0, sqrt(2/fan_in))
    std::normal_distribution<double> he1(0.0, std::sqrt(2.0 / n_features));
    std::normal_distribution<double> he2(0.0, std::sqrt(2.0 / H1));
    std::uniform_real_distribution<double> out_u(-0.01, 0.01);

    for (auto& w : m.W1) w = he1(rng);
    for (auto& w : m.W2) w = he2(rng);
    for (auto& w : m.W3) w = out_u(rng);
}

// -----------------------------------------------------------------------------
// Forward and backward (single sample over a row of flat float X)
// -----------------------------------------------------------------------------
static inline void forward_sample(const MLPModel& m, const float* x, int n_features,
                                  double* a1, double* a2,
                                  double& z_out, double& yhat) {
    for (int i = 0; i < H1; ++i) {
        double s = m.b1[i];
        const double* wr = m.W1.data() + static_cast<size_t>(i) * n_features;
        for (int j = 0; j < n_features; ++j) s += wr[j] * static_cast<double>(x[j]);
        a1[i] = (s > 0.0) ? s : 0.0;
    }
    for (int i = 0; i < H2; ++i) {
        double s = m.b2[i];
        const double* wr = m.W2.data() + static_cast<size_t>(i) * H1;
        for (int j = 0; j < H1; ++j) s += wr[j] * a1[j];
        a2[i] = (s > 0.0) ? s : 0.0;
    }
    double z = m.b3[0];
    const double* wr = m.W3.data();   // OUT == 1
    for (int j = 0; j < H2; ++j) z += wr[j] * a2[j];
    z_out = z;
    yhat  = sigmoid(z);
}

// Uses fused sigmoid+BCE derivative: dL/dz = yhat - t.
static inline void backward_sample(const MLPModel& m, const float* x, int n_features,
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
        dz2[j] = (a2[j] > 0.0) ? da : 0.0;
    }
    for (int i = 0; i < H2; ++i) {
        double* gw_row = gW2 + static_cast<size_t>(i) * H1;
        double v = dz2[i];
        for (int j = 0; j < H1; ++j) gw_row[j] += v * a1[j];
        gb2[i] += v;
    }

    double dz1[H1];
    for (int j = 0; j < H1; ++j) {
        double da = 0.0;
        for (int i = 0; i < H2; ++i) da += m.W2[static_cast<size_t>(i) * H1 + j] * dz2[i];
        dz1[j] = (a1[j] > 0.0) ? da : 0.0;
    }
    for (int i = 0; i < H1; ++i) {
        double* gw_row = gW1 + static_cast<size_t>(i) * n_features;
        double v = dz1[i];
        for (int j = 0; j < n_features; ++j) gw_row[j] += v * static_cast<double>(x[j]);
        gb1[i] += v;
    }
}

// SGD + Polyak momentum + L2 decay on weights (not biases)
static void sgd_momentum_update(MLPModel& m,
                                const std::vector<double>& gW1, const std::vector<double>& gb1,
                                const std::vector<double>& gW2, const std::vector<double>& gb2,
                                const std::vector<double>& gW3, const std::vector<double>& gb3,
                                double lambda, double lr, int batch_size) {
    const double inv_b = 1.0 / static_cast<double>(batch_size);
    const double scale = 1.0 - lr * lambda;
    auto step_w = [&](std::vector<double>& w, std::vector<double>& v,
                      const std::vector<double>& g) {
        for (size_t k = 0; k < w.size(); ++k) {
            v[k] = MOMENTUM * v[k] + inv_b * g[k];
            w[k] = scale * w[k] - lr * v[k];
        }
    };
    auto step_b = [&](std::vector<double>& w, std::vector<double>& v,
                      const std::vector<double>& g) {
        for (size_t k = 0; k < w.size(); ++k) {
            v[k] = MOMENTUM * v[k] + inv_b * g[k];
            w[k] -= lr * v[k];
        }
    };
    step_w(m.W1, m.vW1, gW1);  step_b(m.b1, m.vb1, gb1);
    step_w(m.W2, m.vW2, gW2);  step_b(m.b2, m.vb2, gb2);
    step_w(m.W3, m.vW3, gW3);  step_b(m.b3, m.vb3, gb3);
}

#if MLP_HAS_CBLAS
static void resize_blas_workspace(MLPBlasWorkspace& ws, int rows) {
    ws.A1.resize(static_cast<size_t>(rows) * H1);
    ws.A2.resize(static_cast<size_t>(rows) * H2);
    ws.Z3.resize(rows);
    ws.DZ3.resize(rows);
    ws.D2.resize(static_cast<size_t>(rows) * H2);
    ws.D1.resize(static_cast<size_t>(rows) * H1);
}

static void pack_indexed_batch_double(const std::vector<float>& X,
                                      const Labels& y,
                                      const std::vector<int>& idx,
                                      int batch_start,
                                      int rows,
                                      int n_features,
                                      std::vector<double>& Xb,
                                      std::vector<double>& target) {
    Xb.resize(static_cast<size_t>(rows) * n_features);
    target.resize(rows);
    for (int i = 0; i < rows; ++i) {
        const int sample = idx[batch_start + i];
        const float* src = &X[static_cast<size_t>(sample) * n_features];
        double* dst = &Xb[static_cast<size_t>(i) * n_features];
        for (int j = 0; j < n_features; ++j) dst[j] = static_cast<double>(src[j]);
        target[i] = (y[sample] == 1) ? 1.0 : 0.0;
    }
}

static void pack_contiguous_rows_double(const std::vector<float>& X,
                                        int row_begin,
                                        int row_end,
                                        int n_features,
                                        std::vector<double>& Xb) {
    const int rows = row_end - row_begin;
    Xb.resize(static_cast<size_t>(rows) * n_features);
    for (int i = 0; i < rows; ++i) {
        const float* src = &X[static_cast<size_t>(row_begin + i) * n_features];
        double* dst = &Xb[static_cast<size_t>(i) * n_features];
        for (int j = 0; j < n_features; ++j) dst[j] = static_cast<double>(src[j]);
    }
}

static void apply_bias_relu(double* X, int rows, int cols, const std::vector<double>& bias) {
    for (int i = 0; i < rows; ++i) {
        double* row = X + static_cast<size_t>(i) * cols;
        for (int j = 0; j < cols; ++j) {
            const double v = row[j] + bias[j];
            row[j] = (v > 0.0) ? v : 0.0;
        }
    }
}

static void column_sums(const std::vector<double>& X, int rows, int cols, std::vector<double>& out) {
    std::fill(out.begin(), out.end(), 0.0);
    for (int i = 0; i < rows; ++i) {
        const double* row = &X[static_cast<size_t>(i) * cols];
        for (int j = 0; j < cols; ++j) out[j] += row[j];
    }
}

static void forward_batch_blas(const MLPModel& m,
                               const double* Xb,
                               int rows,
                               int n_features,
                               MLPBlasWorkspace& ws) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, H1, n_features,
                1.0,
                Xb, n_features,
                m.W1.data(), n_features,
                0.0,
                ws.A1.data(), H1);
    apply_bias_relu(ws.A1.data(), rows, H1, m.b1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, H2, H1,
                1.0,
                ws.A1.data(), H1,
                m.W2.data(), H1,
                0.0,
                ws.A2.data(), H2);
    apply_bias_relu(ws.A2.data(), rows, H2, m.b2);

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                rows, H2,
                1.0,
                ws.A2.data(), H2,
                m.W3.data(), 1,
                0.0,
                ws.Z3.data(), 1);
    for (int i = 0; i < rows; ++i) ws.Z3[i] += m.b3[0];
}

static double backward_batch_blas(const MLPModel& m,
                                  const double* Xb,
                                  const double* target,
                                  int rows,
                                  int n_features,
                                  MLPBlasWorkspace& ws,
                                  std::vector<double>& gW1,
                                  std::vector<double>& gb1,
                                  std::vector<double>& gW2,
                                  std::vector<double>& gb2,
                                  std::vector<double>& gW3,
                                  std::vector<double>& gb3) {
    if (rows <= 0) return 0.0;

    forward_batch_blas(m, Xb, rows, n_features, ws);

    double loss = 0.0;
    for (int i = 0; i < rows; ++i) {
        loss += bce_loss(ws.Z3[i], static_cast<int>(target[i]));
        ws.DZ3[i] = sigmoid(ws.Z3[i]) - target[i];
    }

    cblas_dgemv(CblasRowMajor, CblasTrans,
                rows, H2,
                1.0,
                ws.A2.data(), H2,
                ws.DZ3.data(), 1,
                0.0,
                gW3.data(), 1);
    gb3[0] = std::accumulate(ws.DZ3.begin(), ws.DZ3.begin() + rows, 0.0);

    for (int i = 0; i < rows; ++i) {
        double* d2 = &ws.D2[static_cast<size_t>(i) * H2];
        const double* a2 = &ws.A2[static_cast<size_t>(i) * H2];
        const double dz3 = ws.DZ3[i];
        for (int j = 0; j < H2; ++j) {
            const double v = dz3 * m.W3[j];
            d2[j] = (a2[j] > 0.0) ? v : 0.0;
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                H2, H1, rows,
                1.0,
                ws.D2.data(), H2,
                ws.A1.data(), H1,
                0.0,
                gW2.data(), H1);
    column_sums(ws.D2, rows, H2, gb2);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, H1, H2,
                1.0,
                ws.D2.data(), H2,
                m.W2.data(), H1,
                0.0,
                ws.D1.data(), H1);
    for (int i = 0; i < rows; ++i) {
        double* d1 = &ws.D1[static_cast<size_t>(i) * H1];
        const double* a1 = &ws.A1[static_cast<size_t>(i) * H1];
        for (int j = 0; j < H1; ++j) {
            if (a1[j] <= 0.0) d1[j] = 0.0;
        }
    }

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                H1, n_features, rows,
                1.0,
                ws.D1.data(), H1,
                Xb, n_features,
                0.0,
                gW1.data(), n_features);
    column_sums(ws.D1, rows, H1, gb1);

    return loss;
}

static void predict_blas_range(const MLPModel& m,
                               const std::vector<float>& X,
                               Labels& out,
                               int row_begin,
                               int row_end,
                               int n_features) {
    const int block = std::max(1, BLAS_PRED_BLOCK);
    std::vector<double> Xb(static_cast<size_t>(block) * n_features);
    MLPBlasWorkspace ws;
    resize_blas_workspace(ws, block);

    for (int r0 = row_begin; r0 < row_end; r0 += block) {
        const int r1 = std::min(r0 + block, row_end);
        const int rows = r1 - r0;
        pack_contiguous_rows_double(X, r0, r1, n_features, Xb);
        forward_batch_blas(m, Xb.data(), rows, n_features, ws);
        for (int i = 0; i < rows; ++i) out[r0 + i] = (ws.Z3[i] >= 0.0) ? 1 : -1;
    }
}

static Labels predict_serial_blas(const MLPModel& m,
                                  const std::vector<float>& X,
                                  int n_rows,
                                  int n_features) {
    Labels out(n_rows);
    predict_blas_range(m, X, out, 0, n_rows, n_features);
    return out;
}

static Labels predict_parallel_blas(const MLPModel& m,
                                    const std::vector<float>& X,
                                    int n_rows,
                                    int n_features) {
    Labels out(n_rows);
#ifdef _OPENMP
    const int block = std::max(1, BLAS_PRED_BLOCK);
    const int n_blocks = (n_rows + block - 1) / block;
#pragma omp parallel num_threads(N_THREADS)
    {
        std::vector<double> Xb(static_cast<size_t>(block) * n_features);
        MLPBlasWorkspace ws;
        resize_blas_workspace(ws, block);

#pragma omp for schedule(static)
        for (int b = 0; b < n_blocks; ++b) {
            const int r0 = b * block;
            const int r1 = std::min(r0 + block, n_rows);
            const int rows = r1 - r0;
            pack_contiguous_rows_double(X, r0, r1, n_features, Xb);
            forward_batch_blas(m, Xb.data(), rows, n_features, ws);
            for (int i = 0; i < rows; ++i) out[r0 + i] = (ws.Z3[i] >= 0.0) ? 1 : -1;
        }
    }
#else
    predict_blas_range(m, X, out, 0, n_rows, n_features);
#endif
    return out;
}

static MLPModel train_serial_blas(const std::vector<float>& X,
                                  const Labels& y,
                                  int n_rows,
                                  int n_features,
                                  int epochs,
                                  double lr,
                                  double lambda) {
    MLPModel model;
    init_weights(model, n_features, SEED);
    if (n_rows == 0 || n_features == 0 || epochs <= 0 || lr <= 0.0) return model;

    std::vector<int> idx(n_rows);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(SEED);

    std::vector<double> batch_X(static_cast<size_t>(BATCH) * n_features);
    std::vector<double> batch_target(BATCH);
    MLPBlasWorkspace ws;
    resize_blas_workspace(ws, BATCH);

    std::vector<double> gW1(H1 * n_features), gb1(H1);
    std::vector<double> gW2(H2 * H1),         gb2(H2);
    std::vector<double> gW3(OUT * H2),        gb3(OUT);

    const int n_batches = n_rows / BATCH;
    for (int e = 0; e < epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        const double lr_t = lr / std::sqrt(static_cast<double>(e + 1));
        double epoch_loss = 0.0;

        for (int bi = 0; bi < n_batches; ++bi) {
            const int bs = bi * BATCH;
            pack_indexed_batch_double(X, y, idx, bs, BATCH, n_features, batch_X, batch_target);
            epoch_loss += backward_batch_blas(model, batch_X.data(), batch_target.data(),
                                              BATCH, n_features, ws,
                                              gW1, gb1, gW2, gb2, gW3, gb3);
            sgd_momentum_update(model, gW1, gb1, gW2, gb2, gW3, gb3,
                                lambda, lr_t, BATCH);
        }

        std::cout << "epoch " << e << "  lr=" << lr_t << "  loss="
                  << (epoch_loss / static_cast<double>(n_batches * BATCH)) << "\n";
    }
    return model;
}
#endif

// -----------------------------------------------------------------------------
// Inference
// -----------------------------------------------------------------------------
static Labels predict_serial(const MLPModel& m,
                             const std::vector<float>& X,
                             int n_rows,
                             int n_features) {
#if MLP_HAS_CBLAS
    if (USE_BLAS_BACKEND) return predict_serial_blas(m, X, n_rows, n_features);
#endif
    Labels out(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        double a1[H1], a2[H2], z, yhat;
        forward_sample(m, xi, n_features, a1, a2, z, yhat);
        out[i] = (yhat >= 0.5) ? 1 : -1;
    }
    return out;
}

static Labels predict_parallel(const MLPModel& m,
                               const std::vector<float>& X,
                               int n_rows,
                               int n_features) {
#if MLP_HAS_CBLAS
    if (USE_BLAS_BACKEND) return predict_parallel_blas(m, X, n_rows, n_features);
#endif
    Labels out(n_rows);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
#endif
    for (int i = 0; i < n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        double a1[H1], a2[H2], z, yhat;
        forward_sample(m, xi, n_features, a1, a2, z, yhat);
        out[i] = (yhat >= 0.5) ? 1 : -1;
    }
    return out;
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------
static Metrics evaluate(const Labels& truth, const Labels& pred) {
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < static_cast<int>(truth.size()); ++i) {
        if      (truth[i] ==  1 && pred[i] ==  1) ++tp;
        else if (truth[i] == -1 && pred[i] ==  1) ++fp;
        else if (truth[i] == -1 && pred[i] == -1) ++tn;
        else                                       ++fn;
    }
    Metrics m;
    m.acc  = static_cast<double>(tp + tn) / static_cast<double>(truth.size());
    m.prec = (tp + fp) ? static_cast<double>(tp) / (tp + fp) : 0.0;
    m.rec  = (tp + fn) ? static_cast<double>(tp) / (tp + fn) : 0.0;
    m.f1   = (m.prec + m.rec) ? (2.0 * m.prec * m.rec / (m.prec + m.rec)) : 0.0;
    return m;
}

// -----------------------------------------------------------------------------
// Training - Serial (mini-batch SGD + Polyak momentum + decaying LR)
//   This is the single serial baseline; the parallel variants parallelize
//   the same algorithm (no extra "better_serial" tier like the old MLP had).
// -----------------------------------------------------------------------------
static MLPModel train_serial(const std::vector<float>& X,
                             const Labels& y,
                             int n_rows,
                             int n_features,
                             int epochs,
                             double lr,
                             double lambda) {
#if MLP_HAS_CBLAS
    if (USE_BLAS_BACKEND) return train_serial_blas(X, y, n_rows, n_features, epochs, lr, lambda);
#endif
    MLPModel model;
    init_weights(model, n_features, SEED);
    if (n_rows == 0 || n_features == 0 || epochs <= 0 || lr <= 0.0) return model;

    std::vector<int> idx(n_rows);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(SEED);

    std::vector<double> gW1(H1  * n_features), gb1(H1);
    std::vector<double> gW2(H2  * H1),         gb2(H2);
    std::vector<double> gW3(OUT * H2),         gb3(OUT);

    const int n_batches = n_rows / BATCH;

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        const double lr_t = lr / std::sqrt(static_cast<double>(e + 1));
        double epoch_loss = 0.0;

        for (int bi = 0; bi < n_batches; ++bi) {
            const int bs = bi * BATCH;
            std::fill(gW1.begin(), gW1.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gW2.begin(), gW2.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);
            std::fill(gW3.begin(), gW3.end(), 0.0);
            std::fill(gb3.begin(), gb3.end(), 0.0);

            for (int i = 0; i < BATCH; ++i) {
                const int s = idx[bs + i];
                const int t = (y[s] == 1) ? 1 : 0;
                const float* xs = &X[static_cast<size_t>(s) * n_features];
                double a1[H1], a2[H2], z, yhat;
                forward_sample(model, xs, n_features, a1, a2, z, yhat);
                epoch_loss += bce_loss(z, t);
                backward_sample(model, xs, n_features, a1, a2, yhat, t,
                                gW1.data(), gb1.data(),
                                gW2.data(), gb2.data(),
                                gW3.data(), gb3.data());
            }

            sgd_momentum_update(model, gW1, gb1, gW2, gb2, gW3, gb3,
                                lambda, lr_t, BATCH);
        }

        std::cout << "epoch " << e << "  lr=" << lr_t << "  loss="
                  << (epoch_loss / static_cast<double>(n_batches * BATCH)) << "\n";
    }
    return model;
}

// -----------------------------------------------------------------------------
// Training - Parallel (OpenMP, per-thread gradient buffers + serial reduction)
//   Same pattern as svm.cpp train_parallel: no critical section, no mutex.
//   #pragma omp parallel for dispatches mini-batch samples; each thread
//   accumulates into its own gradient buffer (indexed by tid); the primary
//   thread then serially reduces across threads and applies the SGD update.
// -----------------------------------------------------------------------------
static MLPModel train_parallel_omp(const std::vector<float>& X,
                                   const Labels& y,
                                   int n_rows,
                                   int n_features,
                                   int epochs,
                                   double lr,
                                   double lambda) {
#if MLP_HAS_CBLAS
    // For this 128-row mini-batch regime, slicing the batch across outer
    // threads leaves each worker with panels that are too small to amortize
    // BLAS call overhead. Reuse the batch-GEMM trainer instead.
    if (USE_BLAS_BACKEND) return train_serial_blas(X, y, n_rows, n_features, epochs, lr, lambda);
#endif
    MLPModel model;
    init_weights(model, n_features, SEED);
    if (n_rows == 0 || n_features == 0 || epochs <= 0 || lr <= 0.0) return model;

    std::vector<int> idx(n_rows);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(SEED);

    std::vector<double> gW1(H1  * n_features), gb1(H1);
    std::vector<double> gW2(H2  * H1),         gb2(H2);
    std::vector<double> gW3(OUT * H2),         gb3(OUT);

    // Per-thread gradient buffers (allocated once per trainer call)
    std::vector<std::vector<double>> lgW1(N_THREADS, std::vector<double>(H1  * n_features));
    std::vector<std::vector<double>> lgb1(N_THREADS, std::vector<double>(H1));
    std::vector<std::vector<double>> lgW2(N_THREADS, std::vector<double>(H2  * H1));
    std::vector<std::vector<double>> lgb2(N_THREADS, std::vector<double>(H2));
    std::vector<std::vector<double>> lgW3(N_THREADS, std::vector<double>(OUT * H2));
    std::vector<std::vector<double>> lgb3(N_THREADS, std::vector<double>(OUT));
    std::vector<double> lloss(N_THREADS, 0.0);

    const int n_batches = n_rows / BATCH;

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        const double lr_t = lr / std::sqrt(static_cast<double>(e + 1));
        double epoch_loss = 0.0;

        for (int bi = 0; bi < n_batches; ++bi) {
            const int bs = bi * BATCH;

            for (int t = 0; t < N_THREADS; ++t) {
                std::fill(lgW1[t].begin(), lgW1[t].end(), 0.0);
                std::fill(lgb1[t].begin(), lgb1[t].end(), 0.0);
                std::fill(lgW2[t].begin(), lgW2[t].end(), 0.0);
                std::fill(lgb2[t].begin(), lgb2[t].end(), 0.0);
                std::fill(lgW3[t].begin(), lgW3[t].end(), 0.0);
                std::fill(lgb3[t].begin(), lgb3[t].end(), 0.0);
                lloss[t] = 0.0;
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
#endif
            for (int i = 0; i < BATCH; ++i) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                const int s = idx[bs + i];
                const int t = (y[s] == 1) ? 1 : 0;
                const float* xs = &X[static_cast<size_t>(s) * n_features];
                double a1[H1], a2[H2], z, yhat;
                forward_sample(model, xs, n_features, a1, a2, z, yhat);
                lloss[tid] += bce_loss(z, t);
                backward_sample(model, xs, n_features, a1, a2, yhat, t,
                                lgW1[tid].data(), lgb1[tid].data(),
                                lgW2[tid].data(), lgb2[tid].data(),
                                lgW3[tid].data(), lgb3[tid].data());
            }

            // Serial reduction (same pattern as svm.cpp train_parallel)
            std::fill(gW1.begin(), gW1.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gW2.begin(), gW2.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);
            std::fill(gW3.begin(), gW3.end(), 0.0);
            std::fill(gb3.begin(), gb3.end(), 0.0);
            for (int t = 0; t < N_THREADS; ++t) {
                for (size_t k = 0; k < gW1.size(); ++k) gW1[k] += lgW1[t][k];
                for (size_t k = 0; k < gb1.size(); ++k) gb1[k] += lgb1[t][k];
                for (size_t k = 0; k < gW2.size(); ++k) gW2[k] += lgW2[t][k];
                for (size_t k = 0; k < gb2.size(); ++k) gb2[k] += lgb2[t][k];
                for (size_t k = 0; k < gW3.size(); ++k) gW3[k] += lgW3[t][k];
                for (size_t k = 0; k < gb3.size(); ++k) gb3[k] += lgb3[t][k];
                epoch_loss += lloss[t];
            }

            sgd_momentum_update(model, gW1, gb1, gW2, gb2, gW3, gb3,
                                lambda, lr_t, BATCH);
        }

        std::cout << "epoch " << e << "  lr=" << lr_t << "  loss="
                  << (epoch_loss / static_cast<double>(n_batches * BATCH)) << "\n";
    }
    return model;
}

// -----------------------------------------------------------------------------
// Training - Parallel (pthreads, MLP-specific demonstration of fine-grained
// synchronization). Mutex + two barriers per mini-batch. Kept alongside the
// OpenMP variant so the writeup can compare two parallel implementations of
// the same algorithm.
// -----------------------------------------------------------------------------
struct MLPParState {
    const std::vector<float>* X;
    const Labels*             y;
    int                       n_rows;
    int                       n_features;
    MLPModel*                 model;
    std::vector<int>          idx;
    int                       batch_start;
    double                    epoch_loss;
    std::vector<double>       gW1, gb1, gW2, gb2, gW3, gb3;
    double                    shared_loss;
    double                    lr;
    double                    lambda;
    pthread_mutex_t           mutex;
    pthread_barrier_t         barrier_acc;
    pthread_barrier_t         barrier_upd;
    unsigned                  seed;
    int                       epochs;
};

struct MLPTArg {
    MLPParState* st;
    int          tid;
};

static void* mlp_pthread_worker(void* raw) {
    auto*   a  = static_cast<MLPTArg*>(raw);
    auto*   st = a->st;
    const std::vector<float>& X = *st->X;
    const Labels&             y = *st->y;
    const int n_features = st->n_features;

    std::vector<double> lgW1(H1  * n_features), lgb1(H1);
    std::vector<double> lgW2(H2  * H1),         lgb2(H2);
    std::vector<double> lgW3(OUT * H2),         lgb3(OUT);

    const int slice = BATCH / N_THREADS;
    const int s_off = a->tid * slice;
    const int e_off = (a->tid == N_THREADS - 1) ? BATCH : (a->tid + 1) * slice;

    const int n_batches = st->n_rows / BATCH;

    for (int epoch = 0; epoch < st->epochs; ++epoch) {

        // Epoch housekeeping: thread 0 reshuffles + resets counters
        if (a->tid == 0) {
            std::mt19937 rng(st->seed + static_cast<unsigned>(epoch));
            std::shuffle(st->idx.begin(), st->idx.end(), rng);
            st->batch_start = 0;
            st->epoch_loss  = 0.0;
        }
        pthread_barrier_wait(&st->barrier_upd);

        const double lr_t = st->lr / std::sqrt(static_cast<double>(epoch + 1));

        for (int bi = 0; bi < n_batches; ++bi) {

            // Step 1: local forward+backward on this thread's slice
            std::fill(lgW1.begin(), lgW1.end(), 0.0);
            std::fill(lgb1.begin(), lgb1.end(), 0.0);
            std::fill(lgW2.begin(), lgW2.end(), 0.0);
            std::fill(lgb2.begin(), lgb2.end(), 0.0);
            std::fill(lgW3.begin(), lgW3.end(), 0.0);
            std::fill(lgb3.begin(), lgb3.end(), 0.0);
            double lloss = 0.0;

            const int bs = st->batch_start;
            for (int i = s_off; i < e_off; ++i) {
                const int sample = st->idx[bs + i];
                const int t = (y[sample] == 1) ? 1 : 0;
                const float* xs = &X[static_cast<size_t>(sample) * n_features];
                double a1[H1], a2[H2], z, yhat;
                forward_sample(*st->model, xs, n_features, a1, a2, z, yhat);
                lloss += bce_loss(z, t);
                backward_sample(*st->model, xs, n_features, a1, a2, yhat, t,
                                lgW1.data(), lgb1.data(),
                                lgW2.data(), lgb2.data(),
                                lgW3.data(), lgb3.data());
            }

            // Step 2: mutex-protected reduction
            pthread_mutex_lock(&st->mutex);
            for (size_t k = 0; k < st->gW1.size(); ++k) st->gW1[k] += lgW1[k];
            for (size_t k = 0; k < st->gb1.size(); ++k) st->gb1[k] += lgb1[k];
            for (size_t k = 0; k < st->gW2.size(); ++k) st->gW2[k] += lgW2[k];
            for (size_t k = 0; k < st->gb2.size(); ++k) st->gb2[k] += lgb2[k];
            for (size_t k = 0; k < st->gW3.size(); ++k) st->gW3[k] += lgW3[k];
            for (size_t k = 0; k < st->gb3.size(); ++k) st->gb3[k] += lgb3[k];
            st->shared_loss += lloss;
            pthread_mutex_unlock(&st->mutex);

            // Step 3: all threads have contributed
            pthread_barrier_wait(&st->barrier_acc);

            // Step 4: thread 0 applies update
            if (a->tid == 0) {
                sgd_momentum_update(*st->model,
                                    st->gW1, st->gb1,
                                    st->gW2, st->gb2,
                                    st->gW3, st->gb3,
                                    st->lambda, lr_t, BATCH);
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

            // Step 5: weights written; safe to proceed
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

static MLPModel train_parallel_pthreads(const std::vector<float>& X,
                                        const Labels& y,
                                        int n_rows,
                                        int n_features,
                                        int epochs,
                                        double lr,
                                        double lambda) {
#if MLP_HAS_CBLAS
    // Same reasoning as train_parallel_omp(): the BLAS backend wants the full
    // 128-row mini-batch, not N_THREADS tiny slices plus extra barriers.
    if (USE_BLAS_BACKEND) return train_serial_blas(X, y, n_rows, n_features, epochs, lr, lambda);
#endif
    MLPModel model;
    init_weights(model, n_features, SEED);
    if (n_rows == 0 || n_features == 0 || epochs <= 0 || lr <= 0.0) return model;

    MLPParState st;
    st.X          = &X;
    st.y          = &y;
    st.n_rows     = n_rows;
    st.n_features = n_features;
    st.model      = &model;
    st.idx.resize(n_rows);
    std::iota(st.idx.begin(), st.idx.end(), 0);
    st.batch_start = 0;
    st.epoch_loss  = 0.0;
    st.gW1.assign(H1  * n_features, 0.0); st.gb1.assign(H1,  0.0);
    st.gW2.assign(H2  * H1,         0.0); st.gb2.assign(H2,  0.0);
    st.gW3.assign(OUT * H2,         0.0); st.gb3.assign(OUT, 0.0);
    st.shared_loss = 0.0;
    st.lr          = lr;
    st.lambda      = lambda;
    st.seed        = SEED;
    st.epochs      = epochs;

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

// -----------------------------------------------------------------------------
// Printing results (plain-text style, matches svm.cpp + knn.cpp)
// -----------------------------------------------------------------------------
static void print_results(const std::string& tag,
                          double train_ms,
                          double infer_ms,
                          const Metrics& m) {
    std::cout << "[" << tag << "]\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Training time  : " << train_ms << " ms\n";
    std::cout << "  Inference time : " << infer_ms << " ms\n";
    std::cout << "  Total time     : " << (train_ms + infer_ms) << " ms\n";
    std::cout << std::setprecision(4);
    std::cout << "  Accuracy       : " << m.acc  << "\n";
    std::cout << "  Precision      : " << m.prec << "\n";
    std::cout << "  Recall         : " << m.rec  << "\n";
    std::cout << "  F1 Score       : " << m.f1   << "\n\n";
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string train_csv = "data/train_cleaned.csv";
    std::string test_csv  = "data/test_cleaned.csv";
    int    epochs = MAX_EPOCHS;
    double lr     = LR;
    double lambda = LAMBDA;

    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv  = argv[2];
    if (argc > 3) epochs = std::max(0,   parse_int_or_default   (argv[3], MAX_EPOCHS, "epochs"));
    if (argc > 4) lr     = std::max(0.0, parse_double_or_default(argv[4], LR,         "lr"));
    if (argc > 5) lambda = std::max(0.0, parse_double_or_default(argv[5], LAMBDA,     "lambda"));

    if (const char* env = std::getenv("N_THREADS")) {
        int n = std::atoi(env);
        if (n >= 1) N_THREADS = n;
    }
    if (const char* env = std::getenv("MLP_BLAS_BLOCK")) {
        int n = std::atoi(env);
        if (n >= 1) BLAS_PRED_BLOCK = n;
    }
    if (const char* env = std::getenv("MLP_BACKEND")) {
        const std::string backend(env);
        if (backend == "loop" || backend == "cpp" || backend == "pure") {
            USE_BLAS_BACKEND = false;
        } else if (backend == "blas" || backend == "blas-batch" || backend == "gemm") {
#if MLP_HAS_CBLAS
            USE_BLAS_BACKEND = true;
#else
            USE_BLAS_BACKEND = false;
            std::cerr << "Warning: MLP_BACKEND=blas requested, but this binary "
                      << "was not compiled with MLP_USE_BLAS; using sample-loop.\n";
#endif
        }
    }
    if (!USE_BLAS_BACKEND && BATCH % N_THREADS != 0) {
        std::cerr << "Error: BATCH (" << BATCH << ") must be divisible by N_THREADS ("
                  << N_THREADS << "). The pthreads trainer slices BATCH evenly.\n";
        return 1;
    }

    std::cout << "MLP - CS:GO Round Winner Classification\n";
    std::cout << "Topology: n_features -> " << H1 << " -> " << H2
              << " -> " << OUT << " (ReLU, ReLU, sigmoid)\n";
    std::cout << "Config: epochs=" << epochs
              << "  batch=" << BATCH
              << "  lr=" << lr
              << "  lambda=" << lambda
              << "  momentum=" << MOMENTUM
              << "  threads=" << N_THREADS
              << "  backend=" << active_backend_name();
#if MLP_HAS_CBLAS
    std::cout << "  blas_provider=" << blas_provider_name()
              << "  blas_block=" << BLAS_PRED_BLOCK;
#endif
    std::cout << "\n\n";

    const double t0 = now_ms();
    Dataset ds = load_dataset(train_csv, test_csv);
    const double t1 = now_ms();
    std::cout << std::fixed << std::setprecision(2)
              << "Data loading + normalization: " << (t1 - t0) << " ms\n\n";

    const int n_train = static_cast<int>(ds.y_train.size());
    const int n_test  = static_cast<int>(ds.y_test.size());

    // Serial MLP (mini-batch SGD + momentum + decaying LR)
    const double ts0 = now_ms();
    MLPModel serial_model = train_serial(ds.X_train, ds.y_train,
                                         n_train, ds.n_features,
                                         epochs, lr, lambda);
    const double ts1 = now_ms();
    Labels serial_pred = predict_serial(serial_model, ds.X_test, n_test, ds.n_features);
    const double ts2 = now_ms();
    Metrics sm = evaluate(ds.y_test, serial_pred);
    print_results("Serial MLP", ts1 - ts0, ts2 - ts1, sm);

    // Parallel MLP (OpenMP, per-thread buffer + serial reduction)
    const double to0 = now_ms();
    MLPModel omp_model = train_parallel_omp(ds.X_train, ds.y_train,
                                            n_train, ds.n_features,
                                            epochs, lr, lambda);
    const double to1 = now_ms();
    Labels omp_pred = predict_parallel(omp_model, ds.X_test, n_test, ds.n_features);
    const double to2 = now_ms();
    Metrics om = evaluate(ds.y_test, omp_pred);
    print_results("Parallel MLP (OpenMP, " + std::to_string(N_THREADS) + " threads)",
                  to1 - to0, to2 - to1, om);

    // Parallel MLP (pthreads, mutex + barriers)
    const double tp0 = now_ms();
    MLPModel pth_model = train_parallel_pthreads(ds.X_train, ds.y_train,
                                                 n_train, ds.n_features,
                                                 epochs, lr, lambda);
    const double tp1 = now_ms();
    Labels pth_pred = predict_parallel(pth_model, ds.X_test, n_test, ds.n_features);
    const double tp2 = now_ms();
    Metrics pm = evaluate(ds.y_test, pth_pred);
    print_results("Parallel MLP (pthreads, " + std::to_string(N_THREADS) + " threads)",
                  tp1 - tp0, tp2 - tp1, pm);

    // Speedup summary (plain text, matches svm.cpp / knn.cpp)
    const double serial_total = (ts1 - ts0) + (ts2 - ts1);
    const double omp_total    = (to1 - to0) + (to2 - to1);
    const double pth_total    = (tp1 - tp0) + (tp2 - tp1);

    std::cout << "Speedup Summary\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Serial total       : " << serial_total << " ms\n";
    std::cout << "  Parallel OMP total : " << omp_total    << " ms\n";
    std::cout << "  Parallel pth total : " << pth_total    << " ms\n";
    std::cout << "  Speedup (OMP)      : "
              << (omp_total > 0.0 ? serial_total / omp_total : 0.0) << " x\n";
    std::cout << "  Speedup (pthreads) : "
              << (pth_total > 0.0 ? serial_total / pth_total : 0.0) << " x\n";
    std::cout << std::setprecision(4);
    std::cout << "Accuracy parity check (|delta-acc| should be < 0.01):\n";
    std::cout << "  |serial - omp|      = " << std::fabs(sm.acc - om.acc) << "\n";
    std::cout << "  |serial - pthreads| = " << std::fabs(sm.acc - pm.acc) << "\n";

    return 0;
}
