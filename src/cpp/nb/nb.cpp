// =============================================================================
// nb.cpp - Serial and Parallel Naive Bayes (Gaussian + Bernoulli hybrid)
// Dataset : train_cleaned.csv / test_cleaned.csv
// Label   : round_winner in {+1, -1}
// Parallel: train_parallel_omp (OpenMP) + train_parallel_pthreads (mutex+join)
// =============================================================================
// g++ -std=c++17 -O3 -march=native -fopenmp nb.cpp -o nb -lpthread
// ./nb train_cleaned.csv test_cleaned.csv
//
// Hybrid NB: per-feature likelihood chosen by is_binary mask.
//   - Binary features (map_*, bomb_planted, auto 0/1): Bernoulli with Laplace α=1
//   - Continuous features (z-normalized):              Gaussian with ε variance floor
//
// Proposal (§ II.A) calls out NB as a data-parallel workload:
// "Naïve Bayes class conditional frequency counting, where no inter-thread
// communication is required during the main computation phase." Both parallel
// variants exploit this: per-thread accumulators over a static slice of the
// training rows, combined at the end.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------
// Hyperparameters (NB-specific; all compile-time — no CLI hyperparams)
// -----------------------------------------------------------------------------
static int              N_THREADS       = 8;     // overridable via N_THREADS env var (read in main)
static constexpr double LAPLACE_ALPHA   = 1.0;   // Bernoulli add-alpha smoothing
static constexpr double VAR_SMOOTHING   = 1e-9;  // added to every Gaussian variance
static constexpr double BIN_THRESHOLD   = 0.5;   // x > 0.5 → on, else off
static constexpr int    SEED            = 42;    // unused by NB; kept for convention

using Labels = std::vector<int>;

// -----------------------------------------------------------------------------
// Dataset / Metrics (layout identical to svm.cpp, knn.cpp, mlp.cpp, dt.cpp)
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

// -----------------------------------------------------------------------------
// NB model: per-class log-prior, per-feature Gaussian (mean, var) or Bernoulli
// (log P(x=1), log P(x=0)) parameters. Indexed j*2 + class_idx; class_idx=0
// for y=-1, class_idx=1 for y=+1.
// -----------------------------------------------------------------------------
struct NBModel {
    int n_features = 0;
    std::vector<unsigned char> is_binary;   // size n_features

    double log_prior[2] = {0.0, 0.0};       // [0]=class(-1), [1]=class(+1)

    // Gaussian parameters (meaningful when !is_binary[j])
    std::vector<double> mean;               // 2 * n_features
    std::vector<double> var_inv_half;       // 2 * n_features; 1 / (2 * sigma^2)
    std::vector<double> log_var_term;       // 2 * n_features; -0.5*log(2π*sigma^2)

    // Bernoulli parameters (meaningful when is_binary[j])
    std::vector<double> log_p_on;           // 2 * n_features; log P(x=1 | y)
    std::vector<double> log_p_off;          // 2 * n_features; log P(x=0 | y)
};

// -----------------------------------------------------------------------------
// Helpers (resolve_path / split_csv / is_* match svm.cpp + knn.cpp + mlp.cpp)
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

// -----------------------------------------------------------------------------
// Dataset loading and preprocessing
//   Same logic as mlp/mlp.cpp: header-based label column, resolve_path,
//   z-score normalization with skip-list for binary / one-hot / auto 0/1.
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
        if (cols[i] == "round_winner") { label_col = i; break; }
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

    // Flag columns to skip during z-score normalization.
    std::vector<unsigned char> skip_normalize(ds.n_features, 0);
    for (int j = 0; j < ds.n_features; ++j) {
        bool skip = is_one_hot_feature(ds.feature_names[j]) ||
                    is_explicit_binary_indicator(ds.feature_names[j]);
        if (!skip) {
            bool binary_01 = true;
            for (int i = 0; i < n_train; ++i) {
                const float* row = &ds.X_train[static_cast<size_t>(i) * ds.n_features];
                if (!is_binary_01_value(static_cast<double>(row[j]))) {
                    binary_01 = false; break;
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
            mean[j] = 0.0; sd[j] = 1.0;
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
// Recompute the is_binary mask from the (already-normalized) training set.
// Detects the same columns that load_dataset skipped for normalization:
// map_*, bomb_planted, and any column whose training values are all in {0, 1}.
// Passthrough columns keep their original 0/1 values post-normalization so
// detection still works after load_dataset's z-score pass.
// -----------------------------------------------------------------------------
static std::vector<unsigned char> detect_binary(const Dataset& ds, int n_rows) {
    std::vector<unsigned char> is_binary(ds.n_features, 0);
    for (int j = 0; j < ds.n_features; ++j) {
        bool skip = is_one_hot_feature(ds.feature_names[j]) ||
                    is_explicit_binary_indicator(ds.feature_names[j]);
        if (!skip) {
            bool binary_01 = true;
            for (int i = 0; i < n_rows; ++i) {
                const float* row = &ds.X_train[static_cast<size_t>(i) * ds.n_features];
                if (!is_binary_01_value(static_cast<double>(row[j]))) {
                    binary_01 = false; break;
                }
            }
            if (binary_01) skip = true;
        }
        is_binary[j] = static_cast<unsigned char>(skip);
    }
    return is_binary;
}

// -----------------------------------------------------------------------------
// Finalize NBModel from raw counts/sums. Used by all three trainers after
// their respective accumulation phase.
//   n_class[c]   - count of training rows in class c
//   count_on[j*2+c] - count of rows with x[j] > BIN_THRESHOLD in class c
//   sum[j*2+c]   - sum of x[j] over rows in class c (Gaussian only)
//   sumsq[j*2+c] - sum of x[j]^2 over rows in class c (Gaussian only)
// -----------------------------------------------------------------------------
static void finalize_model(NBModel& m,
                           const std::vector<unsigned char>& is_binary,
                           int n_rows,
                           int n_features,
                           const long long n_class[2],
                           const std::vector<long long>& count_on,
                           const std::vector<double>& sum,
                           const std::vector<double>& sumsq) {
    m.n_features = n_features;
    m.is_binary = is_binary;
    m.mean.assign(2 * n_features, 0.0);
    m.var_inv_half.assign(2 * n_features, 0.0);
    m.log_var_term.assign(2 * n_features, 0.0);
    m.log_p_on.assign(2 * n_features, 0.0);
    m.log_p_off.assign(2 * n_features, 0.0);

    // Log priors with add-α smoothing to guard against a vanished class.
    const double denom = static_cast<double>(n_rows) + 2.0 * LAPLACE_ALPHA;
    m.log_prior[0] = std::log((static_cast<double>(n_class[0]) + LAPLACE_ALPHA) / denom);
    m.log_prior[1] = std::log((static_cast<double>(n_class[1]) + LAPLACE_ALPHA) / denom);

    constexpr double LOG_2PI = 1.8378770664093454835606594728112;  // log(2π)

    for (int j = 0; j < n_features; ++j) {
        for (int c = 0; c < 2; ++c) {
            const int idx = j * 2 + c;
            const double nc = static_cast<double>(n_class[c]);
            if (is_binary[j]) {
                const double p = (static_cast<double>(count_on[idx]) + LAPLACE_ALPHA)
                               / (nc + 2.0 * LAPLACE_ALPHA);
                m.log_p_on[idx]  = std::log(p);
                m.log_p_off[idx] = std::log(1.0 - p);
            } else {
                const double mean = (nc > 0.0) ? (sum[idx] / nc) : 0.0;
                double var = (nc > 0.0) ? (sumsq[idx] / nc - mean * mean) : 1.0;
                if (var < VAR_SMOOTHING) var = VAR_SMOOTHING;
                m.mean[idx]         = mean;
                m.var_inv_half[idx] = 1.0 / (2.0 * var);
                m.log_var_term[idx] = -0.5 * (LOG_2PI + std::log(var));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Training - Serial (single-pass accumulation)
// -----------------------------------------------------------------------------
static NBModel train_serial(const std::vector<float>& X,
                            const Labels& y,
                            int n_rows,
                            int n_features,
                            const std::vector<unsigned char>& is_binary) {
    NBModel model;
    if (n_rows == 0 || n_features == 0) return model;

    long long n_class[2] = {0, 0};
    std::vector<long long> count_on(2 * n_features, 0);
    std::vector<double>    sum     (2 * n_features, 0.0);
    std::vector<double>    sumsq   (2 * n_features, 0.0);

    for (int i = 0; i < n_rows; ++i) {
        const int c = (y[i] == 1) ? 1 : 0;
        n_class[c] += 1;
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        for (int j = 0; j < n_features; ++j) {
            const double v = static_cast<double>(xi[j]);
            const int idx = j * 2 + c;
            if (is_binary[j]) {
                if (v > BIN_THRESHOLD) count_on[idx] += 1;
            } else {
                sum[idx]   += v;
                sumsq[idx] += v * v;
            }
        }
    }

    finalize_model(model, is_binary, n_rows, n_features, n_class, count_on, sum, sumsq);
    return model;
}

// -----------------------------------------------------------------------------
// Training - Parallel (OpenMP, per-thread accumulator arena + serial reduction)
//   Mirrors svm.cpp / mlp.cpp train_parallel_omp: each thread writes to its
//   own arena slot indexed by omp_get_thread_num(), main thread sums across
//   threads after the parallel region. No critical section, no atomics.
// -----------------------------------------------------------------------------
static NBModel train_parallel_omp(const std::vector<float>& X,
                                  const Labels& y,
                                  int n_rows,
                                  int n_features,
                                  const std::vector<unsigned char>& is_binary) {
    NBModel model;
    if (n_rows == 0 || n_features == 0) return model;

    // Per-thread accumulator arena, allocated once.
    std::vector<std::vector<long long>> l_n_class (N_THREADS, std::vector<long long>(2, 0));
    std::vector<std::vector<long long>> l_count_on(N_THREADS, std::vector<long long>(2 * n_features, 0));
    std::vector<std::vector<double>>    l_sum     (N_THREADS, std::vector<double>   (2 * n_features, 0.0));
    std::vector<std::vector<double>>    l_sumsq   (N_THREADS, std::vector<double>   (2 * n_features, 0.0));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
#endif
    for (int i = 0; i < n_rows; ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const int c = (y[i] == 1) ? 1 : 0;
        l_n_class[tid][c] += 1;
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        for (int j = 0; j < n_features; ++j) {
            const double v = static_cast<double>(xi[j]);
            const int idx = j * 2 + c;
            if (is_binary[j]) {
                if (v > BIN_THRESHOLD) l_count_on[tid][idx] += 1;
            } else {
                l_sum[tid][idx]   += v;
                l_sumsq[tid][idx] += v * v;
            }
        }
    }

    // Serial reduction across thread arena slots.
    long long n_class[2] = {0, 0};
    std::vector<long long> count_on(2 * n_features, 0);
    std::vector<double>    sum     (2 * n_features, 0.0);
    std::vector<double>    sumsq   (2 * n_features, 0.0);
    for (int t = 0; t < N_THREADS; ++t) {
        n_class[0] += l_n_class[t][0];
        n_class[1] += l_n_class[t][1];
        for (int k = 0; k < 2 * n_features; ++k) {
            count_on[k] += l_count_on[t][k];
            sum[k]      += l_sum[t][k];
            sumsq[k]    += l_sumsq[t][k];
        }
    }

    finalize_model(model, is_binary, n_rows, n_features, n_class, count_on, sum, sumsq);
    return model;
}

// -----------------------------------------------------------------------------
// Training - Parallel (pthreads, fixed pool + mutex-protected reduction)
//   Each worker allocates stack-local accumulators, computes over a static
//   slice of n_rows, then acquires a mutex ONCE at the end to merge into
//   shared counters. pthread_join supplies the final barrier. No per-
//   iteration synchronization — matches the proposal's "no inter-thread
//   communication during the main computation phase" claim (§ II.A).
// -----------------------------------------------------------------------------
struct NBParState {
    const std::vector<float>*         X;
    const Labels*                     y;
    int                               n_rows;
    int                               n_features;
    const std::vector<unsigned char>* is_binary;
    // Shared (mutex-protected at merge time):
    long long                         n_class[2];
    std::vector<long long>            count_on;
    std::vector<double>               sum;
    std::vector<double>               sumsq;
    pthread_mutex_t                   mutex;
};

struct NBTArg {
    NBParState* st;
    int         tid;
};

static void* nb_pthread_worker(void* raw) {
    auto* a = static_cast<NBTArg*>(raw);
    auto* st = a->st;
    const auto& X         = *st->X;
    const auto& y         = *st->y;
    const auto& is_binary = *st->is_binary;
    const int n_rows     = st->n_rows;
    const int n_features = st->n_features;

    const int slice = n_rows / N_THREADS;
    const int s_off = a->tid * slice;
    const int e_off = (a->tid == N_THREADS - 1) ? n_rows : (a->tid + 1) * slice;

    // Stack-local accumulators (no sharing during the main compute phase).
    long long l_n_class[2] = {0, 0};
    std::vector<long long> l_count_on(2 * n_features, 0);
    std::vector<double>    l_sum     (2 * n_features, 0.0);
    std::vector<double>    l_sumsq   (2 * n_features, 0.0);

    for (int i = s_off; i < e_off; ++i) {
        const int c = (y[i] == 1) ? 1 : 0;
        l_n_class[c] += 1;
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        for (int j = 0; j < n_features; ++j) {
            const double v = static_cast<double>(xi[j]);
            const int idx = j * 2 + c;
            if (is_binary[j]) {
                if (v > BIN_THRESHOLD) l_count_on[idx] += 1;
            } else {
                l_sum[idx]   += v;
                l_sumsq[idx] += v * v;
            }
        }
    }

    // Single mutex acquisition to merge this thread's locals into shared.
    pthread_mutex_lock(&st->mutex);
    st->n_class[0] += l_n_class[0];
    st->n_class[1] += l_n_class[1];
    for (int k = 0; k < 2 * n_features; ++k) {
        st->count_on[k] += l_count_on[k];
        st->sum[k]      += l_sum[k];
        st->sumsq[k]    += l_sumsq[k];
    }
    pthread_mutex_unlock(&st->mutex);

    return nullptr;
}

static NBModel train_parallel_pthreads(const std::vector<float>& X,
                                       const Labels& y,
                                       int n_rows,
                                       int n_features,
                                       const std::vector<unsigned char>& is_binary) {
    NBModel model;
    if (n_rows == 0 || n_features == 0) return model;

    NBParState st;
    st.X          = &X;
    st.y          = &y;
    st.n_rows     = n_rows;
    st.n_features = n_features;
    st.is_binary  = &is_binary;
    st.n_class[0] = 0;
    st.n_class[1] = 0;
    st.count_on.assign(2 * n_features, 0);
    st.sum.assign     (2 * n_features, 0.0);
    st.sumsq.assign   (2 * n_features, 0.0);

    pthread_mutex_init(&st.mutex, nullptr);

    std::vector<NBTArg>    args(N_THREADS);
    std::vector<pthread_t> threads(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        args[t].st  = &st;
        args[t].tid = t;
        pthread_create(&threads[t], nullptr, nb_pthread_worker, &args[t]);
    }
    for (int t = 0; t < N_THREADS; ++t) pthread_join(threads[t], nullptr);

    pthread_mutex_destroy(&st.mutex);

    finalize_model(model, is_binary, n_rows, n_features,
                   st.n_class, st.count_on, st.sum, st.sumsq);
    return model;
}

// -----------------------------------------------------------------------------
// Inference: log-posterior scoring over both classes, argmax.
// -----------------------------------------------------------------------------
static inline int nb_classify_one(const NBModel& m, const float* xi, int n_features) {
    double score[2] = { m.log_prior[0], m.log_prior[1] };
    for (int j = 0; j < n_features; ++j) {
        const double v = static_cast<double>(xi[j]);
        if (m.is_binary[j]) {
            const bool on = (v > BIN_THRESHOLD);
            score[0] += on ? m.log_p_on[j*2 + 0] : m.log_p_off[j*2 + 0];
            score[1] += on ? m.log_p_on[j*2 + 1] : m.log_p_off[j*2 + 1];
        } else {
            const double d0 = v - m.mean[j*2 + 0];
            const double d1 = v - m.mean[j*2 + 1];
            score[0] += m.log_var_term[j*2 + 0] - d0 * d0 * m.var_inv_half[j*2 + 0];
            score[1] += m.log_var_term[j*2 + 1] - d1 * d1 * m.var_inv_half[j*2 + 1];
        }
    }
    return (score[1] > score[0]) ? 1 : -1;
}

static Labels predict_serial(const NBModel& m,
                             const std::vector<float>& X,
                             int n_rows,
                             int n_features) {
    Labels out(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        out[i] = nb_classify_one(m, xi, n_features);
    }
    return out;
}

static Labels predict_parallel(const NBModel& m,
                               const std::vector<float>& X,
                               int n_rows,
                               int n_features) {
    Labels out(n_rows);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
#endif
    for (int i = 0; i < n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        out[i] = nb_classify_one(m, xi, n_features);
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
// Printing results (plain-text style, matches svm/knn/mlp/dt)
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
    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv  = argv[2];

    if (const char* env = std::getenv("N_THREADS")) {
        int n = std::atoi(env);
        if (n >= 1) N_THREADS = n;
    }

    std::cout << "NB - CS:GO Round Winner Classification\n";
    std::cout << "Config: likelihood=hybrid(Gaussian+Bernoulli)"
              << "  alpha=" << LAPLACE_ALPHA
              << "  var_eps=" << VAR_SMOOTHING
              << "  threads=" << N_THREADS << "\n\n";

    const double t0 = now_ms();
    Dataset ds = load_dataset(train_csv, test_csv);
    const double t1 = now_ms();
    std::cout << std::fixed << std::setprecision(2)
              << "Data loading + normalization: " << (t1 - t0) << " ms\n\n";

    const int n_train = static_cast<int>(ds.y_train.size());
    const int n_test  = static_cast<int>(ds.y_test.size());

    const auto is_binary = detect_binary(ds, n_train);
    int n_binary = 0;
    for (unsigned char b : is_binary) n_binary += b;
    std::cout << "NB features: " << (ds.n_features - n_binary)
              << " Gaussian, " << n_binary << " Bernoulli\n\n";

    // Serial NB
    const double ts0 = now_ms();
    NBModel serial_model = train_serial(ds.X_train, ds.y_train,
                                        n_train, ds.n_features, is_binary);
    const double ts1 = now_ms();
    Labels serial_pred = predict_serial(serial_model, ds.X_test, n_test, ds.n_features);
    const double ts2 = now_ms();
    Metrics sm = evaluate(ds.y_test, serial_pred);
    print_results("Serial NB", ts1 - ts0, ts2 - ts1, sm);

    // Parallel NB (OpenMP)
    const double to0 = now_ms();
    NBModel omp_model = train_parallel_omp(ds.X_train, ds.y_train,
                                           n_train, ds.n_features, is_binary);
    const double to1 = now_ms();
    Labels omp_pred = predict_parallel(omp_model, ds.X_test, n_test, ds.n_features);
    const double to2 = now_ms();
    Metrics om = evaluate(ds.y_test, omp_pred);
    print_results("Parallel NB (OpenMP, " + std::to_string(N_THREADS) + " threads)",
                  to1 - to0, to2 - to1, om);

    // Parallel NB (pthreads)
    const double tp0 = now_ms();
    NBModel pth_model = train_parallel_pthreads(ds.X_train, ds.y_train,
                                                n_train, ds.n_features, is_binary);
    const double tp1 = now_ms();
    Labels pth_pred = predict_parallel(pth_model, ds.X_test, n_test, ds.n_features);
    const double tp2 = now_ms();
    Metrics pm = evaluate(ds.y_test, pth_pred);
    print_results("Parallel NB (pthreads, " + std::to_string(N_THREADS) + " threads)",
                  tp1 - tp0, tp2 - tp1, pm);

    // Speedup summary
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
