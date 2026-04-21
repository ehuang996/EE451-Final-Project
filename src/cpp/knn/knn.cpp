// =============================================================================
// knn.cpp - Serial and Parallel KNN
// Dataset : train_cleaned.csv / test_cleaned.csv
// Label   : round_winner in {+1, -1}
// Parallel: predict_parallel_omp (OpenMP) + predict_parallel_pthreads (pool)
//          (training stores data + row norms; parallelism is in inference.)
// Kernel  : exact brute force backed by vendor CBLAS SGEMM.
// =============================================================================
// OpenBLAS:
//   g++ -std=c++17 -O3 -march=native -fopenmp -DKNN_USE_BLAS knn.cpp -o knn -lpthread -lopenblas
// MKL:
//   define KNN_USE_MKL and link with the oneMKL flags for your compiler/runtime
// macOS Accelerate:
//   clang++ -std=c++17 -O3 -DKNN_USE_BLAS knn.cpp -o knn -lpthread -framework Accelerate
// ./knn train_cleaned.csv test_cleaned.csv [k]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(KNN_USE_MKL)
#include <mkl.h>
#define KNN_HAS_CBLAS 1
#elif defined(KNN_USE_ACCELERATE) || (defined(KNN_USE_BLAS) && defined(__APPLE__))
#include <Accelerate/Accelerate.h>
#define KNN_HAS_CBLAS 1
#elif defined(KNN_USE_BLAS) || defined(KNN_USE_OPENBLAS) || defined(KNN_USE_CBLAS)
#include <cblas.h>
#define KNN_HAS_CBLAS 1
#else
#define KNN_HAS_CBLAS 0
#endif

#if !KNN_HAS_CBLAS
#error "KNN now requires a BLAS backend. Compile with -DKNN_USE_BLAS, -DKNN_USE_MKL, or the platform equivalent."
#endif

static constexpr int K_NEIGHBORS = 11;
static int N_THREADS = 8;  // overridable via N_THREADS env var (read in main)
static int BLAS_QUERY_BLOCK = 256;  // overridable via KNN_BLAS_BLOCK env var

using Labels = std::vector<int>;

struct Dataset {
    int n_features = 0;
    std::vector<std::string> feature_names;
    std::vector<float> X_train;  // row-major: n_train x n_features
    std::vector<float> X_test;   // row-major: n_test x n_features
    Labels y_train;
    Labels y_test;
};

struct Metrics {
    double acc = 0.0;
    double prec = 0.0;
    double rec = 0.0;
    double f1 = 0.0;
};

struct KNNModel {
    std::vector<float> X;
    std::vector<float> norm2;
    Labels y;
    int n_rows = 0;
    int n_features = 0;
    int k = K_NEIGHBORS;
};

static const char* blas_provider_name() {
#if defined(KNN_USE_MKL)
    return "mkl";
#elif defined(KNN_USE_ACCELERATE) || (defined(KNN_USE_BLAS) && defined(__APPLE__))
    return "accelerate";
#elif defined(KNN_USE_OPENBLAS)
    return "openblas";
#elif defined(KNN_USE_CBLAS) || defined(KNN_USE_BLAS)
    return "cblas";
#else
    return "none";
#endif
}

static const char* active_backend_name() {
    return "blas-sgemm";
}

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
    const std::string test_path = resolve_path(test_path_in);

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
    load_csv_rows(test_path, label_col, ds.n_features, ds.X_test, ds.y_test);

    const int n_train = static_cast<int>(ds.y_train.size());
    const int n_test = static_cast<int>(ds.y_test.size());

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
            sd[j] = 1.0;
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
    normalize_inplace(ds.X_test, n_test);

    int tr_pos = 0, tr_neg = 0, te_pos = 0, te_neg = 0;
    for (int l : ds.y_train) (l == 1 ? tr_pos : tr_neg)++;
    for (int l : ds.y_test) (l == 1 ? te_pos : te_neg)++;

    std::cout << "Dataset Info\n";
    std::cout << "  X_train: " << n_train << " x " << ds.n_features << "\n";
    std::cout << "  X_test : " << n_test << " x " << ds.n_features << "\n";
    std::cout << "  y_train: +1=" << tr_pos << "  -1=" << tr_neg << "\n";
    std::cout << "  y_test : +1=" << te_pos << "  -1=" << te_neg << "\n\n";

    return ds;
}

static std::vector<float> row_norms(const std::vector<float>& X, int n_rows, int n_features) {
    std::vector<float> out(n_rows, 0.0f);
    for (int i = 0; i < n_rows; ++i) {
        const float* row = &X[static_cast<size_t>(i) * n_features];
        float sum = 0.0f;
        for (int j = 0; j < n_features; ++j) sum += row[j] * row[j];
        out[i] = sum;
    }
    return out;
}

static KNNModel make_model(const std::vector<float>& X,
                           const Labels& y,
                           int n_features,
                           int k) {
    KNNModel m;
    m.X = X;
    m.norm2 = row_norms(m.X, static_cast<int>(y.size()), n_features);
    m.y = y;
    m.n_rows = static_cast<int>(y.size());
    m.n_features = n_features;
    m.k = std::min(std::max(k, 1), m.n_rows);
    return m;
}

static KNNModel train_serial(const std::vector<float>& X,
                             const Labels& y,
                             int n_features,
                             int k) {
    return make_model(X, y, n_features, k);
}

static KNNModel train_parallel(const std::vector<float>& X,
                               const Labels& y,
                               int n_features,
                               int k) {
    return make_model(X, y, n_features, k);
}

static void finalize_block_predictions(const int q_count,
                                       const int k,
                                       const std::vector<float>& best_dist,
                                       const std::vector<int>& best_label,
                                       Labels& pred,
                                       const int q_begin) {
    for (int q = 0; q < q_count; ++q) {
        const int off = q * k;
        int pos = 0;
        int neg = 0;
        int nearest_label = 1;
        float nearest_dist = std::numeric_limits<float>::infinity();

        for (int t = 0; t < k; ++t) {
            const int label = best_label[off + t];
            if (label == 1)
                ++pos;
            else
                ++neg;

            const float d = best_dist[off + t];
            if (d < nearest_dist) {
                nearest_dist = d;
                nearest_label = label;
            }
        }

        if (pos > neg)
            pred[q_begin + q] = 1;
        else if (neg > pos)
            pred[q_begin + q] = -1;
        else
            pred[q_begin + q] = nearest_label;
    }
}

static void predict_blas_block(const KNNModel& model,
                               const std::vector<float>& X_test,
                               const std::vector<float>& test_norm2,
                               Labels& pred,
                               int q_begin,
                               int q_end,
                               std::vector<float>& dots,
                               std::vector<float>& best_dist,
                               std::vector<int>& best_label,
                               std::vector<int>& max_idx,
                               std::vector<float>& max_dist) {
    const int q_count = q_end - q_begin;
    const int n_train = model.n_rows;
    const int n_features = model.n_features;
    const int k = model.k;
    if (q_count <= 0) return;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                q_count, n_train, n_features,
                1.0f,
                &X_test[static_cast<size_t>(q_begin) * n_features], n_features,
                model.X.data(), n_features,
                0.0f,
                dots.data(), n_train);

    const int slots = q_count * k;
    std::fill(best_dist.begin(), best_dist.begin() + slots,
              std::numeric_limits<float>::infinity());
    std::fill(best_label.begin(), best_label.begin() + slots, 0);
    std::fill(max_idx.begin(), max_idx.begin() + q_count, 0);
    std::fill(max_dist.begin(), max_dist.begin() + q_count,
              std::numeric_limits<float>::infinity());

    for (int q = 0; q < q_count; ++q) {
        const int off = q * k;
        const float q_norm = test_norm2[q_begin + q];
        const float* dot_row = &dots[static_cast<size_t>(q) * n_train];

        for (int i = 0; i < n_train; ++i) {
            const float d = q_norm + model.norm2[i] - 2.0f * dot_row[i];
            if (d < max_dist[q]) {
                const int replace = max_idx[q];
                best_dist[off + replace] = d;
                best_label[off + replace] = model.y[i];

                int next_idx = 0;
                float next_dist = best_dist[off];
                for (int t = 1; t < k; ++t) {
                    const float candidate = best_dist[off + t];
                    if (candidate > next_dist) {
                        next_dist = candidate;
                        next_idx = t;
                    }
                }
                max_idx[q] = next_idx;
                max_dist[q] = next_dist;
            }
        }
    }

    finalize_block_predictions(q_count, k, best_dist, best_label, pred, q_begin);
}

static void predict_blas_range(const KNNModel& model,
                               const std::vector<float>& X_test,
                               const std::vector<float>& test_norm2,
                               Labels& pred,
                               int q_begin,
                               int q_end) {
    const int k = model.k;
    const int block = std::max(1, BLAS_QUERY_BLOCK);
    std::vector<float> dots(static_cast<size_t>(block) * model.n_rows);
    std::vector<float> best_dist(static_cast<size_t>(block) * k);
    std::vector<int> best_label(static_cast<size_t>(block) * k);
    std::vector<int> max_idx(block);
    std::vector<float> max_dist(block);

    for (int q0 = q_begin; q0 < q_end; q0 += block) {
        const int q1 = std::min(q0 + block, q_end);
        predict_blas_block(model, X_test, test_norm2, pred, q0, q1,
                           dots, best_dist, best_label, max_idx, max_dist);
    }
}

static Labels predict_blas_serial(const KNNModel& model,
                                  const std::vector<float>& X_test,
                                  int n_test) {
    Labels pred(n_test);
    const std::vector<float> test_norm2 = row_norms(X_test, n_test, model.n_features);
    predict_blas_range(model, X_test, test_norm2, pred, 0, n_test);
    return pred;
}

static Labels predict_blas_parallel_omp(const KNNModel& model,
                                        const std::vector<float>& X_test,
                                        int n_test) {
    Labels pred(n_test);
    const std::vector<float> test_norm2 = row_norms(X_test, n_test, model.n_features);
#ifdef _OPENMP
    const int block = std::max(1, BLAS_QUERY_BLOCK);
    const int n_blocks = (n_test + block - 1) / block;
#pragma omp parallel num_threads(N_THREADS)
    {
        const int k = model.k;
        std::vector<float> dots(static_cast<size_t>(block) * model.n_rows);
        std::vector<float> best_dist(static_cast<size_t>(block) * k);
        std::vector<int> best_label(static_cast<size_t>(block) * k);
        std::vector<int> max_idx(block);
        std::vector<float> max_dist(block);

#pragma omp for schedule(static)
        for (int b = 0; b < n_blocks; ++b) {
            const int q0 = b * block;
            const int q1 = std::min(q0 + block, n_test);
            predict_blas_block(model, X_test, test_norm2, pred, q0, q1,
                               dots, best_dist, best_label, max_idx, max_dist);
        }
    }
#else
    predict_blas_range(model, X_test, test_norm2, pred, 0, n_test);
#endif
    return pred;
}

struct KNNBlasPredArg {
    const KNNModel*           model;
    const std::vector<float>* X_test;
    const std::vector<float>* test_norm2;
    Labels*                   pred;
    int                       s_off;
    int                       e_off;
};

static void* knn_blas_pthread_worker(void* raw) {
    auto* a = static_cast<KNNBlasPredArg*>(raw);
    predict_blas_range(*a->model, *a->X_test, *a->test_norm2, *a->pred,
                       a->s_off, a->e_off);
    return nullptr;
}

static Labels predict_blas_parallel_pthreads(const KNNModel& model,
                                             const std::vector<float>& X_test,
                                             int n_test) {
    Labels pred(n_test);
    if (n_test == 0) return pred;

    const int slice = n_test / N_THREADS;
    const std::vector<float> test_norm2 = row_norms(X_test, n_test, model.n_features);
    std::vector<KNNBlasPredArg> args(N_THREADS);
    std::vector<pthread_t> threads(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        args[t].model      = &model;
        args[t].X_test     = &X_test;
        args[t].test_norm2 = &test_norm2;
        args[t].pred       = &pred;
        args[t].s_off      = t * slice;
        args[t].e_off      = (t == N_THREADS - 1) ? n_test : (t + 1) * slice;
        pthread_create(&threads[t], nullptr, knn_blas_pthread_worker, &args[t]);
    }
    for (int t = 0; t < N_THREADS; ++t) pthread_join(threads[t], nullptr);
    return pred;
}

static Labels predict_serial(const KNNModel& model, const std::vector<float>& X_test, int n_test) {
    return predict_blas_serial(model, X_test, n_test);
}

static Labels predict_parallel_omp(const KNNModel& model,
                                   const std::vector<float>& X_test,
                                   int n_test) {
    return predict_blas_parallel_omp(model, X_test, n_test);
}

static Labels predict_parallel_pthreads(const KNNModel& model,
                                        const std::vector<float>& X_test,
                                        int n_test) {
    return predict_blas_parallel_pthreads(model, X_test, n_test);
}

static Metrics evaluate(const Labels& truth, const Labels& pred) {
    int tp = 0, fp = 0, tn = 0, fn = 0;
    for (int i = 0; i < static_cast<int>(truth.size()); ++i) {
        if (truth[i] == 1 && pred[i] == 1)
            ++tp;
        else if (truth[i] == -1 && pred[i] == 1)
            ++fp;
        else if (truth[i] == -1 && pred[i] == -1)
            ++tn;
        else
            ++fn;
    }

    Metrics m;
    m.acc = static_cast<double>(tp + tn) / static_cast<double>(truth.size());
    m.prec = (tp + fp) ? static_cast<double>(tp) / (tp + fp) : 0.0;
    m.rec = (tp + fn) ? static_cast<double>(tp) / (tp + fn) : 0.0;
    m.f1 = (m.prec + m.rec) ? (2.0 * m.prec * m.rec / (m.prec + m.rec)) : 0.0;
    return m;
}

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
    std::cout << "  Accuracy       : " << m.acc << "\n";
    std::cout << "  Precision      : " << m.prec << "\n";
    std::cout << "  Recall         : " << m.rec << "\n";
    std::cout << "  F1 Score       : " << m.f1 << "\n\n";
}

int main(int argc, char* argv[]) {
    std::string train_csv = "data/train_cleaned.csv";
    std::string test_csv = "data/test_cleaned.csv";
    int k = K_NEIGHBORS;

    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv = argv[2];
    if (argc > 3) k = std::max(1, parse_int_or_default(argv[3], K_NEIGHBORS, "k"));

    if (const char* env = std::getenv("N_THREADS")) {
        int n = std::atoi(env);
        if (n >= 1) N_THREADS = n;
    }
    if (const char* env = std::getenv("KNN_BLAS_BLOCK")) {
        int n = std::atoi(env);
        if (n >= 1) BLAS_QUERY_BLOCK = n;
    }

    std::cout << "KNN - CS:GO Round Winner Classification\n";
    std::cout << "Config: k=" << k
              << "  distance=squared_euclidean"
              << "  epochs=N/A (lazy learner)"
              << "  threads=" << N_THREADS
              << "  backend=" << active_backend_name();
    std::cout << "  blas_provider=" << blas_provider_name()
              << "  blas_block=" << BLAS_QUERY_BLOCK;
    std::cout << "\n\n";

    const double t0 = now_ms();
    Dataset ds = load_dataset(train_csv, test_csv);
    const double t1 = now_ms();
    std::cout << std::fixed << std::setprecision(2)
              << "Data loading + normalization: " << (t1 - t0) << " ms\n\n";

    const int n_test = static_cast<int>(ds.y_test.size());

    const double ts0 = now_ms();
    KNNModel serial_model = train_serial(ds.X_train, ds.y_train, ds.n_features, k);
    const double ts1 = now_ms();
    Labels serial_pred = predict_serial(serial_model, ds.X_test, n_test);
    const double ts2 = now_ms();
    Metrics sm = evaluate(ds.y_test, serial_pred);
    print_results("Serial KNN", ts1 - ts0, ts2 - ts1, sm);

    // train_parallel is a trivial duplicate of train_serial (KNN is lazy) —
    // we call it once to populate the model, then exercise both parallel
    // inference variants against the same model.
    KNNModel parallel_model = train_parallel(ds.X_train, ds.y_train, ds.n_features, k);

    // Parallel KNN (OpenMP) — inference only
    const double to0 = now_ms();
    KNNModel omp_model = parallel_model;  // same as serial_model, but time the copy with train
    const double to1 = now_ms();
    Labels omp_pred = predict_parallel_omp(omp_model, ds.X_test, n_test);
    const double to2 = now_ms();
    Metrics om = evaluate(ds.y_test, omp_pred);
    print_results("Parallel KNN (OpenMP, " + std::to_string(N_THREADS) + " threads)",
                  to1 - to0, to2 - to1, om);

    // Parallel KNN (pthreads) — inference only
    const double tp0 = now_ms();
    KNNModel pth_model = parallel_model;
    const double tp1 = now_ms();
    Labels pth_pred = predict_parallel_pthreads(pth_model, ds.X_test, n_test);
    const double tp2 = now_ms();
    Metrics pm = evaluate(ds.y_test, pth_pred);
    print_results("Parallel KNN (pthreads, " + std::to_string(N_THREADS) + " threads)",
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
