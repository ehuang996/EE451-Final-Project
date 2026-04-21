// =============================================================================
// knn.cpp - Serial and Parallel KNN
// Dataset : train_cleaned.csv / test_cleaned.csv
// Label   : round_winner in {+1, -1}
// =============================================================================
// g++ -std=c++17 -O3 -march=native -fopenmp knn.cpp -o knn
// ./knn train_cleaned.csv test_cleaned.csv [k]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int K_NEIGHBORS = 11;
static constexpr int N_THREADS = 8;

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
    Labels y;
    int n_rows = 0;
    int n_features = 0;
    int k = K_NEIGHBORS;
};

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
    if (std::filesystem::exists(path)) return path;
    const std::string alt = "../" + path;
    if (std::filesystem::exists(alt)) return alt;
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

static KNNModel train_serial(const std::vector<float>& X,
                             const Labels& y,
                             int n_features,
                             int k) {
    KNNModel m;
    m.X = X;
    m.y = y;
    m.n_rows = static_cast<int>(y.size());
    m.n_features = n_features;
    m.k = std::min(std::max(k, 1), m.n_rows);
    return m;
}

static KNNModel train_parallel(const std::vector<float>& X,
                               const Labels& y,
                               int n_features,
                               int k) {
    KNNModel m;
    m.X = X;
    m.y = y;
    m.n_rows = static_cast<int>(y.size());
    m.n_features = n_features;
    m.k = std::min(std::max(k, 1), m.n_rows);
    return m;
}

static inline double squared_l2(const float* a, const float* b, int n_features) {
    double sum = 0.0;
    for (int j = 0; j < n_features; ++j) {
        const double d = static_cast<double>(a[j]) - b[j];
        sum += d * d;
    }
    return sum;
}

static int predict_one(const KNNModel& model, const float* xq) {
    const int k = model.k;
    std::vector<double> best_dist(k, std::numeric_limits<double>::infinity());
    std::vector<int> best_label(k, 0);

    int max_idx = 0;
    double max_dist = best_dist[0];
    const auto& X = model.X;
    const auto& y = model.y;

    for (int i = 0; i < model.n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * model.n_features];
        const double d2 = squared_l2(xq, xi, model.n_features);

        if (d2 < max_dist) {
            best_dist[max_idx] = d2;
            best_label[max_idx] = y[i];

            max_idx = 0;
            max_dist = best_dist[0];
            for (int t = 1; t < k; ++t) {
                if (best_dist[t] > max_dist) {
                    max_dist = best_dist[t];
                    max_idx = t;
                }
            }
        }
    }

    int pos = 0;
    int neg = 0;
    int nearest_label = 1;
    double nearest_dist = std::numeric_limits<double>::infinity();
    for (int i = 0; i < k; ++i) {
        if (best_label[i] == 1)
            ++pos;
        else
            ++neg;

        if (best_dist[i] < nearest_dist) {
            nearest_dist = best_dist[i];
            nearest_label = best_label[i];
        }
    }

    if (pos > neg) return 1;
    if (neg > pos) return -1;
    return nearest_label;
}

static Labels predict_serial(const KNNModel& model, const std::vector<float>& X_test, int n_test) {
    Labels pred(n_test);
    for (int i = 0; i < n_test; ++i) {
        const float* xq = &X_test[static_cast<size_t>(i) * model.n_features];
        pred[i] = predict_one(model, xq);
    }
    return pred;
}

static Labels predict_parallel(const KNNModel& model, const std::vector<float>& X_test, int n_test) {
    Labels pred(n_test);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
#endif
    for (int i = 0; i < n_test; ++i) {
        const float* xq = &X_test[static_cast<size_t>(i) * model.n_features];
        pred[i] = predict_one(model, xq);
    }
    return pred;
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
    std::string train_csv = "train_cleaned.csv";
    std::string test_csv = "test_cleaned.csv";
    int k = K_NEIGHBORS;

    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv = argv[2];
    if (argc > 3) k = std::max(1, parse_int_or_default(argv[3], K_NEIGHBORS, "k"));

    std::cout << "KNN - CS:GO Round Winner Classification\n";
    std::cout << "Config: k=" << k
              << "  distance=squared_euclidean"
              << "  epochs=N/A (lazy learner)"
              << "  threads=" << N_THREADS << "\n\n";

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

    const double tp0 = now_ms();
    KNNModel parallel_model = train_parallel(ds.X_train, ds.y_train, ds.n_features, k);
    const double tp1 = now_ms();
    Labels parallel_pred = predict_parallel(parallel_model, ds.X_test, n_test);
    const double tp2 = now_ms();
    Metrics pm = evaluate(ds.y_test, parallel_pred);
    print_results("Parallel KNN (OpenMP, " + std::to_string(N_THREADS) + " threads)",
                  tp1 - tp0,
                  tp2 - tp1,
                  pm);

    const double serial_total = (ts1 - ts0) + (ts2 - ts1);
    const double parallel_total = (tp1 - tp0) + (tp2 - tp1);

    std::cout << "Speedup Summary\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Serial total   : " << serial_total << " ms\n";
    std::cout << "  Parallel total : " << parallel_total << " ms\n";
    std::cout << "  Speedup        : "
              << (parallel_total > 0.0 ? serial_total / parallel_total : 0.0)
              << " x\n";

    return 0;
}
