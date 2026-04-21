// =============================================================================
// dt.cpp - Serial and Parallel Decision Tree (CART, Gini impurity, histogram)
// Dataset : train_cleaned.csv / test_cleaned.csv
// Label   : round_winner in {+1, -1}
// =============================================================================
// g++ -std=c++17 -O3 -march=native -fopenmp dt.cpp -o dt -lpthread
// ./dt train_cleaned.csv test_cleaned.csv [max_depth]
//
// Mirrors the shared Dataset / load_dataset / Metrics / evaluate / print_results
// conventions in svm/svm.cpp, knn/knn.cpp, and mlp/mlp.cpp so a future
// analytics_engine.cpp can drive all five algorithms through one loader.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <pthread.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------------
// Hyperparameters (DT-specific; max_depth overridable via argv)
// -----------------------------------------------------------------------------
static constexpr int    MAX_DEPTH         = 12;   // max tree depth
static constexpr int    MIN_SAMPLES_SPLIT = 20;   // min samples to attempt a split
static constexpr int    MIN_SAMPLES_LEAF  = 5;    // min samples required in a child
static constexpr int    N_BINS            = 64;   // histogram bins per feature
static int              N_THREADS         = 8;    // overridable via N_THREADS env var (read in main)
static constexpr int    PARALLEL_DEPTH    = 3;    // OMP tasks spawn only while depth<this
static constexpr int    SEED              = 42;   // RNG seed
static constexpr double GAIN_EPS          = 1e-7; // min Gini gain to accept a split

using Labels = std::vector<int>;

// -----------------------------------------------------------------------------
// Dataset / Metrics (layout identical to svm.cpp, knn.cpp, mlp.cpp)
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
// Tree / split structures
// -----------------------------------------------------------------------------
struct Node {
    int    feature_index = -1;  // -1 for leaf
    double threshold     = 0.0; // x[feature_index] <= threshold -> left
    int    left          = -1;
    int    right         = -1;
    int    prediction    = 0;   // +1 / -1 majority class at this node
    int    n_samples     = 0;
};

struct DTModel {
    std::vector<Node> nodes;
    int root       = 0;
    int n_features = 0;
};

struct BestSplit {
    double gain      = -std::numeric_limits<double>::infinity();
    int    feature   = -1;
    double threshold = 0.0;
    // Deterministic tie-break: larger gain, then smaller feature, then smaller threshold.
    bool better_than(const BestSplit& o) const noexcept {
        if (gain != o.gain)       return gain > o.gain;
        if (feature != o.feature) return feature < o.feature;
        return threshold < o.threshold;
    }
};

// Cache-line padded so per-thread arrays don't false-share.
struct alignas(64) BestSplitCL {
    BestSplit bs;
    char _pad[64 - sizeof(BestSplit)];
};

// Histogram-binned features. Xb is column-major so per-feature scans walk
// unit-stride bytes — critical for the histogram hot loop.
struct BinnedDataset {
    int n_rows = 0;
    int n_features = 0;
    std::vector<uint8_t> Xb;     // col-major: Xb[j * n_rows + i]
    std::vector<std::vector<double>> edges;  // per-feature upper bin edges
    std::vector<int> n_bins_per_feat;
};

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

static inline double gini(long long pos, long long neg) noexcept {
    long long tot = pos + neg;
    if (tot <= 0) return 0.0;
    double p = static_cast<double>(pos) / static_cast<double>(tot);
    double q = 1.0 - p;
    return 1.0 - p*p - q*q;
}

// -----------------------------------------------------------------------------
// Dataset loading and preprocessing
//   - label column found by header name "round_winner"
//   - z-score normalization computed on training set only
//   - one-hot ("map_*"), explicit binary ("bomb_planted"), and auto-detected
//     0/1-only columns are NOT normalized (passthrough) — matches svm/knn/mlp
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
// Histogram binning (quantile edges on training set, column-major Xb).
//   Low-cardinality features (<= N_BINS unique values) get one bin per value
//   — binary features collapse to 2 bins. Continuous features get quantile bins.
// -----------------------------------------------------------------------------
static BinnedDataset build_binned_dataset(const std::vector<float>& X,
                                          int n_rows,
                                          int n_features,
                                          int n_bins_max) {
    BinnedDataset B;
    B.n_rows = n_rows;
    B.n_features = n_features;
    B.edges.assign(n_features, {});
    B.n_bins_per_feat.assign(n_features, 0);
    B.Xb.assign(static_cast<size_t>(n_features) * n_rows, 0);

    std::vector<float> col(n_rows);
    for (int j = 0; j < n_features; ++j) {
        for (int i = 0; i < n_rows; ++i) col[i] = X[static_cast<size_t>(i) * n_features + j];
        std::vector<float> sorted(col);
        std::sort(sorted.begin(), sorted.end());

        std::vector<double> uniq;
        uniq.reserve(std::min(n_rows, n_bins_max));
        for (float v : sorted) {
            const double d = static_cast<double>(v);
            if (uniq.empty() || d != uniq.back()) uniq.push_back(d);
            if (static_cast<int>(uniq.size()) > n_bins_max) break;
        }

        std::vector<double> edges;
        int nb;
        if (static_cast<int>(uniq.size()) <= n_bins_max) {
            nb = std::max(1, static_cast<int>(uniq.size()));
            for (size_t k = 0; k + 1 < uniq.size(); ++k)
                edges.push_back(0.5 * (uniq[k] + uniq[k + 1]));
        } else {
            nb = n_bins_max;
            for (int b = 1; b < nb; ++b) {
                double frac = static_cast<double>(b) / static_cast<double>(nb);
                int pos = static_cast<int>(frac * n_rows);
                if (pos >= n_rows) pos = n_rows - 1;
                edges.push_back(static_cast<double>(sorted[pos]));
            }
        }

        B.edges[j] = std::move(edges);
        B.n_bins_per_feat[j] = nb;

        uint8_t* colXb = &B.Xb[static_cast<size_t>(j) * n_rows];
        for (int i = 0; i < n_rows; ++i) {
            int b = static_cast<int>(std::upper_bound(B.edges[j].begin(), B.edges[j].end(),
                                                      static_cast<double>(col[i]))
                                     - B.edges[j].begin());
            if (b >= nb) b = nb - 1;
            colXb[i] = static_cast<uint8_t>(b);
        }
    }
    return B;
}

// -----------------------------------------------------------------------------
// Split finding on a single feature (histogram method, column-major Xb).
//   Unit-stride scan of colXb[idx[*]] since Xb is column-major.
// -----------------------------------------------------------------------------
static inline BestSplit scan_feature_histogram(
    int j, const std::vector<int>& idx, const Labels& y,
    const BinnedDataset& B, long long pos_tot, long long neg_tot, double parent_g)
{
    const int nb = B.n_bins_per_feat[j];
    const uint8_t* colXb = &B.Xb[static_cast<size_t>(j) * B.n_rows];
    int hp[256] = {0}, hn[256] = {0};  // N_BINS fits in 256
    for (int i : idx) {
        uint8_t b = colXb[i];
        if (y[i] == 1) ++hp[b]; else ++hn[b];
    }
    const long long n = pos_tot + neg_tot;
    long long cum_p = 0, cum_n = 0;
    BestSplit best;
    for (int b = 0; b < nb - 1; ++b) {
        cum_p += hp[b];
        cum_n += hn[b];
        long long left_n  = cum_p + cum_n;
        long long right_n = n - left_n;
        if (left_n  < MIN_SAMPLES_LEAF) continue;
        if (right_n < MIN_SAMPLES_LEAF) break;
        long long pr = pos_tot - cum_p;
        long long nr = neg_tot - cum_n;
        double gain = parent_g
                    - (static_cast<double>(left_n)  / n) * gini(cum_p, cum_n)
                    - (static_cast<double>(right_n) / n) * gini(pr, nr);
        BestSplit cand;
        cand.gain      = gain;
        cand.feature   = j;
        cand.threshold = B.edges[j][b];
        if (cand.better_than(best)) best = cand;
    }
    return best;
}

// -----------------------------------------------------------------------------
// Partition a node's sample indices by a split on feature j at threshold.
// Reads original flat X (n_features-strided row-major).
// -----------------------------------------------------------------------------
static inline void partition_samples(const std::vector<int>& idx,
                                     const std::vector<float>& X, int n_features,
                                     const Labels& y,
                                     int feat, double threshold,
                                     std::vector<int>& left_idx,
                                     std::vector<int>& right_idx,
                                     long long& pl, long long& nl,
                                     long long& pr, long long& nr)
{
    left_idx.reserve(idx.size() / 2 + 1);
    right_idx.reserve(idx.size() / 2 + 1);
    pl = nl = pr = nr = 0;
    for (int i : idx) {
        const float xv = X[static_cast<size_t>(i) * n_features + feat];
        if (static_cast<double>(xv) <= threshold) {
            left_idx.push_back(i);
            if (y[i] == 1) ++pl; else ++nl;
        } else {
            right_idx.push_back(i);
            if (y[i] == 1) ++pr; else ++nr;
        }
    }
}

// -----------------------------------------------------------------------------
// Serial histogram CART — train_serial's tree builder.
// -----------------------------------------------------------------------------
static void build_hist_serial(DTModel& model, int node_idx, int depth, int max_depth,
                              std::vector<int>& idx,
                              long long pos_tot, long long neg_tot,
                              const BinnedDataset& B,
                              const std::vector<float>& X, int n_features,
                              const Labels& y)
{
    model.nodes[node_idx].n_samples  = static_cast<int>(idx.size());
    model.nodes[node_idx].prediction = (pos_tot >= neg_tot) ? 1 : -1;

    bool stop = (depth >= max_depth)
             || (static_cast<int>(idx.size()) < MIN_SAMPLES_SPLIT)
             || (pos_tot == 0) || (neg_tot == 0);
    if (stop) { model.nodes[node_idx].feature_index = -1; return; }

    double parent_g = gini(pos_tot, neg_tot);
    BestSplit best;
    for (int j = 0; j < n_features; ++j) {
        BestSplit cand = scan_feature_histogram(j, idx, y, B, pos_tot, neg_tot, parent_g);
        if (cand.better_than(best)) best = cand;
    }
    if (best.feature < 0 || best.gain < GAIN_EPS) {
        model.nodes[node_idx].feature_index = -1; return;
    }

    std::vector<int> left_idx, right_idx;
    long long pl, nl, pr, nr;
    partition_samples(idx, X, n_features, y, best.feature, best.threshold,
                      left_idx, right_idx, pl, nl, pr, nr);
    idx.clear(); idx.shrink_to_fit();

    int L = static_cast<int>(model.nodes.size()); model.nodes.emplace_back();
    int R = static_cast<int>(model.nodes.size()); model.nodes.emplace_back();
    model.nodes[node_idx].feature_index = best.feature;
    model.nodes[node_idx].threshold     = best.threshold;
    model.nodes[node_idx].left          = L;
    model.nodes[node_idx].right         = R;

    build_hist_serial(model, L, depth + 1, max_depth, left_idx,  pl, nl, B, X, n_features, y);
    build_hist_serial(model, R, depth + 1, max_depth, right_idx, pr, nr, B, X, n_features, y);
}

// -----------------------------------------------------------------------------
// Parallel OpenMP tree builder.
//   - Inside find_best_split: #pragma omp taskloop over features (shares team,
//     no nested parallel region, so no thread over-subscription).
//   - Outside: #pragma omp task for subtree recursion while depth < PARALLEL_DEPTH.
// -----------------------------------------------------------------------------

struct OmpTreeCtx {
    DTModel*             model;
    const BinnedDataset* B;
    const std::vector<float>* X;
    int                  n_features;
    const Labels*        y;
    int                  max_depth;
    std::atomic<int>     next_idx;   // next free slot in model->nodes
};

static BestSplit find_best_split_omp(
    const std::vector<int>& idx, const BinnedDataset& B, const Labels& y,
    long long pos_tot, long long neg_tot, int depth, int n_features)
{
    const double parent_g = gini(pos_tot, neg_tot);
    std::vector<BestSplitCL> thread_best(N_THREADS);
    const bool parallelize = (depth < PARALLEL_DEPTH);

#ifdef _OPENMP
    if (parallelize) {
        const int grain = std::max(1, n_features / N_THREADS);
        #pragma omp taskloop grainsize(grain) shared(thread_best, idx, y, B) \
                    firstprivate(pos_tot, neg_tot, parent_g, n_features)
        for (int j = 0; j < n_features; ++j) {
            BestSplit cand = scan_feature_histogram(j, idx, y, B, pos_tot, neg_tot, parent_g);
            int tid = omp_get_thread_num();
            if (cand.better_than(thread_best[tid].bs)) thread_best[tid].bs = cand;
        }
    } else
#endif
    {
        (void)parallelize;
        for (int j = 0; j < n_features; ++j) {
            BestSplit cand = scan_feature_histogram(j, idx, y, B, pos_tot, neg_tot, parent_g);
            if (cand.better_than(thread_best[0].bs)) thread_best[0].bs = cand;
        }
    }

    BestSplit best;
    for (int t = 0; t < N_THREADS; ++t)
        if (thread_best[t].bs.better_than(best)) best = thread_best[t].bs;
    return best;
}

static void build_hist_omp(OmpTreeCtx& ctx, int node_idx, int depth,
                           std::shared_ptr<std::vector<int>> idx_ptr,
                           long long pos_tot, long long neg_tot)
{
    auto& idx = *idx_ptr;
    ctx.model->nodes[node_idx].n_samples  = static_cast<int>(idx.size());
    ctx.model->nodes[node_idx].prediction = (pos_tot >= neg_tot) ? 1 : -1;

    bool stop = (depth >= ctx.max_depth)
             || (static_cast<int>(idx.size()) < MIN_SAMPLES_SPLIT)
             || (pos_tot == 0) || (neg_tot == 0);
    if (stop) { ctx.model->nodes[node_idx].feature_index = -1; return; }

    BestSplit best = find_best_split_omp(idx, *ctx.B, *ctx.y, pos_tot, neg_tot,
                                         depth, ctx.n_features);
    if (best.feature < 0 || best.gain < GAIN_EPS) {
        ctx.model->nodes[node_idx].feature_index = -1; return;
    }

    auto left_ptr  = std::make_shared<std::vector<int>>();
    auto right_ptr = std::make_shared<std::vector<int>>();
    long long pl, nl, pr, nr;
    partition_samples(idx, *ctx.X, ctx.n_features, *ctx.y,
                      best.feature, best.threshold,
                      *left_ptr, *right_ptr, pl, nl, pr, nr);
    idx_ptr.reset();  // drop local reference to help free memory

    int L = ctx.next_idx.fetch_add(2, std::memory_order_relaxed);
    int R = L + 1;
    ctx.model->nodes[node_idx].feature_index = best.feature;
    ctx.model->nodes[node_idx].threshold     = best.threshold;
    ctx.model->nodes[node_idx].left          = L;
    ctx.model->nodes[node_idx].right         = R;

    if (depth < PARALLEL_DEPTH) {
#ifdef _OPENMP
        #pragma omp task firstprivate(L, depth, left_ptr, pl, nl) shared(ctx)
        build_hist_omp(ctx, L, depth + 1, left_ptr,  pl, nl);
        #pragma omp task firstprivate(R, depth, right_ptr, pr, nr) shared(ctx)
        build_hist_omp(ctx, R, depth + 1, right_ptr, pr, nr);
        #pragma omp taskwait
#else
        build_hist_omp(ctx, L, depth + 1, left_ptr,  pl, nl);
        build_hist_omp(ctx, R, depth + 1, right_ptr, pr, nr);
#endif
    } else {
        build_hist_omp(ctx, L, depth + 1, left_ptr,  pl, nl);
        build_hist_omp(ctx, R, depth + 1, right_ptr, pr, nr);
    }
}

// -----------------------------------------------------------------------------
// Inference
// -----------------------------------------------------------------------------
static inline int tree_walk(const DTModel& m, const float* x) {
    int n = m.root;
    while (m.nodes[n].feature_index >= 0) {
        n = (static_cast<double>(x[m.nodes[n].feature_index]) <= m.nodes[n].threshold)
            ? m.nodes[n].left : m.nodes[n].right;
    }
    return m.nodes[n].prediction;
}

static Labels predict_serial(const DTModel& m,
                             const std::vector<float>& X,
                             int n_rows,
                             int n_features) {
    Labels out(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        out[i] = tree_walk(m, xi);
    }
    return out;
}

static Labels predict_parallel(const DTModel& m,
                               const std::vector<float>& X,
                               int n_rows,
                               int n_features) {
    Labels out(n_rows);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(N_THREADS)
#endif
    for (int i = 0; i < n_rows; ++i) {
        const float* xi = &X[static_cast<size_t>(i) * n_features];
        out[i] = tree_walk(m, xi);
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
// Training - Serial (histogram-based CART).
//   This is the single serial baseline; the parallel variants parallelize the
//   same algorithm. No "better_serial" tier (matches svm/knn/mlp).
// -----------------------------------------------------------------------------
static DTModel train_serial(const std::vector<float>& X,
                            const Labels& y,
                            int n_rows,
                            int n_features,
                            int max_depth) {
    DTModel model;
    model.n_features = n_features;
    model.nodes.reserve(1 << (max_depth + 1));
    model.nodes.emplace_back();
    model.root = 0;
    if (n_rows == 0 || n_features == 0) return model;

    BinnedDataset B = build_binned_dataset(X, n_rows, n_features, N_BINS);

    std::vector<int> idx(n_rows);
    std::iota(idx.begin(), idx.end(), 0);
    long long pos_tot = 0, neg_tot = 0;
    for (int v : y) (v == 1 ? pos_tot : neg_tot)++;
    build_hist_serial(model, 0, 0, max_depth, idx, pos_tot, neg_tot, B, X, n_features, y);
    return model;
}

// -----------------------------------------------------------------------------
// Training - Parallel (OpenMP, taskloop over features + tasks for subtrees)
// -----------------------------------------------------------------------------
static DTModel train_parallel_omp(const std::vector<float>& X,
                                  const Labels& y,
                                  int n_rows,
                                  int n_features,
                                  int max_depth) {
    DTModel model;
    model.n_features = n_features;
    model.root = 0;
    if (n_rows == 0 || n_features == 0) {
        model.nodes.emplace_back();
        return model;
    }

#ifdef _OPENMP
    omp_set_max_active_levels(2);
#endif

    BinnedDataset B = build_binned_dataset(X, n_rows, n_features, N_BINS);

    // Preallocate the node vector so tasks can fill slots via atomic counter.
    const int max_nodes = 1 << (max_depth + 1);
    model.nodes.resize(max_nodes);

    OmpTreeCtx ctx;
    ctx.model      = &model;
    ctx.B          = &B;
    ctx.X          = &X;
    ctx.n_features = n_features;
    ctx.y          = &y;
    ctx.max_depth  = max_depth;
    ctx.next_idx.store(1, std::memory_order_relaxed);  // root is 0

    auto root_idx = std::make_shared<std::vector<int>>(n_rows);
    std::iota(root_idx->begin(), root_idx->end(), 0);
    long long pos_tot = 0, neg_tot = 0;
    for (int v : y) (v == 1 ? pos_tot : neg_tot)++;

#ifdef _OPENMP
#pragma omp parallel num_threads(N_THREADS)
    {
        #pragma omp single
        build_hist_omp(ctx, 0, 0, root_idx, pos_tot, neg_tot);
    }
#else
    build_hist_omp(ctx, 0, 0, root_idx, pos_tot, neg_tot);
#endif

    model.nodes.resize(ctx.next_idx.load(std::memory_order_relaxed));
    return model;
}

// -----------------------------------------------------------------------------
// Training - Parallel (pthreads, persistent worker pool + per-node barriers)
//   Mirrors mlp.cpp's pthread pattern: fixed pool of N_THREADS, static feature-
//   range partition, mutex-protected reduction, two barriers per node.
//   Subtree recursion is driven serially by the leader thread via an explicit
//   stack — this is the "horizontal only" design; it trades cross-subtree
//   parallelism for simpler synchronization and makes the OMP-vs-pthreads
//   comparison isolate the per-node barrier cost.
// -----------------------------------------------------------------------------

struct PendingNode {
    int              node_idx;
    int              depth;
    std::vector<int> idx;
    long long        pos_tot;
    long long        neg_tot;
};

struct PtState {
    const BinnedDataset*      B;
    const std::vector<float>* X;
    int                       n_features;
    const Labels*             y;
    DTModel*                  model;
    std::vector<PendingNode>  stack;          // leader-owned

    const std::vector<int>*   cur_idx        = nullptr;
    int                       cur_depth      = 0;
    long long                 cur_pos        = 0;
    long long                 cur_neg        = 0;
    double                    cur_parent_g   = 0.0;
    BestSplit                 cur_best;
    bool                      cur_is_leaf    = true;
    bool                      shutdown       = false;

    pthread_mutex_t           mutex;
    pthread_barrier_t         barrier_ready;  // workers done reducing
    pthread_barrier_t         barrier_go;     // leader installed next node
};

struct PtArg {
    PtState* st;
    int      tid;
};

static void pt_merge_global(PtState* st, const BestSplit& cand) {
    pthread_mutex_lock(&st->mutex);
    if (cand.better_than(st->cur_best)) st->cur_best = cand;
    pthread_mutex_unlock(&st->mutex);
}

static void* pt_worker(void* raw) {
    auto* a = static_cast<PtArg*>(raw);
    PtState* st = a->st;
    const int tid = a->tid;
    const int n_features = st->n_features;

    while (true) {
        pthread_barrier_wait(&st->barrier_go);
        if (st->shutdown) break;
        if (st->cur_is_leaf) {
            pthread_barrier_wait(&st->barrier_ready);
            continue;
        }

        const int f_start = tid * n_features / N_THREADS;
        const int f_end   = (tid + 1) * n_features / N_THREADS;

        BestSplit local;
        for (int j = f_start; j < f_end; ++j) {
            BestSplit cand = scan_feature_histogram(
                j, *st->cur_idx, *st->y, *st->B,
                st->cur_pos, st->cur_neg, st->cur_parent_g);
            if (cand.better_than(local)) local = cand;
        }
        pt_merge_global(st, local);
        pthread_barrier_wait(&st->barrier_ready);
    }
    return nullptr;
}

static DTModel train_parallel_pthreads(const std::vector<float>& X,
                                       const Labels& y,
                                       int n_rows,
                                       int n_features,
                                       int max_depth) {
    DTModel model;
    model.n_features = n_features;
    model.nodes.reserve(1 << (max_depth + 1));
    model.nodes.emplace_back();
    model.root = 0;
    if (n_rows == 0 || n_features == 0) return model;

    BinnedDataset B = build_binned_dataset(X, n_rows, n_features, N_BINS);

    PtState st;
    st.B          = &B;
    st.X          = &X;
    st.n_features = n_features;
    st.y          = &y;
    st.model      = &model;

    pthread_mutex_init(&st.mutex, nullptr);
    pthread_barrier_init(&st.barrier_ready, nullptr, static_cast<unsigned>(N_THREADS));
    pthread_barrier_init(&st.barrier_go,    nullptr, static_cast<unsigned>(N_THREADS));

    std::vector<PtArg>     args(N_THREADS);
    std::vector<pthread_t> threads(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) args[t] = {&st, t};
    for (int t = 1; t < N_THREADS; ++t)
        pthread_create(&threads[t], nullptr, pt_worker, &args[t]);

    std::vector<int> root_idx(n_rows);
    std::iota(root_idx.begin(), root_idx.end(), 0);
    long long pos_tot = 0, neg_tot = 0;
    for (int v : y) (v == 1 ? pos_tot : neg_tot)++;
    st.stack.push_back({0, 0, std::move(root_idx), pos_tot, neg_tot});

    while (!st.stack.empty()) {
        PendingNode cur = std::move(st.stack.back());
        st.stack.pop_back();

        st.model->nodes[cur.node_idx].n_samples  = static_cast<int>(cur.idx.size());
        st.model->nodes[cur.node_idx].prediction = (cur.pos_tot >= cur.neg_tot) ? 1 : -1;

        bool stop = (cur.depth >= max_depth)
                 || (static_cast<int>(cur.idx.size()) < MIN_SAMPLES_SPLIT)
                 || (cur.pos_tot == 0) || (cur.neg_tot == 0);

        st.cur_idx      = &cur.idx;
        st.cur_depth    = cur.depth;
        st.cur_pos      = cur.pos_tot;
        st.cur_neg      = cur.neg_tot;
        st.cur_parent_g = gini(cur.pos_tot, cur.neg_tot);
        st.cur_best     = BestSplit{};
        st.cur_is_leaf  = stop;

        pthread_barrier_wait(&st.barrier_go);

        if (!stop) {
            // Leader (tid=0) does its own feature range.
            const int f_start = 0;
            const int f_end   = n_features / N_THREADS;
            BestSplit local;
            for (int j = f_start; j < f_end; ++j) {
                BestSplit cand = scan_feature_histogram(
                    j, cur.idx, y, B, cur.pos_tot, cur.neg_tot, st.cur_parent_g);
                if (cand.better_than(local)) local = cand;
            }
            pt_merge_global(&st, local);
        }
        pthread_barrier_wait(&st.barrier_ready);

        if (stop) {
            st.model->nodes[cur.node_idx].feature_index = -1;
            continue;
        }

        BestSplit best = st.cur_best;
        if (best.feature < 0 || best.gain < GAIN_EPS) {
            st.model->nodes[cur.node_idx].feature_index = -1;
            continue;
        }

        std::vector<int> left_idx, right_idx;
        long long pl, nl, pr, nr;
        partition_samples(cur.idx, X, n_features, y, best.feature, best.threshold,
                          left_idx, right_idx, pl, nl, pr, nr);
        cur.idx.clear(); cur.idx.shrink_to_fit();

        int L = static_cast<int>(st.model->nodes.size()); st.model->nodes.emplace_back();
        int R = static_cast<int>(st.model->nodes.size()); st.model->nodes.emplace_back();
        st.model->nodes[cur.node_idx].feature_index = best.feature;
        st.model->nodes[cur.node_idx].threshold     = best.threshold;
        st.model->nodes[cur.node_idx].left          = L;
        st.model->nodes[cur.node_idx].right         = R;

        // Pre-order traversal: push right first so left runs next.
        st.stack.push_back({R, cur.depth + 1, std::move(right_idx), pr, nr});
        st.stack.push_back({L, cur.depth + 1, std::move(left_idx),  pl, nl});
    }

    // Signal shutdown and let workers exit cleanly.
    st.shutdown = true;
    pthread_barrier_wait(&st.barrier_go);

    for (int t = 1; t < N_THREADS; ++t) pthread_join(threads[t], nullptr);

    pthread_mutex_destroy(&st.mutex);
    pthread_barrier_destroy(&st.barrier_ready);
    pthread_barrier_destroy(&st.barrier_go);

    return model;
}

// -----------------------------------------------------------------------------
// Printing results (plain-text style, matches svm.cpp + knn.cpp + mlp.cpp)
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
    int max_depth = MAX_DEPTH;

    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv  = argv[2];
    if (argc > 3) max_depth = std::max(1, parse_int_or_default(argv[3], MAX_DEPTH, "max_depth"));

    if (const char* env = std::getenv("N_THREADS")) {
        int n = std::atoi(env);
        if (n >= 1) N_THREADS = n;
    }

    std::cout << "DT - CS:GO Round Winner Classification\n";
    std::cout << "Config: max_depth=" << max_depth
              << "  min_split=" << MIN_SAMPLES_SPLIT
              << "  min_leaf=" << MIN_SAMPLES_LEAF
              << "  bins=" << N_BINS
              << "  threads=" << N_THREADS << "\n\n";

    const double t0 = now_ms();
    Dataset ds = load_dataset(train_csv, test_csv);
    const double t1 = now_ms();
    std::cout << std::fixed << std::setprecision(2)
              << "Data loading + normalization: " << (t1 - t0) << " ms\n\n";

    const int n_train = static_cast<int>(ds.y_train.size());
    const int n_test  = static_cast<int>(ds.y_test.size());

    // Serial DT
    const double ts0 = now_ms();
    DTModel serial_model = train_serial(ds.X_train, ds.y_train,
                                        n_train, ds.n_features, max_depth);
    const double ts1 = now_ms();
    Labels serial_pred = predict_serial(serial_model, ds.X_test, n_test, ds.n_features);
    const double ts2 = now_ms();
    Metrics sm = evaluate(ds.y_test, serial_pred);
    print_results("Serial DT", ts1 - ts0, ts2 - ts1, sm);

    // Parallel DT (OpenMP, taskloop + tasks)
    const double to0 = now_ms();
    DTModel omp_model = train_parallel_omp(ds.X_train, ds.y_train,
                                           n_train, ds.n_features, max_depth);
    const double to1 = now_ms();
    Labels omp_pred = predict_parallel(omp_model, ds.X_test, n_test, ds.n_features);
    const double to2 = now_ms();
    Metrics om = evaluate(ds.y_test, omp_pred);
    print_results("Parallel DT (OpenMP, " + std::to_string(N_THREADS) + " threads)",
                  to1 - to0, to2 - to1, om);

    // Parallel DT (pthreads, mutex + barriers)
    const double tp0 = now_ms();
    DTModel pth_model = train_parallel_pthreads(ds.X_train, ds.y_train,
                                                n_train, ds.n_features, max_depth);
    const double tp1 = now_ms();
    Labels pth_pred = predict_parallel(pth_model, ds.X_test, n_test, ds.n_features);
    const double tp2 = now_ms();
    Metrics pm = evaluate(ds.y_test, pth_pred);
    print_results("Parallel DT (pthreads, " + std::to_string(N_THREADS) + " threads)",
                  tp1 - tp0, tp2 - tp1, pm);

    // Speedup summary (plain text, matches svm.cpp / knn.cpp / mlp.cpp)
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
