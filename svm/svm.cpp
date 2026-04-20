// =============================================================================
// svm.cpp — Serial and Parallel Linear SVM (hinge loss + L2 regularisation)
// Dataset  : train_cleaned.csv / test_cleaned.csv
//            (103 features, label: round_winner in {+1, -1})
// Language : C++17
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
static constexpr int    N_FEATURES  = 103;     // number of features in the dataset
static constexpr double LAMBDA      = 1e-4;   // L2 regularisation strength
static constexpr int    MAX_EPOCHS  = 200;     // full-pass epochs over training set
static constexpr double LR          = 0.01;   // learning rate (fixed)
static constexpr int    N_THREADS   = 8;      // worker threads for parallel SVM
static constexpr int    SEED        = 42;     // RNG seed for reproducibility

using Vec    = std::vector<double>;
using Matrix = std::vector<Vec>;
using IVec   = std::vector<int>;

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────
static inline double dot(const Vec& a, const Vec& b) noexcept {
    double s = 0.0;
    for (int j = 0; j < N_FEATURES; ++j) {
        s += a[j] * b[j];
    }
    return s;
}

static double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch()).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset Loading and Preprocessing
// ─────────────────────────────────────────────────────────────────────────────
struct Dataset {
    Matrix X_train, X_test;
    IVec   y_train, y_test; // labels in {+1, -1}
};

Dataset load_dataset(const std::string& train_path, const std::string& test_path) {
    auto parse_row = [&](const std::string& line, bool /*build_map*/)
                     -> std::pair<Vec, int> {
        Vec row;
        row.reserve(N_FEATURES);
        std::istringstream ss(line);
        std::string tok;
        int col = 0, label = 0;
        while (std::getline(ss, tok, ',')) {
            if (col == 95) {
                // round_winner column: already +1 or -1
                label = std::stoi(tok);
            } else {
                row.push_back(std::stod(tok));
            }
            ++col;
        }
        return {std::move(row), label};
    };

    auto load_file = [&](const std::string& path, bool build_map,
                         Matrix& X, IVec& y) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error: cannot open file: " << path << "\n";
            std::exit(1);
        }
        std::string line;
        std::getline(file, line); // skip header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto [row, lbl] = parse_row(line, build_map);
            X.push_back(std::move(row));
            y.push_back(lbl);
        }
    };

    Dataset ds;
    load_file(train_path, /*build_map=*/true,  ds.X_train, ds.y_train);
    load_file(test_path,  /*build_map=*/false, ds.X_test,  ds.y_test);

    int n_train = static_cast<int>(ds.X_train.size());
    int n_test  = static_cast<int>(ds.X_test.size());

    // Compute mean and std-dev on training set only for every feature
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
        if (sd[j] < 1e-9) sd[j] = 1.0;   // guard constant features
    }

    // Normalise both splits in-place
    auto norm_inplace = [&](Matrix& X) {
        for (auto& row : X)
            for (int j = 0; j < N_FEATURES; ++j)
                row[j] = (row[j] - mean[j]) / sd[j];
    };
    norm_inplace(ds.X_train);
    norm_inplace(ds.X_test);

    // Count class distribution
    int tr_pos = 0, tr_neg = 0, te_pos = 0, te_neg = 0;
    for (int l : ds.y_train) (l == 1 ? tr_pos : tr_neg)++;
    for (int l : ds.y_test)  (l == 1 ? te_pos : te_neg)++;

    std::cout << "┌─────────────────────────────────────────────────┐\n";
    std::cout << "│  Dataset Info                                   │\n";
    std::cout << "├─────────────────────────────────────────────────┤\n";
    std::cout << "│  X_train : " << n_train << " x " << N_FEATURES
              << std::string(37 - std::to_string(n_train).size()
                                - std::to_string(N_FEATURES).size(), ' ') << "\n";
    std::cout << "│  X_test  : " << n_test  << " x " << N_FEATURES
              << std::string(37 - std::to_string(n_test).size()
                                - std::to_string(N_FEATURES).size(), ' ') << "\n";
    std::cout << "│  y_train : +1=" << tr_pos << "  -1=" << tr_neg << "\n";
    std::cout << "│  y_test  : +1=" << te_pos << "  -1=" << te_neg << "\n";
    std::cout << "└─────────────────────────────────────────────────┘\n\n";
    return ds;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Structure Declaration
// ─────────────────────────────────────────────────────────────────────────────

struct SVMModel {
    Vec    w = Vec(N_FEATURES, 0.0);
    double b = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────────────

// Serial prediction
IVec predict_serial(const SVMModel& m, const Matrix& X) {
    IVec out;
    out.reserve(X.size());
    for (const auto& x : X)
        out.push_back((dot(m.w, x) + m.b) >= 0.0 ? 1 : -1);
    return out;
}

// OpenMP-parallelised prediction
IVec predict_parallel(const SVMModel& m, const Matrix& X) {
    int n = static_cast<int>(X.size());
    IVec out(n);
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(N_THREADS)
    #endif
    for (int i = 0; i < n; ++i)
        out[i] = (dot(m.w, X[i]) + m.b) >= 0.0 ? 1 : -1;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation Metrics
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
// Serial SVM Training
// ─────────────────────────────────────────────────────────────────────────────

SVMModel train_serial(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    SVMModel model;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        Vec    gw(N_FEATURES, 0.0);
        double gb = 0.0;

        for (int i = 0; i < n; ++i) {
            if (y[i] * (dot(model.w, X[i]) + model.b) < 1.0) {
                for (int j = 0; j < N_FEATURES; ++j)
                    gw[j] += static_cast<double>(y[i]) * X[i][j];
                gb += static_cast<double>(y[i]);
            }
        }

        const double scale = 1.0 - LR * LAMBDA;
        const double step  = LR / n;
        for (int j = 0; j < N_FEATURES; ++j)
            model.w[j] = scale * model.w[j] + step * gw[j];
        model.b += step * gb;
    }
    return model;
}

// Improvements over train_serial:
//   1. Decaying learning rate
//   2. n == 0 guard  — avoids divide-by-zero in step = lr_t / n.
SVMModel train_better_serial(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    assert(n > 0 && "train_better_serial: empty training set");
    SVMModel model;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        // Decaying learning rate: η_t = LR / sqrt(t + 1)
        const double lr_t = LR / std::sqrt(static_cast<double>(epoch + 1));

        Vec    gw(N_FEATURES, 0.0);
        double gb   = 0.0;
        double loss = 0.0;

        for (int i = 0; i < n; ++i) {
            double margin = y[i] * (dot(model.w, X[i]) + model.b);
            if (margin < 1.0) {
                loss += 1.0 - margin;
                for (int j = 0; j < N_FEATURES; ++j)
                    gw[j] += static_cast<double>(y[i]) * X[i][j];
                gb += static_cast<double>(y[i]);
            }
        }
        loss = (LAMBDA / 2.0) * dot(model.w, model.w) + loss / n;
        std::cout << "epoch " << epoch << "  loss=" << loss << "\n";

        const double scale = 1.0 - lr_t * LAMBDA;
        const double step  = lr_t / n;
        for (int j = 0; j < N_FEATURES; ++j) {
            model.w[j] = scale * model.w[j] + step * gw[j];
        }
        model.b += step * gb;
    }
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel SVM Training  (pthreads + mutex for shared gradient accumulation)
// ─────────────────────────────────────────────────────────────────────────────
// Strategy:
//   • Training data is partitioned evenly across N_THREADS pthreads.
//   • Each epoch:
//       1. Every thread independently computes its local gradient contribution
//          over its data partition (no communication needed here).
//       2. Each thread acquires a mutex and adds its local gradient to the
//          shared accumulator — fine-grained locking per the proposal.
//       3. A barrier synchronises all threads after accumulation.
//       4. Thread 0 applies the weight update and resets the accumulator.
//       5. A second barrier ensures no thread starts the next epoch before
//          the model is updated.

struct ParState {
    const Matrix*     X;
    const IVec*       y;
    int               n;
    SVMModel*         model;       // shared model weights (written only by thread 0)
    Vec               gw;          // shared gradient accumulator for w
    double            gb;          // shared gradient accumulator for b
    double            loss;        // shared hinge loss accumulator
    pthread_mutex_t   mutex;       // protects gw / gb / loss during accumulation
    pthread_barrier_t barrier_acc; // all threads have added their local grad
    pthread_barrier_t barrier_upd; // thread 0 has updated model; next epoch may begin
};

struct TArg {
    ParState* st;
    int       tid;   // thread id in [0, N_THREADS)
    int       s, e;  // exclusive slice [s, e) of training data
};

static void* svm_worker(void* raw) {
    auto*      a  = static_cast<TArg*>(raw);
    ParState*  st = a->st;
    const Matrix& X = *st->X;
    const IVec&   y = *st->y;
    int n = st->n;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {

        // ── Step 1: compute local gradient and loss on this thread's slice ──
        // Decaying learning rate: η_t = LR / sqrt(t + 1)
        const double lr_t = LR / std::sqrt(static_cast<double>(epoch + 1));

        Vec    lgw(N_FEATURES, 0.0);
        double lgb   = 0.0;
        double lloss = 0.0;

        for (int i = a->s; i < a->e; ++i) {
            // reads model->w and model->b which are stable (written only
            // by thread 0 after barrier_upd, which all threads passed)
            double margin = y[i] * (dot(st->model->w, X[i]) + st->model->b);
            if (margin < 1.0) {
                lloss += 1.0 - margin;
                for (int j = 0; j < N_FEATURES; ++j)
                    lgw[j] += static_cast<double>(y[i]) * X[i][j];
                lgb += static_cast<double>(y[i]);
            }
        }

        // ── Step 2: mutex-protected accumulation into shared gradient ────────
        pthread_mutex_lock(&st->mutex);
        for (int j = 0; j < N_FEATURES; ++j) st->gw[j] += lgw[j];
        st->gb   += lgb;
        st->loss += lloss;
        pthread_mutex_unlock(&st->mutex);

        // ── Step 3: wait for all threads to finish accumulation ──────────────
        pthread_barrier_wait(&st->barrier_acc);

        // ── Step 4: thread 0 applies the weight update and prints loss ───────
        if (a->tid == 0) {
            double total_loss = (LAMBDA / 2.0) * dot(st->model->w, st->model->w)
                                + st->loss / n;
            std::cout << "epoch " << epoch << "  loss=" << total_loss << "\n";

            const double scale = 1.0 - lr_t * LAMBDA;
            const double step  = lr_t / n;
            for (int j = 0; j < N_FEATURES; ++j)
                st->model->w[j] = scale * st->model->w[j] + step * st->gw[j];
            st->model->b += step * st->gb;
            // reset accumulators for next epoch
            std::fill(st->gw.begin(), st->gw.end(), 0.0);
            st->gb   = 0.0;
            st->loss = 0.0;
        }

        // ── Step 5: all threads wait before starting next epoch ──────────────
        pthread_barrier_wait(&st->barrier_upd);
    }

    return nullptr;
}

SVMModel train_parallel(const Matrix& X, const IVec& y) {
    int n = static_cast<int>(X.size());
    SVMModel model;

    ParState st;
    st.X     = &X;
    st.y     = &y;
    st.n     = n;
    st.model = &model;
    st.gw    = Vec(N_FEATURES, 0.0);
    st.gb    = 0.0;
    st.loss  = 0.0;

    pthread_mutex_init(&st.mutex, nullptr);
    pthread_barrier_init(&st.barrier_acc, nullptr, static_cast<unsigned>(N_THREADS));
    pthread_barrier_init(&st.barrier_upd, nullptr, static_cast<unsigned>(N_THREADS));

    std::vector<TArg>     args(N_THREADS);
    std::vector<pthread_t> threads(N_THREADS);
    int chunk = n / N_THREADS;

    for (int t = 0; t < N_THREADS; ++t) {
        args[t].st  = &st;
        args[t].tid = t;
        args[t].s   = t * chunk;
        args[t].e   = (t == N_THREADS - 1) ? n : (t + 1) * chunk;
        pthread_create(&threads[t], nullptr, svm_worker, &args[t]);
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
    std::cout << "║  SVM — CS:GO Round Winner Classification         ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n";
    std::cout << "Config: epochs=" << MAX_EPOCHS
              << "  lr="     << LR
              << "  lambda=" << LAMBDA
              << "  threads=" << N_THREADS << "\n\n";

    // ── Load data ────────────────────────────────────────────────────────────
    double t_load_start = now_ms();
    Dataset ds = load_dataset(train_csv, test_csv);
    double t_load_end = now_ms();
    std::cout << std::fixed << std::setprecision(2)
              << "Data loading: " << (t_load_end - t_load_start) << " ms\n\n";
    

    // ── Serial SVM ───────────────────────────────────────────────────────────
    double ts0 = now_ms();
    SVMModel serial_model = train_serial(ds.X_train, ds.y_train);
    double ts1 = now_ms();
    IVec serial_preds = predict_serial(serial_model, ds.X_test);
    double ts2 = now_ms();

    Metrics sm = evaluate(ds.y_test, serial_preds);
    print_results("Serial SVM", ts1 - ts0, ts2 - ts1, sm);

    // ── Better Serial SVM ────────────────────────────────────────────────────
    double tb0 = now_ms();
    SVMModel better_model = train_better_serial(ds.X_train, ds.y_train);
    double tb1 = now_ms();
    IVec better_preds = predict_serial(better_model, ds.X_test);
    double tb2 = now_ms();

    Metrics bm = evaluate(ds.y_test, better_preds);
    print_results("Better Serial SVM (shuffled + decaying LR)", tb1 - tb0, tb2 - tb1, bm);

    // ── Parallel SVM ─────────────────────────────────────────────────────────
    double tp0 = now_ms();
    SVMModel parallel_model = train_parallel(ds.X_train, ds.y_train);
    double tp1 = now_ms();
    IVec parallel_preds = predict_parallel(parallel_model, ds.X_test);
    double tp2 = now_ms();

    Metrics pm = evaluate(ds.y_test, parallel_preds);
    print_results("Parallel SVM (" + std::to_string(N_THREADS) + " threads, pthreads+OpenMP)",
                  tp1 - tp0, tp2 - tp1, pm);

    // ── Speedup summary ──────────────────────────────────────────────────────
    double serial_total   = (ts1 - ts0) + (ts2 - ts1);
    double serial_better_total = (tb1 - tb0) + (tb2 - tb1);
    double parallel_total = (tp1 - tp0) + (tp2 - tp1);

    std::cout << "┌─────────────────────────────────────────────────┐\n";
    std::cout << "│  Speedup Analysis                               │\n";
    std::cout << "├─────────────────────────────────────────────────┤\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "│  Serial total    : " << std::setw(10) << serial_total
              << " ms\n";
    std::cout << "│  Better Serial   : " << std::setw(10) << serial_better_total
              << " ms\n";
    std::cout << "│  Parallel total  : " << std::setw(10) << parallel_total
              << " ms\n";
    std::cout << "│  Speedup         : " << std::setw(10)
              << (serial_better_total / parallel_total) << " x\n";
    std::cout << "│  Threads used    : " << std::setw(10) << N_THREADS
              << " \n";
    std::cout << "└─────────────────────────────────────────────────┘\n";

    return 0;
}
