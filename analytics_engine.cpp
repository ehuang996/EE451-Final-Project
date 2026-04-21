// =============================================================================
// analytics_engine.cpp - Master benchmarking driver
// =============================================================================
// Runs each of the five algorithm binaries (./svm, ./knn, ./mlp, ./dt, ./nb)
// as a subprocess, parses their structured [tag] output blocks, and writes a
// consolidated CSV of per-variant timing + quality metrics to
// analytics_results.csv. Prints a summary table to stdout.
//
// Assumes all five binaries already exist in the current working directory
// (job_analytics.sl compiles them first, then runs this driver).
//
// Build:  g++ -std=c++17 -O3 analytics_engine.cpp -o analytics_engine
// Run:    ./analytics_engine [train_cleaned.csv] [test_cleaned.csv]
//         (args forwarded to each algorithm binary verbatim)
//
// CSV schema:
//   algorithm,variant,n_threads,train_ms,infer_ms,total_ms,accuracy,precision,recall,f1
// where:
//   algorithm ∈ {svm, knn, mlp, dt, nb}
//   variant   ∈ {serial, omp, pthreads}
//   n_threads = 1 for serial, parsed from tag for omp/pthreads (typically 8)
// =============================================================================

#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct RunResult {
    std::string algorithm;
    std::string variant;
    int         n_threads = 1;
    double      train_ms  = 0.0;
    double      infer_ms  = 0.0;
    double      total_ms  = 0.0;
    double      accuracy  = 0.0;
    double      precision = 0.0;
    double      recall    = 0.0;
    double      f1        = 0.0;
};

// -----------------------------------------------------------------------------
// Subprocess execution — captures stdout of a shell command.
// -----------------------------------------------------------------------------
static std::string run_command(const std::string& cmd, int& exit_status) {
    std::string output;
    char buf[4096];
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) {
        exit_status = -1;
        return output;
    }
    while (std::fgets(buf, sizeof(buf), fp)) output += buf;
    int rc = pclose(fp);
    exit_status = rc;
    return output;
}

// -----------------------------------------------------------------------------
// Parse helpers
// -----------------------------------------------------------------------------

// Trim leading/trailing whitespace.
static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// Return the numeric value after the first ':' in `line`, or fallback.
static double parse_number_after_colon(const std::string& line, double fallback = 0.0) {
    size_t pos = line.find(':');
    if (pos == std::string::npos) return fallback;
    try {
        return std::stod(line.substr(pos + 1));
    } catch (...) {
        return fallback;
    }
}

// Parse `n_threads` out of a tag like "Parallel SVM (OpenMP, 8 threads)".
// Returns fallback if not found.
static int parse_threads_from_tag(const std::string& tag, int fallback = 1) {
    size_t p = tag.find(" threads");
    if (p == std::string::npos) return fallback;
    // Walk back to find the number.
    size_t q = p;
    while (q > 0 && std::isspace(static_cast<unsigned char>(tag[q - 1]))) --q;
    size_t end = q;
    while (q > 0 && std::isdigit(static_cast<unsigned char>(tag[q - 1]))) --q;
    if (q == end) return fallback;
    try {
        return std::stoi(tag.substr(q, end - q));
    } catch (...) {
        return fallback;
    }
}

// Classify tag text into (algorithm, variant).
//   "Serial SVM"                               -> ("svm",  "serial")
//   "Parallel SVM (OpenMP, 8 threads)"         -> ("svm",  "omp")
//   "Parallel SVM (pthreads, 8 threads)"       -> ("svm",  "pthreads")
//   etc.
static void classify_tag(const std::string& tag,
                         std::string& algorithm,
                         std::string& variant) {
    algorithm = "unknown";
    variant   = "unknown";

    // Algorithm — substring search. DT and NB use a leading space in the
    // needle so they never match "...DT..." embedded inside another token;
    // all actual tags have DT/NB prefixed by "Serial " or "Parallel ".
    struct Pair { const char* needle; const char* algo; };
    static const std::array<Pair, 5> table = {{
        {"SVM", "svm"},
        {"KNN", "knn"},
        {"MLP", "mlp"},
        {" DT", "dt"},
        {" NB", "nb"},
    }};
    for (const auto& p : table) {
        if (tag.find(p.needle) != std::string::npos) {
            algorithm = p.algo;
            break;
        }
    }

    if (tag.find("Serial")   != std::string::npos) variant = "serial";
    if (tag.find("OpenMP")   != std::string::npos) variant = "omp";
    if (tag.find("pthreads") != std::string::npos) variant = "pthreads";
}

// -----------------------------------------------------------------------------
// Parse the stdout of an algorithm binary into RunResults.
//
// Each result block looks like:
//   [Serial SVM]
//     Training time  : 180.21 ms
//     Inference time : 2.13 ms
//     Total time     : 182.34 ms
//     Accuracy       : 0.7317
//     Precision      : 0.7307
//     Recall         : 0.7183
//     F1 Score       : 0.7245
// followed by a blank line, then the next block (or non-block content).
// -----------------------------------------------------------------------------
static std::vector<RunResult> parse_output(const std::string& stdout_text) {
    std::vector<RunResult> results;
    std::istringstream ss(stdout_text);
    std::string line;

    RunResult cur;
    bool in_block = false;

    auto flush = [&] {
        if (in_block && !cur.algorithm.empty() && cur.algorithm != "unknown") {
            results.push_back(cur);
        }
        in_block = false;
        cur = RunResult{};
    };

    while (std::getline(ss, line)) {
        const std::string trimmed = trim(line);

        // Start of a new [tag] block — tag is on a line starting with '['.
        if (!trimmed.empty() && trimmed[0] == '[' && trimmed.back() == ']') {
            flush();
            const std::string tag = trimmed.substr(1, trimmed.size() - 2);
            classify_tag(tag, cur.algorithm, cur.variant);
            cur.n_threads = (cur.variant == "serial")
                            ? 1
                            : parse_threads_from_tag(tag, 8);
            in_block = true;
            continue;
        }

        if (!in_block) continue;

        // Inside a block — extract named metrics.
        if      (starts_with(trimmed, "Training time"))  cur.train_ms  = parse_number_after_colon(line);
        else if (starts_with(trimmed, "Inference time")) cur.infer_ms  = parse_number_after_colon(line);
        else if (starts_with(trimmed, "Total time"))     cur.total_ms  = parse_number_after_colon(line);
        else if (starts_with(trimmed, "Accuracy"))       cur.accuracy  = parse_number_after_colon(line);
        else if (starts_with(trimmed, "Precision"))      cur.precision = parse_number_after_colon(line);
        else if (starts_with(trimmed, "Recall"))         cur.recall    = parse_number_after_colon(line);
        else if (starts_with(trimmed, "F1 Score"))       cur.f1        = parse_number_after_colon(line);
        // Block ends on any non-indented non-empty line we don't recognize.
        else if (!trimmed.empty() && line.size() > 0 && !std::isspace(static_cast<unsigned char>(line[0]))) {
            // New non-indented content (e.g., "Speedup Summary"); end this block.
            flush();
        }
    }
    flush();
    return results;
}

// -----------------------------------------------------------------------------
// CSV output
// -----------------------------------------------------------------------------
static void write_csv(const std::string& path, const std::vector<RunResult>& rows) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "Error: cannot open CSV output file: " << path << "\n";
        return;
    }
    f << "algorithm,variant,n_threads,train_ms,infer_ms,total_ms,accuracy,precision,recall,f1\n";
    f << std::fixed;
    for (const auto& r : rows) {
        f << r.algorithm << ','
          << r.variant   << ','
          << r.n_threads << ','
          << std::setprecision(2)
          << r.train_ms  << ','
          << r.infer_ms  << ','
          << r.total_ms  << ','
          << std::setprecision(4)
          << r.accuracy  << ','
          << r.precision << ','
          << r.recall    << ','
          << r.f1        << '\n';
    }
}

// -----------------------------------------------------------------------------
// Pretty-print a summary table to stdout.
// -----------------------------------------------------------------------------
static void print_summary(const std::vector<RunResult>& rows) {
    std::cout << "\n";
    std::cout << "===============================================================================\n";
    std::cout << "  Analytics Engine — Benchmark Summary\n";
    std::cout << "===============================================================================\n";
    std::cout << std::left
              << std::setw(8)  << "algo"
              << std::setw(11) << "variant"
              << std::setw(6)  << "thr"
              << std::right
              << std::setw(12) << "train(ms)"
              << std::setw(12) << "infer(ms)"
              << std::setw(12) << "total(ms)"
              << std::setw(10) << "acc"
              << std::setw(10) << "f1"
              << "\n";
    std::cout << "-------------------------------------------------------------------------------\n";
    std::cout << std::fixed;
    for (const auto& r : rows) {
        std::cout << std::left
                  << std::setw(8)  << r.algorithm
                  << std::setw(11) << r.variant
                  << std::setw(6)  << r.n_threads
                  << std::right
                  << std::setprecision(2)
                  << std::setw(12) << r.train_ms
                  << std::setw(12) << r.infer_ms
                  << std::setw(12) << r.total_ms
                  << std::setprecision(4)
                  << std::setw(10) << r.accuracy
                  << std::setw(10) << r.f1
                  << "\n";
    }
    std::cout << "===============================================================================\n";

    // Speedup rollup: for each algorithm, compute OMP and pthreads speedups
    // relative to serial. Useful next to the raw times.
    struct Agg { double serial = 0.0, omp = 0.0, pth = 0.0; };
    auto find_algo = [&](const std::string& algo) {
        Agg a;
        for (const auto& r : rows) {
            if (r.algorithm != algo) continue;
            if      (r.variant == "serial")   a.serial = r.total_ms;
            else if (r.variant == "omp")      a.omp    = r.total_ms;
            else if (r.variant == "pthreads") a.pth    = r.total_ms;
        }
        return a;
    };

    std::cout << "\n  Speedup (serial / parallel) per algorithm:\n";
    std::cout << std::left
              << std::setw(10) << "algo"
              << std::right
              << std::setw(12) << "OMP"
              << std::setw(12) << "pthreads"
              << "\n";
    std::cout << "  " << std::string(32, '-') << "\n";
    for (const char* algo : {"svm", "knn", "mlp", "dt", "nb"}) {
        Agg a = find_algo(algo);
        std::cout << "  " << std::left << std::setw(8) << algo
                  << std::right << std::setprecision(2) << std::fixed;
        if (a.omp > 0.0 && a.serial > 0.0)
            std::cout << std::setw(12) << (a.serial / a.omp);
        else
            std::cout << std::setw(12) << "--";
        if (a.pth > 0.0 && a.serial > 0.0)
            std::cout << std::setw(12) << (a.serial / a.pth);
        else
            std::cout << std::setw(12) << "--";
        std::cout << "\n";
    }
    std::cout << "\n";
}

// -----------------------------------------------------------------------------
// Main: invoke each algo binary in turn, parse, accumulate.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string train_csv = "data/train_cleaned.csv";
    std::string test_csv  = "data/test_cleaned.csv";
    std::string csv_out   = "analytics_results.csv";
    if (argc > 1) train_csv = argv[1];
    if (argc > 2) test_csv  = argv[2];
    if (argc > 3) csv_out   = argv[3];

    struct AlgoSpec {
        const char* name;
        const char* binary;
    };
    const std::array<AlgoSpec, 5> algos = {{
        {"SVM", "./svm"},
        {"KNN", "./knn"},
        {"MLP", "./mlp"},
        {"DT",  "./dt"},
        {"NB",  "./nb"},
    }};

    std::cout << "Analytics Engine — EE 451 Final Project\n";
    std::cout << "Train CSV: " << train_csv << "\n";
    std::cout << "Test  CSV: " << test_csv  << "\n";
    std::cout << "Running each algorithm binary and collecting [tag] metrics.\n\n";

    std::vector<RunResult> all_results;

    for (const auto& algo : algos) {
        // Redirect stderr to stdout so any warnings get captured too; this
        // doesn't affect the parser since [tag] blocks are unambiguous.
        const std::string cmd = std::string(algo.binary) + " "
                              + train_csv + " " + test_csv + " 2>&1";
        std::cout << "[run] " << algo.name << " — " << cmd << "\n";

        int status = 0;
        const std::string output = run_command(cmd, status);
        if (status != 0) {
            std::cerr << "  WARNING: " << algo.binary << " exited with status "
                      << status << " (is the binary compiled and on PATH?)\n";
            if (output.empty()) continue;
        }

        std::vector<RunResult> parsed = parse_output(output);
        std::cout << "  parsed " << parsed.size() << " result block(s)\n";

        for (auto& r : parsed) all_results.push_back(std::move(r));
    }

    if (all_results.empty()) {
        std::cerr << "\nNo results parsed. Were the binaries compiled?\n";
        std::cerr << "From repo root, try:\n";
        std::cerr << "  g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/svm/svm.cpp -o svm -lpthread\n";
        std::cerr << "  g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/knn/knn.cpp -o knn -lpthread\n";
        std::cerr << "  g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/mlp/mlp.cpp -o mlp -lpthread\n";
        std::cerr << "  g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/dt/dt.cpp   -o dt  -lpthread\n";
        std::cerr << "  g++ -std=c++17 -O3 -march=native -fopenmp src/cpp/nb/nb.cpp   -o nb  -lpthread\n";
        return 1;
    }

    write_csv(csv_out, all_results);
    std::cout << "\nWrote " << all_results.size() << " rows to " << csv_out << "\n";

    print_summary(all_results);

    return 0;
}
