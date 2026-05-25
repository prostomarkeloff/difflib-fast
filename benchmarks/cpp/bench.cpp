// C++ head-to-head: duckie/difflib (vendored difflib.h) exact Ratcliff-Obershelp ratio, single- and
// multi-threaded, on the same NUL-separated corpora as the Rust benchmarks. Reports pairs/s; compare
// against difflib-fast's `compare` example output for the same corpus + N/P.
//
//   build: clang++ -O3 -std=c++14 -pthread benchmarks/cpp/bench.cpp -o /tmp/cppbench
//   run:   /tmp/cppbench <corpus.bin> [N] [P_pairs]
//
// auto_junk=false -> exact RO (byte-for-byte difflib ratio), the same metric difflib-fast and the
// gestalt_ratio crate compute. Multi-thread = persistent std::threads striding the pair list (no
// per-pass spawn), adaptive timing (run until >=0.4s), matching the Rust harness methodology.
#include "difflib.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using difflib::SequenceMatcher;
using clk = std::chrono::steady_clock;

static double ro_ratio(const std::string& a, const std::string& b) {
    SequenceMatcher<std::string> m(a, b, nullptr, /*auto_junk=*/false); // exact RO
    return m.ratio();
}
static double secs(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "benchmarks/corpora/mypy.canon.bin";
    size_t N = argc > 2 ? std::stoul(argv[2]) : 120;
    size_t budget = argc > 3 ? std::stoul(argv[3]) : 400;

    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string data = ss.str();
    std::vector<std::string> strings;
    size_t start = 0;
    for (size_t i = 0; i <= data.size() && strings.size() < N; ++i) {
        if (i == data.size() || data[i] == '\0') {
            if (i > start) strings.emplace_back(data.substr(start, i - start));
            start = i + 1;
        }
    }
    size_t n = strings.size();
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i < n && pairs.size() < budget; ++i)
        for (size_t j = i + 1; j < n && pairs.size() < budget; ++j) pairs.emplace_back(i, j);
    size_t np = pairs.size();
    size_t mean_len = 0;
    for (auto& s : strings) mean_len += s.size();
    if (n) mean_len /= n;

    // single thread, adaptive
    double ser_rate;
    {
        auto t0 = clk::now();
        long passes = 0;
        volatile double acc = 0;
        while (secs(t0) < 0.4 && passes < 100000) {
            double a = 0;
            for (auto& p : pairs) a += ro_ratio(strings[p.first], strings[p.second]);
            acc = acc + a;
            ++passes;
        }
        ser_rate = passes * (double)np / secs(t0);
    }

    // multi thread: persistent threads stride the pair list until stop, counting pairs processed
    unsigned nt = std::thread::hardware_concurrency();
    if (nt == 0) nt = 8;
    double par_rate;
    {
        std::atomic<bool> stop{false};
        std::vector<long> counts(nt, 0);
        std::vector<std::thread> ts;
        auto t0 = clk::now();
        for (unsigned t = 0; t < nt; ++t) {
            ts.emplace_back([&, t]() {
                long c = 0;
                volatile double acc = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    double a = 0;
                    for (size_t k = t; k < np; k += nt) a += ro_ratio(strings[pairs[k].first], strings[pairs[k].second]);
                    acc = acc + a;
                    for (size_t k = t; k < np; k += nt) ++c;
                }
                counts[t] = c;
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        stop.store(true, std::memory_order_relaxed);
        for (auto& th : ts) th.join();
        double el = secs(t0);
        long total = 0;
        for (long c : counts) total += c;
        par_rate = total / el;
    }

    std::string name(path);
    auto slash = name.find_last_of('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    auto dot = name.find(".canon.bin");
    if (dot != std::string::npos) name = name.substr(0, dot);
    std::printf("%-13s N=%zu mean_len=%zu pairs=%zu  C++ duckie/difflib (auto_junk=false, exact RO)\n", name.c_str(), n, mean_len, np);
    std::printf("  1 thread   : %9.0f pairs/s\n", ser_rate);
    std::printf("  %2u threads : %9.0f pairs/s\n", nt, par_rate);
    return 0;
}
