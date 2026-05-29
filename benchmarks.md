# difflib-fast — benchmarks

Exact **Ratcliff–Obershelp** (Python `difflib.SequenceMatcher(..., autojunk=False).ratio()`) throughput
and head-to-head vs every other implementation we could find. `pairs/s` = pairwise `ratio` decisions
per second. Speedups (`Nx`) are **difflib-fast ÷ competitor**.

## Setup

- **Machine:** Apple M3 Pro — 6 performance + 6 efficiency cores, no SMT (12 logical), macOS.
- **Corpora:** top-level function bodies from 5 real Python repos, AST-canonicalized (`ast.dump`-shape),
  NUL-separated in [`benchmarks/corpora/`](benchmarks/corpora/). This is the realistic input for clone /
  duplicate-definition detection: long, small-alphabet, boilerplate-heavy text — exactly where
  difflib's popular-character `b2j` rescans blow up.

  | repo | files | functions | ~mean canonical length |
  |---|---|---|---|
  | mypy | 437 | 2,201 | ~1,700 |
  | django | 2,910 | 1,874 | ~1,100 |
  | transformers | 4,403 | 6,307 | ~2,100 |
  | ha (Home Assistant) | 9,498 | 14,553 | ~1,400 |
  | sympy | 1,589 | 17,839 | ~1,600 |

- **Metric:** exact RO, `autojunk=False`. Every "exact RO" comparison below is **byte-for-byte
  verified** against difflib-fast (mismatch counts / max|Δ| reported per section).
- **Harnesses** (in [`benchmarks/`](benchmarks/); the `run_all.sh` / `profile.sh` runners are local
  perf tooling, not tracked):
  - `cargo run --release --features bench --bin bench` — difflib-fast self (raw / threshold / cluster).
  - `cargo run --release --example compare` — vs the Rust crates `difflib` + `gestalt_ratio`.
  - `bench_vs_cydifflib.py` — vs CyDifflib, single thread.
  - `bench_parallel.py` — vs CyDifflib, multiprocessing (its only parallel option).
  - `cpp/bench.cpp` — vs C++ `duckie/difflib` *(harness in progress)*.
- **Caveat:** difflib-fast's per-pair `ratio()` is **stateless** (rebuilds its suffix automaton every
  call). These per-pair numbers do **not** use its prebuild + early-exit + cache advantages — the
  clustering path (§1) is much faster still. Slow competitors are measured on smaller `N`/pair budgets
  so each command stays <30s; rates are pair-count-independent, so they remain comparable. Absolute
  difflib-fast numbers differ slightly between harnesses (different sample/`N`); speedups *within* a
  table are same-process, same-pairs, and internally consistent.

---

## 1. difflib-fast throughput (self)

The production path: prebuild each string's suffix automaton once, then an early-exit all-pairs join
(length-block + `quick_ratio` filter + threshold-aware RO), rayon-parallel.

| repo | raw ratio, 1T | threshold @0.5, 1T | threshold @0.5, 12T | clustering @0.5, 12T |
|---|---|---|---|---|
| django | 24,341 /s | 117,142 /s | **936,111 /s** | 483,373 /s |
| sympy | 23,594 | 97,863 | **827,762** | 448,970 |
| ha | 21,538 | 69,641 | 468,255 | 234,977 |
| mypy | 14,127 | 60,605 | 270,902 | 178,194 |
| transformers | 3,521 | 47,272 | 362,307 | 250,569 |

(raw 1T at N=300; threshold 1T at N=1000; 12T at N=2000. transformers is slowest everywhere — model
code has unusually long functions, and per-pair RO is `O(L·log L)`.)

---

## 1b. GPU (Metal) — heterogeneous `cluster_canonicals`

Behind the `gpu` feature on Apple Silicon, `Rationer::cluster_canonicals` runs the suffix-automaton
`matching_stats` walk on the M3 GPU (Metal compute) while the CPU keeps the filters + `longest_in`
recursion + assembly. **Byte-for-byte identical** to the CPU path (cluster counts equal across all
backends, every run). Measured via the `Rationer` API across its three `Concurrency` backends, one
group, threshold 0.6, N=1500, M3 Pro, best-of-2:

| corpus | CPU | GPU | GPU+CPU | GPU speedup |
|---|---|---|---|---|
| mypy | 2.25 s | 1.62 s | 1.63 s | **1.38×** |
| ha | 2.64 s | 2.10 s | 2.13 s | **1.25×** |
| sympy | 1.57 s | 1.34 s | 1.31 s | **1.17×** |
| django | 1.09 s | 0.95 s | 0.98 s | **1.14×** |
| transformers | 2.69 s | 2.49 s | 2.35 s | **1.08×** |

**Honest read.** The GPU only offloads `matching_stats` — ~⅓ of the per-pair cycle budget; the
`longest_in` chain walk (the dominant cost), the `quick_ratio` filters, the SAM build and the
union-find assembly all stay on CPU. By Amdahl that caps the end-to-end win even before the
fixed per-dispatch overhead, so cluster lands at **1.1–1.4×** and does *not* scale up with N
(at N≈2200 the single dispatch's `(fstate, fmatch)` output buffer grows and goes bandwidth-bound,
dropping back toward ~1.15×). The other two operations measured **slower** on the GPU at every size
tried — `ratio_many` 0.82–0.93× (61k–404k pairs; the CPU intern + prebuilt-SAM path is already
efficient) and `cluster_canonicals_multi` 0.59–0.99× (the find-dup-defs many-small-groups shape never
amortizes the dispatch) — so both stay on CPU by default (GPU opt-in via `DFGPU_RATIO_MANY_THRESHOLD`
/ `DFGPU_MULTI_THRESHOLD`). A widely-quoted "1.5–1.7×" is a *kernel-only* `matching_stats` microbench
(the kernel itself is ~3× CPU throughput); it does not survive end-to-end.

---

## 2. vs CyDifflib (Python) — the fast byte-for-byte Cython difflib

[CyDifflib](https://github.com/rapidfuzz/CyDifflib) (by the RapidFuzz author) is the fastest existing
byte-for-byte difflib. Same metric (forced `autojunk=False`); **0 byte-for-byte mismatches** in all
runs.

### 2a. Single thread (per-pair `ratio`)

difflib-fast here is **stateless** (no reuse); CyDifflib gets `set_seq2` reuse (its *best* per-pair
path — amortizes the `b2j` index). difflib-fast wins even handicapped:

| repo | difflib-fast (stateless) | CyDifflib (stateless) | CyDifflib (seq2-reuse, best) | difflib-fast vs CyDifflib-best |
|---|---|---|---|---|
| django | 4,424 /s | 1,257 /s | 1,314 /s | **3.4×** |
| ha | 2,566 | 475 | 514 | **5.0×** |
| mypy | 2,717 | 498 | 519 | **5.2×** |
| transformers | 2,139 | 354 | 340 | **6.3×** |
| sympy | 1,757 | 189 | 184 | **9.6×** |

> CyDifflib's *default* `autojunk=True` is faster but differs from exact RO on ~100% of these pairs
> (mean |Δ| ≈ 0.22) — a different metric, not a drop-in for exact difflib. Longer functions ⇒ bigger
> gap (difflib's popular-character pathology; `autojunk=False` forbids the mitigation).

### 2b. Parallel (all 12 cores)

CyDifflib has **no in-process parallelism** — its source has no `nogil`, no `prange`, no batch/`cdist`
API, and the hot methods hold the GIL (empirically: a 12-thread pool gives **0.98–1.01×**, i.e. zero
speedup). Its only parallel option is **multiprocessing** (one GIL per process), which we give it in
full: all cores, `seq2`-reuse per worker, spawn initializer so the corpus ships once. difflib-fast uses
rayon (in-process, no GIL). Same task (qualifying pairs ≥ 0.5), same input, qualifying counts agree
(e.g. sympy 6,286 vs 6,435 — difflib's argument-order asymmetry on boundary edges).

| repo | N | CyDifflib (multiprocessing, 12 proc) | difflib-fast (rayon, 12) | speedup |
|---|---|---|---|---|
| django | 300 | 12,672 /s | 766,621 /s | **60×** |
| ha | 300 | 7,156 | 500,056 | **70×** |
| mypy | 300 | 4,960 | 366,148 | **74×** |
| sympy | 300 | 5,031 | 1,139,349 | **226×** |
| transformers | 100¹ | 926 | 223,887 | **242×** |

¹ transformers' long functions force a small `N`, so CyDifflib's multiprocessing overhead isn't
amortized (its rate is understated); the single-thread 6.3× is the cleaner transformers figure.

---

## 3. vs other Rust crates (exact RO, same process, system allocator for all)

Two existing Rust crates compute the same metric. Both are pure per-pair functions, so we give them the
**same rayon parallelism** difflib-fast gets (no GIL in Rust).

- [`gestalt_ratio`](https://crates.io/crates/gestalt_ratio) — Ratcliff–Obershelp ratio. **max|Δ| = 0.0**
  → byte-for-byte the same metric. The true exact-RO analogue.
- [`difflib`](https://crates.io/crates/difflib) (port of Python difflib) — **max|Δ| ≈ 0.20–0.29** → a
  *different* result on long inputs (autojunk-style popular-element pruning), i.e. it is both **slower
  and not exact**.

### 3a. Single thread

| repo | difflib-fast | `difflib` crate | vs `difflib` | `gestalt_ratio` | vs `gestalt_ratio` |
|---|---|---|---|---|---|
| mypy | 2,635 /s | 239 /s | **11.0×** | 143 /s | **18.4×** |
| django | 3,611 | 274 | **13.2×** | 128 | **28.1×** |
| ha | 3,212 | 159 | **20.2×** | 83 | **38.6×** |
| sympy | 2,549 | 81 | **31.6×** | 42 | **61.0×** |
| transformers | 1,609 | 32 | **50.2×** | 17 | **96.1×** |

### 3b. 12 threads (rayon for all three)

| repo | difflib-fast | `difflib` crate | vs `difflib` | `gestalt_ratio` | vs `gestalt_ratio` |
|---|---|---|---|---|---|
| mypy | 12,495 /s | 1,386 /s | **9.0×** | 871 /s | **14.3×** |
| django | 15,907 | 1,610 | **9.9×** | 795 | **20.0×** |
| ha | 12,772 | 965 | **13.2×** | 528 | **24.2×** |
| sympy | 9,303 | 448 | **20.8×** | 253 | **36.7×** |
| transformers | 6,042 | 150 | **40.2×** | 60 | **100.2×** |

(Naive crates collapse on long functions — `gestalt_ratio` does **17 pairs/s** on transformers — because
they have no popular-character mitigation. This is exactly the pathology difflib-fast's suffix automaton
removes.)

---

## 4. vs C++ `duckie/difflib`

[`duckie/difflib`](https://github.com/duckie/difflib) — header-only C++11 difflib port, exact RO with
`auto_junk=false`. Vendored at [`benchmarks/cpp/`](benchmarks/cpp/); harness
[`bench.cpp`](benchmarks/cpp/bench.cpp) built with `clang++ -O3 -std=c++14 -pthread`, persistent
`std::thread`s, same adaptive methodology and same `N`/pair budgets as §3. **This is the strongest
competitor** — a well-optimized `b2j` difflib, far ahead of the naive Rust crates. Both sides here are
stateless per-pair (rebuild their index each call); same metric (exact RO).

difflib-fast's `ratio` **dispatches per pair**: a suffix automaton for long/repetitive strings, and an
optimized difflib `b2j` for short/diverse ones (count-sort index, all buffers reused → zero per-pair
allocation, `get_unchecked` inner loop, `set_len` instead of zeroing — all ASM-driven). The dispatch
signal is `W = Σ_c count_a(c)·count_b(c)` (b2j's *match count* — the "match-sensitive" cost parameter
from the Hunt–Szymanski line of work), thresholded per element. So difflib-fast gets the better of both
algorithms on every input. Numbers below are the **dispatched** `ratio` vs `duckie`.

### 4a. Single thread — difflib-fast wins on all 5

| repo | difflib-fast | C++ `duckie/difflib` | speedup |
|---|---|---|---|
| mypy | 5,107 /s | 3,213 /s | **1.6×** |
| django | 4,252 | 3,038 | **1.4×** |
| ha | 3,812 | 1,963 | **1.9×** |
| sympy | 2,202 | 1,057 | **2.1×** |
| transformers | 1,370 | 402 | **3.4×** |

### 4b. 12 threads

| repo | difflib-fast | C++ `duckie/difflib` | speedup |
|---|---|---|---|
| mypy | 23,381 /s | 22,549 /s | **1.04×** |
| ha | 17,217 | 13,791 | **1.25×** |
| sympy | 9,340 | 7,744 | **1.21×** |
| transformers | 4,222 | 3,446 | **1.23×** |
| django | 17,976 | 21,498 | 0.84× (≈tie; MT-noisy) |

**Honest read:** `duckie/difflib` is a strong, well-optimized `b2j` — far ahead of the naive Rust crates,
and the only competitor that gets close. Single-threaded, difflib-fast's dispatched `ratio` beats it on
**all five** (1.4–3.4×): the optimized `b2j` alone is faster than `duckie`'s (Rust + count-sort + reused
buffers vs C++ + `std::unordered_map`), and the automaton takes the long/repetitive bodies where `b2j`
degrades. Multi-threaded, difflib-fast wins 4/5; on `django` it's a noise-level tie (the parallel
stateless-`ratio` microbench is memory-bandwidth-bound, and per-pair `b2j` MT throughput varies ±~25%
run-to-run). The dispatch is **provably imperfect with a cheap signal** — two corpora with identical
character statistics (`distinct≈47`, `concentration≈0.08`) prefer opposite paths at the same `W`, because
the true discriminator is RO recursion depth, which can't be known without doing the comparison; a
speculative "run b2j, abort to the automaton on overrun" variant was tried and is worse (the abort wastes
b2j work). So the committed histogram dispatch is the optimum among cheap policies. Either way,
difflib-fast's decisive edge remains the **clustering join** (§1: 270k–936k pairs/s) where the automaton
is prebuilt once and reused across all n² scans.

---

## 5. vs Python stdlib `difflib`

The original — `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()`, pure Python. Same metric
(byte-for-byte, 0 mismatches). It's slow enough that its rate is measured over a ~24 s window (cycling
the pair set); difflib-fast is its `bench` raw all-pairs throughput (1 thread and all 12 cores).

| repo | stdlib difflib (1 thread) | difflib-fast 1T | difflib-fast 12T | df 1T | df 12T |
|---|---|---|---|---|---|
| mypy | 40 pairs/s | 14,908 | 102,275 | **373×** | **2,557×** |
| django | 77 | 26,470 | 174,614 | 344× | 2,268× |
| ha | 71 | 22,405 | 149,315 | 316× | 2,103× |
| sympy | 24 | 25,660 | 157,563 | **1,069×** | **6,565×** |
| transformers | 15 | 3,670 | 23,939 | 245× | 1,596× |

Pure-Python `SequenceMatcher` does **15–77 exact-RO ratios/s** on this code (sympy worst: long,
repetitive expressions — the popular-character blowup at its peak). difflib-fast is **245–1070×
single-threaded and ~1,600–6,600× on all cores**, for the identical number. This is the gap the crate
exists to close.

---

## Landscape (what computes what)

| library | lang | metric | exact difflib? | parallel | notes |
|---|---|---|---|---|---|
| **difflib-fast** | Rust | Ratcliff–Obershelp | ✅ byte-for-byte | ✅ rayon (in-proc) | suffix automaton; + clustering |
| stdlib `difflib` | Python | Ratcliff–Obershelp | ✅ (the reference, autojunk=False) | ❌ | the original; pure Python |
| CyDifflib | C++/Cython | Ratcliff–Obershelp | ✅ (autojunk=False) | ❌ GIL-bound (MP only) | difflib algorithm |
| `gestalt_ratio` | Rust | Ratcliff–Obershelp | ✅ (Δ=0) | (pure fn) | naive recursion |
| `difflib` crate | Rust | RO + autojunk | ❌ (Δ≈0.25) | (pure fn) | autojunk-style pruning |
| `duckie/difflib` | C++ | Ratcliff–Obershelp | ✅ (auto_junk=false) | manual (`std::thread`) | header-only; well-optimized b2j |
| RapidFuzz / strsim | C++ / Rust | **Indel / Levenshtein** | ❌ different metric | ✅ | faster, but not RO |

**Bottom line** — among byte-for-byte exact-RO implementations:

- vs **Python stdlib `difflib`** (the original it replaces): **245–1070× single-thread**, **~1,600–6,600×
  on all cores** — pure Python manages only 15–77 ratios/s on this code.
- vs **CyDifflib** (Python): **3–10× faster single-thread**, and because CyDifflib is GIL-bound (no
  in-process parallelism — multiprocessing only) **60–242× faster on all cores**.
- vs the Rust **`gestalt_ratio`** crate (same metric): **18–96× single-thread**, 14–100× multi-thread.
- vs the Rust **`difflib`** crate: 11–50× — and that crate isn't even exact (autojunk-style, |Δ|≈0.25).
- vs **C++ `duckie/difflib`** (the real competitor — optimized b2j): **difflib-fast wins all 5
  single-threaded (1.4–3.4×)** via its per-pair dispatch (optimized b2j for short/diverse, automaton
  for long/repetitive); multi-threaded it wins 4/5 with `django` a noise-level tie. On the **clustering
  join** — prebuilt automaton reused across all pairs + threshold early-exit (§1) — difflib-fast is in a
  different regime (270k–936k pairs/s) the per-pair libraries don't address.

The libraries that beat difflib-fast outright on raw speed (RapidFuzz, strsim) do so by computing a
**different metric** (Indel/Levenshtein), not difflib's ratio — so they aren't drop-in replacements for
`difflib.ratio()`.
