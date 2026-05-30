<div align="center">

# difflib-fast

**The exact `difflib` similarity ratio — up to `8,500×` faster.**

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![exact difflib](https://img.shields.io/badge/difflib-byte--for--byte-blue.svg)](https://docs.python.org/3/library/difflib.html)
[![vs difflib: up to 8,500×](https://img.shields.io/badge/vs%20difflib-up%20to%208%2C500%C3%97-brightgreen.svg)](#python-package)

The **same number** Python's `difflib` gives you — byte-for-byte, no `autojunk` approximation. A corpus
stdlib `difflib` would chew on for **~30 minutes** clusters in **~0.2 seconds**.

</div>

```python
import difflib_fast

difflib_fast.ratio("the quick brown fox", "the quick brown dog")   # 0.8947368421052632  (== difflib)
difflib_fast.ratio(pairs)   # list[float] — computed across every core inside Rust, GIL released
```

<div align="center">

| from Python · real corpus · 12 cores | throughput | vs stdlib `difflib` |
|---|---|---|
| `ratio(a, b)` — one call | 2.4k pairs/s | **104×** |
| `ratio(pairs)` — batch, all cores | 15k pairs/s | **628×** |
| `cluster_canonicals(corpus)` — the real workload | 199k pairs/s | **8,541×** |

*23 pairs/s → 199,000 pairs/s on the same task. Same answer. ([how ↓](#how-it-works))*

</div>

And a pure-Rust crate, with **zero** Python dependency by default:

```rust
use difflib_fast::ratio;
// bit-for-bit identical to difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
assert_eq!(ratio("the quick brown fox", "the quick brown dog"), 0.8947368421052632);
```

```toml
difflib-fast = "0.3"
```

---

## The idea

`ratio = 2·M / (len(a) + len(b))`, where `M` is the total size of the Ratcliff–Obershelp matching
blocks. The metric is exact and well-defined — including `difflib`'s tie-break **and its
argument-order asymmetry**, both of which this crate reproduces bit-for-bit.

The catch is how you compute `M`. `difflib` does it by re-scanning every occurrence of each character;
on long, small-alphabet text — canonicalized source code, log lines, DNA — a handful of popular
characters turn that into a quadratic crawl. `difflib-fast` computes the longest common substring with
a **suffix automaton** in `O(|a|+|b|)` regardless of how often a character repeats. Same answer. No
crawl.

> Two independent implementations of `M` live in this crate — the suffix automaton, and a from-scratch
> port of `difflib`'s own recursion. The test suite asserts they are **bit-identical**. That equality
> *is* the correctness guarantee.

---

## What it does

Two things, both exact:

```rust
use difflib_fast::{ratio, cluster_canonicals};

// 1. pairwise similarity — a drop-in for difflib's ratio
let r = ratio("def add(a, b): return a + b", "def add(x, y): return x + y");

// 2. cluster a whole corpus by similarity (single-linkage, exact min pairwise ratio per cluster)
let corpus = vec![
    "def add(a, b): return a + b".to_string(),
    "def add(x, y): return x + y".to_string(),
    "totally unrelated".to_string(),
];
let clusters = cluster_canonicals(&corpus, 0.5);  // → [([0, 1], 0.84…)]
```

The clustering path is the one built for scale: each string's automaton is **prebuilt once** and
reused across the whole `n²` join, dissimilar pairs **early-exit** the moment the threshold is decided,
and the work is spread across cores with `rayon`. A cluster's reported `min_sim` is its exact minimum
pairwise ratio.

---

## API

Two tiers. The **free functions** are the stateless CPU drop-in:

| function | what |
|---|---|
| `ratio(a, b) -> f64` | exact `difflib` ratio (dispatches automaton ⇄ `b2j` per input) |
| `ratio_many(&[(String, String)]) -> Vec<f64>` | exact `ratio` for a batch of pairs, in parallel (rayon) |
| `gestalt::gestalt_ratio(a, b) -> f64` | the suffix-automaton path directly |
| `cluster_canonicals(&[String], threshold) -> Vec<(Vec<usize>, f64)>` | exact single-linkage clusters + min pairwise ratio |
| `cluster_canonicals_lsh(&[String], threshold, num_perm, band_rows)` | scalable MinHash-LSH variant (candidate-gen + exact verify) for very large corpora |

And `Rationer` is the **stateful handle** that owns long-lived resources (rayon pool, and on
macOS the Metal device) once and reuses them across calls — same exact output, with an optional GPU
path:

```rust
use difflib_fast::Rationer;

let r = Rationer::new();                          // builder().build(); default GpuPlusCpu
let clusters = r.cluster_canonicals(&corpus, 0.6); // GPU-accelerated on macOS (see below)
let ratios   = r.ratio_many(&pairs);               // CPU
```

`Rationer::builder().concurrency(Concurrency::Cpu | Gpu | GpuPlusCpu).threads(n).delta(d).build()`
configures it; `PreparedRationer` (`r.prepare(&strings)`) amortizes SAM-build over many index-pair
queries.

### GPU (macOS + Metal)

Behind the `gpu` cargo feature on Apple Silicon, `Rationer::cluster_canonicals` offloads the
suffix-automaton `matching_stats` walk to a Metal compute kernel — **byte-for-byte identical output**,
**~1.1–1.4×** end-to-end vs the (already fast) CPU path on a single large group:

| corpus | `cluster_canonicals` GPU vs CPU |
|---|---|
| mypy | **1.38×** |
| ha | 1.25× |
| sympy | 1.17× |
| django | 1.14× |
| transformers | 1.08× |

It's a modest, honest win: the GPU only does `matching_stats` (~⅓ of the per-pair cost); the
`longest_in` recursion, filtering and assembly stay on CPU, so Amdahl caps it. `ratio_many` and
`cluster_canonicals_multi` measured *slower* on the GPU at every size tested, so they stay on CPU by
default (the GPU paths remain opt-in via `DFGPU_RATIO_MANY_THRESHOLD` / `DFGPU_MULTI_THRESHOLD`). With
the feature off, on non-macOS, or with no Metal device, every call quietly runs on CPU.

```toml
difflib-fast = { version = "0.3", features = ["gpu"] }   # macOS only
```

---

## Correctness

This crate's entire reason to exist is being *exactly* `difflib` — so correctness is enforced, not
hoped for:

- **two implementations, one answer** — the suffix-automaton path and the `b2j` reference port are
  asserted bit-identical on thousands of fuzzed pairs (`fast_matches_reference`);
- **18k-assertion threshold gate** — every early-exit decision matches the full ratio
  (`qualifies_matches_ratio_threshold`);
- **`difflib` reference values**, including non-ASCII.

```bash
cargo test
```

---

## Benchmarks

Exact byte-for-byte RO on real canonicalized Python (top-level function bodies, `ast.dump`-shape),
Apple M3 Pro (6 P + 6 E cores). `pairs/s` = pairwise `ratio` decisions per second. Full methodology,
per-repo tables, and the C++ / Python harnesses are in [`benchmarks.md`](benchmarks.md).

**Clustering throughput** — the production path (prebuilt automaton + threshold early-exit + rayon),
`cargo run --release --features bench --bin bench`:

| repo | raw ratio, 1 thread | threshold @0.5, 1 thread | threshold @0.5, 12 threads |
|---|---|---|---|
| django | 24.3k pairs/s | 117k pairs/s | **936k pairs/s** |
| sympy | 23.6k | 97.9k | **828k** |
| ha | 21.5k | 69.6k | 468k |
| mypy | 14.1k | 60.6k | 271k |
| transformers | 3.5k¹ | 47.3k | 362k |

¹ transformers' model code has unusually long functions (per-pair RO is `O(L·log L)`), so raw
throughput is lower — the same reason `difflib` struggles there.

**vs other exact-RO implementations** — single thread, same metric, across the five repos above:

| competitor | difflib-fast speedup |
|---|---|
| Python stdlib `difflib` (pure Python — the original) | **245–1070×** |
| C++ `duckie/difflib` (well-optimized b2j) | **1.4–3.4×** |
| CyDifflib (Cython `difflib`) | **~3–10×** |
| `difflib` (Rust `difflib` crate) | 11–50× |
| `gestalt_ratio` (Rust Ratcliff–Obershelp crate) | 18–96× |

Against the **original** Python `difflib` it's a different universe: pure-Python `SequenceMatcher`
manages 15–77 exact-RO ratios/s on this code, so difflib-fast is **245–1070× faster single-threaded —
and ~1600–6600× on all 12 cores.** CyDifflib has no in-process parallelism (GIL-bound, no `nogil` /
batch API), so on all 12 cores — difflib-fast (rayon) vs CyDifflib (multiprocessing) on the same
qualifying-pairs task — the gap is **60–242×**. CyDifflib's *default* `autojunk=True` is faster but differs from the exact ratio on ~100%
of these pairs (mean |Δ| ≈ 0.22): a different metric, not a drop-in. The libraries that beat
difflib-fast on raw speed (RapidFuzz, strsim) likewise compute a *different* metric (Indel /
Levenshtein), not `difflib`'s.

---

## Python package

Swap `difflib.SequenceMatcher(...).ratio()` for `difflib_fast.ratio(...)` and the **same number** comes
back — hundreds of times faster per call, and **thousands of times** faster when you score a whole
corpus. That's the entire change: `import difflib_fast`.

**No PyPI** — `pip install difflib-fast` won't work. Two ways to install:

**1. Prebuilt wheel — no Rust toolchain needed.** The wheels are `cp39-abi3`, so one wheel per platform
works on **every CPython ≥ 3.9, including 3.14**. GitHub Releases isn't a package index, so pip can't
pick the wheel for you — grab the one for your platform from the
[Releases page](https://github.com/prostomarkeloff/difflib-fast/releases/latest) and install it by URL:

```bash
# macOS Apple Silicon — swap the filename for your platform (see below):
pip install https://github.com/prostomarkeloff/difflib-fast/releases/download/v0.3.0/difflib_fast-0.3.0-cp39-abi3-macosx_11_0_arm64.whl
```

| platform | wheel suffix |
|---|---|
| macOS Apple Silicon | `…-macosx_11_0_arm64.whl` |
| macOS Intel | `…-macosx_10_12_x86_64.whl` |
| Linux x86_64 | `…-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |
| Linux aarch64 | `…-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` |
| Windows x64 | `…-win_amd64.whl` |

**2. From source — needs a Rust toolchain** (pip drives maturin automatically, no manual build):

```bash
pip install git+https://github.com/prostomarkeloff/difflib-fast
```

```python
import difflib_fast

# one pair → one float, byte-for-byte difflib (autojunk=False)
difflib_fast.ratio("the quick brown fox", "the quick brown dog")   # 0.8947368421052632

# cluster a corpus (single-linkage, exact min pairwise ratio per cluster)
difflib_fast.cluster_canonicals(["def f(a): ...", "def f(x): ...", "other"], 0.5)
# → [([0, 1], 0.86…)]
```

### All cores, no contention

`ratio` is **overloaded**: hand it a *list of pairs* and it returns a list of ratios, computing them
**in parallel across every core inside Rust** — with the GIL released. You don't touch a
`ThreadPoolExecutor`, you don't fight the GIL; you just pass the batch:

```python
pairs = [(a, b) for a in corpus for b in corpus]
difflib_fast.ratio(pairs)              # list[float], one per pair — fanned out over all cores
difflib_fast.ratio(pairs, threads=4)   # …or cap it to 4 workers for this call
```

By default it uses every core; pass `threads=N` to any batch call (`ratio(pairs, …)`,
`cluster_canonicals(…)`) to cap the pool for that call, or set `RAYON_NUM_THREADS` to change the
process-wide default with no code. Thread count never changes the result — only the speed.

This matters because Python *can't* parallelize the stdlib version: `difflib` in a `ThreadPoolExecutor`
stays GIL-bound — **23 → 23 pairs/s, zero speedup**. The batch form sidesteps that entirely: the
parallelism lives in Rust, not in Python threads, so it just scales (numbers up top; full harness in
[`benchmarks/bench_python.py`](benchmarks/bench_python.py), measured on real canonicalized Python,
M3 Pro, 12 threads).

Clustering wins biggest because each string's automaton is built **once** and reused across the whole
`n²` join (dissimilar pairs early-exit) — it's not 12× the per-call speed, it's a different algorithm.

The package is **typed** (`py.typed` + `.pyi` stubs — pyright/mypy see the overloads), gated behind the
`python` cargo feature so the pure-Rust crate keeps **zero** Python dependency by default. Built with
[maturin](https://github.com/PyO3/maturin) (mixed layout: compiled `_difflib_fast` +
`python/difflib_fast/` package); build locally into a venv with `maturin develop --release --features python`.

### GPU from Python (macOS)

The **macOS wheels ship the Metal GPU path**. Use the `Rationer` handle — its `cluster_canonicals`
runs on the GPU when the group is large enough to pay for the dispatch, otherwise CPU; same
byte-for-byte answer either way:

```python
import difflib_fast as df

r = df.Rationer(concurrency="gpu+cpu")        # "cpu" | "gpu" | "gpu+cpu" (default)
r.concurrency                                  # "gpu+cpu" if Metal came up, else "cpu"
r.cluster_canonicals(corpus, 0.6)              # GPU-accelerated (~1.1–1.4× on Apple Silicon)
r.ratio_many(pairs)                            # CPU (the GPU offload loses here)
```

On Linux/Windows wheels, or with no Metal device, a `Rationer` transparently runs everything on CPU.
Build a GPU wheel locally on macOS with
`maturin develop --release --features python,gpu` (the CLI `--features` **replaces** the pyproject
default, so list both).

---

## Also: exact cosine similarity join (`simjoin`)

The same "exact, or it's a bug" discipline, pointed at a different metric. **`simjoin`** is an exact
all-pairs **weighted-cosine** similarity join over sparse non-negative vectors — *every* pair with
`cos ≥ t`, no LSH, no approximation — on the provably-SOTA **L2AP** algorithm (inverted index +
Cauchy–Schwarz prefix pruning; Anastasiu & Karypis, ICDE'14). It's the principled exact replacement for
"shingle candidates → verify" near-duplicate detection: documents = functions, dimensions = canonical
lines, weights = IDF — i.e. **exact Type-3 code-clone detection**.

```python
import difflib_fast as df

# documents as token lists → TF-IDF in Rust → every pair with cosine ≥ 0.8
docs = [["def _fn(_v0):", "return _v0 + 1"],
        ["def _fn(_v0):", "return _v0 + 1"],   # an exact clone of doc 0
        ["import os", "import sys"]]
df.cosine_join(docs, 0.8)          # → [(0, 1, 1.0)]   tuples are (j, i, cos), j < i
df.cosine_join(docs, 0.8, "gpu")   # same join, the dot-products run on the Metal GPU
```

Three backends, one argument (`concurrency=`) — all auto-parallel across every core (rayon, GIL
released, exactly like `ratio`):

| `concurrency` | how | result |
|---|---|---|
| `"cpu"` | L2AP on all cores | exact `f64` |
| `"gpu+cpu"` | CPU prunes ~99% of candidates, GPU verifies the rest (f32 filter), CPU re-scores survivors exactly | **byte-identical to `"cpu"`** |
| `"gpu"` | CPU prunes, GPU verifies, emit the f32 score | ε-exact (≤ 1 differing pair per **millions**) |

On the real **top-300 PyPI** corpus (287,408 functions, 3.1M clone pairs found) the verify is
memory-**bandwidth**-bound, and the Apple GPU's memory-level parallelism wins it: **53 GB/s** of
random-gather sparse dot-products vs the CPU's 22 GB/s, so the GPU backends run the whole join
**~1.8–2× faster than the (already L2AP-tuned) CPU**, byte-for-byte. Brute force would be ~4·10¹⁰ pairs
(hours); this is seconds. `CosineJoiner(docs)` is the stateful handle (build corpus + GPU upload once,
sweep thresholds); full numbers in [`benchmarks.md`](benchmarks.md#6-similarity-join-simjoin).

In Rust: `difflib_fast::simjoin::{Corpus, cosine_join, cosine_join_with, CosineJoiner}` (GPU backends
behind the `gpu` feature). Same correctness gate as the rest of the crate — the indexed join is
asserted **bit-identical to an O(n²) brute-force oracle** on hundreds of fuzzed corpora.

---

## How it works

The metric is **Ratcliff–Obershelp** (Ratcliff & Obershelp, 1988) computed over a **suffix automaton**
(Blumer et al., 1985) via **matching statistics** (Chang & Lawler, 1994); short, diverse inputs take a
lighter `b2j` path instead, chosen per-input by a cheap work estimate. The composition — exact
byte-for-byte RO this way, tuned for an all-pairs clustering join — is original to this crate.

[`src/gestalt.rs`](src/gestalt.rs) is the engine: its module doc and inline comments cover the
automaton, the endpos range structure that lets one prebuilt automaton serve the whole RO recursion,
the threshold-engine early-exit math, and the measured performance floor.

---

<div align="center">

**Exact `difflib`. None of the wait.**

Made with ⚡ by [@prostomarkeloff](https://github.com/prostomarkeloff)

</div>
