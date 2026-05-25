<div align="center">

# difflib-fast

**The exact `difflib` similarity ratio — at suffix-automaton speed.**

[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![exact difflib](https://img.shields.io/badge/difflib-byte--for--byte-blue.svg)](https://docs.python.org/3/library/difflib.html)

</div>

---

`difflib.SequenceMatcher.ratio()` tells you how similar two strings are — the way a human diff sees
it. It is the right metric, and it is **slow**. `difflib-fast` computes the *exact same number*,
byte-for-byte, with a suffix automaton — so it stays linear exactly where `difflib` falls apart.

```rust
use difflib_fast::ratio;

// bit-for-bit identical to difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
assert_eq!(ratio("the quick brown fox", "the quick brown dog"), 0.8947368421052632);
```

```toml
difflib-fast = "0.1"
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

| function | what |
|---|---|
| `ratio(a, b) -> f64` | exact `difflib` ratio (dispatches automaton ⇄ `b2j` per input) |
| `gestalt::gestalt_ratio(a, b) -> f64` | the suffix-automaton path directly |
| `ratio_reference(a, b) -> f64` | the `b2j` reference port (slow; the test oracle) |
| `cluster_canonicals(&[String], threshold) -> Vec<(Vec<usize>, f64)>` | exact single-linkage clusters + min pairwise ratio |
| `cluster_canonicals_chars(&[Vec<char>], threshold)` | same, over pre-collected `char` vectors |
| `cluster_canonicals_lsh(&[String], threshold, num_perm, band_rows)` | scalable MinHash-LSH variant (candidate-gen + exact verify) for very large corpora |

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

## Python bindings

The pure-Rust crate has **zero** Python dependency. Build with the `python` feature +
[maturin](https://github.com/PyO3/maturin) for a `pip install`-able extension exposing `ratio`,
`cluster_canonicals`, and `cluster_canonicals_lsh`:

```bash
maturin develop --release --features python
```

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
