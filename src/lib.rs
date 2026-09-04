//! `difflib-fast` — fast, **byte-for-byte exact** difflib Ratcliff–Obershelp ("gestalt") similarity.
//!
//! [`ratio`] / [`gestalt::gestalt_ratio`] are a drop-in for
//! `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()`:
//!   `ratio = 2·M / (len(a)+len(b))`, where `M` is the total size of the Ratcliff–Obershelp matching
//! blocks. The result — including difflib's tie-break (longest; earliest-a; earliest-b) and its
//! argument-order asymmetry — is reproduced exactly, but `M` is computed via a **suffix automaton**
//! (LCS in O(|a|+|b|) regardless of character frequency) instead of difflib's popular-character
//! `b2j` rescans. On long, small-alphabet text (e.g. canonicalized source code) this is the
//! difference between difflib's pathological case and a linear scan.
//!
//! Beyond the per-pair ratio, [`cluster_canonicals`] does an exact single-linkage **clustering** of a
//! corpus at a similarity threshold — length blocking + `quick_ratio` filter, then only the edges
//! that connect the components are decided (most-similar-first batches over a union-find), on
//! automata built only for the strings actually scanned, in parallel via rayon — and reports each
//! cluster with its exact minimum pairwise ratio, computed under the cluster's running minimum as a
//! cap. [`cluster_canonicals_lsh`]
//! is the scalable `MinHash`-LSH variant (candidate generation + exact verification) for very large
//! corpora past the O(n²) wall.
//!
//! Two independent implementations of `M` back this: the suffix-automaton path ([`gestalt`]) and a
//! straight port of difflib's `b2j` recursion (a test-only reference oracle); the test suite asserts
//! they are bit-identical, which is the crate's core correctness gate.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

pub mod gestalt;
pub use gestalt::gestalt_ratio;

// Exact all-pairs weighted-cosine similarity join (AllPairs/L2AP) — inverted index + prefix
// filtering. The principled replacement for shingle-candidate near-duplicate detection.
pub mod simjoin;

// Heterogeneous CPU+GPU exact RO — Metal compute backend (Apple Silicon). Behind `feature = "gpu"`
// + `cfg(target_os = "macos")`; the rest of the crate falls back to the CPU path when it's off or
// no Metal device is available at runtime.
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod gpu;

// GPU batched sparse-cosine — the offload experiment for `simjoin`'s bandwidth-bound verify step.
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod simjoin_gpu;

// `Rationer` — the stateful, GPU-accelerated similarity/clustering handle. Always available;
// degrades to CPU when the `gpu` feature is off or no Metal device can be acquired.
pub mod rationer;
pub use rationer::{Concurrency, PreparedRationer, Rationer, RationerBuilder};

/// Dispatch threshold: take the `b2j` path while its estimated work per element
/// (`Σ_c count_a·count_b / (|a|+|b|)`) stays at/below this; above it the automaton wins. Tuned on real
/// canonicalized-code corpora. Override at runtime with `DF_WORK_FACTOR`. (The clustering join always
/// uses the automaton — there it's prebuilt once and reused across all n² scans, so it always wins.)
const B2J_WORK_FACTOR: u64 = 34;

/// Fast exact difflib ratio. Bit-identical to
/// `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()`.
///
/// Dispatches by length: short inputs take the lightweight difflib `b2j` recursion (cheap to set up),
/// long inputs take the suffix-automaton LCS (frequency-independent, so it doesn't degrade on
/// repetitive text). Both paths are exact and agree bit-for-bit.
#[must_use]
pub fn ratio(a: &str, b: &str) -> f64 {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    ratio_chars(&av, &bv)
}

/// ASCII char histogram (canonical code is ~all ASCII; non-ASCII folded into one overflow bucket).
fn ascii_counts(s: &[char]) -> ([u32; 128], u32) {
    let mut c = [0u32; 128];
    let mut other = 0u32;
    for &ch in s {
        let u = ch as u32;
        if u < 128 {
            c[u as usize] += 1;
        } else {
            other += 1;
        }
    }
    (c, other)
}

fn work_factor() -> u64 {
    use std::sync::OnceLock;
    static F: OnceLock<u64> = OnceLock::new();
    *F.get_or_init(|| std::env::var("DF_WORK_FACTOR").ok().and_then(|s| s.parse().ok()).unwrap_or(B2J_WORK_FACTOR))
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
fn ratio_chars(a: &[char], b: &[char]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    let (ca, oa) = ascii_counts(a);
    let (cb, ob) = ascii_counts(b);
    // Non-ASCII present (rare in canonical code): the b2j fast path is ASCII-only, so use the
    // automaton, which compares arbitrary code points.
    if oa > 0 || ob > 0 {
        return gestalt::gestalt_ratio_chars(a, b);
    }
    // Dispatch on b2j's estimated work `W = Σ_c count_a(c)·count_b(c)` (exactly the positions the
    // first-block scan visits) per element. Below the threshold b2j is cheaper to set up; above it the
    // repetitive case makes b2j's recursion blow up and the automaton wins. Committing to one path
    // (rather than speculatively running b2j and aborting) avoids wasting work on the clear-automaton
    // cases. The histograms just computed double as the b2j build's counts, so dispatch is near-free.
    let mut w = 0u64;
    for i in 0..128 {
        w += u64::from(ca[i]) * u64::from(cb[i]);
    }
    if w <= work_factor() * total as u64 {
        ratio_b2j_chars(a, b, &cb)
    } else {
        gestalt::gestalt_ratio_chars(a, b)
    }
}

/// The difflib `b2j` ratio path directly (bypasses the length dispatch) — a second,
/// structurally-distinct exact implementation kept as a test oracle for the suffix-automaton path.
#[cfg(test)]
#[must_use]
fn ratio_b2j(a: &str, b: &str) -> f64 {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let (cb, ob) = ascii_counts(&bv);
    if ob > 0 {
        return gestalt::gestalt_ratio_chars(&av, &bv); // b2j fast path is ASCII-only
    }
    ratio_b2j_chars(&av, &bv, &cb)
}

/// Exact difflib [`ratio`] for many `(a, b)` pairs at once, computed **in parallel across all cores**
/// (rayon). `ratio_many(pairs)[i]` equals `ratio(&pairs[i].0, &pairs[i].1)`, bit-for-bit.
///
/// This is the batch primitive: hand it the whole workload and the fan-out happens inside Rust — from
/// Python it runs with the GIL released, so it saturates every core with no `ThreadPoolExecutor` and
/// no per-call Python overhead.
#[must_use]
pub fn ratio_many(pairs: &[(String, String)]) -> Vec<f64> {
    pairs.par_iter().map(|(a, b)| ratio(a, b)).collect()
}

// ───────────────────────── reference b2j path (independent oracle) ─────────────────────────
// A faithful port of difflib's own algorithm (popular-character `b2j` index + the
// `find_longest_match` recursion). Slower (this is what the suffix automaton replaces), kept as a
// second, structurally-different implementation so the tests can assert the fast path matches it.
// Test-only: not part of the public API.

/// Code point → ascending positions in `b`.
#[cfg(test)]
fn build_b2j(b: &[char]) -> HashMap<char, Vec<usize>> {
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, &c) in b.iter().enumerate() {
        b2j.entry(c).or_default().push(j);
    }
    b2j
}

/// difflib `find_longest_match` over `a[alo:ahi] × b[blo:bhi]`; returns `(i, j, k)`.
#[cfg(test)]
#[allow(clippy::similar_names)]
fn find_longest(a: &[char], b2j: &HashMap<char, Vec<usize>>, alo: usize, ahi: usize, blo: usize, bhi: usize) -> (usize, usize, usize) {
    let mut besti = alo;
    let mut bestj = blo;
    let mut bestsize = 0usize;
    let mut j2_prev: HashMap<usize, usize> = HashMap::new();
    for (i, ch) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut j2_cur: HashMap<usize, usize> = HashMap::new();
        if let Some(positions) = b2j.get(ch) {
            for &j in positions {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let prev = if j > blo { *j2_prev.get(&(j - 1)).unwrap_or(&0) } else { 0 };
                let k = prev + 1;
                j2_cur.insert(j, k);
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
        }
        j2_prev = j2_cur;
    }
    (besti, bestj, bestsize)
}

/// Total size of the Ratcliff–Obershelp matching blocks (difflib `get_matching_blocks`).
#[cfg(test)]
#[allow(clippy::many_single_char_names)]
fn matching_count(a: &[char], b: &[char], b2j: &HashMap<char, Vec<usize>>) -> usize {
    let mut total = 0usize;
    let mut stack: Vec<(usize, usize, usize, usize)> = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = stack.pop() {
        let (i, j, k) = find_longest(a, b2j, alo, ahi, blo, bhi);
        if k > 0 {
            total += k;
            if alo < i && blo < j {
                stack.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                stack.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    total
}

/// Reference (b2j) difflib ratio — structurally distinct from the suffix-automaton path; the test
/// suite asserts [`gestalt_ratio`] equals this exactly. Test oracle only (not public API).
#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
#[must_use]
fn ratio_reference(a: &str, b: &str) -> f64 {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let total = av.len() + bv.len();
    if total == 0 {
        return 1.0;
    }
    let b2j = build_b2j(&bv);
    2.0 * (matching_count(&av, &bv, &b2j) as f64) / (total as f64)
}

// ───────────────────────── optimized b2j path (ASCII, short strings) ─────────────────────────
// difflib's own algorithm with CPython's vector `j2len` + touched-index clearing (O(matches)
// per row, not a per-row HashMap), and a **count-sort** b2j index (offsets[128] + a flat positions
// array) instead of a per-char `HashMap<char, Vec>`. All buffers are reused thread-locals → ZERO
// per-pair heap allocation, which is what lets it scale across threads (the HashMap version churned
// the allocator). ASCII-only (the caller routes non-ASCII to the automaton). Exact — equals the SAM
// path byte-for-byte (tested).

#[derive(Default)]
struct B2jScratch {
    offsets: Vec<u32>,   // [129] prefix sums: char c's positions are positions[offsets[c]..offsets[c+1])
    positions: Vec<u32>, // b positions grouped by char, ascending within each char (count-sort order)
    cursor: Vec<u32>,    // write cursors during the count-sort fill
    j2len: Vec<u32>,
    erase: Vec<(u32, u32)>,  // (position, value) set in the previous row → reset to 0 this row
    affect: Vec<(u32, u32)>, // (position, value) to set this row
    stack: Vec<(usize, usize, usize, usize)>,
}

thread_local! {
    /// Reused b2j scratch — no per-pair allocation in the short-string path.
    static B2J: RefCell<B2jScratch> = RefCell::new(B2jScratch::default());
}

/// difflib `b2j` ratio: count-sort index (offsets + positions, all buffers reused → ZERO per-pair
/// allocation) + difflib's `find_longest_match` recursion with a reused vector `j2len`. `cb` = ASCII
/// counts of `b` (computed by the dispatcher, reused here as the count-sort counts); `b` must be ASCII.
/// Second exact implementation; the tests assert it equals the automaton path byte-for-byte.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn ratio_b2j_chars(a: &[char], b: &[char], cb: &[u32; 128]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    B2J.with_borrow_mut(|s| {
        let B2jScratch { offsets, positions, cursor, j2len, erase, affect, stack } = s;
        // count-sort b2j: prefix-sum cb → offsets, then place each b position into its char bucket.
        offsets.clear();
        offsets.resize(129, 0);
        for c in 0..128 {
            offsets[c + 1] = offsets[c] + cb[c];
        }
        cursor.clear();
        cursor.extend_from_slice(&offsets[..128]);
        // size `positions` WITHOUT zeroing: the count-sort below writes all b.len() slots before any
        // read (each b position lands in exactly one bucket), so a resize(_, 0) memset would be pure
        // bandwidth waste — and under many threads that waste is what otherwise caps b2j's scaling.
        positions.clear();
        positions.reserve(b.len());
        #[allow(clippy::uninit_vec)]
        // SAFETY: u32 has no invalid bit patterns; every index 0..b.len() is written by the
        // count-sort below (one write per b position) before find_longest_b2j reads `positions`.
        unsafe {
            positions.set_len(b.len());
        }
        for (j, &ch) in b.iter().enumerate() {
            let c = ch as usize; // ASCII guaranteed
            positions[cursor[c] as usize] = j as u32;
            cursor[c] += 1;
        }
        j2len.clear();
        j2len.resize(b.len() + 1, 0);
        // at most one (key, value) per b-position per row → reserve once so `push` never reallocates.
        erase.reserve(b.len() + 1);
        affect.reserve(b.len() + 1);
        stack.clear();
        stack.push((0, a.len(), 0, b.len()));
        let mut m = 0usize;
        while let Some((alo, ahi, blo, bhi)) = stack.pop() {
            let (bi, bj, bk) = find_longest_b2j(a, offsets, positions, j2len, erase, affect, alo, ahi, blo, bhi);
            if bk > 0 {
                m += bk;
                if alo < bi && blo < bj {
                    stack.push((alo, bi, blo, bj));
                }
                if bi + bk < ahi && bj + bk < bhi {
                    stack.push((bi + bk, ahi, bj + bk, bhi));
                }
            }
        }
        2.0 * m as f64 / total as f64
    })
}

/// difflib `find_longest_match` with a reused vector `j2len` (cleared via the touched-index lists) and
/// the count-sort b2j index (char `c`'s positions = `positions[offsets[c]..offsets[c+1])`).
///
/// The inner loop runs `M` times (the match count), so it is the whole cost — `get_unchecked` there
/// removes the per-iteration `j2len[j]` bounds check the optimizer can't elide (≈10% on b2j, per the
/// disassembly). `affect`/`erase` are pre-reserved by the caller so `push` never reallocates in the
/// loop. A plain `for i in alo..ahi` (not `take().skip()`) keeps the iterator out of the hot path.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation, clippy::similar_names)]
fn find_longest_b2j(
    a: &[char],
    offsets: &[u32],
    positions: &[u32],
    j2len: &mut [u32],
    erase: &mut Vec<(u32, u32)>,
    affect: &mut Vec<(u32, u32)>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut bi, mut bj, mut bk) = (alo, blo, 0usize);
    erase.clear();
    // SAFETY (whole loop): `i ∈ [alo, ahi) ⊆ [0, a.len())`. `c < 128 = offsets.len()-1` is guarded, so
    // `offsets[c]`/`offsets[c+1]` are in bounds and `[lo, hi) ⊆ [0, positions.len())`. Every match
    // position `j` satisfies `blo ≤ j < bhi ≤ b.len()`, and written keys are `j+1 ≤ b.len()`, both
    // `< j2len.len() = b.len()+1`. `erase`/`affect` hold only such keys, so their clears are in bounds.
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        for i in alo..ahi {
            affect.clear();
            let c = *a.get_unchecked(i) as usize;
            if c < 128 {
                let lo = *offsets.get_unchecked(c) as usize;
                let hi = *offsets.get_unchecked(c + 1) as usize;
                for &jj in positions.get_unchecked(lo..hi) {
                    let j = jj as usize;
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = *j2len.get_unchecked(j) as usize + 1;
                    affect.push((j as u32 + 1, k as u32));
                    if k > bk {
                        bi = i + 1 - k;
                        bj = j + 1 - k;
                        bk = k;
                    }
                }
            }
            for &(p, _) in erase.iter() {
                *j2len.get_unchecked_mut(p as usize) = 0;
            }
            for &(p, v) in affect.iter() {
                *j2len.get_unchecked_mut(p as usize) = v;
            }
            std::mem::swap(erase, affect);
        }
        for &(p, _) in erase.iter() {
            *j2len.get_unchecked_mut(p as usize) = 0;
        }
    }
    (bi, bj, bk)
}

// ───────────────────────────── cheap exact upper-bound filters ─────────────────────────────

/// difflib `real_quick_ratio`: a length-only upper bound on `ratio` (cheap skip).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn real_quick_ratio(a: &[char], b: &[char]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * (a.len().min(b.len()) as f64) / (total as f64)
}

/// Sorted `(char, count)` multiset of `a` — precomputed once per string so the `quick_ratio`
/// upper-bound filter is a linear merge over the (small) alphabet instead of a per-pair `HashMap`.
pub(crate) fn char_counts(a: &[char]) -> Vec<(char, u32)> {
    // ASCII (canonical code, nearly always): one histogram pass instead of a sort.
    let mut hist = [0u32; 128];
    if a.iter().all(|&c| {
        let u = c as usize;
        if u < 128 {
            hist[u] += 1;
            true
        } else {
            false
        }
    }) {
        return hist
            .iter()
            .enumerate()
            .filter(|&(_, &k)| k > 0)
            .map(|(i, &k)| (char::from(u8::try_from(i).expect("ascii")), k))
            .collect();
    }
    let mut v = a.to_vec();
    v.sort_unstable();
    let mut out: Vec<(char, u32)> = Vec::new();
    for c in v {
        match out.last_mut() {
            Some(last) if last.0 == c => last.1 += 1,
            _ => out.push((c, 1)),
        }
    }
    out
}

/// difflib `quick_ratio` from precomputed sorted char-counts: `2·Σ min(count_a, count_b)/(|a|+|b|)`,
/// an exact upper bound on `ratio`. Merge of two sorted multisets — O(distinct chars), no hashing.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn quick_ratio_counts(ca: &[(char, u32)], cb: &[(char, u32)], total: usize) -> f64 {
    if total == 0 {
        return 1.0;
    }
    let (mut x, mut y, mut matches) = (0usize, 0usize, 0u32);
    while x < ca.len() && y < cb.len() {
        match ca[x].0.cmp(&cb[y].0) {
            std::cmp::Ordering::Less => x += 1,
            std::cmp::Ordering::Greater => y += 1,
            std::cmp::Ordering::Equal => {
                matches += ca[x].1.min(cb[y].1);
                x += 1;
                y += 1;
            }
        }
    }
    2.0 * f64::from(matches) / total as f64
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn uf_find32(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

// Env-gated diagnostics: set DIFFLIB_FAST_PROGRESS=1 to stream phase timings + progress to stderr
// from inside the Rust hot path (off in production — zero output, ~zero cost).
fn progress_on() -> bool {
    std::env::var_os("DIFFLIB_FAST_PROGRESS").is_some()
}

// ───────────────────────────────────── exact clustering ─────────────────────────────────────

/// Below this many strings (or candidate pairs) a clustering call runs on the calling thread:
/// the rayon fan-out costs more than the work, and the caller is usually already parallel over
/// many such calls.
const SERIAL_BELOW: usize = 32;

/// The strings' suffix automata, built on first use. A pair `(lo, hi)` scans `lo` against
/// `hi`'s automaton, so only the strings that are the `hi` side of some pair that reaches the
/// scan ever need one — in a group of two that is one automaton, not two, and the build is the
/// single most expensive step per string.
pub(crate) struct LazySams<'a> {
    chars: &'a [Vec<char>],
    cells: Vec<std::sync::OnceLock<gestalt::Sam>>,
}

impl<'a> LazySams<'a> {
    pub(crate) fn new(chars: &'a [Vec<char>]) -> Self {
        Self { chars, cells: (0..chars.len()).map(|_| std::sync::OnceLock::new()).collect() }
    }

    /// Wrap automata that already exist (the GPU paths build every one up front).
    pub(crate) fn built(chars: &'a [Vec<char>], sams: Vec<gestalt::Sam>) -> Self {
        debug_assert_eq!(chars.len(), sams.len());
        Self { chars, cells: sams.into_iter().map(std::sync::OnceLock::from).collect() }
    }

    pub(crate) fn get(&self, i: usize) -> &gestalt::Sam {
        self.cells[i].get_or_init(|| gestalt::build_sam(&self.chars[i]))
    }
}

/// Candidate pairs `(lo, hi)`, `lo < hi`, that survive the two exact upper bounds. Length
/// blocking: ratio>=T ⟹ |short|/|long| >= T/(2-T), so in length-sorted order each string only
/// reaches a contiguous run of (not-too-much-longer) strings — the inner loop breaks as soon as
/// `real_quick_ratio` drops below T. Then the char-multiset `quick_ratio`. Together they kill
/// 70–90 % of pairs without a scan, and never drop a qualifying pair.
#[allow(clippy::cast_possible_truncation)]
fn candidate_pairs(chars: &[Vec<char>], counts: &[Vec<(char, u32)>], threshold: f64) -> Vec<(u32, u32)> {
    let n = chars.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| chars[i].len());
    let row = |p: usize, local: &mut Vec<(u32, u32)>| {
        let i = order[p];
        let a = &chars[i];
        for &j in &order[p + 1..] {
            let b = &chars[j];
            if real_quick_ratio(a, b) < threshold {
                break; // lengths only grow ⇒ all remaining partners also fail the bound
            }
            if quick_ratio_counts(&counts[i], &counts[j], a.len() + b.len()) < threshold {
                continue;
            }
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            local.push((lo as u32, hi as u32));
        }
    };
    if n < SERIAL_BELOW {
        let mut out = Vec::new();
        for p in 0..n {
            row(p, &mut out);
        }
        return out;
    }
    (0..n)
        .into_par_iter()
        .flat_map_iter(|p| {
            let mut local = Vec::new();
            row(p, &mut local);
            local
        })
        .collect()
}

/// The edges that connect the clusters: a subset of the qualifying pairs `(i<j, ratio >= T)` with
/// the same connected components — which is all single-linkage clustering needs from them.
///
/// The exact edge test is the expensive step (a scan plus the recursion, and a rejected pair whose
/// ratio is anywhere near the threshold runs most of the recursion before the bound closes). But
/// once two strings are known to be in one component, whether their own pair is an edge changes
/// nothing: the component is the same, and the cluster's minimum is taken over every intra pair
/// anyway, by a capped computation that is far cheaper than an edge test. So the candidates are
/// visited most-similar-first (by their `quick_ratio` bound) in batches — each batch tested in
/// parallel, its edges merged into a union-find before the next — and a pair already connected
/// by then is not tested at all. A dense cluster of `k` strings costs about `k` tests instead of
/// `k²/2`, and its chained non-edges are never rejected the hard way.
///
/// A call with a single candidate computes that pair's exact ratio instead (`Some`): it is the
/// cluster's minimum, and deciding the edge only to scan again would cost more.
#[allow(clippy::cast_possible_truncation)]
fn spanning_edges(
    chars: &[Vec<char>],
    counts: &[Vec<(char, u32)>],
    sams: &LazySams<'_>,
    mut cand: Vec<(u32, u32)>,
    threshold: f64,
) -> Vec<(usize, usize, Option<f64>)> {
    if let [(lo, hi)] = cand[..] {
        let (lo, hi) = (lo as usize, hi as usize);
        return gestalt::gestalt_edge(&chars[lo], &chars[hi], sams.get(hi), threshold)
            .map(|r| (lo, hi, Some(r)))
            .into_iter()
            .collect();
    }
    // most similar first: the quick_ratio bound descending, then the pair for determinism
    let bound = |&(lo, hi): &(u32, u32)| {
        let (lo, hi) = (lo as usize, hi as usize);
        quick_ratio_counts(&counts[lo], &counts[hi], chars[lo].len() + chars[hi].len()).to_bits()
    };
    let mut keyed: Vec<(u64, u32, u32)> = cand.iter().map(|p| (u64::MAX - bound(p), p.0, p.1)).collect();
    keyed.sort_unstable();
    cand.clear();
    cand.extend(keyed.iter().map(|&(_, lo, hi)| (lo, hi)));
    drop(keyed);

    let mut parent: Vec<u32> = (0..chars.len() as u32).collect();
    let find = uf_find32;
    let edge = |&(lo, hi): &(u32, u32)| {
        let (lo, hi) = (lo as usize, hi as usize);
        gestalt::gestalt_edge_bounded(&chars[lo], &chars[hi], sams.get(hi), threshold)
    };
    let mut edges: Vec<(usize, usize, Option<f64>)> = Vec::new();
    if cand.len() < SERIAL_BELOW {
        for &(lo, hi) in &cand {
            let (rl, rh) = (find(&mut parent, lo), find(&mut parent, hi));
            if rl != rh && edge(&(lo, hi)) {
                parent[rl as usize] = rh;
                edges.push((lo as usize, hi as usize, None));
            }
        }
        return edges;
    }
    let total = cand.len();
    let mut pos = 0;
    // the first batch is sized so a large call keeps every worker busy from the start
    let mut batch = (total / 16).clamp(SERIAL_BELOW, 256);
    let mut todo: Vec<(u32, u32)> = Vec::new();
    let progress = progress_on();
    while pos < total {
        let end = (pos + batch).min(total);
        todo.clear();
        for &(lo, hi) in &cand[pos..end] {
            if find(&mut parent, lo) != find(&mut parent, hi) {
                todo.push((lo, hi));
            }
        }
        pos = end;
        batch = (batch * 2).min(EDGE_BATCH_MAX);
        if todo.is_empty() {
            continue;
        }
        // consecutive pairs share the automaton being scanned
        todo.sort_unstable_by_key(|&(lo, hi)| (hi, lo));
        let found: Vec<(u32, u32)> = if todo.len() < SERIAL_BELOW {
            todo.iter().filter(|p| edge(p)).copied().collect()
        } else {
            // the automata this batch scans, built up front so no worker waits on another's build
            let mut need: Vec<u32> = todo.iter().map(|p| p.1).collect();
            need.dedup();
            need.par_iter().for_each(|&hi| {
                sams.get(hi as usize);
            });
            todo.par_iter().filter(|p| edge(p)).copied().collect()
        };
        for &(lo, hi) in &found {
            let (rl, rh) = (find(&mut parent, lo), find(&mut parent, hi));
            if rl != rh {
                parent[rl as usize] = rh;
            }
            edges.push((lo as usize, hi as usize, None));
        }
        if progress {
            #[allow(clippy::cast_precision_loss)]
            let pct = pos as f64 / total as f64 * 100.0;
            eprintln!("    [difflib-fast] edges: {pos}/{total} candidates ({pct:.0}%), {} tested, {} edges", todo.len(), edges.len());
        }
    }
    edges
}

/// Largest batch of candidate pairs tested between two union-find merges in [`spanning_edges`].
const EDGE_BATCH_MAX: usize = 2048;

/// Union-find over qualifying edge pairs → clusters (size >= 2), each with its exact min intra-pair
/// ratio.
///
/// The min is exact and cheap to certify: an intra pair without a cached exact ratio is computed
/// with `gestalt_ratio_capped` against the cluster's **shared** running minimum — the function
/// returns the exact ratio when it is at or below the cap and accept-exits the instant the pair
/// is proven above it. The pair that IS the
/// minimum is always at or below every cap it can meet, so it is always computed exactly; every
/// other pair is either exact or provably not the minimum. Hence the result equals the minimum
/// over all pairs' exact ratios, whatever order the pairs are visited in.
///
/// To make the cap bite early, each cluster's uncached pairs are ordered by their `quick_ratio`
/// (an upper bound: the pair with the smallest bound has the smallest ratio ceiling), the first
/// one is computed alone so its exact value seeds the cap, and the rest run in parallel from there.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(crate) fn assemble(
    n: usize,
    edges: Vec<(usize, usize, Option<f64>)>,
    chars: &[Vec<char>],
    sams: &LazySams<'_>,
) -> Vec<(Vec<usize>, f64)> {
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut parent: Vec<usize> = (0..n).collect();
    for &(i, j, _) in &edges {
        let (ri, rj) = (uf_find(&mut parent, i), uf_find(&mut parent, j));
        if ri != rj {
            parent[ri] = rj;
        }
    }
    let root: Vec<usize> = (0..n).map(|i| uf_find(&mut parent, i)).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by_key(|&i| (root[i], i));
    // clusters = runs of one root, members ascending; `cid[i]` = the cluster of member i
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut cid: Vec<usize> = vec![usize::MAX; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && root[idx[end]] == root[idx[start]] {
            end += 1;
        }
        if end - start >= 2 {
            for &i in &idx[start..end] {
                cid[i] = clusters.len();
            }
            clusters.push(idx[start..end].to_vec());
        }
        start = end;
    }
    if clusters.is_empty() {
        return Vec::new();
    }
    // every edge lies inside one cluster; the ones that carry their exact ratio are that
    // cluster's cached evidence, the others are computed like any other intra pair
    let mut cl_edges: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); clusters.len()];
    for (i, j, r) in edges {
        if let Some(r) = r {
            cl_edges[cid[i]].push((i, j, r));
        }
    }
    // the running minimum per cluster, as f64 bits: ratios are non-negative, so their bit
    // patterns order like the values and `fetch_min` is a float min
    let caps: Vec<AtomicU64> = clusters.iter().map(|_| AtomicU64::new(1.0_f64.to_bits())).collect();
    // the intra pairs without a cached ratio: (cluster, lo, hi)
    let mut jobs: Vec<(u32, u32, u32)> = Vec::new();
    for (k, members) in clusters.iter().enumerate() {
        let es = &mut cl_edges[k];
        es.sort_unstable_by_key(|e| (e.0, e.1));
        let mut cmin = 1.0_f64;
        for e in es.iter() {
            cmin = cmin.min(e.2);
        }
        caps[k].store(cmin.to_bits(), Ordering::Relaxed);
        if es.len() == members.len() * (members.len() - 1) / 2 {
            continue; // every pair carries its exact ratio
        }
        let mut e = 0;
        for (pi, &i) in members.iter().enumerate() {
            for &j in &members[pi + 1..] {
                if e < es.len() && es[e].0 == i && es[e].1 == j {
                    e += 1;
                    continue;
                }
                jobs.push((k as u32, i as u32, j as u32));
            }
        }
    }
    if !jobs.is_empty() {
        // order each cluster's jobs by the quick_ratio bound, ascending
        let mut involved = vec![false; n];
        for &(_, i, j) in &jobs {
            involved[i as usize] = true;
            involved[j as usize] = true;
        }
        let counts: Vec<Vec<(char, u32)>> = if jobs.len() < SERIAL_BELOW {
            (0..n).map(|i| if involved[i] { char_counts(&chars[i]) } else { Vec::new() }).collect()
        } else {
            (0..n).into_par_iter().map(|i| if involved[i] { char_counts(&chars[i]) } else { Vec::new() }).collect()
        };
        let bound = |&(_, i, j): &(u32, u32, u32)| {
            let (i, j) = (i as usize, j as usize);
            quick_ratio_counts(&counts[i], &counts[j], chars[i].len() + chars[j].len())
        };
        let mut keyed: Vec<(u32, u64, u32, u32)> =
            jobs.iter().map(|job| (job.0, bound(job).to_bits(), job.1, job.2)).collect();
        keyed.sort_unstable();
        let run = |&(cluster, _, lo, hi): &(u32, u64, u32, u32)| {
            let (cluster, lo, hi) = (cluster as usize, lo as usize, hi as usize);
            let cap = f64::from_bits(caps[cluster].load(Ordering::Relaxed));
            let ratio = gestalt::gestalt_ratio_capped(&chars[lo], &chars[hi], sams.get(hi), cap);
            caps[cluster].fetch_min(ratio.to_bits(), Ordering::Relaxed);
        };
        // seed every cluster's cap with its lowest-bound pair, then the rest with the caps live
        let mut first: Vec<(u32, u64, u32, u32)> = Vec::new();
        let mut rest: Vec<(u32, u64, u32, u32)> = Vec::with_capacity(keyed.len());
        for job in keyed {
            if first.last().is_none_or(|f| f.0 != job.0) {
                first.push(job);
            } else {
                rest.push(job);
            }
        }
        if first.len() + rest.len() < SERIAL_BELOW {
            first.iter().for_each(run);
            rest.iter().for_each(run);
        } else {
            let mut need: Vec<u32> = rest.iter().map(|j| j.3).collect();
            need.par_sort_unstable();
            need.dedup();
            need.par_iter().for_each(|&hi| {
                sams.get(hi as usize);
            });
            first.par_iter().for_each(run);
            rest.par_iter().for_each(run);
        }
    }
    let mut out: Vec<(Vec<usize>, f64)> = clusters
        .into_iter()
        .zip(caps)
        .map(|(members, cap)| (members, f64::from_bits(cap.into_inner())))
        .collect();
    out.sort_by(|a, b| a.0[0].cmp(&b.0[0]));
    out
}

/// Exact single-linkage clustering over pre-collected `char` vectors: returns each cluster (member
/// indices, sorted) with its exact minimum pairwise ratio. O(n²) early-exit join, rayon-parallel.
#[must_use]
#[doc(hidden)] // low-level Vec<char> entry — used by the bench bin + `Rationer`; prefer `cluster_canonicals`.
pub fn cluster_canonicals_chars(chars: &[Vec<char>], threshold: f64) -> Vec<(Vec<usize>, f64)> {
    let n = chars.len();
    if n < 2 {
        return Vec::new();
    }
    let counts: Vec<Vec<(char, u32)>> = if n < SERIAL_BELOW {
        chars.iter().map(|c| char_counts(c)).collect()
    } else {
        chars.par_iter().map(|c| char_counts(c)).collect()
    };
    let cand = candidate_pairs(chars, &counts, threshold);
    let sams = LazySams::new(chars);
    let edges = spanning_edges(chars, &counts, &sams, cand, threshold);
    assemble(n, edges, chars, &sams)
}

/// `cluster_canonicals(canonicals, threshold)` → `[(member indices, min pairwise ratio)]`.
///
/// Exact single-linkage clustering: `ratio >= threshold` joins two strings; each returned cluster
/// (size >= 2) carries its exact minimum intra-cluster ratio. Bit-identical to the reference
/// pairwise clustering — just far faster (suffix automaton + early-exit + rayon).
#[must_use]
pub fn cluster_canonicals(canonicals: &[String], threshold: f64) -> Vec<(Vec<usize>, f64)> {
    let chars: Vec<Vec<char>> = canonicals.iter().map(|s| s.chars().collect()).collect();
    cluster_canonicals_chars(&chars, threshold)
}

// ─────────────────────────────── scalable MinHash-LSH variant ───────────────────────────────

const SHINGLE_K: usize = 9; // char-k-gram length for MinHash shingles (calibrated on real code)

fn fnv1a_bytes(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn fnv1a_u64s(values: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in values {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Distinct char-k-gram shingle hashes of `s` (the set `MinHash` estimates Jaccard over).
fn shingle_hashes(s: &str) -> Vec<u64> {
    let bytes = s.as_bytes();
    if bytes.len() <= SHINGLE_K {
        return vec![fnv1a_bytes(bytes)];
    }
    let mut set: HashSet<u64> = HashSet::new();
    for window in bytes.windows(SHINGLE_K) {
        set.insert(fnv1a_bytes(window));
    }
    set.into_iter().collect()
}

/// `num` deterministic `(a, b)` hash permutations (fixed seed → reproducible signatures).
fn make_perms(num: usize) -> Vec<(u64, u64)> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..num).map(|_| (next() | 1, next())).collect()
}

fn minhash(shingles: &[u64], perms: &[(u64, u64)]) -> Vec<u64> {
    perms
        .iter()
        .map(|&(a, b)| shingles.iter().map(|&h| a.wrapping_mul(h).wrapping_add(b)).min().unwrap_or(u64::MAX))
        .collect()
}

/// LSH candidate pairs: documents that share a full band signature in any band (an O(n)-ish proxy
/// for "Jaccard above the band threshold" — recall tuned via `band_rows`).
fn lsh_candidates(sigs: &[Vec<u64>], band_rows: usize) -> HashSet<(usize, usize)> {
    let bands = sigs.first().map_or(0, Vec::len).checked_div(band_rows).unwrap_or(0);
    let mut candidates: HashSet<(usize, usize)> = HashSet::new();
    for band in 0..bands {
        let lo = band * band_rows;
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for (d, sig) in sigs.iter().enumerate() {
            buckets.entry(fnv1a_u64s(&sig[lo..lo + band_rows])).or_default().push(d);
        }
        for docs in buckets.values() {
            for a in 0..docs.len() {
                for b in (a + 1)..docs.len() {
                    candidates.insert((docs[a].min(docs[b]), docs[a].max(docs[b])));
                }
            }
        }
    }
    candidates
}

/// `cluster_canonicals_lsh(canonicals, threshold, num_perm, band_rows)`: the scalable path.
///
/// `MinHash`-LSH generates candidate pairs in ~O(n) (skipping the O(n²) dissimilar pairs); each
/// candidate is then **verified with the exact ratio**, so clusters + `min_sim` match the exact path
/// (modulo LSH recall, tuned high via `band_rows`). Filter-verification, in the `BayesLSH`-Lite /
/// `SourcererCC` lineage. Use past the O(n²) wall (>100k strings); for exact recall use
/// [`cluster_canonicals`].
#[must_use]
pub fn cluster_canonicals_lsh(canonicals: &[String], threshold: f64, num_perm: usize, band_rows: usize) -> Vec<(Vec<usize>, f64)> {
    let debug = progress_on();
    let start = std::time::Instant::now();
    let chars: Vec<Vec<char>> = canonicals.iter().map(|s| s.chars().collect()).collect();
    let n = chars.len();
    let perms = make_perms(num_perm);
    let sigs: Vec<Vec<u64>> = canonicals.par_iter().map(|s| minhash(&shingle_hashes(s), &perms)).collect();
    if debug {
        eprintln!("    [difflib-fast] lsh: {n} signatures in {:.2}s", start.elapsed().as_secs_f64());
    }
    let candidates = lsh_candidates(&sigs, band_rows);
    if debug {
        eprintln!("    [difflib-fast] lsh: {} candidate pairs in {:.2}s", candidates.len(), start.elapsed().as_secs_f64());
    }
    let sams: Vec<gestalt::Sam> = chars.par_iter().map(|c| gestalt::build_sam(c)).collect();
    let cand: Vec<(usize, usize)> = candidates.into_iter().collect();
    let pairs: Vec<(usize, usize, Option<f64>)> = cand
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            gestalt::gestalt_edge(&chars[a], &chars[b], &sams[b], threshold).map(|r| (a, b, Some(r)))
        })
        .collect();
    if debug {
        eprintln!("    [difflib-fast] lsh: {} verified pairs in {:.2}s", pairs.len(), start.elapsed().as_secs_f64());
    }
    let sams = LazySams::built(&chars, sams);
    assemble(n, pairs, &chars, &sams)
}

// ───────────────────────────────── optional Python bindings ─────────────────────────────────

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    /// Run `f` on a rayon pool of `threads` workers; `threads == 0` uses the global pool (all cores,
    /// itself tunable process-wide via `RAYON_NUM_THREADS`). A bad pool build falls back to global.
    fn run_on_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        if threads == 0 {
            return f();
        }
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(pool) => pool.install(f),
            Err(_) => f(),
        }
    }

    /// Parse a `"cpu" | "gpu" | "gpu+cpu"` backend string into a [`Concurrency`](super::Concurrency).
    fn parse_concurrency(s: &str) -> PyResult<super::Concurrency> {
        use pyo3::exceptions::PyValueError;
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(super::Concurrency::Cpu),
            "gpu" => Ok(super::Concurrency::Gpu),
            "gpu+cpu" | "gpucpu" | "gpu_cpu" => Ok(super::Concurrency::GpuPlusCpu),
            other => Err(PyValueError::new_err(format!(
                "unknown concurrency {other:?}; expected \"cpu\", \"gpu\", or \"gpu+cpu\""
            ))),
        }
    }

    /// `ratio(a, b)` — fast exact `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()`.
    ///
    /// Releases the GIL for the compute (the inputs are copied to owned `String`s first), so calling
    /// this from many Python threads actually scales across cores instead of serializing on the GIL.
    /// Backs the scalar form of the public `ratio`; for a batch the package routes to `ratio_many`.
    #[pyfunction]
    fn ratio(py: Python<'_>, a: &str, b: &str) -> f64 {
        let (a, b) = (a.to_owned(), b.to_owned());
        py.detach(|| super::ratio(&a, &b))
    }

    /// `ratio_many(pairs, threads=0)` → one exact ratio per `(a, b)` pair, **computed across cores
    /// inside Rust** (rayon, GIL released). Backs the list form of the public `ratio` — the
    /// contention-free batch path, no `ThreadPoolExecutor`. `threads=0` = all cores; `threads=N` caps
    /// it to N for this call.
    #[pyfunction]
    #[pyo3(signature = (pairs, threads=0))]
    #[allow(clippy::needless_pass_by_value)]
    fn ratio_many(py: Python<'_>, pairs: Vec<(String, String)>, threads: usize) -> Vec<f64> {
        py.detach(|| run_on_threads(threads, || super::ratio_many(&pairs)))
    }

    /// `cluster_canonicals(canonicals, threshold, threads=0)` → `[(member indices, min pairwise
    /// ratio)]`.
    ///
    /// Fans the all-pairs join out across cores internally (rayon) — one call, full multicore, no
    /// Python threads needed. The GIL is released during the compute, so it never blocks the rest of
    /// your program. `threads=0` = all cores; `threads=N` caps it to N for this call.
    #[pyfunction]
    #[pyo3(signature = (canonicals, threshold, threads=0))]
    #[allow(clippy::needless_pass_by_value)]
    fn cluster_canonicals(py: Python<'_>, canonicals: Vec<String>, threshold: f64, threads: usize) -> Vec<(Vec<usize>, f64)> {
        py.detach(|| run_on_threads(threads, || super::cluster_canonicals(&canonicals, threshold)))
    }

    /// `cluster_canonicals_lsh(canonicals, threshold, num_perm, band_rows, threads=0)` — scalable LSH
    /// variant. `threads=0` = all cores; `threads=N` caps it to N for this call.
    #[pyfunction]
    #[pyo3(signature = (canonicals, threshold, num_perm, band_rows, threads=0))]
    #[allow(clippy::needless_pass_by_value)]
    fn cluster_canonicals_lsh(py: Python<'_>, canonicals: Vec<String>, threshold: f64, num_perm: usize, band_rows: usize, threads: usize) -> Vec<(Vec<usize>, f64)> {
        py.detach(|| run_on_threads(threads, || super::cluster_canonicals_lsh(&canonicals, threshold, num_perm, band_rows)))
    }

    /// `Rationer(concurrency="gpu+cpu", threads=0, delta=0.0)` — stateful handle that owns the
    /// long-lived backend resources (on macOS+`gpu`: Metal device + power-boost assertion) once and
    /// reuses them across calls. The free functions rebuild per-call state each time; a `Rationer`
    /// pays it once.
    ///
    /// `concurrency` ∈ `"cpu" | "gpu" | "gpu+cpu"`. The GPU is only engaged where it measured a net
    /// win — `cluster_canonicals` on a single large group (~1.1–1.4× on Apple Silicon). `ratio` /
    /// `ratio_many` stay on CPU. On a wheel built without the `gpu` feature (the default Linux/Windows
    /// wheels), or with no Metal device, every call quietly runs on CPU with identical output.
    #[pyclass(name = "Rationer")]
    struct PyRationer {
        inner: super::Rationer,
    }

    #[pymethods]
    impl PyRationer {
        #[new]
        #[pyo3(signature = (concurrency="gpu+cpu", threads=0, delta=0.0))]
        fn new(concurrency: &str, threads: usize, delta: f64) -> PyResult<Self> {
            let c = parse_concurrency(concurrency)?;
            let mut b = super::Rationer::builder().concurrency(c).delta(delta);
            if threads > 0 {
                b = b.threads(threads);
            }
            Ok(Self { inner: b.build() })
        }

        /// The active backend after construction-time fallback: `"cpu"`, `"gpu"`, or `"gpu+cpu"`.
        /// A handle requested as `"gpu"` on a non-Metal build/host reports `"cpu"` here.
        #[getter]
        fn concurrency(&self) -> &'static str {
            match self.inner.concurrency() {
                super::Concurrency::Cpu => "cpu",
                super::Concurrency::Gpu => "gpu",
                super::Concurrency::GpuPlusCpu => "gpu+cpu",
            }
        }

        /// Active approximate-RO `delta` (0.0 = exact).
        #[getter]
        fn delta(&self) -> f64 {
            self.inner.delta()
        }

        /// Single-pair exact ratio (always CPU; one pair offers no GPU win).
        fn ratio(&self, py: Python<'_>, a: &str, b: &str) -> f64 {
            let (a, b) = (a.to_owned(), b.to_owned());
            py.detach(|| self.inner.ratio(&a, &b))
        }

        /// Batched exact ratio over `(a, b)` pairs (CPU rayon; GIL released).
        #[allow(clippy::needless_pass_by_value)]
        fn ratio_many(&self, py: Python<'_>, pairs: Vec<(String, String)>) -> Vec<f64> {
            py.detach(|| self.inner.ratio_many(&pairs))
        }

        /// Exact single-linkage clustering at `threshold`. Routes through the GPU on macOS+`gpu`
        /// when the group is large enough to amortize dispatch; otherwise CPU. Same output as the
        /// free `cluster_canonicals`.
        #[allow(clippy::needless_pass_by_value)]
        fn cluster_canonicals(&self, py: Python<'_>, canonicals: Vec<String>, threshold: f64) -> Vec<(Vec<usize>, f64)> {
            py.detach(|| self.inner.cluster_canonicals(&canonicals, threshold))
        }
    }

    /// `cosine_join(docs, threshold, concurrency="cpu", threads=0)` — exact all-pairs weighted-cosine
    /// similarity join over **token documents**. Each `doc` is a list of string tokens (e.g. a
    /// function's canonical lines); they're turned into TF-IDF sparse vectors in Rust and every pair
    /// with cosine `>= threshold` is returned as `(j, i, cos)` with `j < i`.
    ///
    /// `concurrency` ∈ `"cpu" | "gpu" | "gpu+cpu"`: `"cpu"` is exact f64 everywhere; `"gpu+cpu"` is the
    /// exact f64 GPU-accelerated hybrid (byte-identical to `"cpu"`); `"gpu"` runs the dot on the GPU in
    /// f32 (fastest; differs from exact only on pairs within ~1e-6 of the threshold). On a wheel built
    /// without the `gpu` feature, or with no Metal device, the GPU modes quietly fall back to CPU. The
    /// GIL is released for the whole build+join. For repeated joins on one corpus, use `CosineJoiner`.
    #[pyfunction]
    #[pyo3(signature = (docs, threshold, concurrency="cpu", threads=0))]
    #[allow(clippy::needless_pass_by_value)]
    fn cosine_join(
        py: Python<'_>,
        docs: Vec<Vec<String>>,
        threshold: f64,
        concurrency: &str,
        threads: usize,
    ) -> PyResult<Vec<(usize, usize, f64)>> {
        let mode = parse_concurrency(concurrency)?;
        Ok(py.detach(|| {
            run_on_threads(threads, || {
                let corpus = super::simjoin::Corpus::from_token_docs(&docs);
                super::simjoin::cosine_join_with(&corpus, threshold, mode)
            })
        }))
    }

    /// `CosineJoiner(docs)` — stateful similarity-join handle that builds the TF-IDF corpus and (on a
    /// macOS `gpu` wheel) acquires the Metal device + uploads the corpus **once**, then answers
    /// repeated `join(threshold, concurrency=...)` calls reusing them. The free `cosine_join` rebuilds
    /// everything per call; a `CosineJoiner` pays it once — use it to sweep thresholds.
    #[pyclass(name = "CosineJoiner")]
    struct PyCosineJoiner {
        inner: super::simjoin::CosineJoiner,
    }

    #[pymethods]
    impl PyCosineJoiner {
        #[new]
        #[allow(clippy::needless_pass_by_value)]
        fn new(py: Python<'_>, docs: Vec<Vec<String>>) -> Self {
            let inner = py.detach(|| {
                super::simjoin::CosineJoiner::new(super::simjoin::Corpus::from_token_docs(&docs))
            });
            Self { inner }
        }

        /// Number of documents in the corpus.
        fn __len__(&self) -> usize {
            self.inner.corpus().len()
        }

        /// Whether a Metal GPU backend was acquired (always `False` off a macOS `gpu` wheel). When
        /// `False`, every `join` runs on CPU regardless of the `concurrency` argument.
        #[getter]
        fn has_gpu(&self) -> bool {
            self.inner.has_gpu()
        }

        /// Join at `threshold` under `concurrency` (`"cpu" | "gpu" | "gpu+cpu"`), reusing the handle's
        /// resources. Returns `(j, i, cos)` pairs with `j < i`. GIL released for the compute.
        #[pyo3(signature = (threshold, concurrency="cpu", threads=0))]
        fn join(
            &self,
            py: Python<'_>,
            threshold: f64,
            concurrency: &str,
            threads: usize,
        ) -> PyResult<Vec<(usize, usize, f64)>> {
            let mode = parse_concurrency(concurrency)?;
            Ok(py.detach(|| run_on_threads(threads, || self.inner.join(threshold, mode))))
        }
    }

    /// Compiled core of the `difflib_fast` Python package (re-exported by `difflib_fast/__init__.py`).
    #[pymodule]
    fn _difflib_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(ratio, m)?)?;
        m.add_function(wrap_pyfunction!(ratio_many, m)?)?;
        m.add_function(wrap_pyfunction!(cluster_canonicals, m)?)?;
        m.add_function(wrap_pyfunction!(cluster_canonicals_lsh, m)?)?;
        m.add_function(wrap_pyfunction!(cosine_join, m)?)?;
        m.add_class::<PyRationer>()?;
        m.add_class::<PyCosineJoiner>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unreadable_literal)]
    use super::{cluster_canonicals, gestalt_ratio, ratio, ratio_b2j, ratio_reference};

    #[test]
    fn matches_known_difflib_values() {
        // Cross-checked against difflib.SequenceMatcher(None, a, b, autojunk=False).ratio().
        assert_eq!(gestalt_ratio("", ""), 1.0);
        assert_eq!(gestalt_ratio("", "x"), 0.0);
        assert_eq!(gestalt_ratio("abc", "abc"), 1.0);
        assert_eq!(gestalt_ratio("abc", "abd"), 0.6666666666666666);
        assert_eq!(gestalt_ratio("the quick brown fox", "the quick brown dog"), 0.8947368421052632);
        assert_eq!(gestalt_ratio("ПриветМир", "ПриветМирЪ"), 0.9473684210526315);
    }

    // The fast suffix-automaton path must equal the structurally-distinct b2j reference exactly.
    #[test]
    fn fast_matches_reference() {
        let mut s: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..2000 {
            let mk = |n: usize, rng: &mut dyn FnMut() -> u64| -> String {
                (0..n).map(|_| char::from(b'a' + (rng() % 5) as u8)).collect()
            };
            let (la, lb) = ((next() % 50) as usize, (next() % 50) as usize);
            let a = mk(la, &mut next);
            let b = mk(lb, &mut next);
            let r = ratio_reference(&a, &b);
            assert_eq!(gestalt_ratio(&a, &b), r, "SAM a={a:?} b={b:?}");
            assert_eq!(ratio_b2j(&a, &b), r, "b2j a={a:?} b={b:?}");
        }
    }

    // Both dispatch branches (b2j for short, SAM for long) must agree with the reference on long,
    // repetitive strings that cross B2J_CROSSOVER.
    #[test]
    fn long_strings_all_paths_agree() {
        let mut s: u64 = 0xdead_beef_cafe_1234;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..40 {
            let mk = |n: usize, rng: &mut dyn FnMut() -> u64| -> String {
                (0..n).map(|_| char::from(b'a' + (rng() % 6) as u8)).collect()
            };
            let a = mk(1400 + (next() % 600) as usize, &mut next); // > B2J_CROSSOVER ⇒ SAM branch
            let b = mk(1400 + (next() % 600) as usize, &mut next);
            let r = ratio_reference(&a, &b);
            assert_eq!(gestalt_ratio(&a, &b), r);
            assert_eq!(ratio_b2j(&a, &b), r);
            assert_eq!(ratio(&a, &b), r); // dispatched (SAM here)
        }
    }

    #[test]
    fn clusters_obvious_duplicates() {
        let corpus: Vec<String> = vec![
            "def add(a, b): return a + b".into(),
            "def add(x, y): return x + y".into(),
            "completely unrelated text here".into(),
        ];
        let clusters = cluster_canonicals(&corpus, 0.5);
        assert_eq!(clusters.len(), 1, "the two add() variants should cluster");
        assert_eq!(clusters[0].0, vec![0, 1]);
        assert!(clusters[0].1 >= 0.5);
    }
}
