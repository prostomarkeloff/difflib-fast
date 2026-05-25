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
//! corpus at a similarity threshold — prebuild each string's automaton once, then an early-exit
//! all-pairs join (length blocking + `quick_ratio` filter + threshold-aware RO), in parallel via
//! rayon — and reports each cluster with its exact minimum pairwise ratio. [`cluster_canonicals_lsh`]
//! is the scalable `MinHash`-LSH variant (candidate generation + exact verification) for very large
//! corpora past the O(n²) wall.
//!
//! Two independent implementations of `M` live here: the suffix-automaton path ([`gestalt`]) and a
//! straight port of difflib's `b2j` recursion ([`ratio_reference`]); the test suite asserts they are
//! bit-identical, which is the crate's core correctness gate.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

pub mod gestalt;
pub use gestalt::gestalt_ratio;

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

/// The difflib `b2j` ratio path directly (bypasses the length dispatch) — exposed for benchmarking
/// and as a second, structurally-distinct exact implementation. Prefer [`ratio`] for real use.
#[must_use]
pub fn ratio_b2j(a: &str, b: &str) -> f64 {
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

/// Code point → ascending positions in `b`.
fn build_b2j(b: &[char]) -> HashMap<char, Vec<usize>> {
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, &c) in b.iter().enumerate() {
        b2j.entry(c).or_default().push(j);
    }
    b2j
}

/// difflib `find_longest_match` over `a[alo:ahi] × b[blo:bhi]`; returns `(i, j, k)`.
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
/// suite asserts [`gestalt_ratio`] equals this exactly. Prefer [`ratio`] for real use (far faster).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn ratio_reference(a: &str, b: &str) -> f64 {
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

/// Estimated b2j inner-loop work `Σ_c count_a(c)·count_b(c)` — exactly the positions the first-block
/// scan visits; the dispatch signal, also exposed for benchmarking. (`oa·ob` folds the non-ASCII tail.)
#[must_use]
pub fn b2j_work(a: &[char], b: &[char]) -> u64 {
    let (ca, oa) = ascii_counts(a);
    let (cb, ob) = ascii_counts(b);
    let mut w = u64::from(oa) * u64::from(ob);
    for i in 0..128 {
        w += u64::from(ca[i]) * u64::from(cb[i]);
    }
    w
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
fn real_quick_ratio(a: &[char], b: &[char]) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * (a.len().min(b.len()) as f64) / (total as f64)
}

/// Sorted `(char, count)` multiset of `a` — precomputed once per string so the `quick_ratio`
/// upper-bound filter is a linear merge over the (small) alphabet instead of a per-pair `HashMap`.
fn char_counts(a: &[char]) -> Vec<(char, u32)> {
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
fn quick_ratio_counts(ca: &[(char, u32)], cb: &[(char, u32)], total: usize) -> f64 {
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

// Env-gated diagnostics: set DIFFLIB_FAST_PROGRESS=1 to stream phase timings + progress to stderr
// from inside the Rust hot path (off in production — zero output, ~zero cost).
fn progress_on() -> bool {
    std::env::var_os("DIFFLIB_FAST_PROGRESS").is_some()
}

// ───────────────────────────────────── exact clustering ─────────────────────────────────────

/// Qualifying pairs `(i<j, ratio)` with `ratio >= threshold`, in parallel. The exact upper-bound
/// early-exits (`real_quick_ratio`/`quick_ratio`) skip most pairs without the full O(len²) RO;
/// survivors go through `gestalt_edge` — reject early-exit for non-edges, exact ratio for edges. The
/// edge ratio is kept so `min_sim` reuses it (a dense cluster's intra pairs are ~all edges, so the
/// `min_sim` pass recomputes almost nothing — removing the redundant second scan over the same pairs).
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn qualifying_pairs(chars: &[Vec<char>], sams: &[gestalt::Sam], threshold: f64) -> Vec<(usize, usize, f64)> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n = chars.len();
    let rows = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        if progress_on() {
            let rows = &rows;
            scope.spawn(move || {
                while rows.load(Ordering::Relaxed) < n {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let done = rows.load(Ordering::Relaxed);
                    eprintln!("    [difflib-fast] qualifying_pairs: row {done}/{n} ({:.0}%)", done as f64 / n as f64 * 100.0);
                }
            });
        }
        // Length blocking: ratio>=T ⟹ |short|/|long| >= T/(2-T), so in length-sorted order each
        // string only reaches a contiguous run of (not-too-much-longer) strings — break the inner
        // loop as soon as `real_quick_ratio` drops below T. Turns the O(n²) enumeration into
        // O(n·window); exact (never drops a qualifying pair). `counts` feeds the cheap quick filter.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| chars[i].len());
        let counts: Vec<Vec<(char, u32)>> = chars.par_iter().map(|c| char_counts(c)).collect();
        let pairs = (0..n)
            .into_par_iter()
            .flat_map_iter(|p| {
                let i = order[p];
                let a = &chars[i];
                let mut local: Vec<(usize, usize, f64)> = Vec::new();
                for &j in &order[p + 1..] {
                    let b = &chars[j];
                    if real_quick_ratio(a, b) < threshold {
                        break; // lengths only grow ⇒ all remaining partners also fail the bound
                    }
                    if quick_ratio_counts(&counts[i], &counts[j], a.len() + b.len()) < threshold {
                        continue;
                    }
                    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                    // Reject early-exit for non-edges; exact ratio (cached for min_sim) for edges.
                    if let Some(r) = gestalt::gestalt_edge(&chars[lo], &chars[hi], &sams[hi], threshold) {
                        local.push((lo, hi, r));
                    }
                }
                rows.fetch_add(1, Ordering::Relaxed);
                local
            })
            .collect();
        rows.store(n, Ordering::Relaxed); // unblock the progress thread
        pairs
    })
}

/// Min intra-cluster pairwise ratio (single-linkage's conservative figure), exact. Edge pairs are
/// already in `ratios` (cached from the qualifying pass) — for a dense cluster that is ~every intra
/// pair, so almost nothing is recomputed. The rare missing (non-edge, ratio < threshold) pair is
/// computed with the pruned `gestalt_ratio_capped` (accept-exits any pair above the running min).
/// Parallel over members; each task's cap is its own running min (>= the global min ⇒ exact min kept).
fn cluster_min_sim(members: &[usize], chars: &[Vec<char>], sams: &[gestalt::Sam], ratios: &HashMap<(usize, usize), f64>) -> f64 {
    members
        .par_iter()
        .enumerate()
        .map(|(pos, &i)| {
            let mut local = 1.0_f64;
            for &j in &members[pos + 1..] {
                let key = if i < j { (i, j) } else { (j, i) };
                let r = match ratios.get(&key) {
                    Some(&r) => r, // edge ratio cached by the qualifying pass — no recompute
                    None => gestalt::gestalt_ratio_capped(&chars[key.0], &chars[key.1], &sams[key.1], local),
                };
                local = local.min(r);
            }
            local
        })
        .reduce(|| 1.0_f64, f64::min)
}

/// Union-find over qualifying edge pairs → clusters (size >= 2), each with its exact min intra-pair
/// ratio. The qualifying pass's edge ratios are cached in `ratios` and reused by `cluster_min_sim`.
fn assemble(n: usize, pairs: Vec<(usize, usize, f64)>, chars: &[Vec<char>], sams: &[gestalt::Sam]) -> Vec<(Vec<usize>, f64)> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut ratios: HashMap<(usize, usize), f64> = HashMap::with_capacity(pairs.len());
    for (i, j, r) in pairs {
        ratios.insert((i, j), r);
        let (ri, rj) = (uf_find(&mut parent, i), uf_find(&mut parent, j));
        if ri != rj {
            parent[ri] = rj;
        }
    }
    let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        comps.entry(root).or_default().push(i);
    }
    let mut out: Vec<(Vec<usize>, f64)> = Vec::new();
    for members in comps.values() {
        if members.len() < 2 {
            continue;
        }
        let min_sim = cluster_min_sim(members, chars, sams, &ratios);
        let mut sorted = members.clone();
        sorted.sort_unstable();
        out.push((sorted, min_sim));
    }
    out.sort_by(|a, b| a.0[0].cmp(&b.0[0]));
    out
}

/// Exact single-linkage clustering over pre-collected `char` vectors: returns each cluster (member
/// indices, sorted) with its exact minimum pairwise ratio. O(n²) early-exit join, rayon-parallel.
#[must_use]
pub fn cluster_canonicals_chars(chars: &[Vec<char>], threshold: f64) -> Vec<(Vec<usize>, f64)> {
    let n = chars.len();
    // Prebuild each string's suffix automaton ONCE (n builds), reused as the b-side for all n²
    // pairs — the all-pairs cost becomes n builds + n² scans, not n² builds.
    let sams: Vec<gestalt::Sam> = chars.par_iter().map(|c| gestalt::build_sam(c)).collect();
    let pairs = qualifying_pairs(chars, &sams, threshold);
    assemble(n, pairs, chars, &sams)
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
    let pairs: Vec<(usize, usize, f64)> = cand
        .par_iter()
        .filter_map(|&(i, j)| {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            gestalt::gestalt_edge(&chars[a], &chars[b], &sams[b], threshold).map(|r| (a, b, r))
        })
        .collect();
    if debug {
        eprintln!("    [difflib-fast] lsh: {} verified pairs in {:.2}s", pairs.len(), start.elapsed().as_secs_f64());
    }
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

    /// Compiled core of the `difflib_fast` Python package (re-exported by `difflib_fast/__init__.py`).
    #[pymodule]
    fn _difflib_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(ratio, m)?)?;
        m.add_function(wrap_pyfunction!(ratio_many, m)?)?;
        m.add_function(wrap_pyfunction!(cluster_canonicals, m)?)?;
        m.add_function(wrap_pyfunction!(cluster_canonicals_lsh, m)?)?;
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
