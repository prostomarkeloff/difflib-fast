//! `simjoin` — exact all-pairs **weighted-cosine similarity join**: given a corpus of sparse
//! non-negative vectors, find every pair `(i, j)` whose cosine similarity is `≥ t`.
//!
//! This is the `AllPairs` / `L2AP` family (Bayardo et al. WWW'07; Anastasiu & Karypis ICDE'14): an
//! inverted index with **prefix filtering** so that the vast majority of vector pairs are never
//! compared. It is the principled, exact replacement for shingle-candidate + verify-all
//! near-duplicate detection (e.g. Type-3 code clones = functions × IDF-weighted lines).
//!
//! ## Correctness gate
//! [`cosine_join`] (the indexed algorithm) must return the **exact same** pair set, with
//! bit-identical similarities, as [`cosine_join_bruteforce`] (the naive `O(n²)` oracle). Both score
//! a pair with the same [`cos_full`] sorted-merge dot, so the values match to the bit and a pair is
//! never dropped or gained at the threshold. This equality is asserted on fuzzed corpora — the same
//! "two implementations, one answer" discipline the RO path uses.
//!
//! ## Method (this reference)
//! Vectors are L2-normalised (so `cos = dot`) and their dimensions relabelled to a global rank by
//! **increasing max weight** (common low-weight dims first, rare high-weight dims last — so only the
//! rare tail is indexed, keeping postings short). For a probe vector we look up candidates through
//! the index, then verify with the full dot. When *indexing* a vector we skip the leading prefix
//! whose max possible contribution to any dot stays `< t`:
//! `Σ_{k<b} w_k · maxw[dim_k] < t` ⇒ a pair matching only in that prefix can't reach `t`
//! (weights ≥ 0), so it is guaranteed to also share an indexed dim. The skipped prefix holds the few
//! common dims (huge postings); only the rarer tail is indexed — that is the whole speed-up. The
//! tighter Cauchy–Schwarz / L2-norm bounds (true L2AP) and accumulation-time pruning are the next
//! optimisation layer on top of this exact base.

// Index/rank ids are bounded by the corpus dimension count (≤ u32 by construction); ranks are
// assigned densely from a sorted dim list. The `as` casts below are intentional and in-range.
// Dense numerical code: `i`/`d`/`w`/`y`/`a`/`s`/`t` mirror the cosine/prefix-bound formulas and read
// clearer than verbose names here — allow the single-char-names pedantic lint module-wide.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::Concurrency;

/// One join's result: `(row_a, row_b, cosine)` triples with `row_a < row_b`, arbitrary order.
pub type JoinPairs = Vec<(usize, usize, f64)>;

/// A corpus of L2-normalised sparse non-negative vectors in CSR form, with dimensions relabelled to
/// a global rank (decreasing max weight). Built once; joinable at any threshold.
pub struct Corpus {
    n: usize,
    ndims: usize,
    /// `n + 1` row offsets into `dims`/`wts`.
    indptr: Vec<usize>,
    /// Relabelled (rank) dimension ids, ascending within each row.
    dims: Vec<u32>,
    /// L2-normalised weights, aligned with `dims`.
    wts: Vec<f64>,
    /// Per relabelled dim: the max weight it takes across the corpus (the prefix-filter bound).
    maxw: Vec<f64>,
}

/// Merge duplicate dims of one row (weights summed in input order within each dim group, dims in
/// ascending dim order — deterministic, unlike the per-call-random `HashMap` iteration order of the
/// previous implementation) and L2-normalise. A row whose norm is 0 (all weights 0) becomes empty.
#[must_use]
fn norm_row(r: &[(u32, f64)]) -> Vec<(u32, f64)> {
    let mut v = r.to_vec();
    v.sort_by_key(|&(d, _)| d); // stable: preserves input order within each duplicate group
    let mut out: Vec<(u32, f64)> = Vec::with_capacity(v.len());
    let mut sq = 0.0;
    let mut iter = v.into_iter();
    if let Some((mut d, mut w)) = iter.next() {
        for (d2, w2) in iter {
            if d2 == d {
                w += w2;
            } else {
                sq += w * w;
                out.push((d, w));
                d = d2;
                w = w2;
            }
        }
        sq += w * w;
        out.push((d, w));
    }
    let norm = sq.sqrt();
    if norm > 0.0 {
        for (_, w) in &mut out {
            *w /= norm;
        }
        out
    } else {
        Vec::new()
    }
}

impl Corpus {
    /// Number of vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if the corpus has no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `(dims, weights)` of vector `i` — dims ascending by global rank, weights L2-normalised.
    #[must_use]
    fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let (s, e) = (self.indptr[i], self.indptr[i + 1]);
        (&self.dims[s..e], &self.wts[s..e])
    }

    /// CSR view for GPU offload: `(indptr, dims, wts_f32)`, with `indptr` cast to `u32` and the
    /// L2-normalised weights cast to `f32` (Apple GPUs have no `f64`). For the
    /// [`crate::simjoin_gpu`] throughput experiment only — the f32 cast means GPU dots are *not*
    /// bit-identical to the CPU `f64` path.
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    #[must_use]
    pub fn csr_f32(&self) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let indptr = self.indptr.iter().map(|&x| x as u32).collect();
        let wts = self.wts.iter().map(|&w| w as f32).collect();
        (indptr, self.dims.clone(), wts)
    }

    /// Build a corpus from **token documents** — each document a list of string tokens — as TF-IDF
    /// sparse vectors: dim = a distinct token, weight = `(token count in doc) × ln(n / df_token)`
    /// (`df` = number of documents containing the token). This is the principled input for a Type-3
    /// code-clone join (documents = functions, tokens = canonicalised lines) and the shape most
    /// callers actually have. A token appearing in every document gets `idf = 0` (contributes
    /// nothing), as expected.
    #[must_use]
    pub fn from_token_docs<S: AsRef<str>>(docs: &[Vec<S>]) -> Corpus {
        let n = docs.len();
        let mut dim: HashMap<&str, u32> = HashMap::new();
        let mut df: Vec<u32> = Vec::new();
        // Assign a dim id to each distinct token and count document frequency (once per doc/token).
        let mut doc_ids: Vec<Vec<u32>> = Vec::with_capacity(n);
        for doc in docs {
            let mut ids = Vec::with_capacity(doc.len());
            let mut seen: HashSet<u32> = HashSet::new();
            for tok in doc {
                let id = *dim.entry(tok.as_ref()).or_insert_with(|| {
                    let i = df.len() as u32;
                    df.push(0);
                    i
                });
                ids.push(id);
                if seen.insert(id) {
                    df[id as usize] += 1;
                }
            }
            doc_ids.push(ids);
        }
        let idf: Vec<f64> = df.iter().map(|&d| (n as f64 / f64::from(d.max(1))).ln()).collect();
        // Emit (dim, idf) once per token occurrence; `from_rows` sums duplicates → tf·idf per dim.
        let rows: Vec<Vec<(u32, f64)>> = doc_ids
            .iter()
            .map(|ids| ids.iter().map(|&id| (id, idf[id as usize])).collect())
            .collect();
        Corpus::from_rows(&rows)
    }

    /// Build a corpus from raw `(dim, weight)` rows. Duplicate dims within a row are summed; each row
    /// is L2-normalised; dims are relabelled to a global rank by decreasing max weight. Weights are
    /// expected non-negative (the prefix-filter bound requires it — IDF weights satisfy this).
    ///
    /// The build is rayon-parallel (per-row normalise, global max-weight reduction, per-row relabel
    /// into the CSR) and **deterministic**: rows are merged in dim order, the rank is a function of
    /// the exact per-dim max weight (ties by dim id), and parallel reductions only combine
    /// order-independent operations — the resulting corpus is a pure function of `rows`.
    ///
    /// Dims that occur in exactly one row (`df = 1`) are dropped from the rank and the rows **after**
    /// normalisation: no pair shares them, so they contribute exactly `0.0` to every pairwise dot and
    /// their postings are never walked — the join output is bit-identical with or without them, while
    /// the index and the rows get smaller and the pruning bounds tighter. Normalising over the full
    /// row first is what keeps the surviving weights — and every emitted score — unchanged.
    #[must_use]
    pub fn from_rows(rows: &[Vec<(u32, f64)>]) -> Corpus {
        Self::from_rows_with(rows, true)
    }

    /// Build a corpus from normalized rows with optional **hapax stripping**. Hapax dims (present in
    /// exactly one row) are dropped from the rank and the rows **after** L2 normalisation — no pair
    /// shares them, so they contribute exactly `0.0` to every pairwise dot, their postings are never
    /// walked (probes visit only postings of their own dims), and dropping their mass only tightens
    /// the Cauchy–Schwarz bounds (which are filters). The join output is therefore bit-identical
    /// with stripping on or off; the strip just shrinks the index/rows and prunes harder. Normalising
    /// over the full row first keeps the surviving weights — and every emitted score — unchanged
    /// (removing the mass *before* normalisation would rescale the weights and change the indexing).
    /// `strip_hapax = false` is the un-stripped reference build, used by tests to pin the equality.
    fn from_rows_with(rows: &[Vec<(u32, f64)>], strip_hapax: bool) -> Corpus {
        let n = rows.len();
        // 1. Merge duplicate dims + L2-normalise each row (kept as (orig_dim, weight)), in parallel.
        let normed: Vec<Vec<(u32, f64)>> = rows.par_iter().map(|r| norm_row(r)).collect();
        // 2. Document frequency per original dim (over the normalised rows) — a parallel
        //    fold+reduce of exact counts, so the chunking cannot change the result.
        let df: HashMap<u32, u32> = normed
            .par_iter()
            .fold(HashMap::new, |mut m, v| {
                for &(d, _) in v {
                    *m.entry(d).or_insert(0) += 1;
                }
                m
            })
            .reduce(HashMap::new, |mut a, mut b| {
                if a.len() < b.len() {
                    std::mem::swap(&mut a, &mut b);
                }
                for (d, c) in b {
                    *a.entry(d).or_insert(0) += c;
                }
                a
            });
        // A dim earns a rank only if some pair could share it. `strip_hapax = false` keeps every
        // dim (test-only reference).
        let shared = |d: u32| !strip_hapax || df.get(&d).is_some_and(|&c| c >= 2);
        // 3. Max normalised weight per original dim — a parallel fold+reduce. `max` is exact and
        //    order-independent, so the chunking cannot change any value.
        let maxw_orig: HashMap<u32, f64> = normed
            .par_iter()
            .fold(HashMap::new, |mut m, v| {
                for &(d, w) in v {
                    if shared(d) {
                        let e = m.entry(d).or_insert(0.0);
                        if w > *e {
                            *e = w;
                        }
                    }
                }
                m
            })
            .reduce(HashMap::new, |mut a, mut b| {
                if a.len() < b.len() {
                    std::mem::swap(&mut a, &mut b);
                }
                for (d, w) in b {
                    let e = a.entry(d).or_insert(0.0);
                    if w > *e {
                        *e = w;
                    }
                }
                a
            });
        // 4. Rank dims by (max weight ASC, dim asc) → dense global order. Ascending so the common,
        //    low-weight dims land at the FRONT: they fill the un-indexed prefix (their tiny
        //    `w·maxw` keeps the cumulative bound under `t` for many of them), and only the rare,
        //    high-weight tail is indexed — short postings. (Reversing this indexes the common dims
        //    and their huge postings, which is correct but orders of magnitude slower.)
        //    The comparator is a total order on distinct dims, so the unstable parallel sort yields
        //    the same unique sorted sequence the serial stable sort it replaces would.
        let mut dims_sorted: Vec<(u32, f64)> = maxw_orig.into_iter().collect();
        dims_sorted.par_sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0))
        });
        let rank: HashMap<u32, u32> =
            dims_sorted.iter().enumerate().map(|(i, &(d, _))| (d, i as u32)).collect();
        let ndims = dims_sorted.len();
        let maxw: Vec<f64> = dims_sorted.iter().map(|&(_, w)| w).collect();
        // 5. CSR with relabelled dims, ascending within each row. Relabel + sort each row in
        //    parallel (rank lookups are the cost), then a serial flatten into the flat CSR arrays —
        //    a pure memcpy; a parallel fill into disjoint slices would need unsafe borrows for no
        //    measurable win. Hapax dims have no rank, so the `rank.get` filter drops them.
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        let mut off = 0usize;
        for v in &normed {
            let kept = if strip_hapax {
                v.iter().filter(|&&(d, _)| rank.contains_key(&d)).count()
            } else {
                v.len()
            };
            off += kept;
            indptr.push(off);
        }
        let relabelled: Vec<Vec<(u32, f64)>> = normed
            .par_iter()
            .map(|v| {
                let mut rv: Vec<(u32, f64)> =
                    v.iter().filter_map(|&(d, w)| rank.get(&d).map(|&r| (r, w))).collect();
                rv.sort_unstable_by_key(|&(d, _)| d);
                rv
            })
            .collect();
        let mut dims = Vec::with_capacity(off);
        let mut wts = Vec::with_capacity(off);
        for rv in relabelled {
            for (d, w) in rv {
                dims.push(d);
                wts.push(w);
            }
        }
        Corpus { n, ndims, indptr, dims, wts, maxw }
    }
}

/// Sorted-merge dot product of two rows (dims ascending). For L2-normalised non-negative vectors
/// this is exactly their cosine similarity. The single scoring routine shared by the indexed join
/// and the brute-force oracle, so both agree to the bit.
#[must_use]
#[cfg_attr(feature = "profiling", inline(never))]
fn cos_full((da, wa): (&[u32], &[f64]), (db, wb): (&[u32], &[f64])) -> f64 {
    // Rows are equal-length dim/weight slices (built by `Corpus::from_rows`).
    debug_assert_eq!(da.len(), wa.len());
    debug_assert_eq!(db.len(), wb.len());
    let (la, lb) = (da.len(), db.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut s = 0.0f64;
    // Branchless sorted-merge: the 3-way `cmp` branches mispredict on random dim order, and the
    // weight loads carry bounds checks the optimiser can't elide. Here we always load both weights
    // (`i<la=wa.len()`, `j<lb=wb.len()`) and mask the product by dim-equality — `s += 0.0` for
    // unequal dims adds the *same* terms in the *same* increasing-dim order, so the result is
    // bit-identical to the branchy merge while shedding every data-dependent branch.
    while i < la && j < lb {
        // SAFETY: `i < la == wa.len()` and `j < lb == wb.len()`.
        let (ai, bj) = unsafe { (*da.get_unchecked(i), *db.get_unchecked(j)) };
        let (wai, wbj) = unsafe { (*wa.get_unchecked(i), *wb.get_unchecked(j)) };
        let eq = f64::from(u32::from(ai == bj));
        s += eq * wai * wbj;
        i += usize::from(ai <= bj);
        j += usize::from(ai >= bj);
    }
    s
}

/// Per-vector prune data, read in the hot verify loop when a vector appears as a candidate. Packed
/// into one array (not two parallel `Vec`s) so a candidate's `pnorm` and `split` come from a single
/// scattered cache line instead of two — verify is memory-latency-bound on these random accesses.
#[derive(Clone, Copy)]
struct Bound {
    /// ‖un-indexed prefix of y‖₂ — Cauchy–Schwarz cap on the dot mass the accumulator misses (the
    /// prefix dims of `y` were never indexed, so never accumulated).
    pnorm: f64,
    /// Rank of `y`'s first *indexed* dim (`u32::MAX` if `y` indexed nothing). Every prefix dim of
    /// `y` has rank `<` this — lets the probe restrict its norm to that rank range.
    split: u32,
}

/// Per-vector data cached as each vector is indexed; read by the prune bound when the vector later
/// appears as a candidate. (One `Cached` for the whole join.)
struct Cached {
    bound: Vec<Bound>,
}

/// Reused scratch buffers (allocated once for the whole join, not per probe).
struct Scratch {
    /// `acc[y]` = partial dot of the probe with `y` over shared *indexed* dims; `-1.0` = untouched
    /// sentinel (a real partial dot of non-negative weights is always `≥ 0`).
    acc: Vec<f64>,
    /// Candidate ids the current probe touched (the keys to reset in `acc`).
    touched: Vec<u32>,
    /// Probe prefix L2 norms for this probe: `xpn[k] = ‖wi[..k]‖₂`, length `nnz+1`.
    xpn: Vec<f64>,
    /// Sparse-mode accumulator: `(candidate, partial dot)` pairs sorted by candidate id — the
    /// touched list and the accumulator merged (classic L2AP). Used instead of `acc`/`touched` when
    /// the corpus's average per-probe candidate count is low; the probe falls back to the dense pair
    /// if the list overflows [`LIST_ACC_MAX`].
    list: Vec<(u32, f64)>,
}

/// Accumulator strategy for the probe phase, chosen per join from the corpus's shape (and forced
/// per-mode in tests — the two must agree bit-for-bit).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AccMode {
    /// Dense `acc[n]` array: O(1) scatter per posting entry. Wins when probes touch many candidates.
    Dense,
    /// Sorted `(y, acc)` list: few candidates per probe ⇒ the list stays tiny and L1-resident, and
    /// the n-wide dense array's footprint never competes with the postings for L2.
    Sparse,
}

/// Average per-probe posting touches at/below which [`cosine_join`] picks the [`AccMode::Sparse`]
/// accumulator. The touch volume is estimated from the built index as `Σ len²/2` (posting `d` is
/// walked by about half the probes, each walk visiting `len` entries) divided by `n` — the per-probe
/// mean walk volume, a corpus-shape signal. Per-probe outliers are caught by [`LIST_ACC_MAX`]'s
/// dense fallback.
const LIST_ACC_MAX_TOUCHES: usize = 32;

/// Pick the accumulator mode from a built inverted index: sparse list when the average probe walks
/// few posting entries, dense scatter otherwise.
#[must_use]
fn acc_mode_for(index: &[Vec<(u32, f64)>], n: usize) -> AccMode {
    let mut walks = 0u64;
    for p in index {
        let len = p.len() as u64;
        walks += len * len;
    }
    if walks / 2 / (n.max(1) as u64) < LIST_ACC_MAX_TOUCHES as u64 {
        AccMode::Sparse
    } else {
        AccMode::Dense
    }
}

/// Hard cap on one probe's sparse list: beyond this the per-insert memmove dominates the O(1)
/// dense scatter it replaces, so [`accumulate_list`] bails out and the probe is redone densely
/// (re-walking its postings — rare by construction; the additions repeat in the same order, so the
/// values are unchanged).
const LIST_ACC_MAX: usize = 512;

/// Exact all-pairs cosine join via inverted index + **L2AP** prefix filtering and accumulation-time
/// pruning. Returns `(j, i, cos)` with `j < i` for every pair with `cos ≥ t`. Bit-identical pair set
/// to [`cosine_join_bruteforce`].
///
/// For each probe we accumulate a partial dot over shared *indexed* dims ([`accumulate`]), then for
/// each touched candidate compute a Cauchy–Schwarz upper bound on the true cosine and skip the exact
/// [`cos_full`] when it cannot reach `t` ([`verify_pruned`]). The bound is a filter only — survivors
/// are scored exactly, so the output is byte-for-byte the brute-force result. On skewed data the
/// bound prunes the ~99.9 % of candidates that collide on a single rare dim, so `cos_full` (the
/// former 90 % hotspot) runs only on genuine near-matches.
///
/// The full inverted index is built once (postings ascending by id), then every vector is probed in
/// **parallel**: probe `i` walks each posting only while `y < i` (postings are id-sorted), so it sees
/// exactly the earlier vectors — each pair `(j, i)` with `j < i` is found once, from the larger id.
/// This is the same candidate set the sequential index-as-you-go build produces, so the result is
/// unchanged; the returned `Vec` is in arbitrary order (sort if a canonical order is needed).
#[must_use]
pub fn cosine_join(c: &Corpus, t: f64) -> Vec<(usize, usize, f64)> {
    let n = c.n;
    // Postings carry the indexed weight `(y, w_y[d])` so the scan can accumulate a partial dot.
    let mut index: Vec<Vec<(u32, f64)>> = vec![Vec::new(); c.ndims];
    let mut cached = Cached { bound: vec![Bound { pnorm: 0.0, split: u32::MAX }; n] };
    for i in 0..n {
        let (di, wi) = c.row(i);
        index_suffix(c, i, (di, wi), t, &mut index, &mut cached);
    }
    // Accumulator strategy from the corpus shape: few candidates per probe on average (sparse /
    // duplicate-heavy corpora) ⇒ the sorted list wins; many (dense rows) ⇒ the O(1) dense scatter.
    let mode = acc_mode_for(&index, n);
    cosine_join_mode(c, t, &index, &cached, mode)
}

/// The join body under an explicit [`AccMode`]: probe vectors in parallel; each worker keeps one
/// reusable `Scratch`. `with_min_len` batches many probes per rayon task so the per-probe work
/// (tiny) isn't dwarfed by task-splitting / scheduling overhead (`swtch_pri` in the profile).
#[must_use]
fn cosine_join_mode(
    c: &Corpus,
    t: f64,
    index: &[Vec<(u32, f64)>],
    cached: &Cached,
    mode: AccMode,
) -> Vec<(usize, usize, f64)> {
    let n = c.n;
    (0..n)
        .into_par_iter()
        .with_min_len(256)
        .map_init(
            || Scratch { acc: vec![-1.0; n], touched: Vec::new(), xpn: Vec::new(), list: Vec::new() },
            |scratch, i| {
                let mut out = Vec::new();
                probe_half(c, i, t, index, cached, mode, scratch, &mut out);
                out
            },
        )
        .flatten()
        .collect()
}

/// One probe's accumulate + prune + score against `c`'s index, under `mode`, appending to `out`.
/// Shared by the single join and the fused two-corpus join so the per-half work is structurally
/// identical (the fused join relies on that for its bit-identical halves).
#[allow(clippy::too_many_arguments, clippy::inline_always)]
#[inline(always)] // per-probe dispatch only — must fold into the rayon closure, not add a call
fn probe_half(
    c: &Corpus,
    i: usize,
    t: f64,
    index: &[Vec<(u32, f64)>],
    cached: &Cached,
    mode: AccMode,
    s: &mut Scratch,
    out: &mut Vec<(usize, usize, f64)>,
) {
    let (di, wi) = c.row(i);
    match mode {
        AccMode::Dense => {
            accumulate(index, (di, wi), i as u32, s);
            verify_pruned(c, i, t, s, cached, out);
        }
        AccMode::Sparse => {
            if accumulate_list(index, (di, wi), i as u32, s) {
                verify_list(c, i, t, s, cached, out);
            } else {
                // The sparse list overflowed — redo this probe with the dense pass (identical
                // additions, identical order, so the values are unchanged).
                accumulate(index, (di, wi), i as u32, s);
                verify_pruned(c, i, t, s, cached, out);
            }
        }
    }
}

/// [`cosine_join`] under an explicitly forced [`AccMode`]. Test-only: the fuzz gates must cover both
/// modes, and the shape-based pick exercises only one of them per corpus.
#[cfg(test)]
#[must_use]
fn cosine_join_for_mode(c: &Corpus, t: f64, mode: AccMode) -> Vec<(usize, usize, f64)> {
    let n = c.n;
    let mut index: Vec<Vec<(u32, f64)>> = vec![Vec::new(); c.ndims];
    let mut cached = Cached { bound: vec![Bound { pnorm: 0.0, split: u32::MAX }; n] };
    for i in 0..n {
        let (di, wi) = c.row(i);
        index_suffix(c, i, (di, wi), t, &mut index, &mut cached);
    }
    cosine_join_mode(c, t, &index, &cached, mode)
}

/// Fused two-corpus cosine join: one probe pass over `a` and `b` — two corpora with the **same row
/// count**, the typical "two views of the same objects" shape (e.g. line-vectors and part-vectors of
/// the same functions). Returns `(pairs_a, pairs_b)`; each half is **bit-identical** to a separate
/// [`cosine_join`] on its corpus — the per-half phases ([`accumulate`]/[`accumulate_list`],
/// [`verify_pruned`]/[`verify_list`]) run on the same inputs in the same order, only the rayon
/// schedule and the per-worker [`Scratch`] are shared (one `acc` buffer serves both halves: half A's
/// entries are reset by its verify pass before half B accumulates).
///
/// Wins over two separate joins by paying the per-probe fixed costs once: one parallel schedule, one
/// `Scratch` per worker, and the probe iteration bookkeeping is shared. Pair order within each half
/// is arbitrary (as in [`cosine_join`]). The accumulator mode is chosen per corpus from its shape
/// (see [`cosine_join`]), so the halves may use different modes.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn cosine_join_double(a: &Corpus, b: &Corpus, t: f64) -> (JoinPairs, JoinPairs) {
    assert_eq!(a.n, b.n, "cosine_join_double: corpora must share the row count");
    let n = a.n;
    let mut index_a: Vec<Vec<(u32, f64)>> = vec![Vec::new(); a.ndims];
    let mut index_b: Vec<Vec<(u32, f64)>> = vec![Vec::new(); b.ndims];
    let mut cached_a = Cached { bound: vec![Bound { pnorm: 0.0, split: u32::MAX }; n] };
    let mut cached_b = Cached { bound: vec![Bound { pnorm: 0.0, split: u32::MAX }; n] };
    for i in 0..n {
        let (di, wi) = a.row(i);
        index_suffix(a, i, (di, wi), t, &mut index_a, &mut cached_a);
        let (di, wi) = b.row(i);
        index_suffix(b, i, (di, wi), t, &mut index_b, &mut cached_b);
    }
    let mode_a = acc_mode_for(&index_a, n);
    let mode_b = acc_mode_for(&index_b, n);
    (0..n)
        .into_par_iter()
        .with_min_len(256)
        .map_init(
            || Scratch { acc: vec![-1.0; n], touched: Vec::new(), xpn: Vec::new(), list: Vec::new() },
            |scratch, i| {
                let mut out_a = Vec::new();
                probe_half(a, i, t, &index_a, &cached_a, mode_a, scratch, &mut out_a);
                let mut out_b = Vec::new();
                probe_half(b, i, t, &index_b, &cached_b, mode_b, scratch, &mut out_b);
                (out_a, out_b)
            },
        )
        .fold(
            || (Vec::new(), Vec::new()),
            |(mut va, mut vb), (oa, ob)| {
                va.extend(oa);
                vb.extend(ob);
                (va, vb)
            },
        )
        .reduce(
            || (Vec::new(), Vec::new()),
            |(mut a1, mut b1), (mut a2, mut b2)| {
                a1.append(&mut a2);
                b1.append(&mut b2);
                (a1, b1)
            },
        )
}

/// Run the cosine join under a chosen [`Concurrency`] backend. Returns `(j, i, cos)` pairs with
/// `j < i` and `cos ≥ t`, scores as `f64` (the `Gpu` mode's f32 cosines are widened losslessly).
///
/// - [`Concurrency::Cpu`] — [`cosine_join`]: exact `f64`, all-CPU, every platform.
/// - [`Concurrency::GpuPlusCpu`] — exact `f64` hybrid: CPU generates survivor pairs, the GPU f32
///   cosine *filters* the clear rejects, the CPU recomputes the exact `f64` score on what passes.
///   **Byte-identical to `Cpu`**; both engines fully used. ~1.7–2× on bandwidth-bound real data.
/// - [`Concurrency::Gpu`] — GPU-dominant `f32`: CPU generates survivor pairs, the GPU scores them and
///   the result is emitted directly (no f64 re-verify). Fastest (~2×); differs from the exact answer
///   only on pairs whose true cosine is within ~`1e-6` of `t` (measured: ≤1 pair in millions).
///
/// When the `gpu` feature is off, the target isn't macOS, or no Metal device can be acquired, the GPU
/// modes transparently fall back to [`cosine_join`] (same as `Rationer`). This convenience entry
/// **compiles + uploads the GPU corpus on every call** — fine for a one-shot join, but for repeated
/// joins on one corpus build a [`CosineJoiner`] once and call [`CosineJoiner::join`], which holds the
/// device + kernel + uploaded CSR across calls (and avoids the driver instability of compiling a
/// Metal library hundreds of times in a tight loop).
#[must_use]
pub fn cosine_join_with(c: &Corpus, t: f64, mode: Concurrency) -> Vec<(usize, usize, f64)> {
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    {
        if matches!(mode, Concurrency::Cpu) {
            return cosine_join(c, t);
        }
        let (indptr, dims, wts) = c.csr_f32();
        let Some(gpu) = crate::simjoin_gpu::BatchCosineGpu::new(&indptr, &dims, &wts) else {
            return cosine_join(c, t); // no Metal device → CPU fallback
        };
        match mode {
            Concurrency::GpuPlusCpu => cosine_join_gpu(c, t, &gpu),
            // `Gpu`: emit the GPU f32 cosines directly (widened to f64), no exact re-verify.
            Concurrency::Gpu => cosine_join_gpu_f32(c, t, &gpu)
                .into_iter()
                .map(|(a, b, s)| (a, b, f64::from(s)))
                .collect(),
            Concurrency::Cpu => unreachable!("handled above"),
        }
    }
    #[cfg(not(all(feature = "gpu", target_os = "macos")))]
    {
        let _ = mode; // GPU modes degrade to the CPU join when the feature is off / not macOS.
        cosine_join(c, t)
    }
}

/// A reusable cosine-join handle that owns the corpus and — under `feature = "gpu"` on macOS — the
/// Metal device, compiled `batch_cosine` kernel, and the corpus CSR uploaded to unified memory, all
/// acquired **once** at construction. Repeated [`join`](CosineJoiner::join)s at different thresholds
/// then skip the per-call kernel compile + CSR upload that [`cosine_join_with`] pays (only the
/// `t`-specific inverted index is rebuilt each call, on the CPU). Always constructible; degrades to
/// the pure-CPU join when the `gpu` feature is off or no Metal device is available — mirroring
/// `Rationer`. This is the right entry point for sweeping thresholds or joining repeatedly.
pub struct CosineJoiner {
    corpus: Corpus,
    /// Owned GPU resources (device + kernel + CSR in UMA); `None` when no Metal device was acquired.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    gpu: Option<crate::simjoin_gpu::BatchCosineGpu>,
}

impl CosineJoiner {
    /// Build a joiner over `corpus`, acquiring the GPU device + uploading the corpus CSR once if the
    /// `gpu` feature is on and a Metal device is present.
    #[must_use]
    pub fn new(corpus: Corpus) -> Self {
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            let (indptr, dims, wts) = corpus.csr_f32();
            let gpu = crate::simjoin_gpu::BatchCosineGpu::new(&indptr, &dims, &wts);
            Self { corpus, gpu }
        }
        #[cfg(not(all(feature = "gpu", target_os = "macos")))]
        {
            Self { corpus }
        }
    }

    /// The owned corpus (e.g. for `len()` or to run other queries against it).
    #[must_use]
    pub fn corpus(&self) -> &Corpus {
        &self.corpus
    }

    /// Whether a Metal GPU backend was acquired. Always `false` without `feature = "gpu"` on macOS;
    /// when `false`, every [`join`](CosineJoiner::join) runs on the CPU regardless of `mode`.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            self.gpu.is_some()
        }
        #[cfg(not(all(feature = "gpu", target_os = "macos")))]
        {
            false
        }
    }

    /// Run the join at threshold `t` under `mode`, reusing the handle's GPU resources. Returns the
    /// same results as [`cosine_join_with`] (Cpu/GpuPlusCpu exact, Gpu f32→f64); falls back to the
    /// CPU join when the GPU is unavailable.
    #[must_use]
    pub fn join(&self, t: f64, mode: Concurrency) -> Vec<(usize, usize, f64)> {
        #[cfg(all(feature = "gpu", target_os = "macos"))]
        {
            match (mode, self.gpu.as_ref()) {
                (Concurrency::GpuPlusCpu, Some(g)) => cosine_join_gpu(&self.corpus, t, g),
                (Concurrency::Gpu, Some(g)) => cosine_join_gpu_f32(&self.corpus, t, g)
                    .into_iter()
                    .map(|(a, b, s)| (a, b, f64::from(s)))
                    .collect(),
                _ => cosine_join(&self.corpus, t), // Cpu mode, or no Metal device
            }
        }
        #[cfg(not(all(feature = "gpu", target_os = "macos")))]
        {
            let _ = mode;
            cosine_join(&self.corpus, t)
        }
    }
}

/// FP slack for the prune bound: the Cauchy–Schwarz upper bound holds in exact arithmetic, but the
/// accumulated dot and the `sqrt` norms each carry rounding error. We only *skip* `cos_full` when
/// the bound is below `t` by more than this slack, so a true pair (exact cosine `≥ t`) is never
/// pruned. Not skipping is always correctness-safe (just a wasted `cos_full`), so the slack trades a
/// negligible number of extra verifies for safety and never changes the emitted pair set.
/// `1e-9 ≫` the `~1e-15` accumulated error over ~15 terms.
const PRUNE_SLACK: f64 = 1e-9;

/// Probe row length at/above which the cheap monotone pre-bound in [`verify_pruned`] is worth it.
/// The pre-bound skips the per-candidate `partition_point` for candidates that can't reach `t`, but
/// `partition_point` over a *short* probe row is already cheap, so on sparse corpora (mean nnz ≈ 11,
/// e.g. `PyPI` type3) the pre-bound's own `fmul`/`fadd`/compare is pure overhead that rarely prunes
/// (match-dense ⇒ high survivor rate). It only pays on **dense** rows (e.g. find-dup-defs
/// patternology, mean nnz ≈ 61) where `partition_point` is costly and the survivor rate is tiny.
/// Gating on `di.len()` (a per-probe, loop-invariant test) keeps the sparse regime regression-free.
const PREBOUND_MIN_DIMS: usize = 24;

/// Phase 1 — accumulate: for each indexed dim of the probe, add `w_probe·w_y` into `acc[y]` for
/// every **earlier** `y` (`y < cutoff`, the probe's own id) indexing that dim. Postings are id-sorted,
/// so we `break` at the first `y ≥ cutoff`. Leaves `acc` `-1.0` everywhere except the touched ids
/// (listed in `touched`, reset in [`verify_pruned`]). One scattered `acc[]` FMA per posting.
#[cfg_attr(feature = "profiling", inline(never))]
fn accumulate(index: &[Vec<(u32, f64)>], (di, wi): (&[u32], &[f64]), cutoff: u32, s: &mut Scratch) {
    // Split the borrows so `acc` and `touched` are independent: otherwise the optimiser, seeing
    // `s.touched.push(...)` go through the same `&mut Scratch`, can't prove the push leaves
    // `s.acc`'s base pointer untouched and reloads `acc.ptr` from the struct on EVERY posting entry
    // (one extra load across the ~5M-entry accumulation on dense corpora). With `acc` a separate
    // `&mut [f64]` local the base stays in a register; a `touched` realloc can't alias it.
    let Scratch { acc, touched, .. } = s;
    let acc = acc.as_mut_slice();
    touched.clear();
    // At most one distinct candidate per acc slot, so `n == acc.len()` slots is enough headroom to
    // append every first-touch without bounds checks or a realloc inside the loop.
    touched.reserve(acc.len());
    let tptr = touched.as_mut_ptr();
    let mut tlen = 0usize;
    for (&d, &w) in di.iter().zip(wi) {
        for &(y, wy) in &index[d as usize] {
            if y >= cutoff {
                break;
            }
            let yu = y as usize;
            // SAFETY: `y` is a vector id pushed by `index_suffix`, so `yu < n == acc.len()`.
            let a = unsafe { acc.get_unchecked_mut(yu) };
            // Branchless first-touch: the `*a < 0.0` test fed a data-dependent branch that mispredicts
            // (first-touch vs repeat interleave unpredictably across the probe's dims). Instead select
            // the base (`0.0` on first touch, current partial dot otherwise) with a conditional move,
            // and append `y` to `touched` by an UNCONDITIONAL store that's only *committed* when the
            // length is bumped — `tlen += first`. Same accumulator values and same touched order as
            // the branchy form, so the result is bit-identical; just no mispredicting branch.
            let first = *a < 0.0;
            let base = if first { 0.0 } else { *a };
            *a = base + w * wy;
            // SAFETY: `tlen ≤ distinct candidates so far < acc.len() ≤ touched.capacity()`.
            unsafe { *tptr.add(tlen) = y };
            tlen += usize::from(first);
        }
    }
    // SAFETY: `tlen` first-touch stores were written into the reserved region, in order.
    unsafe { touched.set_len(tlen) };
}

/// Fill `xpn` with the probe's running prefix L2 norms (`xpn[k] = ‖wi[..k]‖₂`) and return the full
/// probe norm plus whether the cheap monotone pre-bound is worth it for this row length. One sqrt
/// per probe dim, reused across all its candidates.
///
/// Since `xpn` is monotonic, `xpn[kstar] ≤ xnorm` for every candidate, so `a + xnorm·pnorm ≥ a +
/// xpn[kstar]·pnorm` (the exact bound). A candidate failing the cheap `a + xnorm·pnorm < need` test
/// therefore also fails the exact bound — prune it without the per-candidate `partition_point`. The
/// survivor set is bit-identical; only the binary search is skipped for the ~99% of touched
/// candidates that can't reach `t` on dense corpora.
#[must_use]
fn probe_prefix_norms(wi: &[f64], xpn: &mut Vec<f64>) -> (f64, bool) {
    xpn.clear();
    xpn.push(0.0);
    let mut sq = 0.0f64;
    for &w in wi {
        sq += w * w;
        xpn.push(sq.sqrt());
    }
    (sq.sqrt(), wi.len() >= PREBOUND_MIN_DIMS)
}

/// Per-candidate Cauchy–Schwarz prune test, shared by the dense verify pass, the sparse verify pass
/// and the GPU survivor collector: `true` iff candidate `y` (accumulated partial dot `a`) survives —
/// i.e. its exact [`cos_full`] score must be computed. The bound is a filter only; the emitted value
/// is always the exact dot, so the pair set is bit-identical to brute force whatever this decides.
#[must_use]
#[inline(always)]
#[allow(clippy::inline_always)] // this runs once per touched candidate — call overhead is the cost
fn candidate_survives(
    di: &[u32],
    xpn: &[f64],
    xnorm: f64,
    prebound: bool,
    need: f64,
    a: f64,
    bd: Bound,
) -> bool {
    // Cheap monotone pre-bound (dense rows only) — skip the binary search when even `xnorm` can't
    // clear `need`. `prebound` is loop-invariant, so sparse rows pay nothing.
    if prebound && a + xnorm * bd.pnorm < need {
        return false;
    }
    // Number of probe dims with rank < split[y] → index into xpn (di sorted ascending).
    let kstar = di.partition_point(|&d| d < bd.split);
    // SAFETY: kstar ≤ di.len() == wi.len() = xpn.len()-1.
    // (An added maxweight cap `min(…, Σ wx·maxw)` was tried + reverted: never tighter than the
    // L2 cap on either synthetic or real data — it sums over all probe-prefix dims, not just the
    // shared ones — so it pruned nothing and cost ~16% on the real PyPI corpus.)
    a + unsafe { xpn.get_unchecked(kstar) } * bd.pnorm >= need
}

/// Phase 2 — prune + verify (dense mode). For each touched candidate `y`, reset its accumulator and
/// test the **L2AP `l2` bound**: the dot mass missing from `acc[y]` (the dims in `prefix(y)`) is at
/// most `‖x_{rank<split[y]}‖ · ‖prefix(y)‖` by Cauchy–Schwarz, where `x_{rank<split[y]}` is the probe
/// restricted to the rank range `prefix(y)` lives in. Since the probe's mass sits in its rare
/// (high-rank) dims, that restricted norm is tiny — a far tighter cap than the whole-probe `‖x‖=1`.
/// Only if `acc[y] + that bound ≥ t` (minus FP slack) do we score exactly with [`cos_full`]. Filter
/// only — the emitted value is the exact dot, so the pair set is bit-identical to brute force.
#[cfg_attr(feature = "profiling", inline(never))]
fn verify_pruned(
    c: &Corpus,
    i: usize,
    t: f64,
    s: &mut Scratch,
    cached: &Cached,
    out: &mut Vec<(usize, usize, f64)>,
) {
    let (di, wi) = c.row(i);
    let (xnorm, prebound) = probe_prefix_norms(wi, &mut s.xpn);
    let need = t - PRUNE_SLACK;
    let Scratch { acc, touched, xpn, .. } = s;
    // (Software-prefetching the candidate row a few ahead was tried + reverted: no measurable change
    // — with all cores gathering at once the join is memory-*bandwidth*-bound, not per-access
    // latency-bound, so prefetch can't add throughput.)
    for &y in touched.iter() {
        let yu = y as usize;
        // SAFETY: `yu < n` (same provenance as in `accumulate`).
        let a = unsafe { std::mem::replace(acc.get_unchecked_mut(yu), -1.0) };
        // SAFETY: `yu < n`. One scattered load fetches both prune fields.
        let bd = unsafe { *cached.bound.get_unchecked(yu) };
        if candidate_survives(di, xpn, xnorm, prebound, need, a, bd) {
            let cos = cos_full((di, wi), c.row(yu));
            if cos >= t {
                out.push((yu, i, cos));
            }
        }
    }
}

/// Sparse-mode phase 1 — the same partial-dot accumulation as [`accumulate`], but into the sorted
/// `(y, acc)` list instead of the dense `acc[n]` array. Wins when a probe touches few candidates:
/// the list stays tiny and L1-resident, and the n-wide dense array's footprint never competes with
/// the postings for L2. The additions happen in the same per-dim/posting order, so the accumulated
/// values are bit-identical to the dense pass. Returns `false` on [`LIST_ACC_MAX`] overflow — the
/// caller must redo the probe densely (identical additions, identical order).
#[cfg_attr(feature = "profiling", inline(never))]
fn accumulate_list(
    index: &[Vec<(u32, f64)>],
    (di, wi): (&[u32], &[f64]),
    cutoff: u32,
    s: &mut Scratch,
) -> bool {
    let Scratch { list, .. } = s;
    list.clear();
    for (&d, &w) in di.iter().zip(wi) {
        for &(y, wy) in &index[d as usize] {
            if y >= cutoff {
                break;
            }
            let term = w * wy;
            match list.binary_search_by_key(&y, |&(yy, _)| yy) {
                Ok(k) => list[k].1 += term,
                Err(k) => {
                    if list.len() >= LIST_ACC_MAX {
                        list.clear();
                        return false;
                    }
                    list.insert(k, (y, term));
                }
            }
        }
    }
    true
}

/// Sparse-mode phase 2 — the same prune + score as [`verify_pruned`], over the sorted accumulator
/// list. No reset needed: the next probe's [`accumulate_list`] clears the list. Candidate order is
/// by id (the dense pass visits first-touch order); both are "arbitrary order" for the caller, and
/// the per-candidate arithmetic is identical, so the emitted pair set + scores are bit-identical.
#[cfg_attr(feature = "profiling", inline(never))]
fn verify_list(
    c: &Corpus,
    i: usize,
    t: f64,
    s: &mut Scratch,
    cached: &Cached,
    out: &mut Vec<(usize, usize, f64)>,
) {
    let (di, wi) = c.row(i);
    let (xnorm, prebound) = probe_prefix_norms(wi, &mut s.xpn);
    let need = t - PRUNE_SLACK;
    let Scratch { list, xpn, .. } = s;
    for &(y, a) in list.iter() {
        let yu = y as usize;
        // SAFETY: `yu < n` (candidate ids come from the postings, same as in `accumulate`).
        let bd = unsafe { *cached.bound.get_unchecked(yu) };
        if candidate_survives(di, xpn, xnorm, prebound, need, a, bd) {
            let cos = cos_full((di, wi), c.row(yu));
            if cos >= t {
                out.push((yu, i, cos));
            }
        }
    }
}

/// Phase 3 — index this vector's suffix: skip the leading prefix whose max possible contribution to
/// any dot stays `< t` (`Σ w_k·maxw[dim_k] < t`), index only the rarer tail (short postings = the
/// whole speed-up), and cache `pnorm[i] = ‖prefix‖₂` and `split[i]` = first indexed rank for the
/// [`verify_pruned`] bound.
#[cfg_attr(feature = "profiling", inline(never))]
fn index_suffix(
    c: &Corpus,
    i: usize,
    (di, wi): (&[u32], &[f64]),
    t: f64,
    index: &mut [Vec<(u32, f64)>],
    cached: &mut Cached,
) {
    // Largest safe prefix under the maxweight bound: `Σ_{k<b} w_k·maxw[dim_k] < t` ⇒ the prefix
    // can't contribute `t` to any dot (weights ≥ 0). (A norm-based extension `‖x_{<b}‖ < t` was
    // tried and reverted: it gave zero candidate reduction in the realistic `t<1` regime — the
    // maxweight bound already indexes less — and it is FP-fragile at `t=1` where `‖x‖` rounds below
    // `1` and indexes nothing, dropping exact-duplicate pairs.)
    let mut rs = 0.0f64;
    let mut b = 0usize;
    for k in 0..di.len() {
        let bound = wi[k] * c.maxw[di[k] as usize];
        if rs + bound >= t {
            break;
        }
        rs += bound;
        b = k + 1;
    }
    let mut p = 0.0f64;
    for &w in &wi[..b] {
        p += w * w;
    }
    cached.bound[i] = Bound {
        pnorm: p.sqrt(),
        split: if b < di.len() { di[b] } else { u32::MAX },
    };
    for k in b..di.len() {
        index[di[k] as usize].push((i as u32, wi[k]));
    }
}

/// FP margin for the GPU f32 cosine *filter* in [`cosine_join_gpu`]: a survivor is dropped only when
/// its GPU f32 cosine is below `t` by more than this. The f32 dot's error is `~1e-6` relative, so a
/// `1e-4` absolute margin never drops a true pair; the CPU then recomputes the exact `f64` score on
/// whatever passes, so the emitted pair set + scores stay bit-identical to [`cosine_join`].
#[cfg(all(target_os = "macos", feature = "gpu"))]
const GPU_FILTER_MARGIN: f64 = 1e-4;

/// Like the verify half of [`verify_pruned`], but instead of scoring, **collects** each surviving
/// `(candidate, probe)` pair (`candidate < probe`) for batch scoring elsewhere. The bound test is
/// [`candidate_survives`] — the same function `verify_pruned` uses — so the survivor set is
/// identical by construction.
#[cfg(all(target_os = "macos", feature = "gpu"))]
fn collect_survivors(c: &Corpus, i: usize, t: f64, s: &mut Scratch, cached: &Cached, out: &mut Vec<(u32, u32)>) {
    let (di, wi) = c.row(i);
    let (xnorm, prebound) = probe_prefix_norms(wi, &mut s.xpn);
    let need = t - PRUNE_SLACK;
    let Scratch { acc, touched, xpn, .. } = s;
    for &y in touched.iter() {
        let yu = y as usize;
        // SAFETY: `yu < n` (same provenance as in `accumulate`).
        let a = unsafe { std::mem::replace(acc.get_unchecked_mut(yu), -1.0) };
        let bd = unsafe { *cached.bound.get_unchecked(yu) };
        if candidate_survives(di, xpn, xnorm, prebound, need, a, bd) {
            out.push((y, i as u32)); // (candidate, probe), candidate < probe
        }
    }
}

/// **CPU+GPU hybrid join** (feature `gpu`, macOS). Returns the **exact same** pair set + scores as
/// [`cosine_join`] (and thus the brute-force oracle) — only faster when the verify is bandwidth-bound.
///
/// Pipeline: CPU builds the index and, in parallel, accumulates + bounds every probe to a list of
/// surviving `(candidate, probe)` pairs (no `cos_full`). The GPU then computes an f32 cosine for the
/// whole batch (its memory-level parallelism clears the random-gather dots ~3× faster than the CPU),
/// and the CPU recomputes the exact `f64` `cos_full` **only** on the pairs whose GPU score clears
/// `t − margin` — typically a few percent of survivors. Because the GPU is a *conservative filter*
/// (margin ≫ f32 error, so no true pair is ever dropped) and every emitted score is the exact CPU
/// `f64` value, the output is byte-for-byte identical to [`cosine_join`].
#[cfg(all(target_os = "macos", feature = "gpu"))]
#[must_use]
pub fn cosine_join_gpu(
    c: &Corpus,
    t: f64,
    gpu: &crate::simjoin_gpu::BatchCosineGpu,
) -> Vec<(usize, usize, f64)> {
    let (pa, pb) = survivor_pairs(c, t);
    if pa.is_empty() {
        return Vec::new();
    }
    // GPU phase: f32 cosine over the whole survivor batch (conservative filter).
    let gcos = gpu.cosine_batch(&pa, &pb);
    let need = t - GPU_FILTER_MARGIN;
    // CPU phase: exact f64 re-verify only on pairs the GPU filter passes.
    (0..pa.len())
        .into_par_iter()
        .with_min_len(1024)
        .filter_map(|k| {
            if f64::from(gcos[k]) < need {
                return None;
            }
            let (a, b) = (pa[k] as usize, pb[k] as usize);
            let cos = cos_full(c.row(a), c.row(b));
            (cos >= t).then_some((a, b, cos))
        })
        .collect()
}

/// **Pure-f32** CPU+GPU join (feature `gpu`, macOS): same survivor generation as [`cosine_join_gpu`]
/// but emits the GPU's **f32** cosine directly, with **no exact f64 re-verify**. Trades byte-parity
/// for speed (no re-verify, and `cos_full` never runs on the GPU survivors). The result differs from
/// [`cosine_join`] only on pairs whose true cosine lies within ~`1e-6` (f32 rounding) of `t` — for a
/// similarity join with an arbitrary threshold that is immaterial. Use when an ε-exact answer is
/// acceptable; use [`cosine_join_gpu`] when bit-exactness is required.
#[cfg(all(target_os = "macos", feature = "gpu"))]
#[must_use]
pub fn cosine_join_gpu_f32(
    c: &Corpus,
    t: f64,
    gpu: &crate::simjoin_gpu::BatchCosineGpu,
) -> Vec<(usize, usize, f32)> {
    let (pa, pb) = survivor_pairs(c, t);
    if pa.is_empty() {
        return Vec::new();
    }
    let gcos = gpu.cosine_batch(&pa, &pb);
    let tf = t as f32;
    (0..pa.len())
        .into_par_iter()
        .with_min_len(1024)
        .filter_map(|k| (gcos[k] >= tf).then_some((pa[k] as usize, pb[k] as usize, gcos[k])))
        .collect()
}

/// CPU half shared by the GPU joins: build the index, then accumulate + bound every probe in
/// parallel to the list of surviving `(candidate, probe)` pairs (candidate `<` probe), split into
/// two `u32` arrays ready for [`crate::simjoin_gpu::BatchCosineGpu::cosine_batch`].
#[cfg(all(target_os = "macos", feature = "gpu"))]
fn survivor_pairs(c: &Corpus, t: f64) -> (Vec<u32>, Vec<u32>) {
    let n = c.n;
    let mut index: Vec<Vec<(u32, f64)>> = vec![Vec::new(); c.ndims];
    let mut cached = Cached { bound: vec![Bound { pnorm: 0.0, split: u32::MAX }; n] };
    for i in 0..n {
        let (di, wi) = c.row(i);
        index_suffix(c, i, (di, wi), t, &mut index, &mut cached);
    }
    let pairs: Vec<(u32, u32)> = (0..n)
        .into_par_iter()
        .with_min_len(256)
        .map_init(
            || Scratch { acc: vec![-1.0; n], touched: Vec::new(), xpn: Vec::new(), list: Vec::new() },
            |scratch, i| {
                let (di, wi) = c.row(i);
                accumulate(&index, (di, wi), i as u32, scratch);
                let mut out = Vec::new();
                collect_survivors(c, i, t, scratch, &cached, &mut out);
                out
            },
        )
        .flatten()
        .collect();
    pairs.into_iter().unzip()
}

/// Diagnostic (feature `profiling`, off the hot path): counts that quantify the prune. Returns
/// `(candidates, survivors, pairs)` — candidates touched by the accumulator, survivors that pass the
/// Cauchy–Schwarz bound (i.e. the `cos_full` calls actually made), and real pairs. `survivors /
/// candidates` is the prune pass-rate (lower = better); `survivors` is the verify volume we pay for.
#[cfg(feature = "profiling")]
#[must_use]
pub fn cosine_join_counts(c: &Corpus, t: f64) -> (u64, u64, u64) {
    let n = c.n;
    let mut index: Vec<Vec<(u32, f64)>> = vec![Vec::new(); c.ndims];
    let mut cached = Cached { bound: vec![Bound { pnorm: 0.0, split: u32::MAX }; n] };
    for i in 0..n {
        let (di, wi) = c.row(i);
        index_suffix(c, i, (di, wi), t, &mut index, &mut cached);
    }
    let mut s = Scratch { acc: vec![-1.0; n], touched: Vec::new(), xpn: Vec::new(), list: Vec::new() };
    let (mut ncand, mut survivors, mut pairs) = (0u64, 0u64, 0u64);
    let need = t - PRUNE_SLACK;
    for i in 0..n {
        let (di, wi) = c.row(i);
        accumulate(&index, (di, wi), i as u32, &mut s);
        ncand += s.touched.len() as u64;
        let (xnorm, prebound) = probe_prefix_norms(wi, &mut s.xpn);
        for &y in &s.touched {
            let yu = y as usize;
            let a = std::mem::replace(&mut s.acc[yu], -1.0);
            let bd = cached.bound[yu];
            if candidate_survives(di, &s.xpn, xnorm, prebound, need, a, bd) {
                survivors += 1;
                if cos_full((di, wi), c.row(yu)) >= t {
                    pairs += 1;
                }
            }
        }
    }
    (ncand, survivors, pairs)
}

/// Naive `O(n²)` oracle: score every pair with [`cos_full`], keep `cos ≥ t`. The correctness
/// reference [`cosine_join`] is validated against.
#[must_use]
pub fn cosine_join_bruteforce(c: &Corpus, t: f64) -> Vec<(usize, usize, f64)> {
    let mut out: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..c.n {
        for j in 0..i {
            let s = cos_full(c.row(i), c.row(j));
            if s >= t {
                out.push((j, i, s));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        cosine_join, cosine_join_bruteforce, cosine_join_double, cosine_join_for_mode, AccMode,
        Corpus,
    };

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn sort_pairs(mut v: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, u64)> {
        v.sort_by_key(|a| (a.0, a.1));
        v.into_iter().map(|(a, b, s)| (a, b, s.to_bits())).collect()
    }

    #[test]
    fn indexed_join_matches_bruteforce() {
        let mut next = xorshift(0x9e37_79b9_7f4a_7c15);
        for _ in 0..400 {
            let n = (next() % 40 + 2) as usize;
            let dim_space = next() % 15 + 1;
            let rows: Vec<Vec<(u32, f64)>> = (0..n)
                .map(|_| {
                    let nnz = (next() % 8) as usize;
                    (0..nnz)
                        .map(|_| ((next() % dim_space) as u32, (next() % 10 + 1) as f64))
                        .collect()
                })
                .collect();
            let c = Corpus::from_rows(&rows);
            for &t in &[0.1_f64, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let got = sort_pairs(cosine_join(&c, t));
                let want = sort_pairs(cosine_join_bruteforce(&c, t));
                assert_eq!(got, want, "n={n} t={t}");
            }
        }
    }

    /// `Corpus::from_rows` must be a deterministic function of its input: building the same rows
    /// twice yields the same corpus, so the join output is bit-identical. (The previous
    /// HashMap-based per-row merge summed weights in per-call-random hash order, so this did not
    /// hold; it is now a pinned contract.)
    #[test]
    fn from_rows_is_deterministic() {
        let mut next = xorshift(0x6f6c_6c61_6279_6e6f);
        for _ in 0..20 {
            let n = (next() % 30 + 2) as usize;
            let dim_space = next() % 12 + 1;
            let rows: Vec<Vec<(u32, f64)>> = (0..n)
                .map(|_| {
                    let nnz = (next() % 9) as usize;
                    let mut r: Vec<(u32, f64)> = (0..nnz)
                        .map(|_| ((next() % dim_space) as u32, (next() % 10 + 1) as f64))
                        .collect();
                    // Force duplicate dims so the merge path runs and the sum order matters.
                    if nnz > 1 {
                        r[1].0 = r[0].0;
                    }
                    r
                })
                .collect();
            let a = Corpus::from_rows(&rows);
            let b = Corpus::from_rows(&rows);
            for &t in &[0.3_f64, 0.7] {
                assert_eq!(
                    sort_pairs(cosine_join(&a, t)),
                    sort_pairs(cosine_join(&b, t)),
                    "n={n} t={t}"
                );
            }
        }
    }

    /// Stripping `df = 1` dims from the corpus must be invisible to the join: a hapax dim is shared
    /// by no pair, so it contributes exactly `0.0` to every pairwise dot (the branchless `cos_full`
    /// accumulates `+0.0` no-ops on non-negative sums), its posting is never walked, and the tighter
    /// Cauchy–Schwarz bounds only prune more. The stripped and un-stripped builds must therefore
    /// return bit-identical pair sets, and each must match its own brute-force oracle.
    #[test]
    fn hapax_stripping_is_invisible() {
        let mut next = xorshift(0x51a7_4e5f_0c92_83d6);
        for _ in 0..200 {
            let n = (next() % 40 + 2) as usize;
            let dim_space = next() % 12 + 1;
            let rows: Vec<Vec<(u32, f64)>> = (0..n)
                .map(|i| {
                    let nnz = (next() % 8) as usize;
                    let mut r: Vec<(u32, f64)> = (0..nnz)
                        .map(|_| ((next() % dim_space) as u32, (next() % 10 + 1) as f64))
                        .collect();
                    // Every row carries one unique dim — a guaranteed hapax the strip must remove
                    // invisibly (its mass participates in the L2 normalisation of the full row).
                    r.push((1000 + i as u32, 1.0));
                    r
                })
                .collect();
            let full = Corpus::from_rows_with(&rows, false);
            let stripped = Corpus::from_rows_with(&rows, true);
            for &t in &[0.1_f64, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let want = sort_pairs(cosine_join_bruteforce(&stripped, t));
                assert_eq!(
                    sort_pairs(cosine_join(&stripped, t)),
                    want,
                    "stripped vs oracle, n={n} t={t}"
                );
                assert_eq!(
                    sort_pairs(cosine_join(&full, t)),
                    want,
                    "stripped vs full, n={n} t={t}"
                );
                assert_eq!(
                    sort_pairs(cosine_join(&full, t)),
                    sort_pairs(cosine_join_bruteforce(&full, t)),
                    "full vs its own oracle, n={n} t={t}"
                );
            }
        }
    }

    /// Both accumulator strategies must agree bit-for-bit with brute force. The shape-based mode
    /// pick in `cosine_join` means any given corpus exercises only one strategy — force each here
    /// (the dense pass is what the large-corpus shape selects, the sparse list the duplicate-heavy
    /// consumer shape).
    #[test]
    fn both_acc_modes_match_bruteforce() {
        let mut next = xorshift(0x5eed_0f1a_2b3c_4d5e);
        for _ in 0..200 {
            let n = (next() % 40 + 2) as usize;
            let dim_space = next() % 15 + 1;
            let rows: Vec<Vec<(u32, f64)>> = (0..n)
                .map(|_| {
                    let nnz = (next() % 8) as usize;
                    (0..nnz)
                        .map(|_| ((next() % dim_space) as u32, (next() % 10 + 1) as f64))
                        .collect()
                })
                .collect();
            let c = Corpus::from_rows(&rows);
            for &t in &[0.1_f64, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let want = sort_pairs(cosine_join_bruteforce(&c, t));
                for mode in [AccMode::Dense, AccMode::Sparse] {
                    let got = sort_pairs(cosine_join_for_mode(&c, t, mode));
                    assert_eq!(got, want, "n={n} t={t} mode={mode:?}");
                }
            }
        }
    }

    /// The fused two-corpus join must return, per half, the bit-identical pair set of a separate
    /// `cosine_join` on that corpus — fusion shares only schedule and scratch, never arithmetic.
    #[test]
    fn double_join_matches_single() {
        let mut next = xorshift(0xd0a2_7b7e_9c31_5f84);
        for _ in 0..200 {
            let n = (next() % 40 + 2) as usize;
            let dim_space = next() % 15 + 1;
            let mut gen = || -> Vec<Vec<(u32, f64)>> {
                (0..n)
                    .map(|_| {
                        let nnz = (next() % 8) as usize;
                        (0..nnz)
                            .map(|_| ((next() % dim_space) as u32, (next() % 10 + 1) as f64))
                            .collect()
                    })
                    .collect()
            };
            let (rows_a, rows_b) = (gen(), gen());
            let a = Corpus::from_rows(&rows_a);
            let b = Corpus::from_rows(&rows_b);
            for &t in &[0.1_f64, 0.3, 0.5, 0.7, 0.9, 1.0] {
                let (ga, gb) = cosine_join_double(&a, &b, t);
                assert_eq!(sort_pairs(ga), sort_pairs(cosine_join(&a, t)), "half a n={n} t={t}");
                assert_eq!(sort_pairs(gb), sort_pairs(cosine_join(&b, t)), "half b n={n} t={t}");
            }
        }
    }

    /// The CPU+GPU hybrid [`super::cosine_join_gpu`] must return bit-identical results to the pure-CPU
    /// [`cosine_join`] on fuzzed corpora — the GPU is only a conservative filter; every emitted score
    /// is the exact CPU `f64` value. Skips when no Metal device is present.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    #[test]
    fn gpu_hybrid_matches_cpu() {
        use super::CosineJoiner;
        use crate::Concurrency;
        let mut next = xorshift(0x1357_9bdf_0246_8ace);
        for _ in 0..60 {
            let n = (next() % 60 + 4) as usize;
            let dim_space = next() % 20 + 2;
            let rows: Vec<Vec<(u32, f64)>> = (0..n)
                .map(|_| {
                    let nnz = (next() % 10) as usize;
                    (0..nnz)
                        .map(|_| ((next() % dim_space) as u32, (next() % 10 + 1) as f64))
                        .collect()
                })
                .collect();
            // One reusable handle per corpus — `join` is called repeatedly across thresholds, which
            // also exercises that the handle reuses its GPU resources (no per-call library compile).
            let joiner = CosineJoiner::new(Corpus::from_rows(&rows));
            if !joiner.has_gpu() {
                eprintln!("no Metal device — skipping gpu_hybrid_matches_cpu");
                return;
            }
            for &t in &[0.1_f64, 0.3, 0.5, 0.7, 0.9, 1.0] {
                let want = sort_pairs(cosine_join(joiner.corpus(), t));
                // Exact GPU+CPU hybrid is byte-identical to the plain join; `Cpu` mode too.
                assert_eq!(
                    sort_pairs(joiner.join(t, Concurrency::GpuPlusCpu)),
                    want,
                    "GpuPlusCpu n={n} t={t}"
                );
                assert_eq!(sort_pairs(joiner.join(t, Concurrency::Cpu)), want, "Cpu n={n} t={t}");
            }
        }
    }
}
