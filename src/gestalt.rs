//! Gestalt-Chain: fast **exact** Ratcliff–Obershelp via a suffix automaton.
//!
//! difflib's `ratio` is slow because `find_longest_match` is O(|a|·|b|) (it rescans every
//! occurrence of popular characters). The matched-block total M is the recursive
//! longest-common-substring decomposition; a **suffix automaton** finds the longest common
//! substring in O(|a|+|b|) regardless of character frequency, and recursing on the left/
//! right remainders reproduces difflib's M with the *same* tie-break (longest; earliest in
//! a; earliest in b). So `gestalt_ratio` equals `difflib.SequenceMatcher.ratio()` exactly.
//!
//! Hot-path engineering for the all-pairs join:
//!   * transitions are stored **CSR** (per-state sorted slice) → O(log deg) lookup, O(len)
//!     memory, no hashing — and crucially no O(deg) scan at the high-degree root, which the
//!     scan hammers when strings are dissimilar;
//!   * the b-side automaton is **prebuilt once per string** (`build_sam`) and reused for
//!     every pair, so the all-pairs cost is n builds + n² scans, not n² builds.
//!   * **prefetch hints attempted on the per-iteration `node[state]` load** — the SAM walk is a
//!     data-dependent pointer chase the hardware prefetcher cannot anticipate. A `prfm pldl1keep`
//!     (`AArch64`) / `_mm_prefetch` (x86) experiment was MEASURED on M3 Pro and did NOT pay off
//!     (-10% to -2% across mypy/sympy/django) — the SAMs fit in L2 already, and the prefetch
//!     instruction burns execution slots without producing a hit-rate improvement. See the tombstone
//!     near `matching_stats_into` for details.
//!
//! Operates on `char` (code points), so it is bit-identical to difflib on non-ASCII.

// Note: a software-prefetch experiment (prfm pldl1keep / _mm_prefetch for the next iteration's
// node[state] load) was tried and DID NOT pay off on M3 Pro — the prefetch instruction itself
// burned execution slots without producing a measurable hit-rate improvement, because the SAM
// pages fit comfortably in L2 for the per-pair working set (~100KB for the two SAMs in play)
// and the hardware prefetcher already keeps the hot stretches warm. Net: -10% to -2% across
// mypy/sympy/django. Left here as a tombstone so future readers don't waste a day on it.

/// Size of the root's direct transition table — covers ASCII (the canonical text's alphabet).
const ROOT_TBL: usize = 128;

/// `#[cfg(feature = "instrument")]` per-call counters for data-driven perf decisions. Compiles to
/// no-op in default builds (no atomic ops, no codegen). Enabled via `cargo build --features
/// instrument`. The counters use `Ordering::Relaxed` because we only care about totals at
/// end-of-workload (no causal ordering needed) — the relaxed cost is one atomic add per inc.
///
/// Cross-thread aggregation: every rayon worker writes into the same global atomics; we read them
/// after the workload completes via `instrument::dump()`. Use `instrument::reset()` between
/// successive workload runs in the same process so the numbers stay per-workload.
#[cfg(feature = "instrument")]
pub mod instrument {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Histogram bucket count for chain walk + recursion depth. Anything deeper than 63 falls in
    /// the last bucket — measured tail on canonical Python is < 30 in practice, so 64 is plenty.
    pub const HIST_BUCKETS: usize = 64;

    pub static LONGEST_IN_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static PAIRS_PROCESSED: AtomicU64 = AtomicU64::new(0);
    pub static MAX_LE_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static MAX_LE_FAST_PATH: AtomicU64 = AtomicU64::new(0);   // lastpos<=x or firstpos>x shortcut
    pub static MAX_LE_LINEAR: AtomicU64 = AtomicU64::new(0);      // cnt <= LINEAR_MAX scan
    pub static MAX_LE_LINEAR_LEN_SUM: AtomicU64 = AtomicU64::new(0); // total entries scanned
    pub static MAX_LE_SEGTREE: AtomicU64 = AtomicU64::new(0);     // merge-sort-tree walk
    pub static MIN_IN_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static MIN_IN_FAST_PATH: AtomicU64 = AtomicU64::new(0);
    pub static MIN_IN_LINEAR: AtomicU64 = AtomicU64::new(0);
    pub static MIN_IN_LINEAR_LEN_SUM: AtomicU64 = AtomicU64::new(0);
    pub static MIN_IN_SEGTREE: AtomicU64 = AtomicU64::new(0);
    pub static FMATCH_ZERO: AtomicU64 = AtomicU64::new(0);
    pub static FMATCH_NONZERO: AtomicU64 = AtomicU64::new(0);
    pub static FMATCH_SUM: AtomicU64 = AtomicU64::new(0); // sum of all non-zero fmatch values

    pub static CHAIN_DEPTHS: [AtomicU64; HIST_BUCKETS] = {
        // const init via repeated AtomicU64::new(0) — array initializer.
        const ZERO: AtomicU64 = AtomicU64::new(0);
        [ZERO; HIST_BUCKETS]
    };
    pub static RECURSION_DEPTHS: [AtomicU64; HIST_BUCKETS] = {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        [ZERO; HIST_BUCKETS]
    };

    pub fn reset() {
        LONGEST_IN_CALLS.store(0, Ordering::Relaxed);
        PAIRS_PROCESSED.store(0, Ordering::Relaxed);
        MAX_LE_CALLS.store(0, Ordering::Relaxed);
        MAX_LE_FAST_PATH.store(0, Ordering::Relaxed);
        MAX_LE_LINEAR.store(0, Ordering::Relaxed);
        MAX_LE_LINEAR_LEN_SUM.store(0, Ordering::Relaxed);
        MAX_LE_SEGTREE.store(0, Ordering::Relaxed);
        MIN_IN_CALLS.store(0, Ordering::Relaxed);
        MIN_IN_FAST_PATH.store(0, Ordering::Relaxed);
        MIN_IN_LINEAR.store(0, Ordering::Relaxed);
        MIN_IN_LINEAR_LEN_SUM.store(0, Ordering::Relaxed);
        MIN_IN_SEGTREE.store(0, Ordering::Relaxed);
        FMATCH_ZERO.store(0, Ordering::Relaxed);
        FMATCH_NONZERO.store(0, Ordering::Relaxed);
        FMATCH_SUM.store(0, Ordering::Relaxed);
        for b in &CHAIN_DEPTHS {
            b.store(0, Ordering::Relaxed);
        }
        for b in &RECURSION_DEPTHS {
            b.store(0, Ordering::Relaxed);
        }
    }

    /// Dump all counters as a human-readable multi-line string. Includes derived stats
    /// (avg per pair, % linear vs seg-tree, percentile of chain depth, etc.).
    #[must_use]
    pub fn dump() -> String {
        let mut s = String::with_capacity(2048);
        let pairs = PAIRS_PROCESSED.load(Ordering::Relaxed).max(1);
        let li = LONGEST_IN_CALLS.load(Ordering::Relaxed);
        let mxc = MAX_LE_CALLS.load(Ordering::Relaxed).max(1);
        let mxfp = MAX_LE_FAST_PATH.load(Ordering::Relaxed);
        let mxln = MAX_LE_LINEAR.load(Ordering::Relaxed);
        let mxlnsum = MAX_LE_LINEAR_LEN_SUM.load(Ordering::Relaxed);
        let mxsg = MAX_LE_SEGTREE.load(Ordering::Relaxed);
        let mic = MIN_IN_CALLS.load(Ordering::Relaxed);
        let micfp = MIN_IN_FAST_PATH.load(Ordering::Relaxed);
        let micln = MIN_IN_LINEAR.load(Ordering::Relaxed);
        let miclnsum = MIN_IN_LINEAR_LEN_SUM.load(Ordering::Relaxed);
        let micsg = MIN_IN_SEGTREE.load(Ordering::Relaxed);
        let fz = FMATCH_ZERO.load(Ordering::Relaxed);
        let fnz = FMATCH_NONZERO.load(Ordering::Relaxed);
        let fsum = FMATCH_SUM.load(Ordering::Relaxed);
        let ftotal = (fz + fnz).max(1);

        s.push_str(&format!("=== gestalt::instrument dump ===\n"));
        s.push_str(&format!(
            "pairs processed:  {}\n",
            PAIRS_PROCESSED.load(Ordering::Relaxed),
        ));
        s.push_str(&format!(
            "longest_in calls: {} ({:.1} per pair)\n",
            li,
            li as f64 / pairs as f64,
        ));
        s.push_str(&format!(
            "max_le calls:     {} ({:.1} per pair, {:.1} per longest_in)\n",
            mxc,
            mxc as f64 / pairs as f64,
            mxc as f64 / li.max(1) as f64,
        ));
        s.push_str(&format!(
            "  fast-path:      {} ({:.1}%)\n",
            mxfp,
            mxfp as f64 / mxc as f64 * 100.0,
        ));
        s.push_str(&format!(
            "  linear scan:    {} ({:.1}%)   avg scan len: {:.1}\n",
            mxln,
            mxln as f64 / mxc as f64 * 100.0,
            mxlnsum as f64 / mxln.max(1) as f64,
        ));
        s.push_str(&format!(
            "  seg-tree:       {} ({:.1}%)\n",
            mxsg,
            mxsg as f64 / mxc as f64 * 100.0,
        ));
        s.push_str(&format!(
            "min_in calls:     {} ({:.1} per pair)\n",
            mic,
            mic as f64 / pairs as f64,
        ));
        s.push_str(&format!(
            "  fast-path:      {} ({:.1}%)\n",
            micfp,
            micfp as f64 / mic.max(1) as f64 * 100.0,
        ));
        s.push_str(&format!(
            "  linear scan:    {} ({:.1}%)   avg scan len: {:.1}\n",
            micln,
            micln as f64 / mic.max(1) as f64 * 100.0,
            miclnsum as f64 / micln.max(1) as f64,
        ));
        s.push_str(&format!(
            "  seg-tree:       {} ({:.1}%)\n",
            micsg,
            micsg as f64 / mic.max(1) as f64 * 100.0,
        ));
        s.push_str(&format!(
            "fmatch values:    {:.1}% zero  ({} z / {} nz)  avg-nz {:.2}\n",
            fz as f64 / ftotal as f64 * 100.0,
            fz,
            fnz,
            fsum as f64 / fnz.max(1) as f64,
        ));

        let mut chain_total = 0u64;
        let chain: Vec<u64> = CHAIN_DEPTHS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        for &v in &chain {
            chain_total += v;
        }
        s.push_str(&format!(
            "chain walk depths ({} total walks):\n",
            chain_total,
        ));
        let pct = |target: f64| -> usize {
            let mut acc = 0u64;
            for (i, &v) in chain.iter().enumerate() {
                acc += v;
                if acc as f64 >= target * chain_total as f64 {
                    return i;
                }
            }
            chain.len() - 1
        };
        s.push_str(&format!(
            "  p50={} p90={} p95={} p99={} max-bucket={}\n",
            pct(0.50),
            pct(0.90),
            pct(0.95),
            pct(0.99),
            chain.iter().rposition(|&v| v > 0).unwrap_or(0),
        ));
        // Compact histogram preview (first 16 buckets)
        s.push_str("  hist[0..16]: ");
        for &v in chain.iter().take(16) {
            s.push_str(&format!("{} ", v));
        }
        s.push('\n');

        let mut rec_total = 0u64;
        let rec: Vec<u64> = RECURSION_DEPTHS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        for &v in &rec {
            rec_total += v;
        }
        s.push_str(&format!(
            "recursion depths ({} stack pushes):\n",
            rec_total,
        ));
        s.push_str("  hist[0..16]: ");
        for &v in rec.iter().take(16) {
            s.push_str(&format!("{} ", v));
        }
        s.push('\n');

        s
    }
}

/// Helper that the instrument hooks call when the feature is enabled; no-op otherwise.
#[cfg(feature = "instrument")]
#[inline(always)]
fn instr_inc(c: &std::sync::atomic::AtomicU64, by: u64) {
    c.fetch_add(by, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(feature = "instrument")]
#[inline(always)]
fn instr_hist(buckets: &[std::sync::atomic::AtomicU64; instrument::HIST_BUCKETS], depth: usize) {
    buckets[depth.min(instrument::HIST_BUCKETS - 1)]
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// Suffix automaton with CSR (sorted-per-state) transitions — built once, queried by scans.
///
/// For the range-restricted recursion (fix b), each state also carries its **endpos** as a
/// contiguous slice `[dfs_in, dfs_in+dfs_cnt)` of `epos` (the end-positions in b, laid out by
/// a DFS of the suffix-link tree so a subtree is contiguous). A merge-sort tree over `epos`
/// answers "is there an end-position in [lo,hi] within this state's subtree, and the min/max
/// such" — so the whole RO recursion runs on this one prebuilt SAM, with **no sub-builds**.
#[derive(Clone)]
pub struct Sam {
    // Per-state hot fields packed into one cache-line-friendly struct `[len, link, edge_lo, edge_hi]`
    // so a state visit in the scan/recursion loads len + link + transition-range with a SINGLE cache
    // miss (these were 5 separate arrays — under multi-thread DRAM-bandwidth pressure, fewer distinct
    // lines per visit is the dominant win). `link` is u32: the root's link (-1) is stored as 0 and
    // never read, since both the scan and `longest_in` stop at state 0 before following its link.
    node: Vec<[u32; 4]>,
    // Transitions packed as `(char as u64) << 32 | to`, sorted by char within each state's range
    // [edge_lo, edge_hi). Co-locating char+target means the binary-search key and the taken edge's
    // target live on the same cache line (was two parallel arrays `csr_char`/`csr_to`).
    edges: Vec<u64>,
    firstpos: Vec<u32>,  // per state: *smallest* end-position (build-time; folded into epmeta)
    lastpos: Vec<u32>,   // per state: *largest* end-position (build-time; folded into epmeta)
    // Direct lookup for the root's ASCII transitions (`root_next[c] = state`, or -1). The root is
    // the high-degree state hit after every match reset; this makes its transition O(1) instead
    // of a binary search. Non-ASCII root chars (rare) fall back to the edge search, so it's general.
    root_next: Vec<i32>,
    dfs_in: Vec<u32>,    // per state: start index into the endpos array of its endpos range
    dfs_cnt: Vec<u32>,   // per state: number of end-positions in its subtree
    // merge-sort tree over the DFS-ordered end-positions, flattened CSR-style: node `i`'s sorted
    // range is seg_data[seg_off[i]..seg_off[i+1]] (node 1 = root). A state's endpos is the leaf
    // range [dfs_in, dfs_in+dfs_cnt); the tree answers range predecessor/successor adaptively in
    // O(log²·slice) — and `firstpos`/`lastpos` give an O(1) fast path that skips it when the query
    // bound doesn't split the state's endpos span (the common case for wide windows).
    seg_data: Vec<u32>,
    seg_off: Vec<u32>,
    seg_n: usize,        // number of leaves (power of two) in the merge-sort tree
    // Precomputed `len(link[s])` per state — used by `longest_in`'s chain walk's `band_min` check.
    // Without this, every chain step does TWO node loads: `node[cur]` for (len, link, ...), then
    // `node[node[cur].link]` for its `len`. Storing the link's length in a parallel array lets the
    // hot loop fold both into one cache line. Saves ~1 memory access per chain step on M3 — chain
    // walks are pointer-chase-bound, so this is a measurable win on `longest_in` (the 70%-of-CPU
    // function on prepared ratio_many).
    link_len: Vec<u32>,
    // DFS-ordered end-positions (each state's endpos = contiguous slice [dfs_in, dfs_in+dfs_cnt)).
    // For small endpos sets a cache-friendly linear scan of this beats the scattered tree descent.
    epos: Vec<u32>,
    // Query-hot per-state fields packed into one cache line `[firstpos, lastpos, dfs_in, dfs_cnt]`
    // so `max_le`/`min_in` load all four with a single cache miss (they were 4 separate arrays).
    epmeta: Vec<[u32; 4]>,
    // Optimization B (Phase 4): chain-walk hot slot. Layout:
    //   [0]   = len (same as node[s][0])
    //   [1]   = link (same as node[s][1])
    //   [2]   = link_len[s]
    //   [3]   = padding
    //   [4..8] = epmeta[s] = [firstpos, lastpos, dfs_in, dfs_cnt]
    //
    // 32 bytes / state — one cache-line-aligned load supplies everything `longest_in`'s chain
    // walk + `max_le`/`min_in`'s fast path need for one state. Replaces three separate scattered
    // accesses (`node[cur]`, `link_len[cur]`, `epmeta[cur]`) on the chain-walk hot path; PMU said
    // L1D miss rate was 0.51 % and accounted for 18 % of cycles, distributed across these three
    // per-state arrays. Co-locating folds 3 cache-line misses per chain step into 1.
    //
    // Memory cost: +32 bytes / state ≈ +1.6 MB on a 50 k-state SAM. Total `Sam` size grows ~20 %.
    // Bench `bench_new delta-sweep` validates a wall-time reduction on the exact path.
    chain_slot: Vec<[u32; 8]>,
}

/// Below this endpos-set size, `max_le`/`min_in` linear-scan the contiguous DFS-ordered endpos
/// (cache-friendly) instead of the merge-sort tree (scattered) — most queried sets are this small.
const LINEAR_MAX: usize = 256;

impl Sam {
    fn empty() -> Self {
        Sam {
            node: Vec::new(),
            edges: Vec::new(),
            firstpos: Vec::new(),
            lastpos: Vec::new(),
            root_next: Vec::new(),
            dfs_in: Vec::new(),
            dfs_cnt: Vec::new(),
            seg_data: Vec::new(),
            seg_off: Vec::new(),
            seg_n: 0,
            epos: Vec::new(),
            epmeta: Vec::new(),
            link_len: Vec::new(),
            chain_slot: Vec::new(),
        }
    }

    /// Node `i`'s sorted endpos range in the flattened merge-sort tree.
    fn seg_node(&self, i: usize) -> &[u32] {
        &self.seg_data[self.seg_off[i] as usize..self.seg_off[i + 1] as usize]
    }

    /// Read-only view of the packed `[len, link, edge_lo, edge_hi]` per state — needed by the
    /// GPU port (`gpu::matching_stats_gpu`) to serialize the SAM into a Metal buffer. The kernel
    /// reads this slice via index calculations, so we expose it raw (one `[u32; 4]` per state).
    #[must_use]
    pub fn nodes(&self) -> &[[u32; 4]] {
        &self.node
    }

    /// Read-only view of the packed edge slice: `(char << 32) | target_state`, sorted by char
    /// within each state's `[edge_lo, edge_hi)` range. The GPU kernel does binary search over
    /// this slice exactly as `csr_lookup` does on the CPU.
    #[must_use]
    pub fn edges_packed(&self) -> &[u64] {
        &self.edges
    }

    /// Read-only view of the root's direct ASCII transition table (`root_next[c] = state`, or
    /// `-1` for missing). 128 entries per SAM. The GPU kernel uses this to skip the binary
    /// search at the root state, exactly as the CPU does.
    #[must_use]
    pub fn root_next_table(&self) -> &[i32] {
        &self.root_next
    }

    /// `state`'s endpos slice as the merge-sort-tree leaf range [l, r), from packed `epmeta`.
    fn endpos_range_m(m: &[u32; 4], seg_n: usize) -> (usize, usize) {
        let s = m[2] as usize; // dfs_in
        (s + seg_n, s + m[3] as usize + seg_n) // dfs_cnt
    }

    /// Largest end-position `<= x` among `state`'s endpos, with pre-loaded metadata. Used by
    /// `longest_in`'s chain walk where `m` was already pulled from `chain_slot[cur]` (Optimization
    /// B) — saves an `epmeta[state]` load that would hit a separate cache line.
    fn max_le_with_meta(&self, m: &[u32; 4], x: u32) -> Option<u32> {
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MAX_LE_CALLS, 1);
        // O(1) fast path: x doesn't split the state's [firstpos, lastpos] span.
        if m[1] <= x {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MAX_LE_FAST_PATH, 1);
            return Some(m[1]);
        }
        if m[0] > x {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MAX_LE_FAST_PATH, 1);
            return None;
        }
        let cnt = m[3] as usize;
        if cnt <= LINEAR_MAX {
            #[cfg(feature = "instrument")]
            {
                instr_inc(&instrument::MAX_LE_LINEAR, 1);
                instr_inc(&instrument::MAX_LE_LINEAR_LEN_SUM, cnt as u64);
            }
            let lo = m[2] as usize;
            let mut best = 0u32;
            // SAFETY: `m` comes from a valid SAM state's epmeta or chain_slot, so the slice
            // [m[2], m[2]+m[3]) is within `self.epos` (built that way at SAM construction).
            #[allow(clippy::undocumented_unsafe_blocks)]
            for &v in unsafe { self.epos.get_unchecked(lo..lo + cnt) } {
                best = best.max(if v <= x { v } else { 0 });
            }
            return Some(best);
        }
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MAX_LE_SEGTREE, 1);
        let (mut l, mut r) = Self::endpos_range_m(m, self.seg_n);
        let mut best: Option<u32> = None;
        while l < r {
            if l & 1 == 1 {
                best = best.max(node_max_le(self.seg_node(l), x));
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                best = best.max(node_max_le(self.seg_node(r), x));
            }
            l >>= 1;
            r >>= 1;
        }
        best
    }

    /// Smallest end-position in `[lo, hi]` with pre-loaded metadata. See `max_le_with_meta`.
    fn min_in_with_meta(&self, m: &[u32; 4], lo: u32, hi: u32) -> Option<u32> {
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MIN_IN_CALLS, 1);
        if m[0] >= lo {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MIN_IN_FAST_PATH, 1);
            return (m[0] <= hi).then_some(m[0]);
        }
        if m[1] < lo {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MIN_IN_FAST_PATH, 1);
            return None;
        }
        let cnt = m[3] as usize;
        if cnt <= LINEAR_MAX {
            #[cfg(feature = "instrument")]
            {
                instr_inc(&instrument::MIN_IN_LINEAR, 1);
                instr_inc(&instrument::MIN_IN_LINEAR_LEN_SUM, cnt as u64);
            }
            let off = m[2] as usize;
            let mut best = u32::MAX;
            // SAFETY: `m` is from a valid SAM state; endpos slice within bounds (same as max_le).
            #[allow(clippy::undocumented_unsafe_blocks)]
            for &v in unsafe { self.epos.get_unchecked(off..off + cnt) } {
                best = best.min(if v >= lo && v <= hi { v } else { u32::MAX });
            }
            return (best != u32::MAX).then_some(best);
        }
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MIN_IN_SEGTREE, 1);
        let (mut l, mut r) = Self::endpos_range_m(m, self.seg_n);
        let mut best: Option<u32> = None;
        while l < r {
            if l & 1 == 1 {
                best = merge_min(best, node_min_in(self.seg_node(l), lo, hi));
                l += 1;
            }
            if r & 1 == 1 {
                r -= 1;
                best = merge_min(best, node_min_in(self.seg_node(r), lo, hi));
            }
            l >>= 1;
            r >>= 1;
        }
        best
    }

}

/// Largest value `<= x` in a sorted slice, or None.
fn node_max_le(sorted: &[u32], x: u32) -> Option<u32> {
    let k = sorted.partition_point(|&v| v <= x);
    (k > 0).then(|| sorted[k - 1])
}

/// Smallest value in `[lo, hi]` in a sorted slice, or None.
fn node_min_in(sorted: &[u32], lo: u32, hi: u32) -> Option<u32> {
    let k = sorted.partition_point(|&v| v < lo);
    sorted.get(k).copied().filter(|&v| v <= hi)
}

fn merge_min(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Transient builder (Ukkonen online SAM with a linked-list transition arena).
struct Builder {
    edge_char: Vec<char>,
    edge_to: Vec<u32>,
    edge_next: Vec<i32>,
    head: Vec<i32>,
    link: Vec<i32>,
    len: Vec<u32>,
    firstpos: Vec<u32>,
    primary: Vec<bool>, // true for the per-position state (its firstpos is a real end-position)
    last: u32,
}

impl Builder {
    fn empty() -> Self {
        Builder {
            edge_char: Vec::new(),
            edge_to: Vec::new(),
            edge_next: Vec::new(),
            head: Vec::new(),
            link: Vec::new(),
            len: Vec::new(),
            firstpos: Vec::new(),
            primary: Vec::new(),
            last: 0,
        }
    }

    fn new(cap: usize) -> Self {
        let mut b = Builder::empty();
        b.reset(cap);
        b
    }

    /// Clear the arenas (keeping capacity) and re-seed the root — for buffer reuse.
    fn reset(&mut self, cap: usize) {
        self.edge_char.clear();
        self.edge_to.clear();
        self.edge_next.clear();
        self.head.clear();
        self.link.clear();
        self.len.clear();
        self.firstpos.clear();
        self.primary.clear();
        self.edge_char.reserve(3 * cap);
        self.edge_to.reserve(3 * cap);
        self.edge_next.reserve(3 * cap);
        self.primary.push(false); // root state 0 is not a per-position state
        self.head.push(-1);
        self.link.push(-1);
        self.len.push(0);
        self.firstpos.push(0);
        self.last = 0;
    }

    #[allow(clippy::cast_sign_loss)]
    fn find(&self, state: u32, c: char) -> Option<u32> {
        let mut e = self.head[state as usize];
        while e != -1 {
            let idx = e as usize;
            if self.edge_char[idx] == c {
                return Some(self.edge_to[idx]);
            }
            e = self.edge_next[idx];
        }
        None
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn add_edge(&mut self, state: u32, c: char, to: u32) {
        self.edge_char.push(c);
        self.edge_to.push(to);
        self.edge_next.push(self.head[state as usize]);
        self.head[state as usize] = (self.edge_char.len() - 1) as i32;
    }

    #[allow(clippy::cast_sign_loss)]
    fn set_edge(&mut self, state: u32, c: char, to: u32) {
        let mut e = self.head[state as usize];
        while e != -1 {
            let idx = e as usize;
            if self.edge_char[idx] == c {
                self.edge_to[idx] = to;
                return;
            }
            e = self.edge_next[idx];
        }
        self.add_edge(state, c, to);
    }

    #[allow(clippy::cast_possible_truncation)]
    fn new_state(&mut self, len: u32, link: i32, firstpos: u32, primary: bool) -> u32 {
        self.head.push(-1);
        self.len.push(len);
        self.link.push(link);
        self.firstpos.push(firstpos);
        self.primary.push(primary);
        (self.head.len() - 1) as u32
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn extend(&mut self, c: char, pos: usize) {
        let cur = self.new_state(self.len[self.last as usize] + 1, -1, pos as u32, true);
        let mut p = self.last as i32;
        while p != -1 && self.find(p as u32, c).is_none() {
            self.add_edge(p as u32, c, cur);
            p = self.link[p as usize];
        }
        if p == -1 {
            self.link[cur as usize] = 0;
        } else {
            let q = self.find(p as u32, c).unwrap();
            if self.len[p as usize] + 1 == self.len[q as usize] {
                self.link[cur as usize] = q as i32;
            } else {
                let clone = self.new_state(self.len[p as usize] + 1, self.link[q as usize], self.firstpos[q as usize], false);
                let mut e = self.head[q as usize];
                while e != -1 {
                    let idx = e as usize;
                    self.add_edge(clone, self.edge_char[idx], self.edge_to[idx]);
                    e = self.edge_next[idx];
                }
                while p != -1 && self.find(p as u32, c) == Some(q) {
                    self.set_edge(p as u32, c, clone);
                    p = self.link[p as usize];
                }
                self.link[q as usize] = clone as i32;
                self.link[cur as usize] = clone as i32;
            }
        }
        self.last = cur;
    }

    /// Convert the linked-list transitions into the packed `node`/`edges` layout (per-state edges
    /// sorted by char, co-located char+target), reusing `out`'s allocations (no per-build malloc churn).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::needless_range_loop)]
    fn finalize_into(&self, out: &mut Sam) {
        let nstates = self.head.len();
        let nedges = self.edge_char.len();
        out.firstpos.clear();
        out.firstpos.extend_from_slice(&self.firstpos);
        // per-state edge counts → exclusive prefix-sum offsets (local; folded into `node` below)
        let mut off = vec![0u32; nstates + 1];
        for state in 0..nstates {
            let mut e = self.head[state];
            while e != -1 {
                off[state + 1] += 1;
                e = self.edge_next[e as usize];
            }
        }
        for state in 0..nstates {
            off[state + 1] += off[state];
        }
        // edges: (char << 32 | to), sorted by char within each state's [off[s], off[s+1]) range.
        // sorting the packed u64 sorts by char (high bits) since a state's chars are distinct.
        out.edges.clear();
        out.edges.resize(nedges, 0);
        let mut scratch: Vec<u64> = Vec::new(); // reused across states
        for state in 0..nstates {
            scratch.clear();
            let mut e = self.head[state];
            while e != -1 {
                let idx = e as usize;
                scratch.push((u64::from(self.edge_char[idx] as u32) << 32) | u64::from(self.edge_to[idx]));
                e = self.edge_next[idx];
            }
            scratch.sort_unstable();
            let base = off[state] as usize;
            for (k, &packed) in scratch.iter().enumerate() {
                out.edges[base + k] = packed;
            }
        }
        // per-state packed node [len, link(clamped: root -1 → 0, never read), edge_lo, edge_hi]
        out.node.clear();
        out.node.reserve(nstates);
        out.link_len.clear();
        out.link_len.reserve(nstates);
        for s in 0..nstates {
            let link_idx_signed = self.link[s];
            let link = if link_idx_signed < 0 { 0 } else { link_idx_signed as u32 };
            out.node.push([self.len[s], link, off[s], off[s + 1]]);
            // Precompute len(link[s]) — saves the second node-load in `longest_in`'s chain walk.
            // Root (link = -1) is never read here because the chain walk breaks before entering it.
            let llen = if link_idx_signed < 0 { 0 } else { self.len[link_idx_signed as usize] };
            out.link_len.push(llen);
        }
        // root direct transition table (ASCII): fill from the root's edges
        out.root_next.clear();
        out.root_next.resize(ROOT_TBL, -1);
        for k in off[0] as usize..off[1] as usize {
            let e = out.edges[k];
            let ci = (e >> 32) as usize;
            if ci < ROOT_TBL {
                out.root_next[ci] = i32::try_from((e & 0xFFFF_FFFF) as u32).expect("state fits i32");
            }
        }
        self.build_endpos(out, nstates);
    }

    /// Build the endpos range structure (fix b): suffix-link children → DFS-order end-positions
    /// (each state's endpos = a contiguous array range) → wavelet matrix over those positions.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn build_endpos(&self, out: &mut Sam, nstates: usize) {
        // suffix-link children, CSR
        let mut child_head = vec![0u32; nstates + 1];
        for s in 1..nstates {
            child_head[self.link[s] as usize + 1] += 1;
        }
        for s in 0..nstates {
            child_head[s + 1] += child_head[s];
        }
        let mut child_arr = vec![0u32; nstates.saturating_sub(1)];
        let mut fill = child_head.clone();
        for s in 1..nstates {
            let p = self.link[s] as usize;
            child_arr[fill[p] as usize] = s as u32;
            fill[p] += 1;
        }
        // iterative DFS (enter/exit) from root → dfs_in / dfs_cnt + DFS-ordered end-positions;
        // on exit (post-order, children done) accumulate lastpos = max end-position in subtree.
        out.dfs_in.clear();
        out.dfs_in.resize(nstates, 0);
        out.dfs_cnt.clear();
        out.dfs_cnt.resize(nstates, 0);
        out.lastpos.clear();
        out.lastpos.resize(nstates, 0);
        out.epos.clear();
        let mut stack: Vec<(u32, bool)> = vec![(0, false)];
        while let Some((s, exit)) = stack.pop() {
            let su = s as usize;
            if exit {
                out.dfs_cnt[su] = out.epos.len() as u32 - out.dfs_in[su];
                let mut lp = if self.primary[su] { self.firstpos[su] } else { 0 };
                for k in child_head[su]..child_head[su + 1] {
                    lp = lp.max(out.lastpos[child_arr[k as usize] as usize]);
                }
                out.lastpos[su] = lp;
                continue;
            }
            out.dfs_in[su] = out.epos.len() as u32;
            if self.primary[su] {
                out.epos.push(self.firstpos[su]);
            }
            stack.push((s, true));
            for k in child_head[su]..child_head[su + 1] {
                stack.push((child_arr[k as usize], false));
            }
        }
        // merge-sort tree over the DFS-ordered end-positions, flattened CSR-style (build is a
        // negligible fraction of runtime; queries hit the contiguous `seg_data` layout).
        let n = out.epos.len();
        let mut seg_n = 1usize;
        while seg_n < n.max(1) {
            seg_n <<= 1;
        }
        out.seg_n = seg_n;
        let mut tree: Vec<Vec<u32>> = vec![Vec::new(); 2 * seg_n];
        for i in 0..n {
            tree[seg_n + i] = vec![out.epos[i]];
        }
        for i in (1..seg_n).rev() {
            let (lhs, rhs) = (&tree[2 * i], &tree[2 * i + 1]);
            let mut acc = Vec::with_capacity(lhs.len() + rhs.len());
            let (mut x, mut y) = (0usize, 0usize);
            while x < lhs.len() && y < rhs.len() {
                if lhs[x] <= rhs[y] {
                    acc.push(lhs[x]);
                    x += 1;
                } else {
                    acc.push(rhs[y]);
                    y += 1;
                }
            }
            acc.extend_from_slice(&lhs[x..]);
            acc.extend_from_slice(&rhs[y..]);
            tree[i] = acc;
        }
        out.seg_off.clear();
        out.seg_off.reserve(2 * seg_n + 1);
        out.seg_data.clear();
        let mut off = 0u32;
        out.seg_off.push(0);
        for node in &tree {
            out.seg_data.extend_from_slice(node);
            off += u32::try_from(node.len()).expect("epos fits u32");
            out.seg_off.push(off);
        }
        // pack the query-hot per-state fields into one cache-line-friendly array
        out.epmeta.clear();
        out.epmeta.reserve(nstates);
        for s in 0..nstates {
            out.epmeta.push([out.firstpos[s], out.lastpos[s], out.dfs_in[s], out.dfs_cnt[s]]);
        }
        // Optimization B: interleave (node.len, node.link, link_len, epmeta) into one 32-byte slot
        // per state so the chain walk's per-state data lands on a single cache line. Built lazily
        // after epmeta + link_len; safe to do here because all the source arrays are finalized.
        out.chain_slot.clear();
        out.chain_slot.reserve(nstates);
        for s in 0..nstates {
            let nd = out.node[s];
            let m = out.epmeta[s];
            out.chain_slot.push([
                nd[0],            // len
                nd[1],            // link
                out.link_len[s],  // band_min source
                0,                // padding for 32-byte alignment
                m[0],             // firstpos
                m[1],             // lastpos
                m[2],             // dfs_in
                m[3],             // dfs_cnt
            ]);
        }
    }

    fn finalize(self) -> Sam {
        let mut out = Sam::empty();
        self.finalize_into(&mut out);
        out
    }
}

/// Build (and finalize) the suffix automaton of `b` — prebuild once, reuse across pairs.
#[must_use]
pub fn build_sam(b: &[char]) -> Sam {
    let mut bld = Builder::new(b.len());
    for (i, &c) in b.iter().enumerate() {
        bld.extend(c, i);
    }
    bld.finalize()
}

/// Longest substring of `a[al..ar]` that occurs in `b[bl..br)`, using the **precomputed**
/// window-independent match (`fstate`/`fmatch` = the SAM state and match length ending at each
/// a-position, from one full-`a` scan). For each position the chain walk starts at the precomputed
/// state, capping the usable length to the a-fragment (`i-al+1`) and the b-window via endpos
/// queries. Returns (`a_start`, absolute `b_start`, len) with difflib's tie-break. No re-scanning.
///
/// `chain_cap` (Phase 3 approximation knob) bounds the suffix-link chain walk to `chain_cap`
/// states ascended; deeper chains return whatever match was already found. With the empirically
/// measured distribution (p99 depth 7, p95 depth 5; see `gestalt::instrument`'s histograms), a
/// cap of 7 keeps ~99% of chains intact, capping at 5 keeps ~95%, etc. Setting `chain_cap =
/// u32::MAX` recovers the exact behaviour. The caller (`gestalt_edge_with_ms` and friends)
/// derives `chain_cap` from the user-supplied `delta` parameter via `delta_to_chain_cap`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::too_many_arguments)]
fn longest_in(
    sam: &Sam,
    fstate: &[u32],
    fmatch: &[u32],
    al: usize,
    ar: usize,
    bl: usize,
    br: usize,
    chain_cap: u32,
) -> (usize, usize, usize) {
    #[cfg(feature = "instrument")]
    instr_inc(&instrument::LONGEST_IN_CALLS, 1);
    let blo = bl as u32;
    let hi = br as u32 - 1; // caller guarantees bl < br, so br >= 1
    let (mut best_len, mut best_a, mut best_b) = (0usize, 0usize, 0usize);
    // SAFETY: i ∈ [al, ar) ⊆ [0, n) = fstate.len() = fmatch.len(); `cur` is always a valid SAM
    // state (fstate entry or a suffix link), so chain_slot[cur] is in bounds (= nstates entries).
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        for i in al..ar {
            let cap = i - al + 1; // a-match ending at i can't start before `al`
            let eff = (*fmatch.get_unchecked(i) as usize).min(cap);
            if eff <= best_len {
                continue; // can't beat the best — skip (dominant pruning)
            }
            // walk the suffix-link chain from the precomputed state up; `curlen` is the usable length
            // at the current state (capped to the a-fragment), shrinking as we ascend.
            let mut cur = *fstate.get_unchecked(i);
            // Optimization K1: `cur != 0` is an invariant here. `cur == 0` (the SAM root) is only
            // stored in `fstate[i]` when `fmatch[i] == 0`, and in that case `eff == 0 <= best_len`
            // already pruned this iteration via the `continue` above. From the second chain step
            // on, `cur` came from `link != 0` (we check before assigning). Hoisting the
            // entry-time `if cur == 0 { break }` out saves ≈4 k arm64 instructions per pair on
            // the bench corpus (-0.9 % retired) — no cycle change on this single-thread profile
            // because the chain walk is latency-bound by the `node[cur] → link → node[link]`
            // pointer chase, but it frees front-end issue bandwidth for the threaded path.
            let mut chain_depth: u32 = 0;
            loop {
                chain_depth += 1;
                // Phase 3 approximate-RO cap: stop walking the chain past `chain_cap` ascended
                // states. With cap=u32::MAX (default delta=0) this never fires; smaller caps
                // trade tail accuracy for fewer pointer chases. Measurement on canonical Python:
                // p95 chain depth = 5, p99 = 7 — capping at 5/7 truncates <5%/<1% of chains.
                if chain_depth > chain_cap {
                    break;
                }
                // OPTIMIZATION B: single 32-byte load of `chain_slot[cur]` brings the per-state
                // hot fields into registers in ONE cache-line touch:
                //   [0] = len   [1] = link   [2] = link_len   [3] = padding
                //   [4..8] = epmeta (firstpos, lastpos, dfs_in, dfs_cnt) for max_le/min_in
                // Replaces the three scattered loads (node[cur], link_len[cur], epmeta[cur]) that
                // the chain walk used to do, each on its own cache line — PMU attribution showed
                // L1D misses at 18 % of cycle budget split between those three arrays.
                let cs = *sam.chain_slot.get_unchecked(cur as usize);
                let curlen = eff.min(cs[0] as usize);
                if curlen <= best_len {
                    break;
                }
                let band_min = cs[2] as usize + 1;
                if curlen >= band_min {
                    // Reuse the epmeta bytes already in `cs` — no extra load needed.
                    let m = [cs[4], cs[5], cs[6], cs[7]];
                    if let Some(pmax) = sam.max_le_with_meta(&m, hi) {
                        if pmax >= blo {
                            let l_window = (pmax - blo) as usize + 1; // window-cap on match len
                            let l = curlen.min(l_window); // = min(chain-cap, window-cap)
                            if l >= band_min {
                                // earliest-b (min_in) only when we beat the best — rare, off the path.
                                if l > best_len {
                                    // OPTIMIZATION A: when the window cap binds (l == l_window),
                                    // lo_q = blo + l - 1 = pmax. Since pmax is THE max v <= hi in
                                    // this state's endpos, the set ∩ [pmax, hi] is exactly {pmax},
                                    // so pmin = pmax trivially — skip the linear scan in `min_in`.
                                    // Only the chain-cap branch (l < l_window) needs a real scan.
                                    let pmin = if l == l_window {
                                        Some(pmax)
                                    } else {
                                        sam.min_in_with_meta(&m, blo + l as u32 - 1, hi)
                                    };
                                    if let Some(pmin) = pmin {
                                        best_len = l;
                                        best_a = i + 1 - l;
                                        best_b = pmin as usize + 1 - l;
                                    }
                                }
                                break; // deepest qualifying state ⇒ longest in-window match here
                            }
                        }
                    }
                }
                let link = cs[1];
                if link == 0 {
                    break; // reached the root (state 0) — no shorter qualifying state above
                }
                cur = link;
            }
            #[cfg(feature = "instrument")]
            instr_hist(&instrument::CHAIN_DEPTHS, chain_depth as usize);
        }
    }
    (best_a, best_b, best_len)
}

/// Map a user-facing `delta` (max acceptable RO ratio loss, in absolute units) to a chain-walk
/// depth cap for `longest_in`. Empirically calibrated against `gestalt::instrument`'s chain
/// depth histogram on the mypy/sympy corpora:
///
/// | delta | cap | covers |
/// |---:|---:|---:|
/// | 0.00 | `u32::MAX` | exact (default) |
/// | 0.01 | 12 | p99.5+ |
/// | 0.05 | 7  | p99   |
/// | 0.10 | 5  | p95   |
/// | 0.20 | 3  | p85   |
/// | 0.50 | 2  | p70   |
/// | 1.00 | 1  | only fstate, no walk |
///
/// Formula: `cap = ceil(1 / sqrt(delta))` clamped to ≥1 for delta > 0. Pure heuristic — the
/// property test in `tests/approx_ro.rs` verifies the actual loss stays below `delta` on a
/// representative corpus.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn delta_to_chain_cap(delta: f64) -> u32 {
    if delta <= 0.0 {
        return u32::MAX;
    }
    if delta >= 1.0 {
        return 1;
    }
    let cap = (1.0_f64 / delta.sqrt()).ceil() as u32;
    cap.max(1)
}

/// Window-independent matching statistics of `a` vs `sam_b`, filled into reused buffers: for each
/// i, `(state, matched)` where `matched` = longest suffix of `a[..=i]` occurring anywhere in b.
/// One O(|a|) scan, reused by every recursion node (no per-node re-scan). Reusing the caller's
/// buffers avoids a per-pair allocation (was ~10% of the all-pairs join).
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn matching_stats_into(a: &[char], sam_b: &Sam, fstate: &mut Vec<u32>, fmatch: &mut Vec<u32>) {
    let n = a.len();
    // Size the buffers WITHOUT zero-filling: the scan below writes every index [0, n) before any read,
    // so the resize(n, 0) zero-pass was pure waste (it was ~half the scan's store traffic).
    // SAFETY: u32 has no invalid bit patterns, and fstate[i]/fmatch[i] are written for every i in 0..n
    // (the loop below) strictly before longest_in ever reads them.
    fstate.clear();
    fstate.reserve(n);
    fmatch.clear();
    fmatch.reserve(n);
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        fstate.set_len(n);
        fmatch.set_len(n);
    }
    // Hoist the SAM arrays into locals so the compiler keeps base pointers in registers and
    // doesn't reload them through `sam_b` each iteration; transition is hand-inlined.
    let node = sam_b.node.as_slice();
    let edges = sam_b.edges.as_slice();
    let root = sam_b.root_next.as_slice();
    let mut state = 0u32;
    let mut matched = 0u32;
    // SAFETY: `state` is always a valid SAM state index (< nstates = node.len()): it starts at the
    // root (0) and only ever becomes a transition target (an edge's low bits, a valid state) or a
    // suffix link (node[..][1], a valid state). So node[state] and node[link] are in bounds; the edge
    // range [edge_lo, edge_hi) ⊆ [0, edges.len()); `ci < ROOT_TBL == root.len()`; `i < n == fstate.len()`.
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        for i in 0..n {
            let c = *a.get_unchecked(i);
            loop {
                // inline `transition(state, c)` → next state, or -1. The non-root branch loads
                // node[state] ONCE for both the edge range and (on miss) the suffix link — one cache line.
                let nx: i64 = if state == 0 {
                    let ci = c as usize;
                    if ci < ROOT_TBL {
                        i64::from(*root.get_unchecked(ci))
                    } else {
                        let nd = node.get_unchecked(0);
                        csr_lookup(edges, nd[2] as usize, nd[3] as usize, c)
                    }
                } else {
                    let nd = node.get_unchecked(state as usize);
                    csr_lookup(edges, nd[2] as usize, nd[3] as usize, c)
                };
                if nx >= 0 {
                    state = nx as u32;
                    matched += 1;
                    break;
                }
                if state == 0 {
                    matched = 0;
                    break;
                }
                let link = node.get_unchecked(state as usize)[1]; // same line as the edge-range load
                state = link;
                matched = node.get_unchecked(link as usize)[0]; // len(link)
            }
            *fstate.get_unchecked_mut(i) = state;
            *fmatch.get_unchecked_mut(i) = matched;
        }
    }
}

/// Binary search the packed edge slice `edges[lo..hi]` (sorted by char in the high 32 bits) for `c`;
/// returns the target state (low 32 bits) as i64, or -1. Inlined into the scan.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn csr_lookup(edges: &[u64], mut lo: usize, hi: usize, c: char) -> i64 {
    let mut hi = hi;
    let key = c as u32;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // SAFETY: callers pass lo,hi from a state's edge range ⊆ [0, edges.len()]; `mid ∈ [lo, hi)`.
        let e = unsafe { *edges.get_unchecked(mid) };
        let mc = (e >> 32) as u32; // char (code point) in the high 32 bits
        if mc == key {
            return i64::from((e & 0xFFFF_FFFF) as u32); // target state in the low 32 bits
        }
        if mc < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    -1
}

thread_local! {
    /// Reused (fstate, fmatch) buffers for the per-pair matching statistics — no per-pair alloc.
    static MS_BUF: std::cell::RefCell<(Vec<u32>, Vec<u32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    /// Reused recursion stack of (al, ar, bl, br) windows — retains capacity across pairs so the
    /// RO recursion never heap-allocates or reallocs per pair (was `__rust_alloc` + `grow_one`).
    static STACK_BUF: std::cell::RefCell<Vec<(usize, usize, usize, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Reused DP buffer for `ub_from_fmatch` (currently unused — kept for re-enabling H later).
    #[allow(dead_code)]
    static UB_BUF: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Tight upper bound on the RO matched-length M, computed from the per-position `fmatch` array
/// via O(na) weighted-interval-scheduling DP. Each `fmatch[i]` records the longest substring of
/// `b` ending at `a[i]`; an actual RO decomposition picks non-overlapping such matches, so M is
/// bounded above by the max non-overlapping sum we can extract from `fmatch`.
///
/// `dp[i]` = best sum using only positions `0..=i`. Recurrence:
///   * not taking position i: `dp[i-1]`
///   * taking the fmatch[i]-length block ending at i (a[i+1-l..=i]): `dp[i-l] + l` where
///     `l = fmatch[i].min(i+1)` (clamped to what fits in `a[..=i]`)
///
/// **CURRENTLY UNUSED** — tried in `gestalt_edge_with_ms` (optimization H) and reverted: on
/// the `cluster_canonicals` threshold path the O(na) DP cost (+ thread-local buffer access) was
/// larger than the wall savings from skipping recursion, because the existing
/// `m + pending < need` check inside the recursion already bails cheaply for non-edges. Kept
/// as a tombstone in case a future call shape (e.g., larger threshold + denser matches) makes
/// it worth re-trying. To re-enable: insert the call before `STACK_BUF.with_borrow_mut` in
/// `gestalt_edge_with_ms` and gate on `need > 0`.
#[allow(dead_code)]
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn ub_from_fmatch(fmatch: &[u32], buf: &mut Vec<u32>) -> u32 {
    let n = fmatch.len();
    if n == 0 {
        return 0;
    }
    buf.clear();
    buf.resize(n, 0);
    // SAFETY: buf and fmatch are both len = n; indices below stay in [0, n).
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        let mut prev: u32 = 0;
        for i in 0..n {
            let l = (*fmatch.get_unchecked(i)).min(i as u32 + 1);
            let with_take = if l == 0 {
                0
            } else if (l as usize) > i {
                l // entire prefix [0..=i] is the block, no `dp[i-l]` term
            } else {
                buf.get_unchecked(i - l as usize).saturating_add(l)
            };
            let v = prev.max(with_take);
            *buf.get_unchecked_mut(i) = v;
            prev = v;
        }
        prev
    }
}

/// Test-only access to `matching_stats_into` — used by `corpus_sa` tests to verify the SA-based
/// fmatch is byte-for-byte identical to the SAM-based fmatch.
#[doc(hidden)]
pub fn matching_stats_for_test(a: &[char], sam_b: &Sam, fstate: &mut Vec<u32>, fmatch: &mut Vec<u32>) {
    matching_stats_into(a, sam_b, fstate, fmatch);
}

/// Cost probe: run only the matching-statistics scan of `a` vs `sam_b` (the unavoidable per-pair
/// floor — RO's first block is an LCS, Θ(|a|)) and return a checksum so it isn't optimized out.
/// Measures the pure scan throughput separate from the RO recursion.
#[must_use]
pub fn matching_stats_cost(a: &[char], sam_b: &Sam) -> u64 {
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        matching_stats_into(a, sam_b, fstate, fmatch);
        fmatch.iter().map(|&x| u64::from(x)).sum()
    })
}

fn gestalt_m_with(a: &[char], b: &[char], sam_b: &Sam) -> usize {
    let n = a.len();
    if n == 0 {
        return 0;
    }
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        matching_stats_into(a, sam_b, fstate, fmatch);
        gestalt_m_recur(a, b, sam_b, fstate, fmatch)
    })
}

#[allow(clippy::cast_sign_loss, clippy::many_single_char_names)]
fn gestalt_m_recur(a: &[char], b: &[char], sam_b: &Sam, fstate: &[u32], fmatch: &[u32]) -> usize {
    let n = a.len();
    let mut total = 0usize;
    STACK_BUF.with_borrow_mut(|stack| {
        stack.clear();
        stack.push((0, n, 0, b.len()));
        while let Some((al, ar, bl, br)) = stack.pop() {
            if al >= ar || bl >= br {
                continue;
            }
            let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br, u32::MAX);
            if l == 0 {
                continue;
            }
            total += l;
            stack.push((al, i, bl, j));
            stack.push((i + l, ar, j + l, br));
        }
    });
    total
}

/// Threshold-aware exact decision: does `RO(a,b) ≥ threshold`? Computes the matched total M
/// with **two-sided early-exit** — accept the instant `M ≥ need`, reject the instant the upper
/// bound `M + Σ min(window lengths) < need`. Exact (the bound never drops a qualifying pair),
/// and dissimilar pairs abort long before the full decomposition. `need = ⌈threshold·(|a|+|b|)/2⌉`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
#[must_use]
pub fn gestalt_qualifies(a: &[char], b: &[char], sam_b: &Sam, threshold: f64) -> bool {
    let nb = b.len();
    let total = a.len() + nb;
    if total == 0 {
        return true;
    }
    let need = (threshold * total as f64 / 2.0).ceil() as usize;
    if need == 0 {
        return true;
    }
    let n = a.len();
    if n == 0 || nb == 0 {
        return false; // M = 0 < need
    }
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        matching_stats_into(a, sam_b, fstate, fmatch);
        gestalt_qualifies_ms(n, nb, sam_b, threshold, fstate, fmatch)
    })
}

/// The threshold early-exit recursion given **precomputed** matching statistics (`fstate`/`fmatch`
/// for `a` of length `na` vs `sam_b`). Split out so the scan can be done in an MLP batch separately.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
#[must_use]
pub fn gestalt_qualifies_ms(na: usize, nb: usize, sam_b: &Sam, threshold: f64, fstate: &[u32], fmatch: &[u32]) -> bool {
    let total = na + nb;
    if total == 0 {
        return true;
    }
    let need = (threshold * total as f64 / 2.0).ceil() as usize;
    if need == 0 {
        return true;
    }
    if na == 0 || nb == 0 {
        return false;
    }
    STACK_BUF.with_borrow_mut(|stack| {
        let mut m = 0usize;
        let mut pending = na.min(nb);
        stack.clear();
        stack.push((0, na, 0, nb));
        while let Some((al, ar, bl, br)) = stack.pop() {
            pending -= (ar - al).min(br - bl);
            if al >= ar || bl >= br {
                continue;
            }
            let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br, u32::MAX);
            if l == 0 {
                continue;
            }
            m += l;
            if m >= need {
                return true;
            }
            pending += (i - al).min(j - bl) + (ar - i - l).min(br - j - l);
            if m + pending < need {
                return false;
            }
            stack.push((al, i, bl, j));
            stack.push((i + l, ar, j + l, br));
        }
        m >= need
    })
}


/// Edge test that also yields the exact ratio: `Some(ratio)` iff `RO(a,b) >= threshold`, else `None`.
/// Keeps the **reject** early-exit (abort the instant the upper bound `M + Σ min(window) < need`) but
/// computes full M on the qualifying branch so the cached ratio feeds the cluster `min_sim` — caching
/// edge ratios here means `min_sim` only recomputes the rare non-edge (chained) intra-cluster pairs,
/// not the whole dense blob. Bit-identical to `(let r = ratio; (r >= threshold).then_some(r))`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
#[must_use]
pub fn gestalt_edge(a: &[char], b: &[char], sam_b: &Sam, threshold: f64) -> Option<f64> {
    let na = a.len();
    let nb = b.len();
    let total = na + nb;
    if total == 0 {
        return Some(1.0);
    }
    let need = (threshold * total as f64 / 2.0).ceil() as usize;
    if na == 0 || nb == 0 {
        return (need == 0).then_some(0.0);
    }
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        matching_stats_into(a, sam_b, fstate, fmatch);
        gestalt_edge_with_ms(a, b, sam_b, fstate, fmatch, threshold)
    })
}

/// Stage-4b helper: same recursion + early-exit as `gestalt_edge`, but operates on a
/// **caller-provided** `(fstate, fmatch)` instead of running `matching_stats_into` inline.
///
/// The GPU dispatch produces these arrays in batch for many pairs at once; the CPU side then
/// does the small stack walk per pair via this entry point. Keeping the recursion on the CPU is
/// the right split because `longest_in`'s suffix-link walk has data-dependent depth (poor GPU
/// fit), while `matching_stats_into` is a wide independent per-pair walk (great GPU fit).
///
/// `fstate` / `fmatch` must be `a.len()` long and have been filled for THIS exact `(a, sam_b)`
/// pair — passing arrays computed for a different `a` or `b` is a logic bug, no runtime check.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn gestalt_edge_with_ms(
    a: &[char],
    b: &[char],
    sam_b: &Sam,
    fstate: &[u32],
    fmatch: &[u32],
    threshold: f64,
) -> Option<f64> {
    // Backwards-compatible exact wrapper. New callers wanting approximation pass through
    // `gestalt_edge_with_ms_delta` directly with their delta.
    gestalt_edge_with_ms_delta(a, b, sam_b, fstate, fmatch, threshold, 0.0)
}

/// Approximate-RO variant: same as [`gestalt_edge_with_ms`] but caps the suffix-link chain walk
/// inside `longest_in` to roughly `1/√delta` ascents. `delta = 0.0` means exact (no cap, default).
/// `delta ∈ (0, 1]` caps the depth; the returned ratio's worst-case absolute deviation from the
/// exact RO is bounded by ~delta (empirically verified on canonical-Python corpora by
/// `tests/approx_ro.rs`). The chain depth distribution on real workloads is heavy-headed
/// (p99 ≈ 7), so even small delta values rarely actually fire the cap.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn gestalt_edge_with_ms_delta(
    a: &[char],
    b: &[char],
    sam_b: &Sam,
    fstate: &[u32],
    fmatch: &[u32],
    threshold: f64,
    delta: f64,
) -> Option<f64> {
    let chain_cap = delta_to_chain_cap(delta);
    gestalt_edge_with_ms_inner(a, b, sam_b, fstate, fmatch, threshold, chain_cap)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::many_single_char_names)]
fn gestalt_edge_with_ms_inner(
    a: &[char],
    b: &[char],
    sam_b: &Sam,
    fstate: &[u32],
    fmatch: &[u32],
    threshold: f64,
    chain_cap: u32,
) -> Option<f64> {
    let na = a.len();
    let nb = b.len();
    let total = na + nb;
    if total == 0 {
        return Some(1.0);
    }
    let need = (threshold * total as f64 / 2.0).ceil() as usize;
    if na == 0 || nb == 0 {
        return (need == 0).then_some(0.0);
    }
    debug_assert_eq!(fstate.len(), na, "fstate length must equal a.len()");
    debug_assert_eq!(fmatch.len(), na, "fmatch length must equal a.len()");
    #[cfg(feature = "instrument")]
    {
        instr_inc(&instrument::PAIRS_PROCESSED, 1);
        let (mut zero, mut nz, mut sum) = (0u64, 0u64, 0u64);
        for &f in fmatch {
            if f == 0 {
                zero += 1;
            } else {
                nz += 1;
                sum += f as u64;
            }
        }
        instr_inc(&instrument::FMATCH_ZERO, zero);
        instr_inc(&instrument::FMATCH_NONZERO, nz);
        instr_inc(&instrument::FMATCH_SUM, sum);
    }
    // Optimization H (tight fmatch-based UB) was tried and reverted — measured a NET regression
    // on cluster_canonicals threshold path: the O(na) weighted-interval-scheduling DP added
    // ~25 µs per call across rayon workers, and the bail rate on filter-survivors was too low
    // to amortize. The existing `m + pending < need` check inside the recursion loop already
    // aborts cheaply for non-edges (after 1-2 longest_in calls in the common case). See
    // `src/new/PERF_MAP.md`'s "Tombstones" section and the `ub_from_fmatch` helper just above —
    // kept around but unused in case a different call shape makes it worth re-trying.
    STACK_BUF.with_borrow_mut(|stack| {
        let mut m = 0usize;
        let mut pending = na.min(nb);
        stack.clear();
        stack.push((0, na, 0, nb));
        #[cfg(feature = "instrument")]
        let mut max_depth: usize = 1;
        while let Some((al, ar, bl, br)) = stack.pop() {
            #[cfg(feature = "instrument")]
            {
                if stack.len() + 1 > max_depth {
                    max_depth = stack.len() + 1;
                }
            }
            pending -= (ar - al).min(br - bl);
            if al >= ar || bl >= br {
                continue;
            }
            let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br, chain_cap);
            if l == 0 {
                continue;
            }
            m += l;
            pending += (i - al).min(j - bl) + (ar - i - l).min(br - j - l);
            if m + pending < need {
                #[cfg(feature = "instrument")]
                instr_hist(&instrument::RECURSION_DEPTHS, max_depth);
                return None; // upper bound below the bar ⇒ certified non-edge, abort
            }
            stack.push((al, i, bl, j));
            stack.push((i + l, ar, j + l, br));
        }
        #[cfg(feature = "instrument")]
        instr_hist(&instrument::RECURSION_DEPTHS, max_depth);
        (m >= need).then(|| 2.0 * m as f64 / total as f64)
    })
}

/// Cluster `min_sim` helper: returns the **exact** ratio when `RO(a,b) <= cap`, otherwise a value
/// `> cap` (it accept-early-exits the instant M exceeds the cap's bound). Used to find a cluster's
/// minimum pairwise ratio with `cur = cur.min(gestalt_ratio_capped(a, b, sam, cur))`: in a dense
/// cluster the dominant high-ratio pairs blow past `cur` and are pruned after the first block or two,
/// so full M is computed only for the genuinely-low pairs. Bit-identical minimum to the full ratio.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
#[must_use]
pub fn gestalt_ratio_capped(a: &[char], b: &[char], sam_b: &Sam, cap: f64) -> f64 {
    let na = a.len();
    let nb = b.len();
    let total = na + nb;
    if total == 0 {
        return 1.0; // two empty strings ⇒ ratio 1.0
    }
    if na == 0 || nb == 0 {
        return 0.0; // M = 0 ⇒ ratio 0 (a valid minimum candidate, <= cap for any cap >= 0)
    }
    // ratio > cap ⟺ 2M/total > cap ⟺ M > cap·total/2 ⟺ M >= ⌊cap·total/2⌋ + 1.
    let exceed = (cap * total as f64 / 2.0).floor() as usize + 1;
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        matching_stats_into(a, sam_b, fstate, fmatch);
        STACK_BUF.with_borrow_mut(|stack| {
            let mut m = 0usize;
            stack.clear();
            stack.push((0, na, 0, nb));
            while let Some((al, ar, bl, br)) = stack.pop() {
                if al >= ar || bl >= br {
                    continue;
                }
                let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br, u32::MAX);
                if l == 0 {
                    continue;
                }
                m += l;
                if m >= exceed {
                    return 2.0; // ratio > cap ⇒ cannot be the minimum; prune (any value > cap works)
                }
                stack.push((al, i, bl, j));
                stack.push((i + l, ar, j + l, br));
            }
            2.0 * m as f64 / total as f64 // full exact M ⇒ exact ratio (<= cap)
        })
    })
}

/// Ratio of `a` vs the string the prebuilt `sam_b` was built from (= `b`).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn gestalt_ratio_prebuilt(a: &[char], b: &[char], sam_b: &Sam) -> f64 {
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * gestalt_m_with(a, b, sam_b) as f64 / total as f64
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn gestalt_ratio_chars(a: &[char], b: &[char]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let sam_b = build_sam(b);
    gestalt_ratio_prebuilt(a, b, &sam_b)
}

/// `gestalt_ratio(a, b) -> float`: exact difflib ratio, computed via suffix-automaton LCS.
#[must_use]
pub fn gestalt_ratio(a: &str, b: &str) -> f64 {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    gestalt_ratio_chars(&av, &bv)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unreadable_literal)]
    use super::{build_sam, gestalt_edge, gestalt_qualifies, gestalt_ratio_capped, gestalt_ratio_chars};

    fn r(a: &str, b: &str) -> f64 {
        gestalt_ratio_chars(&a.chars().collect::<Vec<_>>(), &b.chars().collect::<Vec<_>>())
    }

    #[test]
    fn matches_difflib_reference_values() {
        assert_eq!(r("", ""), 1.0);
        assert_eq!(r("", "x"), 0.0);
        assert_eq!(r("abc", "abc"), 1.0);
        assert_eq!(r("abc", "abd"), 0.6666666666666666);
        assert_eq!(r("the quick brown fox", "the quick brown dog"), 0.8947368421052632);
        assert_eq!(r("tide", "diet"), 0.25);
        assert_eq!(r("ПриветМир", "ПриветМирЪ"), 0.9473684210526315);
        assert_eq!(r("aaaaabbbbbccccc", "aaaaaxbbbbbxccccc"), 0.9375);
    }

    // `gestalt_qualifies` (threshold early-exit) must be byte-for-byte with `ratio >= T`.
    #[test]
    fn qualifies_matches_ratio_threshold() {
        // xorshift PRNG → deterministic pseudo-random ASCII strings over a small alphabet
        let mut s: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..3000 {
            let la = (next() % 40) as usize;
            let lb = (next() % 40) as usize;
            let mk = |n: usize, rng: &mut dyn FnMut() -> u64| -> Vec<char> {
                (0..n).map(|_| char::from(b'a' + (rng() % 4) as u8)).collect()
            };
            let a = mk(la, &mut next);
            let b = mk(lb, &mut next);
            let sam_b = build_sam(&b);
            let ratio = gestalt_ratio_chars(&a, &b);
            for &t in &[0.0_f64, 0.25, 0.5, 0.75, 0.9, 1.0] {
                assert_eq!(
                    gestalt_qualifies(&a, &b, &sam_b, t),
                    ratio >= t,
                    "a={a:?} b={b:?} t={t} ratio={ratio}"
                );
                // gestalt_edge(cap=t): Some(exact ratio) iff qualifying; bit-exact ratio when Some.
                let edge = gestalt_edge(&a, &b, &sam_b, t);
                assert_eq!(edge.is_some(), ratio >= t, "edge.is_some a={a:?} b={b:?} t={t} ratio={ratio}");
                if let Some(r) = edge {
                    assert_eq!(r, ratio, "edge ratio a={a:?} b={b:?} t={t}");
                }
                // gestalt_ratio_capped(cap=t): exact ratio when ratio <= t, else a value > t (pruned).
                let capped = gestalt_ratio_capped(&a, &b, &sam_b, t);
                if ratio <= t {
                    assert_eq!(capped, ratio, "capped exact a={a:?} b={b:?} cap={t} ratio={ratio}");
                } else {
                    assert!(capped > t, "capped prune a={a:?} b={b:?} cap={t} ratio={ratio} got={capped}");
                }
            }
        }
    }
}
