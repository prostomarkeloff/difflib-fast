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
//!     every pair, so the all-pairs cost is n builds + n² scans, not n² builds;
//!   * the scan reads one 32-byte slot per state whose inline transition continues along `b`,
//!     so a match that keeps extending costs one load per character (`Sam::fast`);
//!   * the recursion's narrow windows (`b` side of a few characters) go to a direct row-by-row
//!     comparison (`longest_direct`) instead of climbing the automaton for every position, and
//!     the early-exit recursions take the largest window first so their bounds close sooner;
//!   * the scan can stop the moment a single match proves the pair above a bound
//!     (`matching_stats_bounded`), and a common prefix or suffix decides many pairs before it.
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
#[allow(clippy::all, clippy::pedantic)] // diagnostics only; never in a default build
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

    /// Per log2(min(|aw|,|bw|)) bucket: longest_in calls, positions iterated, chain walks, chain steps, calls that found nothing.
    pub static WIN_CALLS: [AtomicU64; 24] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 24] };
    pub static WIN_POS: [AtomicU64; 24] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 24] };
    pub static WIN_WALKS: [AtomicU64; 24] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 24] };
    pub static WIN_STEPS: [AtomicU64; 24] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 24] };
    pub static WIN_ZERO: [AtomicU64; 24] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 24] };
    /// Chain steps + longest_in calls by pair category: 0 exact, 1 edge test, 2 capped.
    pub static CAT_STEPS: [AtomicU64; 3] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 3] };
    pub static CAT_CALLS: [AtomicU64; 3] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 3] };
    pub static CAT_PAIRS: [AtomicU64; 3] = { const ZERO: AtomicU64 = AtomicU64::new(0); [ZERO; 3] };
    thread_local! {
        pub static TL_STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        pub static TL_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    pub fn pair_done(cat: usize) {
        let s = TL_STEPS.with(|c| c.replace(0));
        let n = TL_CALLS.with(|c| c.replace(0));
        CAT_STEPS[cat].fetch_add(s, Ordering::Relaxed);
        CAT_CALLS[cat].fetch_add(n, Ordering::Relaxed);
        CAT_PAIRS[cat].fetch_add(1, Ordering::Relaxed);
    }
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

        for (k, name) in ["exact", "edge", "capped"].iter().enumerate() {
            s.push_str(&format!("category {name:>6}: pairs {:>9} longest_in {:>10} steps {:>12}\n", CAT_PAIRS[k].load(Ordering::Relaxed), CAT_CALLS[k].load(Ordering::Relaxed), CAT_STEPS[k].load(Ordering::Relaxed)));
        }
        s.push_str("windows by log2(min side): calls / positions / walks / steps / found-nothing\n");
        for k in 0..24 {
            let c = WIN_CALLS[k].load(Ordering::Relaxed);
            if c == 0 { continue; }
            s.push_str(&format!("  2^{:<2} {:>10} {:>12} {:>12} {:>12} {:>10}\n", k, c, WIN_POS[k].load(Ordering::Relaxed), WIN_WALKS[k].load(Ordering::Relaxed), WIN_STEPS[k].load(Ordering::Relaxed), WIN_ZERO[k].load(Ordering::Relaxed)));
        }
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
#[allow(clippy::inline_always)]
fn instr_inc(c: &std::sync::atomic::AtomicU64, by: u64) {
    c.fetch_add(by, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(feature = "instrument")]
#[inline(always)]
#[allow(clippy::inline_always)]
fn instr_hist(buckets: &[std::sync::atomic::AtomicU64; instrument::HIST_BUCKETS], depth: usize) {
    buckets[depth.min(instrument::HIST_BUCKETS - 1)]
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// Suffix automaton with CSR (sorted-per-state) transitions — built once, queried by scans.
///
/// For the range-restricted recursion (fix b), each state also carries its **endpos** as a
/// contiguous slice `[dfs_in, dfs_in+dfs_cnt)` of `epos` (the end-positions in b, laid out so
/// that every subtree of the suffix-link tree is contiguous). Small sets are scanned linearly;
/// the few states whose set is larger than [`LINEAR_MAX`] carry a sorted copy in `big_sorted`,
/// and "is there an end-position in [lo,hi] within this state's subtree, and the min/max such"
/// is one binary search there — so the whole RO recursion runs on this one prebuilt SAM, with
/// **no sub-builds**.
#[derive(Clone)]
pub struct Sam {
    // Transitions packed as `(char as u64) << 32 | to`, sorted by char within each state's range
    // [edge_lo, edge_hi). Co-locating char+target means the binary-search key and the taken edge's
    // target live on the same cache line (was two parallel arrays `csr_char`/`csr_to`).
    edges: Vec<u64>,
    // Direct lookup for the root's ASCII transitions (`root_next[c] = state`, or -1). The root is
    // the high-degree state hit after every match reset; this makes its transition O(1) instead
    // of a binary search. Non-ASCII root chars (rare) fall back to the edge search, so it's general.
    root_next: Vec<i32>,
    // The scan's per-state slot, one 32-byte line:
    //   [0] c0  [1] t0   first inline transition — for a per-position state the one that continues
    //                     along `b`, so a match that keeps extending costs one load per character
    //   [2] c1  [3] t1   second inline transition (the state's next one in char order)
    //   [4] edge_lo  [5] edge_hi   the state's range in `edges`, for a state with more than two
    //   [6] link  [7] link_len     the fallback: suffix link and `len(link)`, the matched length after it
    // Nearly every state has one or two transitions, so a miss on the inline pair means a
    // fallback, not a search; `c0`/`c1` are `u32::MAX` (no code point) when absent. (A 128-bit
    // character set in place of the second transition and the range was measured slower.)
    fast: Vec<ScanSlot>,
    // Per position of `b`, the state reached by reading `b[..=pos]` from the root (the state
    // created for that position). A scan of a string sharing a prefix with `b` starts there.
    pos_state: Vec<u32>,
    // End-positions laid out by subtree (each state's endpos = contiguous slice
    // [dfs_in, dfs_in+dfs_cnt)). For small endpos sets a cache-friendly linear scan of this beats
    // any tree descent.
    epos: Vec<u32>,
    // Chain-walk hot slot, 32 bytes per state:
    //   [0] len   [1] link   [2] link_len   [3] sorted_off (u32::MAX if dfs_cnt <= LINEAR_MAX)
    //   [4] firstpos   [5] lastpos   [6] dfs_in   [7] dfs_cnt
    // One cache-line-aligned load supplies everything `longest_in`'s chain walk + `max_le`/`min_in`
    // need for one state. Replaces three separate scattered accesses; PMU said L1D miss rate was
    // 0.51 % and accounted for 18 % of cycles, distributed across those per-state arrays.
    chain_slot: Vec<[u32; 8]>,
    // Sorted end-positions of the states with more than `LINEAR_MAX` of them (short, frequent
    // substrings near the root), concatenated; a state's copy starts at `chain_slot[s][3]`.
    // Replaces a merge-sort tree over all positions: those queries were under 1 % of the endpos
    // queries on real code, and the tree cost one allocation per leaf to build.
    big_sorted: Vec<u32>,
}

/// One state's scan slot (see `Sam::fast`), aligned so a state never straddles two cache lines.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct ScanSlot([u32; 8]);

/// Below this endpos-set size, `max_le`/`min_in` linear-scan the state's contiguous endpos slice
/// (cache-friendly) instead of binary-searching a sorted copy — most queried sets are this small.
const LINEAR_MAX: usize = 256;

impl Sam {
    fn empty() -> Self {
        Sam {
            edges: Vec::new(),
            root_next: Vec::new(),
            fast: Vec::new(),
            pos_state: Vec::new(),
            epos: Vec::new(),
            chain_slot: Vec::new(),
            big_sorted: Vec::new(),
        }
    }

    /// The packed `[len, link, edge_lo, edge_hi]` per state — what the GPU port
    /// (`gpu::matching_stats_gpu`) serializes into a Metal buffer. `link` is the root's link (-1)
    /// stored as 0, never read. Assembled from the CPU tables on request.
    #[must_use]
    pub fn nodes(&self) -> Vec<[u32; 4]> {
        self.chain_slot
            .iter()
            .zip(&self.fast)
            .map(|(cs, f)| [cs[0], cs[1], f.0[4], f.0[5]])
            .collect()
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

    /// Largest end-position `<= x` among the state's endpos, from its pre-loaded chain slot
    /// `cs` (see `chain_slot`). Used by `longest_in`'s chain walk where `cs` was already pulled
    /// from `chain_slot[cur]` — no second per-state load.
    fn max_le_slot(&self, cs: &[u32; 8], x: u32) -> Option<u32> {
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MAX_LE_CALLS, 1);
        // O(1) fast path: x doesn't split the state's [firstpos, lastpos] span.
        if cs[5] <= x {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MAX_LE_FAST_PATH, 1);
            return Some(cs[5]);
        }
        if cs[4] > x {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MAX_LE_FAST_PATH, 1);
            return None;
        }
        let cnt = cs[7] as usize;
        if cnt <= LINEAR_MAX {
            #[cfg(feature = "instrument")]
            {
                instr_inc(&instrument::MAX_LE_LINEAR, 1);
                instr_inc(&instrument::MAX_LE_LINEAR_LEN_SUM, cnt as u64);
            }
            let lo = cs[6] as usize;
            let mut best = 0u32;
            // SAFETY: `cs` is a valid SAM state's chain slot, so the slice [cs[6], cs[6]+cs[7]) is
            // within `self.epos` (built that way at SAM construction).
            #[allow(clippy::undocumented_unsafe_blocks)]
            for &v in unsafe { self.epos.get_unchecked(lo..lo + cnt) } {
                best = best.max(if v <= x { v } else { 0 });
            }
            return Some(best);
        }
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MAX_LE_SEGTREE, 1);
        let so = cs[3] as usize;
        node_max_le(&self.big_sorted[so..so + cnt], x)
    }

    /// Smallest end-position in `[lo, hi]`, from the pre-loaded chain slot. See `max_le_slot`.
    fn min_in_slot(&self, cs: &[u32; 8], lo: u32, hi: u32) -> Option<u32> {
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MIN_IN_CALLS, 1);
        if cs[4] >= lo {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MIN_IN_FAST_PATH, 1);
            return (cs[4] <= hi).then_some(cs[4]);
        }
        if cs[5] < lo {
            #[cfg(feature = "instrument")]
            instr_inc(&instrument::MIN_IN_FAST_PATH, 1);
            return None;
        }
        let cnt = cs[7] as usize;
        if cnt <= LINEAR_MAX {
            #[cfg(feature = "instrument")]
            {
                instr_inc(&instrument::MIN_IN_LINEAR, 1);
                instr_inc(&instrument::MIN_IN_LINEAR_LEN_SUM, cnt as u64);
            }
            let off = cs[6] as usize;
            let mut best = u32::MAX;
            // SAFETY: `cs` is from a valid SAM state; endpos slice within bounds (same as max_le).
            #[allow(clippy::undocumented_unsafe_blocks)]
            for &v in unsafe { self.epos.get_unchecked(off..off + cnt) } {
                best = best.min(if v >= lo && v <= hi { v } else { u32::MAX });
            }
            return (best != u32::MAX).then_some(best);
        }
        #[cfg(feature = "instrument")]
        instr_inc(&instrument::MIN_IN_SEGTREE, 1);
        let so = cs[3] as usize;
        node_min_in(&self.big_sorted[so..so + cnt], lo, hi)
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

/// Transitions a builder state keeps inline; the rest overflow into the list arena. Nearly every
/// state of a code string's automaton has one or two, so `find` almost never touches the arena.
const INLINE: usize = 4;

/// "No code point" marker for an empty inline transition slot (no `char` is `u32::MAX`).
const NONE: u32 = u32::MAX;

/// Transient builder (Ukkonen online SAM). One per thread, reused across builds: every arena and
/// scratch vector keeps its capacity, so building a SAM allocates only the vectors the finished
/// `Sam` owns.
struct Builder {
    // per state: up to `INLINE` transitions as `[c0, t0, c1, t1, ...]`, `NONE` when empty
    inl: Vec<[u32; 2 * INLINE]>,
    deg: Vec<u32>, // per state: number of transitions, inline and overflowed
    // overflow arena (a state's transitions beyond `INLINE`), linked per state from `head`
    edge_char: Vec<u32>,
    edge_to: Vec<u32>,
    edge_next: Vec<i32>,
    head: Vec<i32>,
    // The root's ASCII transitions as a direct table (`-1` = none): the root has by far the most
    // transitions (one per distinct character) and every extension whose character is new to
    // the current suffixes walks up to it.
    root_tbl: Vec<i32>,
    link: Vec<i32>,
    len: Vec<u32>,
    firstpos: Vec<u32>,
    primary: Vec<bool>, // true for the per-position state (its firstpos is a real end-position)
    last: u32,
    // finalize scratch
    off: Vec<u32>,
    child_head: Vec<u32>,
    child_arr: Vec<u32>,
    lastpos: Vec<u32>,
}

thread_local! {
    /// The per-thread SAM builder — arenas and scratch retained across `build_sam` calls.
    static BUILDER: std::cell::RefCell<Builder> = std::cell::RefCell::new(Builder::empty());
}

impl Builder {
    fn empty() -> Self {
        Builder {
            inl: Vec::new(),
            deg: Vec::new(),
            edge_char: Vec::new(),
            edge_to: Vec::new(),
            edge_next: Vec::new(),
            head: Vec::new(),
            root_tbl: Vec::new(),
            link: Vec::new(),
            len: Vec::new(),
            firstpos: Vec::new(),
            primary: Vec::new(),
            last: 0,
            off: Vec::new(),
            child_head: Vec::new(),
            child_arr: Vec::new(),
            lastpos: Vec::new(),
        }
    }

    /// Clear the arenas (keeping capacity) and re-seed the root — for buffer reuse.
    fn reset(&mut self, cap: usize) {
        self.inl.clear();
        self.deg.clear();
        self.edge_char.clear();
        self.edge_to.clear();
        self.edge_next.clear();
        self.head.clear();
        self.root_tbl.clear();
        self.root_tbl.resize(ROOT_TBL, -1);
        self.link.clear();
        self.len.clear();
        self.firstpos.clear();
        self.primary.clear();
        self.inl.reserve(2 * cap + 1);
        self.deg.reserve(2 * cap + 1);
        self.head.reserve(2 * cap + 1);
        self.link.reserve(2 * cap + 1);
        self.len.reserve(2 * cap + 1);
        self.firstpos.reserve(2 * cap + 1);
        self.primary.reserve(2 * cap + 1);
        self.new_state(0, -1, 0, false); // root state 0 is not a per-position state
        self.last = 0;
    }

    #[allow(clippy::cast_sign_loss)]
    fn find(&self, state: u32, key: u32) -> Option<u32> {
        if state == 0 && (key as usize) < ROOT_TBL {
            let t = self.root_tbl[key as usize];
            return (t >= 0).then_some(t as u32);
        }
        let e = &self.inl[state as usize];
        // empty slots hold `NONE`, which no key equals
        if e[0] == key {
            return Some(e[1]);
        }
        if e[2] == key {
            return Some(e[3]);
        }
        if e[4] == key {
            return Some(e[5]);
        }
        if e[6] == key {
            return Some(e[7]);
        }
        if self.deg[state as usize] as usize > INLINE {
            let mut x = self.head[state as usize];
            while x != -1 {
                let idx = x as usize;
                if self.edge_char[idx] == key {
                    return Some(self.edge_to[idx]);
                }
                x = self.edge_next[idx];
            }
        }
        None
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn add_edge(&mut self, state: u32, key: u32, to: u32) {
        let s = state as usize;
        let d = self.deg[s] as usize;
        if d < INLINE {
            self.inl[s][2 * d] = key;
            self.inl[s][2 * d + 1] = to;
        } else {
            self.edge_char.push(key);
            self.edge_to.push(to);
            self.edge_next.push(self.head[s]);
            self.head[s] = (self.edge_char.len() - 1) as i32;
        }
        self.deg[s] = (d + 1) as u32;
        if state == 0 && (key as usize) < ROOT_TBL {
            self.root_tbl[key as usize] = to as i32;
        }
    }

    /// Redirect the existing transition `state --key-->` to `to` (the clone step).
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    fn set_edge(&mut self, state: u32, key: u32, to: u32) {
        let s = state as usize;
        if state == 0 && (key as usize) < ROOT_TBL {
            self.root_tbl[key as usize] = to as i32;
        }
        let e = &mut self.inl[s];
        for k in 0..INLINE {
            if e[2 * k] == key {
                e[2 * k + 1] = to;
                return;
            }
        }
        let mut x = self.head[s];
        while x != -1 {
            let idx = x as usize;
            if self.edge_char[idx] == key {
                self.edge_to[idx] = to;
                return;
            }
            x = self.edge_next[idx];
        }
        self.add_edge(state, key, to);
    }

    #[allow(clippy::cast_possible_truncation)]
    fn new_state(&mut self, len: u32, link: i32, firstpos: u32, primary: bool) -> u32 {
        self.inl.push([NONE; 2 * INLINE]);
        self.deg.push(0);
        self.head.push(-1);
        self.len.push(len);
        self.link.push(link);
        self.firstpos.push(firstpos);
        self.primary.push(primary);
        (self.head.len() - 1) as u32
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn extend(&mut self, c: char, pos: usize) {
        let key = c as u32;
        let cur = self.new_state(self.len[self.last as usize] + 1, -1, pos as u32, true);
        let mut p = self.last as i32;
        while p != -1 && self.find(p as u32, key).is_none() {
            self.add_edge(p as u32, key, cur);
            p = self.link[p as usize];
        }
        if p == -1 {
            self.link[cur as usize] = 0;
        } else {
            let q = self.find(p as u32, key).unwrap();
            if self.len[p as usize] + 1 == self.len[q as usize] {
                self.link[cur as usize] = q as i32;
            } else {
                let clone = self.new_state(self.len[p as usize] + 1, self.link[q as usize], self.firstpos[q as usize], false);
                // the clone starts with q's transitions
                let qi = q as usize;
                let inline = self.inl[qi];
                for k in 0..(self.deg[qi] as usize).min(INLINE) {
                    self.add_edge(clone, inline[2 * k], inline[2 * k + 1]);
                }
                let mut x = self.head[qi];
                while x != -1 {
                    let idx = x as usize;
                    self.add_edge(clone, self.edge_char[idx], self.edge_to[idx]);
                    x = self.edge_next[idx];
                }
                while p != -1 && self.find(p as u32, key) == Some(q) {
                    self.set_edge(p as u32, key, clone);
                    p = self.link[p as usize];
                }
                self.link[q as usize] = clone as i32;
                self.link[cur as usize] = clone as i32;
            }
        }
        self.last = cur;
    }

    /// Convert the builder's transitions into the packed `edges` layout (per-state edges sorted
    /// by char, co-located char+target) and derive every query table, into a fresh `Sam`. `b` is
    /// the string the automaton was built from — the per-position states' inline transition is
    /// the one that continues along it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::needless_range_loop)]
    fn finalize(&mut self, b: &[char]) -> Sam {
        let mut out = Sam::empty();
        let nstates = self.head.len();
        // per-state edge counts → exclusive prefix-sum offsets
        let off = &mut self.off;
        off.clear();
        off.resize(nstates + 1, 0);
        for state in 0..nstates {
            off[state + 1] = off[state] + self.deg[state];
        }
        let nedges = off[nstates] as usize;
        // edges: (char << 32 | to), sorted by char within each state's [off[s], off[s+1]) range.
        // sorting the packed u64 sorts by char (high bits) since a state's chars are distinct.
        // Nearly every state has one or two transitions: packed in place and insertion-sorted.
        out.edges.resize(nedges, 0);
        self.pack_edges(&mut out, nstates);
        self.tables(&mut out, nstates, b);
        self.build_endpos(&mut out, nstates, b.len());
        out
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::needless_range_loop)]
    fn pack_edges(&mut self, out: &mut Sam, nstates: usize) {
        let off = self.off.as_slice();
        let (inl, deg, head, edge_char, edge_to, edge_next) = (
            self.inl.as_slice(),
            self.deg.as_slice(),
            self.head.as_slice(),
            self.edge_char.as_slice(),
            self.edge_to.as_slice(),
            self.edge_next.as_slice(),
        );
        let edges = out.edges.as_mut_slice();
        for state in 0..nstates {
            let base = off[state] as usize;
            let mut k = base;
            let mut place = |packed: u64, k: &mut usize| {
                // insertion into the sorted prefix [base, *k)
                let mut p = *k;
                while p > base && edges[p - 1] > packed {
                    edges[p] = edges[p - 1];
                    p -= 1;
                }
                edges[p] = packed;
                *k += 1;
            };
            let e = &inl[state];
            for slot in 0..(deg[state] as usize).min(INLINE) {
                place((u64::from(e[2 * slot]) << 32) | u64::from(e[2 * slot + 1]), &mut k);
            }
            let mut x = head[state];
            while x != -1 {
                let idx = x as usize;
                place((u64::from(edge_char[idx]) << 32) | u64::from(edge_to[idx]), &mut k);
                x = edge_next[idx];
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::needless_range_loop)]
    fn tables(&mut self, out: &mut Sam, nstates: usize, b: &[char]) {
        let off = self.off.as_slice();
        let edges = out.edges.as_slice();
        let (primary, firstpos, link_of, len_of) =
            (self.primary.as_slice(), self.firstpos.as_slice(), self.link.as_slice(), self.len.as_slice());
        // root direct transition table (ASCII): fill from the root's edges
        out.root_next.resize(ROOT_TBL, -1);
        for &e in &edges[off[0] as usize..off[1] as usize] {
            let ci = (e >> 32) as usize;
            if ci < ROOT_TBL {
                out.root_next[ci] = i32::try_from((e & 0xFFFF_FFFF) as u32).expect("state fits i32");
            }
        }
        // the scan slot: two inline transitions, the edge range, the suffix link and its length
        out.fast.reserve(nstates);
        for s in 0..nstates {
            let (lo, hi) = (off[s] as usize, off[s + 1] as usize);
            let es = &edges[lo..hi];
            let split = |e: u64| ((e >> 32) as u32, (e & 0xFFFF_FFFF) as u32);
            // e0: the continuation along `b` for a per-position state, else the first edge
            let mut e0 = es.first().map_or((NONE, 0), |&e| split(e));
            if primary[s] {
                let next = firstpos[s] as usize + 1;
                if next < b.len() {
                    let t = csr_lookup(edges, lo, hi, b[next]);
                    if t >= 0 {
                        e0 = (b[next] as u32, t as u32);
                    }
                }
            }
            // e1: the first edge in char order that is not e0
            let e1 = es.iter().map(|&e| split(e)).find(|&(c, _)| c != e0.0).unwrap_or((NONE, 0));
            let link_idx_signed = link_of[s];
            let link = if link_idx_signed < 0 { 0 } else { link_idx_signed as u32 };
            let llen = if link_idx_signed < 0 { 0 } else { len_of[link_idx_signed as usize] };
            out.fast.push(ScanSlot([e0.0, e0.1, e1.0, e1.1, lo as u32, hi as u32, link, llen]));
        }
        out.pos_state.resize(b.len(), 0);
        for s in 0..nstates {
            if primary[s] {
                out.pos_state[firstpos[s] as usize] = s as u32;
            }
        }
    }

    /// Build the endpos range structure (fix b): lay the end-positions out so every state's endpos
    /// set is one contiguous slice `[dfs_in, dfs_in+dfs_cnt)` of `epos` (its subtree in the
    /// suffix-link tree), fill the chain slot per state, and keep a sorted copy for the few states
    /// holding more than `LINEAR_MAX` positions.
    ///
    /// No tree walk: a state's suffix link is always shorter than the state, so the states in
    /// length order are a topological order of the link tree. One pass in decreasing length
    /// accumulates subtree sizes and the largest position of each subtree; one pass in increasing
    /// length hands each state a range inside its parent's, its own position first, its children's
    /// ranges after. Any layout in which subtrees are contiguous serves the queries: a slice is
    /// read as a set.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::needless_range_loop)]
    fn build_endpos(&mut self, out: &mut Sam, nstates: usize, blen: usize) {
        let (len_of, link_of, primary, firstpos) =
            (self.len.as_slice(), self.link.as_slice(), self.primary.as_slice(), self.firstpos.as_slice());
        // states in length order (counting sort: len <= blen)
        self.child_head.clear();
        self.child_head.resize(blen + 2, 0);
        let start = self.child_head.as_mut_slice();
        for s in 0..nstates {
            start[len_of[s] as usize + 1] += 1;
        }
        for l in 0..=blen {
            start[l + 1] += start[l];
        }
        self.child_arr.clear();
        self.child_arr.resize(nstates, 0);
        let by_len = self.child_arr.as_mut_slice();
        self.off.clear();
        self.off.extend_from_slice(&start[..=blen]);
        let cursor = self.off.as_mut_slice();
        for s in 0..nstates {
            let l = len_of[s] as usize;
            by_len[cursor[l] as usize] = s as u32;
            cursor[l] += 1;
        }
        // decreasing length: subtree size (`cnt`) and largest position (`lastpos`) flow to the link
        self.lastpos.clear();
        self.lastpos.resize(nstates, 0);
        let cnt = self.lastpos.as_mut_slice(); // scratch: subtree sizes
        out.chain_slot.clear();
        out.chain_slot.reserve(nstates);
        for s in 0..nstates {
            let f = out.fast[s].0;
            let own = if primary[s] { firstpos[s] } else { 0 };
            out.chain_slot.push([len_of[s], f[6], f[7], u32::MAX, firstpos[s], own, 0, 0]);
            cnt[s] = u32::from(primary[s]);
        }
        let slots = out.chain_slot.as_mut_slice();
        for &s in by_len[1..].iter().rev() {
            let s = s as usize;
            let p = link_of[s] as usize;
            cnt[p] += cnt[s];
            let lp = slots[s][5];
            if lp > slots[p][5] {
                slots[p][5] = lp;
            }
        }
        // increasing length: ranges, parents before children; `cursor[s]` = next free slot in s's range
        self.off.clear();
        self.off.resize(nstates, 0);
        let cursor = self.off.as_mut_slice();
        out.epos.clear();
        out.epos.resize(blen, 0);
        let epos = out.epos.as_mut_slice();
        slots[0][6] = 0;
        slots[0][7] = cnt[0];
        cursor[0] = 0;
        for &s in by_len.iter() {
            let s = s as usize;
            if s != 0 {
                let p = link_of[s] as usize;
                let at = cursor[p];
                cursor[p] += cnt[s];
                slots[s][6] = at;
                slots[s][7] = cnt[s];
                cursor[s] = at;
            }
            if primary[s] {
                epos[cursor[s] as usize] = firstpos[s];
                cursor[s] += 1;
            }
        }
        self.sort_big(out, nstates, blen);
    }

    #[allow(clippy::cast_possible_truncation)]
    fn sort_big(&mut self, out: &mut Sam, nstates: usize, blen: usize) {
        // sorted copies for the big endpos sets (a handful of states: short, frequent substrings);
        // positions are below `blen`, so a radix sort needs one pass per byte of that
        let passes = ((usize::BITS - blen.leading_zeros()) as usize).div_ceil(8).max(1);
        let scratch = &mut self.child_arr;
        for s in 0..nstates {
            let cnt = out.chain_slot[s][7] as usize;
            if cnt > LINEAR_MAX {
                let start = out.chain_slot[s][6] as usize;
                let so = out.big_sorted.len();
                out.big_sorted.extend_from_slice(&out.epos[start..start + cnt]);
                radix_sort_u32(&mut out.big_sorted[so..], scratch, passes);
                out.chain_slot[s][3] = so as u32;
            }
        }
    }
}

/// LSD radix sort of `v` (byte digits, `passes` low bytes significant), `scratch` as the buffer.
fn radix_sort_u32(v: &mut [u32], scratch: &mut Vec<u32>, passes: usize) {
    let n = v.len();
    scratch.clear();
    scratch.resize(n, 0);
    let mut counts = [0usize; 256];
    for pass in 0..passes {
        let shift = 8 * pass;
        counts.fill(0);
        for &x in v.iter() {
            counts[((x >> shift) & 0xFF) as usize] += 1;
        }
        let mut sum = 0;
        for c in &mut counts {
            let k = *c;
            *c = sum;
            sum += k;
        }
        for &x in v.iter() {
            let d = ((x >> shift) & 0xFF) as usize;
            scratch[counts[d]] = x;
            counts[d] += 1;
        }
        v.copy_from_slice(&scratch[..n]);
    }
}

/// Build (and finalize) the suffix automaton of `b` — prebuild once, reuse across pairs.
#[must_use]
pub fn build_sam(b: &[char]) -> Sam {
    BUILDER.with_borrow_mut(|bld| {
        bld.reset(b.len());
        for (i, &c) in b.iter().enumerate() {
            bld.extend(c, i);
        }
        bld.finalize(b)
    })
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
///
/// `wlen[i]` is the caller's running upper bound on the in-window match ending at `i`: the
/// windows of a Ratcliff–Obershelp recursion only ever shrink, so whatever this walk learns about
/// position `i` — the exact longest match inside this window, or that nothing longer than the
/// length it was pruned at exists — bounds every descendant window too, and the descendant's walk
/// for `i` is pruned by it up front. The precomputed `fmatch[i]` is the longest match ending at
/// `i` *anywhere* in `b`; on real code that is almost always long, so without this refinement a
/// narrow window re-walks the chain for every position, level after level.
///
/// A window whose `b` side is at most [`DIRECT_B_MAX`] wide goes to [`longest_direct`] instead:
/// half the recursion's windows are that narrow on real code, and there the chain walk climbs
/// far for every `a` position (the precomputed matches lie outside the window) and mostly finds
/// nothing, while a direct row-by-row comparison costs a handful of vector ops per position.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::too_many_arguments, clippy::too_many_lines)]
fn longest_in(
    a: &[char],
    b: &[char],
    sam: &Sam,
    fstate: &[u32],
    fmatch: &[u32],
    wlen: &mut [u32],
    al: usize,
    ar: usize,
    bl: usize,
    br: usize,
    chain_cap: u32,
) -> (usize, usize, usize) {
    #[cfg(feature = "instrument")]
    instr_inc(&instrument::LONGEST_IN_CALLS, 1);
    if br - bl <= DIRECT_B_MAX {
        return longest_direct(a, b, al, ar, bl, br);
    }
    let blo = bl as u32;
    let hi = br as u32 - 1; // caller guarantees bl < br, so br >= 1
    let (mut best_len, mut best_a, mut best_b) = (0usize, 0usize, 0usize);
    #[cfg(feature = "instrument")]
    let (mut n_pos, mut n_walks, mut n_steps) = (0u64, 0u64, 0u64);
    // SAFETY: i ∈ [al, ar) ⊆ [0, n) = fstate.len() = fmatch.len(); `cur` is always a valid SAM
    // state (fstate entry or a suffix link), so chain_slot[cur] is in bounds (= nstates entries).
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        // A match ending at `i+k` is at most `k` longer than one ending at `i` (drop its last `k`
        // characters), and each of the three bounds below grows by at most one per position. So
        // once position `i` is bounded by `eff <= best_len`, positions `i+1 ..= i+(best_len-eff)`
        // cannot beat the best either and are jumped over: on similar strings a long block early
        // in the window turns the walk over the rest of it into a few strides.
        let mut i = al;
        while i < ar {
            let cap = i - al + 1; // a-match ending at i can't start before `al`
            let known = *wlen.get_unchecked(i) as usize; // what enclosing windows established
            let eff = (*fmatch.get_unchecked(i) as usize).min(cap).min(known);
            #[cfg(feature = "instrument")]
            {
                n_pos += 1;
            }
            if eff <= best_len {
                i += 1 + (best_len - eff); // can't beat the best — skip (dominant pruning)
                continue;
            }
            #[cfg(feature = "instrument")]
            {
                n_walks += 1;
            }
            // what this walk establishes for `i`: the exact in-window match when it finds one, else
            // the length it stopped at (nothing longer exists in this window, hence in any child)
            let mut learned = 0usize;
            // walk the suffix-link chain from the precomputed state up; `curlen` is the usable length
            // at the current state (capped to the a-fragment), shrinking as we ascend.
            let mut cur = *fstate.get_unchecked(i);
            // Optimization K1: `cur != 0` is an invariant here. `cur == 0` (the SAM root) is only
            // stored in `fstate[i]` when `fmatch[i] == 0`, and in that case `eff == 0 <= best_len`
            // already pruned this iteration via the `continue` above. From the second chain step
            // on, `cur` came from `link != 0` (we check before assigning). Hoisting the
            // entry-time `if cur == 0 { break }` out saves ≈4 k arm64 instructions per pair on
            // the bench corpus (-0.9 % retired) — no cycle change on this single-thread profile
            // because the chain walk is latency-bound by the `chain_slot[cur] → link →
            // chain_slot[link]` pointer chase, but it frees front-end issue bandwidth.
            let mut chain_depth: u32 = 0;
            loop {
                chain_depth += 1;
                #[cfg(feature = "instrument")]
                {
                    n_steps += 1;
                }
                // Phase 3 approximate-RO cap: stop walking the chain past `chain_cap` ascended
                // states. With cap=u32::MAX (default delta=0) this never fires; smaller caps
                // trade tail accuracy for fewer pointer chases. Measurement on canonical Python:
                // p95 chain depth = 5, p99 = 7 — capping at 5/7 truncates <5%/<1% of chains.
                if chain_depth > chain_cap {
                    learned = known; // approximate walk: learn nothing
                    break;
                }
                // One 32-byte load of `chain_slot[cur]` brings the per-state hot fields into
                // registers in ONE cache-line touch:
                //   [0] = len   [1] = link   [2] = link_len   [3] = sorted_off
                //   [4..8] = firstpos, lastpos, dfs_in, dfs_cnt for max_le/min_in
                // (These were three scattered loads on three cache lines; PMU attribution had
                // L1D misses at 18 % of the cycle budget split between them.)
                let cs = *sam.chain_slot.get_unchecked(cur as usize);
                let curlen = eff.min(cs[0] as usize);
                if curlen <= best_len {
                    learned = curlen;
                    break;
                }
                let band_min = cs[2] as usize + 1;
                if curlen >= band_min {
                    // The endpos metadata is already in `cs` — no extra load needed.
                    if let Some(pmax) = sam.max_le_slot(&cs, hi) {
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
                                        sam.min_in_slot(&cs, blo + l as u32 - 1, hi)
                                    };
                                    if let Some(pmin) = pmin {
                                        best_len = l;
                                        best_a = i + 1 - l;
                                        best_b = pmin as usize + 1 - l;
                                    }
                                }
                                learned = l;
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
            if learned < known {
                *wlen.get_unchecked_mut(i) = learned as u32;
            }
            #[cfg(feature = "instrument")]
            instr_hist(&instrument::CHAIN_DEPTHS, chain_depth as usize);
            // `learned` bounds this position's match, so the same stride applies from here
            i += 1 + best_len.saturating_sub(learned);
        }
    }
    #[cfg(feature = "instrument")]
    {
        let w = (ar - al).min(br - bl);
        let k = (usize::BITS - w.leading_zeros()) as usize;
        let k = k.min(23);
        instr_inc(&instrument::WIN_CALLS[k], 1);
        instr_inc(&instrument::WIN_POS[k], n_pos);
        instr_inc(&instrument::WIN_WALKS[k], n_walks);
        instr_inc(&instrument::WIN_STEPS[k], n_steps);
        if best_len == 0 {
            instr_inc(&instrument::WIN_ZERO[k], 1);
        }
        instrument::TL_STEPS.with(|c| c.set(c.get() + n_steps));
        instrument::TL_CALLS.with(|c| c.set(c.get() + 1));
    }
    (best_a, best_b, best_len)
}

/// Widest `b` window [`longest_direct`] handles; wider ones walk the automaton. Measured on the
/// name-gated workload: 4 and 8 are within noise of each other, 16 is slower, 64 is slower than
/// no direct path at all — the row cost grows with the width while the walk's does not.
const DIRECT_B_MAX: usize = 8;

/// Longest common substring of `a[al..ar]` and `b[bl..br)` by direct comparison, for a narrow
/// `b` window: difflib's own `find_longest_match` recurrence (`k[i][j] = k[i-1][j-1] + 1` on
/// equal characters) over two rows of at most `DIRECT_B_MAX + 1` cells. The inner loop has no
/// loop-carried dependency, so it vectorizes; the row's maximum is located only when it beats the
/// best so far. Same result and tie-break as the chain walk: longest, then earliest in `a`, then
/// earliest in `b` — rows go up in `i`, cells up in `j`, and only a strictly longer match replaces
/// the best.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
fn longest_direct(a: &[char], b: &[char], al: usize, ar: usize, bl: usize, br: usize) -> (usize, usize, usize) {
    let wb = br - bl;
    let bw: Vec<u32> = b[bl..br].iter().map(|&c| c as u32).collect();
    let mut rows = [[0u32; DIRECT_B_MAX + 1]; 2];
    let (mut best_len, mut best_a, mut best_b) = (0usize, 0usize, 0usize);
    for (r, i) in (al..ar).enumerate() {
        let (lo, hi) = rows.split_at_mut(1);
        let (prev, cur) = if r & 1 == 0 { (&lo[0], &mut hi[0]) } else { (&hi[0], &mut lo[0]) };
        let ai = a[i] as u32;
        let mut rowmax = 0u32;
        for ((c, &p), &bj) in cur[1..=wb].iter_mut().zip(&prev[..wb]).zip(&bw) {
            let k = if ai == bj { p + 1 } else { 0 };
            *c = k;
            rowmax = rowmax.max(k);
        }
        if rowmax as usize > best_len {
            let j = cur[1..=wb].iter().position(|&k| k == rowmax).unwrap_or(0);
            best_len = rowmax as usize;
            best_a = i + 1 - best_len;
            best_b = bl + j + 1 - best_len;
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

/// Fill the matching statistics of `a`'s common prefix with `b` (the string `sam_b` was built
/// from): position `i` of the prefix is matched in full by `b`'s own prefix, so its state is the
/// one created for position `i` and its match length `i + 1`. Returns the prefix length.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn prefix_fill(a: &[char], sam_b: &Sam, fstate: &mut [u32], fmatch: &mut [u32]) -> usize {
    let b = sam_b.pos_state.as_slice();
    let n = a.len().min(b.len());
    let mut pre = 0;
    // `b`'s characters are not stored; the state's first inline transition continues along `b`
    // and its target is the next position's state, which is what the comparison needs
    let fast = sam_b.fast.as_slice();
    let mut st = 0u32;
    while pre < n {
        let f = fast[st as usize].0;
        let key = a[pre] as u32;
        let next = if st == 0 {
            let ci = key as usize;
            if ci < ROOT_TBL && sam_b.root_next[ci] >= 0 { sam_b.root_next[ci] as u32 } else { break }
        } else if f[0] == key && f[1] == b[pre] {
            f[1]
        } else {
            break;
        };
        if next != b[pre] {
            break;
        }
        fstate[pre] = next;
        fmatch[pre] = pre as u32 + 1;
        st = next;
        pre += 1;
    }
    pre
}

/// Window-independent matching statistics of `a` vs `sam_b`, filled into reused buffers: for each
/// i, `(state, matched)` where `matched` = longest suffix of `a[..=i]` occurring anywhere in b.
/// One O(|a|) scan, reused by every recursion node (no per-node re-scan). Reusing the caller's
/// buffers avoids a per-pair allocation (was ~10% of the all-pairs join).
///
/// Per character the scan first tries the state's inline transition (`Sam::fast`): on similar
/// strings a match extends for hundreds of characters, and each of those is then one 16-byte
/// load instead of a node load plus a binary search over the edge slice.
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
    let edges = sam_b.edges.as_slice();
    let root = sam_b.root_next.as_slice();
    let fast = sam_b.fast.as_slice();
    // A common prefix of `a` and `b` is walked by the automaton along `b`'s own states with the
    // match growing by one per character; those entries are written directly (same-named code
    // shares long prefixes) and the walk starts where the prefix ends.
    let pre = prefix_fill(a, sam_b, fstate, fmatch);
    let mut state = if pre == 0 { 0 } else { sam_b.pos_state[pre - 1] };
    let mut matched = pre as u32;
    // SAFETY: `state` is always a valid SAM state index (< nstates = fast.len()): it starts at
    // the root (0) or a position's state and only ever becomes a transition target (an inline
    // target or an edge's low bits, both valid states) or a suffix link (`fast[..][6]`, a valid
    // state). So fast[state] is in bounds; the edge range [edge_lo, edge_hi) ⊆ [0, edges.len());
    // `ci < ROOT_TBL == root.len()`; `i < n == fstate.len()`.
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        for i in pre..n {
            let c = *a.get_unchecked(i);
            let key = c as u32;
            loop {
                if state == 0 {
                    let ci = c as usize;
                    let nx: i64 = if ci < ROOT_TBL {
                        i64::from(*root.get_unchecked(ci))
                    } else {
                        let f = fast.get_unchecked(0).0;
                        csr_lookup(edges, f[4] as usize, f[5] as usize, c)
                    };
                    if nx >= 0 {
                        state = nx as u32;
                        matched += 1;
                    } else {
                        matched = 0;
                    }
                    break;
                }
                let f = fast.get_unchecked(state as usize).0;
                if f[0] == key {
                    state = f[1];
                    matched += 1;
                    break;
                }
                if f[2] == key {
                    state = f[3];
                    matched += 1;
                    break;
                }
                if f[5] - f[4] > 2 {
                    let nx = csr_lookup(edges, f[4] as usize, f[5] as usize, c);
                    if nx >= 0 {
                        state = nx as u32;
                        matched += 1;
                        break;
                    }
                }
                state = f[6]; // suffix link
                matched = f[7]; // len(link)
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
    if hi - lo <= 8 {
        // a short range: one predictable loop beats a binary search's data-dependent branches
        for k in lo..hi {
            // SAFETY: k ∈ [lo, hi) ⊆ [0, edges.len()] (callers pass a state's edge range).
            let e = unsafe { *edges.get_unchecked(k) };
            if (e >> 32) as u32 == key {
                return i64::from((e & 0xFFFF_FFFF) as u32);
            }
        }
        return -1;
    }
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

/// A pending recursion window, ordered by the size of its smaller side (a max-heap key).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Window {
    key: usize,
    al: usize,
    ar: usize,
    bl: usize,
    br: usize,
}

impl Window {
    fn new(al: usize, ar: usize, bl: usize, br: usize) -> Self {
        Self { key: (ar - al).min(br - bl), al, ar, bl, br }
    }
}

impl Ord for Window {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key).then_with(|| other.al.cmp(&self.al))
    }
}

impl PartialOrd for Window {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

thread_local! {
    /// Reused window heap for the early-exit recursions (largest window first).
    static HEAP_BUF: std::cell::RefCell<std::collections::BinaryHeap<Window>> =
        const { std::cell::RefCell::new(std::collections::BinaryHeap::new()) };
    /// Reused per-position window bound for `longest_in` (see its `wlen` parameter).
    static WLEN_BUF: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with a fresh (all-`u32::MAX`) window bound for a string of `na` positions.
fn with_wlen<R>(na: usize, f: impl FnOnce(&mut [u32]) -> R) -> R {
    WLEN_BUF.with_borrow_mut(|w| {
        w.clear();
        w.resize(na, u32::MAX);
        f(w)
    })
}

/// What a bounded scan concluded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scan {
    /// The buffers hold the complete matching statistics.
    Full,
    /// Some position's match reached `accept_at`: that common substring alone bounds M from below.
    Accept,
}

/// `matching_stats_into` with an **exact** early exit decided while `a` is still being walked:
/// `Accept` the moment `fmatch[i] >= accept_at`. Ratcliff–Obershelp's first block is the longest
/// common substring, so M is at least any common substring's length — a match of `accept_at`
/// characters proves `M >= accept_at` before a single recursion step. On `Full` the buffers hold
/// the complete statistics, bit-identical to `matching_stats_into`.
///
/// (A reject exit from the non-overlapping-interval bound on `fmatch` was tried here and never
/// fired on real code: dissimilar functions still share long substrings, so every position's
/// `fmatch` is long and the bound stays near `|a|`; what makes their ratio low is the order the
/// blocks must come in, which only the recursion sees.)
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn matching_stats_bounded(
    a: &[char],
    sam_b: &Sam,
    fstate: &mut Vec<u32>,
    fmatch: &mut Vec<u32>,
    accept_at: u32,
) -> Scan {
    let n = a.len();
    fstate.clear();
    fstate.reserve(n);
    fmatch.clear();
    fmatch.reserve(n);
    // SAFETY: u32 has no invalid bit patterns; every index [0, n) of both buffers is written by
    // the loop below before it is read.
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        fstate.set_len(n);
        fmatch.set_len(n);
    }
    let edges = sam_b.edges.as_slice();
    let root = sam_b.root_next.as_slice();
    let fast = sam_b.fast.as_slice();
    let pre = prefix_fill(a, sam_b, fstate, fmatch);
    if pre as u32 >= accept_at {
        return Scan::Accept;
    }
    let mut state = if pre == 0 { 0 } else { sam_b.pos_state[pre - 1] };
    let mut matched = pre as u32;
    // SAFETY: as in `matching_stats_into` — `state` is always a valid state index, edge ranges
    // are within `edges`, `ci < ROOT_TBL`, `i < n`.
    #[allow(clippy::undocumented_unsafe_blocks)]
    unsafe {
        for i in pre..n {
            let c = *a.get_unchecked(i);
            let key = c as u32;
            loop {
                if state == 0 {
                    let ci = c as usize;
                    let nx: i64 = if ci < ROOT_TBL {
                        i64::from(*root.get_unchecked(ci))
                    } else {
                        let f = fast.get_unchecked(0).0;
                        csr_lookup(edges, f[4] as usize, f[5] as usize, c)
                    };
                    if nx >= 0 {
                        state = nx as u32;
                        matched += 1;
                    } else {
                        matched = 0;
                    }
                    break;
                }
                let f = fast.get_unchecked(state as usize).0;
                if f[0] == key {
                    state = f[1];
                    matched += 1;
                    break;
                }
                if f[2] == key {
                    state = f[3];
                    matched += 1;
                    break;
                }
                if f[5] - f[4] > 2 {
                    let nx = csr_lookup(edges, f[4] as usize, f[5] as usize, c);
                    if nx >= 0 {
                        state = nx as u32;
                        matched += 1;
                        break;
                    }
                }
                state = f[6]; // suffix link
                matched = f[7]; // len(link)
            }
            *fstate.get_unchecked_mut(i) = state;
            *fmatch.get_unchecked_mut(i) = matched;
            if matched >= accept_at {
                return Scan::Accept;
            }
        }
    }
    Scan::Full
}

/// Length of the longest common prefix of `a` and `b` plus that of their longest common suffix
/// (the two never overlap when the sum is capped at `min(|a|, |b|)`): both are common substrings,
/// so Ratcliff–Obershelp's first block — the longest common substring — is at least the larger of
/// them, and `M` is at least that. Cheap (two straight compares) and on same-named code often
/// enough to decide a pair before the scan.
fn common_ends(a: &[char], b: &[char]) -> usize {
    let n = a.len().min(b.len());
    let pre = a.iter().zip(b).take(n).take_while(|(x, y)| x == y).count();
    if pre == n {
        return n;
    }
    let suf = a.iter().rev().zip(b.iter().rev()).take(n - pre).take_while(|(x, y)| x == y).count();
    pre.max(suf)
}

/// Exact edge test `RO(a,b) >= threshold`: the bounded scan accepts a pair with a single block as
/// long as `need` before any recursion, the rest run the two-sided early-exit recursion. No ratio
/// comes out — the cluster minimum pass computes the exact values it needs under its own cap.
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#[must_use]
pub fn gestalt_edge_bounded(a: &[char], b: &[char], sam_b: &Sam, threshold: f64) -> bool {
    let na = a.len();
    let nb = b.len();
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
    if common_ends(a, b) >= need {
        return true;
    }
    let need32 = u32::try_from(need).unwrap_or(u32::MAX);
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        let r = match matching_stats_bounded(a, sam_b, fstate, fmatch, need32) {
            Scan::Accept => true,
            Scan::Full => gestalt_qualifies_ms(a, b, sam_b, threshold, fstate, fmatch),
        };
        #[cfg(feature = "instrument")]
        instrument::pair_done(1);
        r
    })
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
        with_wlen(n, |wlen| {
            stack.clear();
            stack.push((0, n, 0, b.len()));
            while let Some((al, ar, bl, br)) = stack.pop() {
                if al >= ar || bl >= br {
                    continue;
                }
                let (i, j, l) = longest_in(a, b, sam_b, fstate, fmatch, wlen, al, ar, bl, br, u32::MAX);
                if l == 0 {
                    continue;
                }
                total += l;
                stack.push((al, i, bl, j));
                stack.push((i + l, ar, j + l, br));
            }
        });
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
        gestalt_qualifies_ms(a, b, sam_b, threshold, fstate, fmatch)
    })
}

/// The threshold early-exit recursion given **precomputed** matching statistics (`fstate`/`fmatch`
/// for `a` vs `sam_b`, the automaton of `b`). Split out so the scan can be done separately.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]
#[must_use]
pub fn gestalt_qualifies_ms(a: &[char], b: &[char], sam_b: &Sam, threshold: f64, fstate: &[u32], fmatch: &[u32]) -> bool {
    let na = a.len();
    let nb = b.len();
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
    HEAP_BUF.with_borrow_mut(|heap| {
        with_wlen(na, |wlen| {
            let mut m = 0usize;
            let mut pending = na.min(nb);
            heap.clear();
            heap.push(Window::new(0, na, 0, nb));
            // largest window first: its block removes the most from the bound, so both exits
            // — `m >= need` and `m + pending < need` — are reached in fewer windows
            while let Some(w) = heap.pop() {
                let (al, ar, bl, br) = (w.al, w.ar, w.bl, w.br);
                pending -= (ar - al).min(br - bl);
                if al >= ar || bl >= br {
                    continue;
                }
                let (i, j, l) = longest_in(a, b, sam_b, fstate, fmatch, wlen, al, ar, bl, br, u32::MAX);
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
                heap.push(Window::new(al, i, bl, j));
                heap.push(Window::new(i + l, ar, j + l, br));
            }
            m >= need
        })
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
        let r = gestalt_edge_with_ms(a, b, sam_b, fstate, fmatch, threshold);
        #[cfg(feature = "instrument")]
        instrument::pair_done(0);
        r
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
                sum += u64::from(f);
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
        with_wlen(na, |wlen| {
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
                let (i, j, l) = longest_in(a, b, sam_b, fstate, fmatch, wlen, al, ar, bl, br, chain_cap);
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
    if common_ends(a, b) >= exceed {
        return 2.0;
    }
    MS_BUF.with_borrow_mut(|(fstate, fmatch)| {
        // A single common substring of `exceed` characters already proves ratio > cap.
        let accept_at = u32::try_from(exceed).unwrap_or(u32::MAX);
        let scan = matching_stats_bounded(a, sam_b, fstate, fmatch, accept_at);
        let r = if scan == Scan::Accept { 2.0 } else { capped_recursion(a, b, sam_b, fstate, fmatch, exceed) };
        #[cfg(feature = "instrument")]
        instrument::pair_done(2);
        r
    })
}

/// The recursion of [`gestalt_ratio_capped`] over precomputed matching statistics: the exact
/// ratio if `M < exceed`, else `2.0` the instant the running total reaches `exceed`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn capped_recursion(a: &[char], b: &[char], sam_b: &Sam, fstate: &[u32], fmatch: &[u32], exceed: usize) -> f64 {
    let na = a.len();
    let nb = b.len();
    let total = na + nb;
    HEAP_BUF.with_borrow_mut(|heap| {
        with_wlen(na, |wlen| {
            let mut m = 0usize;
            heap.clear();
            heap.push(Window::new(0, na, 0, nb));
            // largest window first: the biggest blocks come earliest, so the prune fires sooner
            while let Some(w) = heap.pop() {
                let (al, ar, bl, br) = (w.al, w.ar, w.bl, w.br);
                if al >= ar || bl >= br {
                    continue;
                }
                let (i, j, l) = longest_in(a, b, sam_b, fstate, fmatch, wlen, al, ar, bl, br, u32::MAX);
                if l == 0 {
                    continue;
                }
                m += l;
                if m >= exceed {
                    return 2.0; // ratio > cap ⇒ cannot be the minimum; prune (any value > cap works)
                }
                heap.push(Window::new(al, i, bl, j));
                heap.push(Window::new(i + l, ar, j + l, br));
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
