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
//!
//! Operates on `char` (code points), so it is bit-identical to difflib on non-ASCII.

/// Size of the root's direct transition table — covers ASCII (the canonical text's alphabet).
const ROOT_TBL: usize = 128;


/// Suffix automaton with CSR (sorted-per-state) transitions — built once, queried by scans.
///
/// For the range-restricted recursion (fix b), each state also carries its **endpos** as a
/// contiguous slice `[dfs_in, dfs_in+dfs_cnt)` of `epos` (the end-positions in b, laid out by
/// a DFS of the suffix-link tree so a subtree is contiguous). A merge-sort tree over `epos`
/// answers "is there an end-position in [lo,hi] within this state's subtree, and the min/max
/// such" — so the whole RO recursion runs on this one prebuilt SAM, with **no sub-builds**.
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
    // DFS-ordered end-positions (each state's endpos = contiguous slice [dfs_in, dfs_in+dfs_cnt)).
    // For small endpos sets a cache-friendly linear scan of this beats the scattered tree descent.
    epos: Vec<u32>,
    // Query-hot per-state fields packed into one cache line `[firstpos, lastpos, dfs_in, dfs_cnt]`
    // so `max_le`/`min_in` load all four with a single cache miss (they were 4 separate arrays).
    epmeta: Vec<[u32; 4]>,
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
        }
    }

    /// Node `i`'s sorted endpos range in the flattened merge-sort tree.
    fn seg_node(&self, i: usize) -> &[u32] {
        &self.seg_data[self.seg_off[i] as usize..self.seg_off[i + 1] as usize]
    }

    /// `state`'s endpos slice as the merge-sort-tree leaf range [l, r), from packed `epmeta`.
    fn endpos_range_m(m: &[u32; 4], seg_n: usize) -> (usize, usize) {
        let s = m[2] as usize; // dfs_in
        (s + seg_n, s + m[3] as usize + seg_n) // dfs_cnt
    }

    /// Largest end-position `<= x` among `state`'s endpos, or None.
    fn max_le(&self, state: u32, x: u32) -> Option<u32> {
        // SAFETY: `state` is a valid SAM state (< nstates = epmeta.len()); the endpos slice
        // [dfs_in, dfs_in+dfs_cnt) is within epos (built that way). Hot path, called ~30M times.
        #[allow(clippy::undocumented_unsafe_blocks)]
        let m = unsafe { self.epmeta.get_unchecked(state as usize) }; // [firstpos,lastpos,dfs_in,dfs_cnt]
        // O(1) fast path: x doesn't split the state's [firstpos, lastpos] span.
        if m[1] <= x {
            return Some(m[1]); // whole set <= x → answer is the max (lastpos)
        }
        if m[0] > x {
            return None; // whole set > x (firstpos) → nothing <= x
        }
        let cnt = m[3] as usize;
        if cnt <= LINEAR_MAX {
            // small endpos set: branchless (vectorizable) scan of the contiguous slice. firstpos ≤ x
            // is guaranteed above, so a qualifying element exists and `best` is the true answer.
            let lo = m[2] as usize;
            let mut best = 0u32;
            #[allow(clippy::undocumented_unsafe_blocks)]
            for &v in unsafe { self.epos.get_unchecked(lo..lo + cnt) } {
                best = best.max(if v <= x { v } else { 0 });
            }
            return Some(best);
        }
        // large endpos set: query the merge-sort tree.
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

    /// Smallest end-position in `[lo, hi]` among `state`'s endpos, or None.
    fn min_in(&self, state: u32, lo: u32, hi: u32) -> Option<u32> {
        // SAFETY: as in `max_le` — `state` < nstates = epmeta.len(); endpos slice within epos.
        #[allow(clippy::undocumented_unsafe_blocks)]
        let m = unsafe { self.epmeta.get_unchecked(state as usize) };
        // O(1) fast path: the smallest end-position is already >= lo.
        if m[0] >= lo {
            return (m[0] <= hi).then_some(m[0]); // firstpos
        }
        if m[1] < lo {
            return None; // whole set < lo (lastpos) → nothing in [lo, hi]
        }
        let cnt = m[3] as usize;
        if cnt <= LINEAR_MAX {
            // small endpos set: branchless (vectorizable) scan for the smallest value in [lo, hi]
            let off = m[2] as usize;
            let mut best = u32::MAX;
            #[allow(clippy::undocumented_unsafe_blocks)]
            for &v in unsafe { self.epos.get_unchecked(off..off + cnt) } {
                best = best.min(if v >= lo && v <= hi { v } else { u32::MAX });
            }
            return (best != u32::MAX).then_some(best);
        }
        // lo is interior to the span: query the tree for the smallest value in [lo, hi].
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
        for s in 0..nstates {
            let link = if self.link[s] < 0 { 0 } else { self.link[s] as u32 };
            out.node.push([self.len[s], link, off[s], off[s + 1]]);
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
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn longest_in(sam: &Sam, fstate: &[u32], fmatch: &[u32], al: usize, ar: usize, bl: usize, br: usize)
    -> (usize, usize, usize) {
    let blo = bl as u32;
    let hi = br as u32 - 1; // caller guarantees bl < br, so br >= 1
    let (mut best_len, mut best_a, mut best_b) = (0usize, 0usize, 0usize);
    let node = sam.node.as_slice();
    // SAFETY: i ∈ [al, ar) ⊆ [0, n) = fstate.len() = fmatch.len(); `cur` is always a valid SAM
    // state (fstate entry or a suffix link), so node[cur] and node[link] (link is a state) are in bounds.
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
            loop {
                if cur == 0 {
                    break;
                }
                let nd = *node.get_unchecked(cur as usize); // [len, link, edge_lo, edge_hi]
                let curlen = eff.min(nd[0] as usize);
                if curlen <= best_len {
                    break;
                }
                let band_min = *node.get_unchecked(nd[1] as usize); // node[link]
                let band_min = band_min[0] as usize + 1; // len(link) + 1
                if curlen >= band_min {
                    if let Some(pmax) = sam.max_le(cur, hi) {
                        if pmax >= blo {
                            let l = curlen.min((pmax - blo) as usize + 1); // start = p-l+1 >= bl
                            if l >= band_min {
                                // earliest-b (min_in) only when we beat the best — rare, off the path.
                                if l > best_len {
                                    if let Some(pmin) = sam.min_in(cur, blo + l as u32 - 1, hi) {
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
                let link = nd[1];
                if link == 0 {
                    break; // reached the root (state 0) — no shorter qualifying state above
                }
                cur = link;
            }
        }
    }
    (best_a, best_b, best_len)
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
            let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br);
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
            let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br);
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
                let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br);
                if l == 0 {
                    continue;
                }
                m += l;
                pending += (i - al).min(j - bl) + (ar - i - l).min(br - j - l);
                if m + pending < need {
                    return None; // upper bound below the bar ⇒ certified non-edge, abort
                }
                stack.push((al, i, bl, j));
                stack.push((i + l, ar, j + l, br));
            }
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
                let (i, j, l) = longest_in(sam_b, fstate, fmatch, al, ar, bl, br);
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
