//! Benchmark harness for `difflib-fast`: throughput + clustering on a NUL-separated corpus.
//!
//! Usage: `bench <corpus.bin> [N] [T] [mode]`
//!   - no `T`          → raw all-pairs full ratio (`gestalt_ratio_prebuilt`): pairs/s + checksum
//!   - `T` (e.g. 0.5)  → threshold qualifying-pairs (early-exit): pairs/s + qualifying count
//!   - `T` + `cluster` → exact single-linkage clustering (the production path): clusters + sizes
//!   - `T` + `scan`    → matching-statistics scan-only floor (the unavoidable per-pair cost)
//!
//! env: `BENCH_MT=1` (rayon-parallel threshold mode) · `RAYON_NUM_THREADS=K` · `BENCH_REPS=R`
//! (repeat the sweep under one timer for stable timing on short runs).

use std::time::Instant;

use difflib_fast::cluster_canonicals_chars;
use difflib_fast::gestalt::{build_sam, gestalt_qualifies, gestalt_ratio_prebuilt, matching_stats_cost, Sam};
use rayon::prelude::*;

// macOS's default malloc madvises freed pages back aggressively (~25% syscall time once parallel);
// mimalloc caches them. A binary may set the global allocator (a library must not).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// ASCII char profile (q=1 `quick_ratio` upper bound: `2·Σmin/(|a|+|b|)`).
fn profile(c: &[char]) -> [u32; 128] {
    let mut p = [0u32; 128];
    for &ch in c {
        let u = ch as u32;
        if u < 128 {
            p[u as usize] += 1;
        }
    }
    p
}

/// Qualifying-pairs mode: length blocking + `quick_ratio` filter + early-exit RO over the
/// length-sorted upper triangle. `BENCH_MT` ⇒ rayon-parallel over the outer index.
#[allow(clippy::cast_precision_loss)]
fn run_threshold(canon: &[Vec<char>], sams: &[Sam], n: usize, pairs: f64, t_thr: f64, scan_only: bool) {
    let prof: Vec<[u32; 128]> = canon.iter().map(|c| profile(c)).collect();
    let lens: Vec<usize> = canon.iter().map(Vec::len).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| lens[i]);
    let mt = std::env::var("BENCH_MT").is_ok();
    let process_p = |p: usize| -> (usize, usize) {
        let i = order[p];
        let (li, pi) = (lens[i], &prof[i]);
        let (mut qual, mut verified) = (0usize, 0usize);
        for &j in &order[p + 1..] {
            let lj = lens[j];
            if 2.0 * (li.min(lj) as f64) < t_thr * ((li + lj) as f64) {
                break; // length blocking: lengths only grow ⇒ rest fail too
            }
            let mut inter = 0u32;
            for c in 0..128 {
                inter += pi[c].min(prof[j][c]);
            }
            if 2.0 * f64::from(inter) < t_thr * ((li + lj) as f64) {
                continue; // quick_ratio < T ⇒ certified non-edge
            }
            verified += 1;
            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            if scan_only {
                qual += (matching_stats_cost(&canon[lo], &sams[hi]) & 1) as usize;
            } else if gestalt_qualifies(&canon[lo], &canon[hi], &sams[hi], t_thr) {
                qual += 1;
            }
        }
        (qual, verified)
    };
    let reps: usize = std::env::var("BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let t = Instant::now();
    let (mut qual, mut verified) = (0usize, 0usize);
    for _ in 0..reps {
        let (q, v) = if mt {
            (0..n).into_par_iter().map(&process_p).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        } else {
            (0..n).map(&process_p).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        };
        qual = q;
        verified = v;
    }
    let dt = t.elapsed().as_secs_f64();
    let threads = if mt { rayon::current_num_threads() } else { 1 };
    eprintln!(
        "threshold {t_thr}{}: {dt:.3}s  ({:.0} pairs/s)  qualifying={qual}  verified={verified} ({:.1}% reached RO)",
        if mt { format!(" [mt×{threads}]") } else { String::new() },
        pairs * reps as f64 / dt,
        verified as f64 / pairs * 100.0
    );
}

/// `cluster` mode: run the exact production clustering and report throughput + cluster-size
/// distribution (so a "one giant cluster" pathology is visible). Always rayon-parallel.
#[allow(clippy::cast_precision_loss)]
fn run_cluster(canon: &[Vec<char>], pairs: f64, t_thr: f64) {
    let reps: usize = std::env::var("BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let t = Instant::now();
    let mut clusters = Vec::new();
    for _ in 0..reps {
        clusters = cluster_canonicals_chars(canon, t_thr);
    }
    let dt = t.elapsed().as_secs_f64();
    let mut sizes: Vec<usize> = clusters.iter().map(|(m, _)| m.len()).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let members: usize = sizes.iter().sum();
    let threads = rayon::current_num_threads();
    eprintln!(
        "cluster {t_thr} [×{threads}]: {dt:.3}s  ({:.0} pairs/s)  clusters={}  members={members}  largest={:?}",
        pairs / dt,
        clusters.len(),
        &sizes[..sizes.len().min(6)]
    );
}

#[allow(clippy::cast_precision_loss)]
fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "benchmarks/corpora/mypy.canon.bin".to_owned());
    let limit: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let threshold: Option<f64> = std::env::args().nth(3).and_then(|s| s.parse().ok());
    let mode4 = std::env::args().nth(4);
    let scan_only = mode4.as_deref() == Some("scan");
    let cluster_mode = mode4.as_deref() == Some("cluster");

    let data = std::fs::read_to_string(&path).expect("read corpus file");
    let mut canon: Vec<Vec<char>> = data.split('\0').filter(|s| !s.is_empty()).map(|s| s.chars().collect()).collect();
    canon.truncate(limit);
    let n = canon.len();
    let pairs = (n * (n - 1) / 2) as f64;
    eprintln!("corpus {path}: {n} strings, {pairs:.0} pairs");

    if cluster_mode {
        run_cluster(&canon, pairs, threshold.unwrap_or(0.5));
        return;
    }

    let t = Instant::now();
    let sams: Vec<Sam> = canon.iter().map(|c| build_sam(c)).collect();
    eprintln!("prebuild {n} SAMs: {:.3}s", t.elapsed().as_secs_f64());

    if let Some(t_thr) = threshold {
        run_threshold(&canon, &sams, n, pairs, t_thr, scan_only);
    } else {
        let mt = std::env::var("BENCH_MT").is_ok();
        let row = |i: usize| -> f64 { ((i + 1)..n).map(|j| gestalt_ratio_prebuilt(&canon[i], &canon[j], &sams[j])).sum() };
        let t = Instant::now();
        let acc: f64 = if mt { (0..n).into_par_iter().map(row).sum() } else { (0..n).map(row).sum() };
        let dt = t.elapsed().as_secs_f64();
        let threads = if mt { rayon::current_num_threads() } else { 1 };
        eprintln!(
            "all-pairs gestalt{}: {dt:.2}s  ({:.0} pairs/s)  (acc={acc:.0})",
            if mt { format!(" [mt×{threads}]") } else { String::new() },
            pairs / dt
        );
    }
}
