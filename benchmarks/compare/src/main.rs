//! Head-to-head vs the other exact-RO Rust crates, in one process, same corpus/pairs, system
//! allocator for all (fair). Measures raw all-pairs `ratio` throughput **single-threaded and
//! multi-threaded** — the competitors get the same rayon parallelism difflib-fast does (they are
//! pure per-pair functions, so rayon parallelizes them trivially; there is no GIL here).
//!
//!   `difflib_fast::ratio`   — this crate (suffix automaton)
//!   `difflib` (by `DimaKudosh`)  — port of Python's difflib to Rust
//!   `gestalt_ratio`         — Ratcliff–Obershelp ratio in Rust
//!
//! Usage: `cargo run --release --example compare -- <corpus.bin> [N] [P_pairs]`
#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use difflib::sequencematcher::SequenceMatcher;
use rayon::prelude::*;

fn dl_ratio(a: &str, b: &str) -> f64 {
    f64::from(SequenceMatcher::new(a, b).ratio())
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "benchmarks/corpora/mypy.canon.bin".to_owned());
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(120);
    let budget: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let data = std::fs::read_to_string(&path).expect("read corpus");
    let strings: Vec<String> = data.split('\0').filter(|s| !s.is_empty()).take(n).map(String::from).collect();
    let n = strings.len();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
            if pairs.len() >= budget {
                break 'outer;
            }
        }
    }
    let np = pairs.len();
    let name = path.rsplit('/').next().unwrap_or(&path).replace(".canon.bin", "");
    let mean_len = strings.iter().map(String::len).sum::<usize>() / n.max(1);

    // value agreement vs difflib-fast (difflib crate returns f32 → tiny precision delta expected),
    // capped subset so it stays bounded when the naive crates are slow on long strings.
    let (mut maxd_dl, mut maxd_ge) = (0.0f64, 0.0f64);
    for &(i, j) in pairs.iter().take(150) {
        let our = difflib_fast::ratio(&strings[i], &strings[j]);
        maxd_dl = maxd_dl.max((our - dl_ratio(&strings[i], &strings[j])).abs());
        maxd_ge = maxd_ge.max((our - gestalt_ratio::gestalt_ratio(&strings[i], &strings[j])).abs());
    }

    // Adaptive timing: repeat the all-pairs pass until ≥0.3s elapsed, so fast (difflib-fast) and slow
    // (naive crates) impls both get a stable rate without a fixed pair count favoring either.
    let measure = |parallel: bool, f: &(dyn Fn(&str, &str) -> f64 + Sync)| -> f64 {
        let t = Instant::now();
        let mut passes = 0u64;
        let mut acc = 0.0f64;
        while t.elapsed().as_secs_f64() < 0.3 && passes < 100_000 {
            acc += if parallel {
                pairs.par_iter().map(|&(i, j)| f(&strings[i], &strings[j])).sum::<f64>()
            } else {
                pairs.iter().map(|&(i, j)| f(&strings[i], &strings[j])).sum::<f64>()
            };
            passes += 1;
        }
        std::hint::black_box(acc);
        passes as f64 * np as f64 / t.elapsed().as_secs_f64()
    };
    let ser = |f: &(dyn Fn(&str, &str) -> f64 + Sync)| -> f64 { measure(false, f) };
    let par = |f: &(dyn Fn(&str, &str) -> f64 + Sync)| -> f64 { measure(true, f) };

    let b2j = &|a: &str, b: &str| difflib_fast::ratio_b2j(a, b);
    let sam = &|a: &str, b: &str| difflib_fast::gestalt_ratio(a, b);
    let dl = &dl_ratio;
    let ge = &|a: &str, b: &str| gestalt_ratio::gestalt_ratio(a, b);
    let threads = rayon::current_num_threads();

    let hybrid = &|a: &str, b: &str| difflib_fast::ratio(a, b);
    // dispatch-signal diagnostics: avg b2j_work/len, avg distinct chars, avg concentration (maxfreq/len)
    let chv: Vec<Vec<char>> = strings.iter().map(|s| s.chars().collect()).collect();
    let avg_wr: f64 = pairs.iter().map(|&(i, j)| difflib_fast::b2j_work(&chv[i], &chv[j]) as f64 / (chv[i].len() + chv[j].len()).max(1) as f64).sum::<f64>() / np as f64;
    let mut sum_distinct = 0.0;
    let mut sum_conc = 0.0;
    for cs in &chv {
        let mut cnt = [0u32; 128];
        for &c in cs {
            let u = c as usize;
            if u < 128 { cnt[u] += 1; }
        }
        sum_distinct += cnt.iter().filter(|&&x| x > 0).count() as f64;
        sum_conc += f64::from(*cnt.iter().max().unwrap_or(&0)) / cs.len().max(1) as f64;
    }
    let (avg_distinct, avg_conc) = (sum_distinct / n as f64, sum_conc / n as f64);
    let (h1, hp) = (ser(hybrid), par(hybrid));
    let (b1, bp) = (ser(b2j), par(b2j));
    let (m1, mp) = (ser(sam), par(sam));
    let best_ser = b1.max(m1);
    let best_par = bp.max(mp);
    let wf = std::env::var("DF_WORK_FACTOR").unwrap_or_else(|_| "default".into());
    println!("{name:13} N={n} mean_len={mean_len} W/len={avg_wr:.0} distinct={avg_distinct:.0} conc={avg_conc:.3} K={wf}  (maxΔ df-difflib={maxd_dl:.1e}, df-gestalt={maxd_ge:.1e})");
    println!("  dispatched 1T {h1:8.0}/s ({:.0}% of best)  {threads}T {hp:9.0}/s ({:.0}% of best)", 100.0 * h1 / best_ser, 100.0 * hp / best_par);
    println!("  b2j {b1:8.0}/s {bp:9.0}/s   SAM {m1:8.0}/s {mp:9.0}/s   |  difflib {:.0}/{:.0}  gestalt {:.0}/{:.0}", ser(dl), par(dl), ser(ge), par(ge));
}
