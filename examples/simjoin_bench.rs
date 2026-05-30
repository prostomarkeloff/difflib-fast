//! Bench harness for `simjoin::cosine_join` on a synthetic Zipfian-skewed sparse corpus (few common
//! dims, many rare — like IDF-weighted lines/tokens). Reproducible, no I/O, scalable.
//!
//! `cargo run --release --example simjoin_bench -- [n] [nnz] [ndims] [threshold] [reps]`
//! defaults: n=100000 nnz=14 ndims=20000 t=0.7 reps=3

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::time::Instant;

use difflib_fast::simjoin::{cosine_join, Corpus};
#[cfg(feature = "profiling")]
use difflib_fast::simjoin::cosine_join_counts;

fn arg<T: std::str::FromStr>(i: usize, def: T) -> T {
    std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(def)
}

/// Deterministic xorshift → IDF-weighted sparse rows. Each vector draws `nnz` distinct dims with a
/// cubic bias toward low ids (so low ids are common, high ids rare); the per-dim weight is its IDF
/// `ln(n / df)` computed from the generated corpus — the realistic weighted-cosine input shape.
fn gen(n: usize, nnz: usize, ndims: usize, seed: u64) -> Vec<Vec<(u32, f64)>> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // 1. distinct dims per vector (cubic skew → Zipfian-ish frequencies). ~5% of vectors are planted
    //    near-duplicates of an earlier one (so real cosine clusters exist → the verify path runs).
    let mut sets: Vec<Vec<u32>> = Vec::with_capacity(n);
    for i in 0..n {
        let dup = i > 0 && (next() % 100) < 5;
        let mut v: Vec<u32> = if dup {
            let src = (next() as usize) % i;
            let mut c = sets[src].clone();
            if !c.is_empty() {
                let k = (next() as usize) % c.len();
                c[k] = (next() % ndims as u64) as u32; // mutate one dim
            }
            c
        } else {
            (0..nnz)
                .map(|_| {
                    let u = (next() >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
                    ((u * u * u) * ndims as f64) as u32 % ndims as u32
                })
                .collect()
        };
        v.sort_unstable();
        v.dedup();
        sets.push(v);
    }
    let mut df = vec![0u32; ndims];
    for v in &sets {
        for &d in v {
            df[d as usize] += 1;
        }
    }
    // 2. weight each present dim by its IDF.
    sets.into_iter()
        .map(|v| {
            v.into_iter()
                .map(|d| {
                    let idf = (n as f64 / f64::from(df[d as usize]).max(1.0)).ln();
                    (d, idf)
                })
                .collect()
        })
        .collect()
}

fn main() {
    let n: usize = arg(1, 100_000);
    let nnz: usize = arg(2, 14);
    let ndims: usize = arg(3, 20_000);
    let t: f64 = arg(4, 0.7);
    let reps: usize = arg(5, 3);

    let rows = gen(n, nnz, ndims, 0x1234_5678_9abc_def1);
    let build0 = Instant::now();
    let corpus = Corpus::from_rows(&rows);
    let build_ms = build0.elapsed().as_secs_f64() * 1000.0;

    // Strategy diagnostic (profiling builds only): posting touches / candidates / pairs. The
    // candidates-per-pair ratio decides whether to prune harder or speed the dot up.
    #[cfg(feature = "profiling")]
    if std::env::var("STATS").is_ok() {
        let (ncand, survivors, pairs) = cosine_join_counts(&corpus, t);
        eprintln!(
            "STATS n={n} t={t} | candidates={ncand} survivors(cos_full)={survivors} pairs={pairs} \
             | prune_pass={:.4} cos_full_saved={:.4} survivor_precision={:.3}",
            survivors as f64 / ncand.max(1) as f64,
            1.0 - survivors as f64 / ncand.max(1) as f64,
            pairs as f64 / survivors.max(1) as f64,
        );
    }

    let mut ms: Vec<f64> = Vec::with_capacity(reps);
    let mut npairs = 0usize;
    for _ in 0..reps {
        let t0 = Instant::now();
        let pairs = cosine_join(&corpus, t);
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        npairs = pairs.len();
        std::hint::black_box(&pairs);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "n={n} nnz={nnz} ndims={ndims} t={t} | build={build_ms:.0}ms | join: min={:.1}ms median={:.1}ms | pairs={npairs}",
        ms[0],
        ms[reps / 2],
    );
}
