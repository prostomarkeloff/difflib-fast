//! Bench for the fused two-corpus join [`difflib_fast::simjoin::cosine_join_double`] — the typical
//! "two views of the same objects" shape (e.g. line-vectors and part-vectors of the same functions:
//! two corpora with the same row count). Times one fused join against two separate
//! [`cosine_join`]s and checks each half matches the separate result.
//!
//! `cargo run --release --example simjoin_double -- <a.bin> [b.bin] [t] [reps]`
//! defaults: b.bin = a.bin, t=0.8, reps=3. `SJ_NSUB=<n>` truncates both corpora; thread count via
//! `RAYON_NUM_THREADS`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use std::time::Instant;

use difflib_fast::simjoin::{cosine_join, cosine_join_double, Corpus};

fn le_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

fn arg<T: std::str::FromStr>(i: usize, def: T) -> T {
    std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(def)
}

/// Parse the `SIMJOIN1` binary into raw `(dim, weight)` rows (same format as `examples/simjoin_pypi`).
fn load(path: &str) -> Vec<Vec<(u32, f64)>> {
    let b = std::fs::read(path).expect("read corpus");
    assert!(b.len() >= 16, "corpus too small");
    let magic = u64::from_le_bytes(b[0..8].try_into().unwrap());
    assert_eq!(magic, 0x5349_4D4A_4F49_4E31, "bad magic");
    let n = le_u32(&b, 8) as usize;
    let mut p = 16usize;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let nnz = le_u32(&b, p) as usize;
        p += 4;
        let mut row = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            let d = le_u32(&b, p);
            p += 4;
            let w = f64::from(f32::from_le_bytes(b[p..p + 4].try_into().unwrap()));
            p += 4;
            row.push((d, w));
        }
        rows.push(row);
    }
    rows
}

/// Canonical order for the parity check (join output order is arbitrary).
fn sort_pairs(mut v: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, u64)> {
    v.sort_by_key(|a| (a.0, a.1));
    v.into_iter().map(|(a, b, s)| (a, b, s.to_bits())).collect()
}

fn main() {
    let path_a: String = arg(1, "perf-local/pypi-type3.simjoin.bin".to_string());
    let path_b: String = arg(2, path_a.clone());
    let t: f64 = arg(3, 0.8);
    let reps: usize = arg(4, 3);

    let mut rows_a = load(&path_a);
    let mut rows_b = load(&path_b);
    if let Ok(nsub) = std::env::var("SJ_NSUB") {
        let nsub: usize = nsub.parse().expect("SJ_NSUB");
        rows_a.truncate(nsub);
        rows_b.truncate(nsub);
    }
    assert_eq!(rows_a.len(), rows_b.len(), "corpora must share the row count");
    let n = rows_a.len();

    let corpus_a = Corpus::from_rows(&rows_a);
    let corpus_b = Corpus::from_rows(&rows_b);

    let mut fused: Vec<f64> = Vec::with_capacity(reps);
    let mut separate: Vec<f64> = Vec::with_capacity(reps);
    let (mut pa, mut pb) = (0usize, 0usize);
    for r in 0..reps {
        let (ga, gb) = {
            let t0 = Instant::now();
            let r = cosine_join_double(&corpus_a, &corpus_b, t);
            fused.push(t0.elapsed().as_secs_f64() * 1000.0);
            r
        };
        let (sa, sb) = {
            let t0 = Instant::now();
            let r = (cosine_join(&corpus_a, t), cosine_join(&corpus_b, t));
            separate.push(t0.elapsed().as_secs_f64() * 1000.0);
            r
        };
        if r == 0 {
            pa = ga.len();
            pb = gb.len();
            // The fused halves must be bit-identical to the separate joins (real-data smoke).
            assert_eq!(sort_pairs(ga), sort_pairs(sa), "half a differs from the separate join");
            assert_eq!(sort_pairs(gb), sort_pairs(sb), "half b differs from the separate join");
        }
    }
    fused.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    separate.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (f, s) = (fused[fused.len() / 2], separate[separate.len() / 2]);
    eprintln!(
        "double n={n} t={t} | pairs=({pa},{pb}) | fused={f:.1}ms separate={s:.1}ms speedup={:.2}x",
        s / f
    );
}
