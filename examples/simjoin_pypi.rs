//! Real-data `simjoin` bench: load a binary Type-3 corpus (built by `scripts/simjoin-corpus.py`
//! from the top-300 PyPI snapshot) and time `cosine_join` on it.
//!
//! `cargo run --release --example simjoin_pypi -- <corpus.bin> [threshold] [reps]`
//! defaults: threshold=0.8 reps=3.  Thread count via `RAYON_NUM_THREADS`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use std::time::Instant;

use difflib_fast::simjoin::{cosine_join, Corpus};
#[cfg(feature = "profiling")]
use difflib_fast::simjoin::cosine_join_counts;

fn le_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

fn arg<T: std::str::FromStr>(i: usize, def: T) -> T {
    std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(def)
}

/// Parse the `SIMJOIN1` binary into raw `(dim, weight)` rows (weights kept as `f64`; `Corpus`
/// L2-normalises and merges duplicate dims).
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
            let w = f64::from(f32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()));
            p += 8;
            row.push((d, w));
        }
        rows.push(row);
    }
    rows
}

fn main() {
    let path: String = arg(1, "perf-local/pypi-type3.simjoin.bin".to_string());
    let t: f64 = arg(2, 0.8);
    let reps: usize = arg(3, 3);

    let rows = load(&path);
    let n = rows.len();
    let nnz: usize = rows.iter().map(Vec::len).sum();

    let b0 = Instant::now();
    let corpus = Corpus::from_rows(&rows);
    let build_ms = b0.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "profiling")]
    if std::env::var("STATS").is_ok() {
        let (ncand, survivors, pairs) = cosine_join_counts(&corpus, t);
        eprintln!(
            "STATS n={n} t={t} | candidates={ncand} survivors(cos_full)={survivors} pairs={pairs} \
             | prune_pass={:.4} survivor_precision={:.3}",
            survivors as f64 / ncand.max(1) as f64,
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
        "pypi-type3 n={n} nnz_total={nnz} mean_nnz={:.1} t={t} | build={build_ms:.0}ms | \
         join: min={:.1}ms median={:.1}ms | pairs={npairs}",
        nnz as f64 / n as f64,
        ms[0],
        ms[reps / 2],
    );
}
