//! GPU-vs-CPU throughput experiment for `simjoin`'s bandwidth-bound verify step.
//!
//! Loads the binary corpus, builds the CSR, generates a batch of `P` random candidate pairs, and
//! times the f32 sparse-cosine dot over that batch on (a) the GPU (`batch_cosine` kernel, incl. pair
//! upload), (b) the CPU serially, (c) the CPU with rayon. Reports pairs/sec each — answering whether
//! the Apple GPU clears the random-gather dots faster than the tuned CPU.
//!
//! `cargo run --release --features gpu --example simjoin_gpu_bench -- <corpus.bin> [npairs] [reps]`

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::comparison_chain,
    clippy::too_many_lines
)]

use std::time::Instant;

use std::collections::HashSet;

use difflib_fast::simjoin::{cosine_join, cosine_join_gpu, cosine_join_gpu_f32, Corpus};
use difflib_fast::simjoin_gpu::BatchCosineGpu;
use rayon::prelude::*;

fn le_u32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}

fn arg<T: std::str::FromStr>(i: usize, def: T) -> T {
    std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(def)
}

fn load(path: &str) -> Vec<Vec<(u32, f64)>> {
    let b = std::fs::read(path).expect("read corpus");
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

#[inline]
fn cpu_dot(indptr: &[u32], dims: &[u32], wts: &[f32], a: u32, b: u32) -> f32 {
    let (mut ia, ea) = (indptr[a as usize] as usize, indptr[a as usize + 1] as usize);
    let (mut ib, eb) = (indptr[b as usize] as usize, indptr[b as usize + 1] as usize);
    let mut s = 0.0f32;
    while ia < ea && ib < eb {
        let (da, db) = (dims[ia], dims[ib]);
        if da == db {
            s += wts[ia] * wts[ib];
            ia += 1;
            ib += 1;
        } else if da < db {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    s
}

fn main() {
    let path: String = arg(1, "perf-local/pypi-type3.simjoin.bin".to_string());
    let np: usize = arg(2, 20_000_000);
    let reps: usize = arg(3, 3);

    let rows = load(&path);
    let corpus = Corpus::from_rows(&rows);
    let (indptr, dims, wts) = corpus.csr_f32();
    let n = corpus.len();

    // Deterministic random valid pairs (each gathers two random CSR rows — the bandwidth pattern).
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut pa = Vec::with_capacity(np);
    let mut pb = Vec::with_capacity(np);
    for _ in 0..np {
        pa.push((next() as usize % n) as u32);
        pb.push((next() as usize % n) as u32);
    }

    let Some(gpu) = BatchCosineGpu::new(&indptr, &dims, &wts) else {
        eprintln!("no Metal device — skipping");
        return;
    };
    eprintln!("device: {} | n={n} nnz={} | npairs={np}", gpu.device_name(), dims.len());

    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let rate = |ms: f64| np as f64 / (ms / 1000.0) / 1e6; // M pairs/sec
    // Effective gather bandwidth: bytes of row data actually read = Σ (nnz_a+nnz_b) × 8 B per pair
    // (u32 dim + f32 weight). This is the dominant traffic on the random-gather verify.
    let row_bytes: u64 = (0..np)
        .map(|k| {
            let la = u64::from(indptr[pa[k] as usize + 1] - indptr[pa[k] as usize]);
            let lb = u64::from(indptr[pb[k] as usize + 1] - indptr[pb[k] as usize]);
            (la + lb) * 8
        })
        .sum();
    let gbs = |ms: f64| row_bytes as f64 / (ms / 1000.0) / 1e9; // GB/s of gathered row data

    // GPU (includes pair upload — a real hybrid must materialise pairs too).
    let mut g = Vec::new();
    let mut gpu_out = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        gpu_out = gpu.cosine_batch(&pa, &pb);
        g.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let gpu_ms = med(g);

    // CPU serial.
    let mut cs = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for k in 0..np {
            acc += cpu_dot(&indptr, &dims, &wts, pa[k], pb[k]);
        }
        cs.push(t0.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(acc);
    }
    let cpu_serial_ms = med(cs);

    // CPU rayon.
    let mut cp = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let acc: f32 = (0..np)
            .into_par_iter()
            .map(|k| cpu_dot(&indptr, &dims, &wts, pa[k], pb[k]))
            .sum();
        cp.push(t0.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(acc);
    }
    let cpu_par_ms = med(cp);

    // Sanity: GPU f32 vs CPU f32 should agree to ~1e-4 (same precision/algorithm).
    let mut maxdiff = 0.0f32;
    for k in 0..np.min(100_000) {
        let c = cpu_dot(&indptr, &dims, &wts, pa[k], pb[k]);
        maxdiff = maxdiff.max((gpu_out[k] - c).abs());
    }

    eprintln!(
        "row_bytes={:.0}MB over {np} pairs (mean {:.0}B/pair)",
        row_bytes as f64 / 1e6,
        row_bytes as f64 / np as f64,
    );
    eprintln!(
        "GPU(+upload): {gpu_ms:.1}ms = {:.0} Mpairs/s = {:.1} GB/s | \
         CPU serial: {cpu_serial_ms:.1}ms = {:.0} M/s = {:.1} GB/s | \
         CPU rayon: {cpu_par_ms:.1}ms = {:.0} M/s = {:.1} GB/s | GPU/CPUrayon {:.2}x | maxdiff={maxdiff:.1e}",
        rate(gpu_ms),
        gbs(gpu_ms),
        rate(cpu_serial_ms),
        gbs(cpu_serial_ms),
        rate(cpu_par_ms),
        gbs(cpu_par_ms),
        cpu_par_ms / gpu_ms,
    );

    // --- Full hybrid join vs CPU join, at the threshold given in env SJ_T (default 0.8) ---
    let jt: f64 = std::env::var("SJ_T").ok().and_then(|s| s.parse().ok()).unwrap_or(0.8);

    let c0 = Instant::now();
    let cpu_pairs = cosine_join(&corpus, jt);
    let cpu_join_ms = c0.elapsed().as_secs_f64() * 1000.0;

    let h0 = Instant::now();
    let gpu_pairs = cosine_join_gpu(&corpus, jt, &gpu);
    let hyb_join_ms = h0.elapsed().as_secs_f64() * 1000.0;

    // Parity: the hybrid MUST return the exact same pair set + scores as the pure-CPU join.
    let mut a = cpu_pairs;
    let mut b = gpu_pairs;
    a.sort_by_key(|x| (x.0, x.1));
    b.sort_by_key(|x| (x.0, x.1));
    let same = a.len() == b.len()
        && a.iter().zip(&b).all(|(x, y)| x.0 == y.0 && x.1 == y.1 && x.2.to_bits() == y.2.to_bits());

    eprintln!(
        "JOIN t={jt} pairs={} | CPU: {cpu_join_ms:.0}ms | hybrid GPU+CPU: {hyb_join_ms:.0}ms \
         | speedup {:.2}x | parity={}",
        a.len(),
        cpu_join_ms / hyb_join_ms,
        if same { "BIT-IDENTICAL ✓" } else { "MISMATCH ✗" },
    );

    // --- Pure-f32 join: is "f32 everywhere" actually worse? Measure pair-set delta + speed. ---
    let f0 = Instant::now();
    let f32_pairs = cosine_join_gpu_f32(&corpus, jt, &gpu);
    let f32_join_ms = f0.elapsed().as_secs_f64() * 1000.0;

    // a = the exact f64 pair set (sorted). Compare membership + worst score gap on shared pairs.
    let f64_set: HashSet<(usize, usize)> = a.iter().map(|p| (p.0, p.1)).collect();
    let f32_set: HashSet<(usize, usize)> = f32_pairs.iter().map(|p| (p.0, p.1)).collect();
    let only_f64 = f64_set.difference(&f32_set).count();
    let only_f32 = f32_set.difference(&f64_set).count();
    let f64_score: std::collections::HashMap<(usize, usize), f64> =
        a.iter().map(|p| ((p.0, p.1), p.2)).collect();
    let max_gap = f32_pairs
        .iter()
        .filter_map(|p| f64_score.get(&(p.0, p.1)).map(|&e| (f64::from(p.2) - e).abs()))
        .fold(0.0f64, f64::max);
    let diff = only_f64 + only_f32;
    eprintln!(
        "F32-ONLY t={jt} pairs={} | {f32_join_ms:.0}ms (vs CPU {cpu_join_ms:.0}ms = {:.2}x, \
         vs hybrid {hyb_join_ms:.0}ms = {:.2}x) | differing pairs: {diff} of {} ({:.4}%: \
         {only_f64} dropped, {only_f32} added) | max score gap on shared: {max_gap:.1e}",
        f32_pairs.len(),
        cpu_join_ms / f32_join_ms,
        hyb_join_ms / f32_join_ms,
        a.len(),
        100.0 * diff as f64 / a.len() as f64,
    );
}
