//! Property test for `gestalt_edge_with_ms_delta` — the approximate-RO knob.
//!
//! Builds RO ratios for a sample of mypy.canon.bin pairs under several `delta` values and
//! asserts the worst-case absolute deviation from the exact ratio stays under `delta + slack`.
//! The slack covers small floating-point rounding in the recursion and the chain-cap's
//! discretization (cap is an integer; delta is continuous).
//!
//! Skipped when `benchmarks/corpora/mypy.canon.bin` isn't present (the bench corpus is excluded
//! from the published crate).

use difflib_fast::gestalt::{
    build_sam, delta_to_chain_cap, gestalt_edge_with_ms, gestalt_edge_with_ms_delta,
    matching_stats_for_test,
};

#[test]
fn delta_zero_matches_exact() {
    // Sanity: delta=0.0 must give bit-identical results vs the no-delta entry point.
    let a = "def add(a, b): return a + b".chars().collect::<Vec<_>>();
    let b = "def add(x, y): return x + y".chars().collect::<Vec<_>>();
    let sam_b = build_sam(&b);
    let mut fstate = Vec::new();
    let mut fmatch = Vec::new();
    matching_stats_for_test(&a, &sam_b, &mut fstate, &mut fmatch);
    let exact = gestalt_edge_with_ms(&a, &b, &sam_b, &fstate, &fmatch, 0.0).unwrap();
    let approx = gestalt_edge_with_ms_delta(&a, &b, &sam_b, &fstate, &fmatch, 0.0, 0.0).unwrap();
    assert_eq!(exact.to_bits(), approx.to_bits(), "delta=0 must be bit-identical");
}

#[test]
fn delta_cap_table() {
    // Quick sanity check on the delta → chain_cap mapping. Empirical p99 chain depth is ≈7,
    // so the cap should drop below 10 by delta≈0.05.
    assert_eq!(delta_to_chain_cap(0.0), u32::MAX);
    assert_eq!(delta_to_chain_cap(1.0), 1);
    assert!(delta_to_chain_cap(0.01) >= 10);
    assert!(delta_to_chain_cap(0.05) >= 4);
    assert!(delta_to_chain_cap(0.10) >= 3);
}

#[test]
fn delta_accuracy_on_mypy_corpus() {
    // Property test: pick a sample of pairs from the bench corpus and verify the approximate
    // ratio stays within `delta + slack` of the exact ratio.
    let Ok(data) = std::fs::read_to_string("benchmarks/corpora/mypy.canon.bin") else {
        eprintln!("mypy.canon.bin not present — skipping approx_ro property test");
        return;
    };
    let strings: Vec<&str> = data
        .split('\0')
        .filter(|s| !s.is_empty() && s.is_ascii())
        .take(200)
        .collect();
    if strings.len() < 4 {
        return;
    }
    let chars: Vec<Vec<char>> = strings.iter().map(|s| s.chars().collect()).collect();
    let sams: Vec<_> = chars.iter().map(|c| build_sam(c)).collect();

    // Sample 500 pairs from the upper triangle to cover dissimilar + similar mix.
    let n = chars.len();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    'outer: for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
            if pairs.len() >= 500 {
                break 'outer;
            }
        }
    }

    for &delta in &[0.0_f64, 0.01, 0.05, 0.10, 0.20] {
        // Slack covers float rounding in the chain-cap heuristic — the empirical loss tends to
        // run BELOW delta on real workloads (cap is conservative).
        let slack: f64 = 0.05;
        let mut max_err: f64 = 0.0;
        let mut max_err_pair = (0, 0);
        let mut both_below_threshold = 0;
        let threshold = 0.5;
        for &(i, j) in &pairs {
            let a = &chars[i];
            let b = &chars[j];
            let sam_b = &sams[j];
            let mut fstate = Vec::new();
            let mut fmatch = Vec::new();
            matching_stats_for_test(a, sam_b, &mut fstate, &mut fmatch);
            let exact = gestalt_edge_with_ms(a, b, sam_b, &fstate, &fmatch, 0.0).unwrap_or(0.0);
            let approx = gestalt_edge_with_ms_delta(a, b, sam_b, &fstate, &fmatch, 0.0, delta)
                .unwrap_or(0.0);
            let err = (exact - approx).abs();
            if err > max_err {
                max_err = err;
                max_err_pair = (i, j);
            }
            if exact < threshold && approx < threshold {
                both_below_threshold += 1;
            }
        }
        let cap = delta_to_chain_cap(delta);
        eprintln!(
            "delta={delta:.3}  cap={cap:>5}  max_abs_err={max_err:.4} @ pair {max_err_pair:?}  \
             (both<threshold on {both_below_threshold}/{} pairs)",
            pairs.len(),
        );
        assert!(
            max_err <= delta + slack,
            "delta={delta} max_err={max_err} exceeds delta+slack={}",
            delta + slack
        );
    }
}
