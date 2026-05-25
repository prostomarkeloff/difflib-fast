#!/usr/bin/env python3
"""Python-side benchmark + correctness check for difflib-fast.

Shows three things, on real canonicalized Python source:
  1. `ratio(a, b)` is byte-for-byte `difflib.SequenceMatcher(autojunk=False).ratio()` — verified.
  2. how much faster the same number is, single-threaded, than stdlib `difflib`.
  3. how to use every core *without contention* — `ratio(pairs)` (the list form) parallelizes inside
     Rust with the GIL released, so it needs no `ThreadPoolExecutor`; and because `ratio` itself
     releases the GIL, a thread pool over it scales too (unlike pure-Python `difflib`, which can't).

Run:  .venv/bin/python benchmarks/bench_python.py [corpus.canon.bin] [n_pairs]
"""

from __future__ import annotations

import difflib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import difflib_fast


def load_canonicals(path: Path) -> list[str]:
    """A `.canon.bin` corpus is NUL-separated canonicalized function bodies."""
    blob = path.read_bytes().decode("utf-8", "replace")
    return [s for s in blob.split("\0") if s]


def difflib_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def timed(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> None:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "corpora" / "mypy300.canon.bin"
    )
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
    ncores = os.cpu_count() or 1

    canon = load_canonicals(corpus)
    print(f"corpus: {corpus.name}  ({len(canon)} canonicals)   cores: {ncores}\n")

    # A fixed pair list: every (i, j) with i < j, truncated to n_pairs.
    pairs: list[tuple[str, str]] = []
    for i in range(len(canon)):
        for j in range(i + 1, len(canon)):
            pairs.append((canon[i], canon[j]))
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break
    print(f"workload: {len(pairs)} pairs\n")

    # ── 1. correctness: byte-for-byte difflib, scalar and batch forms ─────────────────────────────
    sample = pairs[:1500]
    mismatches = sum(
        1 for a, b in sample if difflib_fast.ratio(a, b) != difflib_ratio(a, b)
    )
    batch = difflib_fast.ratio(sample)  # list form
    batch_ok = all(batch[i] == difflib_fast.ratio(*sample[i]) for i in range(len(sample)))
    print("1. correctness (vs difflib, autojunk=False)")
    print(f"   ratio(a, b)  : {len(sample) - mismatches}/{len(sample)} exact "
          f"({'PASS' if mismatches == 0 else 'FAIL'})")
    print(f"   ratio(pairs) : {'PASS' if batch_ok else 'FAIL'} (== ratio per pair)\n")

    # ── 2. single-thread throughput vs stdlib difflib ─────────────────────────────────────────────
    # difflib is slow, so time it on a sample and normalize to pairs/s.
    dff_sample = pairs[: min(2000, len(pairs))]
    t_dff, _ = timed(lambda: [difflib_ratio(a, b) for a, b in dff_sample])
    dff_pps = len(dff_sample) / t_dff

    t_seq, _ = timed(lambda: [difflib_fast.ratio(a, b) for a, b in pairs])
    seq_pps = len(pairs) / t_seq

    print("2. single thread, pairs/s")
    print(f"   difflib (stdlib)      : {dff_pps:>12,.0f}")
    print(f"   difflib_fast.ratio    : {seq_pps:>12,.0f}   ({seq_pps / dff_pps:>6.0f}x difflib)\n")

    # ── 3. all cores, no contention: the list form does the threading inside Rust ─────────────────
    t_many, _ = timed(lambda: difflib_fast.ratio(pairs))
    many_pps = len(pairs) / t_many

    # ThreadPoolExecutor over scalar ratio() — works only because ratio releases the GIL.
    def tpe(fn, items, workers):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            chunks = [items[k::workers] for k in range(workers)]
            list(ex.map(lambda ch: [fn(a, b) for a, b in ch], chunks))

    t_tpe, _ = timed(lambda: tpe(difflib_fast.ratio, pairs, ncores))
    tpe_pps = len(pairs) / t_tpe

    # contrast: pure-Python difflib in a thread pool does NOT scale (GIL never released).
    t_dff_tpe, _ = timed(lambda: tpe(difflib_ratio, dff_sample, ncores))
    dff_tpe_pps = len(dff_sample) / t_dff_tpe

    print(f"3. all {ncores} cores, pairs/s")
    print(f"   difflib_fast.ratio(pairs)  (rayon, inside Rust): {many_pps:>12,.0f}"
          f"   ({many_pps / seq_pps:>5.1f}x vs 1 thread)")
    print(f"   difflib_fast.ratio in ThreadPoolExecutor       : {tpe_pps:>12,.0f}"
          f"   ({tpe_pps / seq_pps:>5.1f}x vs 1 thread, GIL released)")
    print(f"   difflib in ThreadPoolExecutor (GIL-bound)      : {dff_tpe_pps:>12,.0f}"
          f"   ({dff_tpe_pps / dff_pps:>5.1f}x vs 1 thread — no scaling)\n")

    # ── 4. the real workload: cluster a whole corpus (prebuilt automata, reused across all pairs) ──
    n = len(canon)
    all_pairs = n * (n - 1) // 2
    t_clu, clusters = timed(lambda: difflib_fast.cluster_canonicals(canon, 0.5))
    clu_pps = all_pairs / t_clu
    # task-level: how long the same all-pairs clustering would take stdlib difflib at dff_pps.
    dff_task_s = all_pairs / dff_pps

    print(f"4. cluster the whole corpus ({n} canonicals = {all_pairs:,} pairs @ 0.5)")
    print(f"   difflib_fast.cluster_canonicals : {t_clu * 1000:>8.1f} ms"
          f"   ({clu_pps:>12,.0f} eff. pairs/s, {len(clusters)} clusters)")
    print(f"   same task via stdlib difflib    : {dff_task_s:>8.1f} s   (estimated)\n")

    print("headline")
    print(f"   per call : ratio(a, b)   {seq_pps / dff_pps:>6,.0f}x stdlib difflib")
    print(f"   batch    : ratio(pairs)  {many_pps / dff_pps:>6,.0f}x stdlib difflib (all cores, no contention)")
    print(f"   corpus   : cluster_canonicals {clu_pps / dff_pps:>6,.0f}x stdlib difflib (the real workload)")


if __name__ == "__main__":
    main()
