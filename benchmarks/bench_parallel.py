"""Parallel head-to-head: difflib-fast (rayon, in-process) vs CyDifflib (multiprocessing).

CyDifflib has NO in-process parallelism — its source has no `nogil`/`prange`/batch API, and the hot
methods hold the GIL (they operate on Python objects). So threads can't scale it; the fair "as much
as possible" parallelism for CyDifflib is `multiprocessing` (one GIL per process). We give it all
cores, seq2-reuse inside each worker, and a spawn initializer so the corpus is shipped once per
worker (not per task) — its best realistic parallel form.

Task (both sides): the all-pairs QUALIFYING-PAIRS join — count pairs with exact RO ratio >= T, using
the same length-block + quick_ratio early-exit funnel. difflib-fast's side is the Rust `bench`
binary in BENCH_MT mode (run separately, same N/T); this script measures CyDifflib.

Usage: python bench_parallel.py <corpus.bin> [N] [T]
"""
from __future__ import annotations
import sys, time, random, os
from cydifflib import SequenceMatcher as CySM

# globals populated per-process (main + each spawned worker via the initializer)
_SUB: list[str] = []
_ORDER: list[int] = []
_N = 0
_T = 0.5


def _init(sub, order, t):
    global _SUB, _ORDER, _N, _T
    _SUB, _ORDER, _N, _T = sub, order, len(order), t


def _row_qual(p: int) -> int:
    """Qualifying-pair count for length-sorted row p (seq2-reuse + funnel), exact autojunk=False RO."""
    i = _ORDER[p]
    m = CySM(autojunk=False)
    m.set_seq2(_SUB[i])
    q = 0
    for qq in range(p + 1, _N):
        j = _ORDER[qq]
        m.set_seq1(_SUB[j])
        if m.real_quick_ratio() < _T:
            break
        if m.quick_ratio() < _T:
            continue
        if m.ratio() >= _T:
            q += 1
    return q


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

    path = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    cores = os.cpu_count() or 8
    name = path.rsplit("/", 1)[-1].replace(".canon.bin", "")

    data = open(path, "rb").read()
    strings = [p.decode("utf-8", "surrogatepass") for p in data.split(b"\x00") if p]
    sub = strings[:N]  # first N — identical to the Rust bench's truncate(N), so inputs match exactly
    n = len(sub)
    order = sorted(range(n), key=lambda i: len(sub[i]))
    pairs = n * (n - 1) // 2
    _init(sub, order, T)

    # GIL check at a small slice: thread-pool vs single thread. If ~no speedup, CyDifflib is GIL-bound.
    g = min(n, 80)
    t0 = time.perf_counter()
    _ = sum(_row_qual(p) for p in range(g))
    st_small = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cores) as ex:
        _ = sum(ex.map(_row_qual, range(g)))
    tt_small = time.perf_counter() - t0

    # Full multiprocessing run over all rows (its best parallel form).
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cores, initializer=_init, initargs=(sub, order, T)) as ex:
        qual = sum(ex.map(_row_qual, range(n), chunksize=2))
    mp = time.perf_counter() - t0

    print(f"{name:13} N={n} pairs={pairs} T={T} cores={cores}")
    print(f"  GIL check (first {g} rows): single {st_small:.2f}s vs {cores}-thread {tt_small:.2f}s  -> thread speedup {st_small/tt_small:.2f}x (≈1 ⇒ GIL-bound, threads useless)")
    print(f"  cydifflib multiprocessing ({cores} procs): {mp:.2f}s  ({pairs/mp:.0f} pairs/s)  qualifying={qual}")


if __name__ == "__main__":
    main()
