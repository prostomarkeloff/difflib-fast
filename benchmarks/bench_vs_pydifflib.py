"""Best-effort throughput of Python's stdlib `difflib` (exact RO, autojunk=False) on a real corpus.

stdlib difflib is pure Python and slow, so we measure its rate over a fixed time budget (cycling the
pair set so the whole window is used), rather than finishing a fixed number of pairs. Same metric as
difflib-fast (byte-for-byte verified on a small sample). Compare the printed pairs/s against
difflib-fast's `bench` numbers (raw all-pairs, 1 thread and all cores).

Usage: python bench_vs_pydifflib.py <corpus.bin> [N_strings] [budget_seconds]
"""
from __future__ import annotations
import sys, time, difflib
import difflib_fast as df

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0

data = open(path, "rb").read()
strings = [p.decode("utf-8", "surrogatepass") for p in data.split(b"\x00") if p][:N]
n = len(strings)
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
mean_len = sum(len(s) for s in strings) // max(n, 1)
name = path.rsplit("/", 1)[-1].replace(".canon.bin", "")

# byte-for-byte: difflib-fast == stdlib difflib(autojunk=False) on a tiny bounded sample
mism = sum(
    df.ratio(strings[i], strings[j]) != difflib.SequenceMatcher(None, strings[i], strings[j], autojunk=False).ratio()
    for i, j in pairs[:12]
)

# stdlib difflib rate over `budget` seconds, cycling the pair set to fill the window
t0 = time.perf_counter()
done = 0
while time.perf_counter() - t0 < budget:
    for i, j in pairs:
        difflib.SequenceMatcher(None, strings[i], strings[j], autojunk=False).ratio()
        done += 1
        if (done & 63) == 0 and time.perf_counter() - t0 >= budget:
            break
    if not pairs:
        break
dt = time.perf_counter() - t0

print(f"{name:13} N={n} mean_len={mean_len} byte-for-byte_mismatches={mism}/12  "
      f"stdlib difflib (autojunk=False): {done / dt:.0f} pairs/s  ({done} pairs in {dt:.1f}s)")
