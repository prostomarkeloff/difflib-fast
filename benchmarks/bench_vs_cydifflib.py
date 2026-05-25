"""Fair head-to-head: difflib-fast vs CyDifflib on real canonicalized-code corpora.

Both compute the SAME metric (exact Ratcliff-Obershelp; CyDifflib forced to autojunk=False), in the
SAME process, single thread, on the SAME corpus. Three per-pair rates are reported:

  df.ratio (stateless)      - difflib-fast, rebuilds its suffix automaton every call (no reuse)
  cydifflib (stateless)     - fresh SequenceMatcher(None,a,b,autojunk=False) every call (no reuse)
  cydifflib (seq2-reuse)    - set_seq2 once per b, set_seq1 per a: amortizes b2j (CyDifflib's BEST)

The headline is df.ratio (no reuse) vs cydifflib (seq2-reuse, its best): if difflib-fast wins even
without reuse, the comparison is unambiguously in its favor. Byte-for-byte equality is asserted first.

Usage: python bench_vs_cydifflib.py <corpus.bin> [N_strings] [P_pairs_budget]
"""
from __future__ import annotations
import sys, time, random
from cydifflib import SequenceMatcher as CySM
import difflib_fast as df

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
P = int(sys.argv[3]) if len(sys.argv) > 3 else 2500   # cap pairs so each method stays well under 30s

data = open(path, "rb").read()
strings = [p.decode("utf-8", "surrogatepass") for p in data.split(b"\x00") if p]
random.seed(13)
sub = random.sample(strings, min(N, len(strings)))
n = len(sub)
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][:P]
np_ = len(pairs)
mean_len = sum(len(s) for s in sub) // n
name = path.rsplit("/", 1)[-1].replace(".canon.bin", "")

# byte-for-byte: df.ratio == cydifflib(autojunk=False), same argument order
mism = 0
for i, j in pairs:
    if df.ratio(sub[i], sub[j]) != CySM(None, sub[i], sub[j], autojunk=False).ratio():
        mism += 1

# df.ratio stateless
t0 = time.perf_counter()
for i, j in pairs:
    df.ratio(sub[i], sub[j])
r_df = np_ / (time.perf_counter() - t0)

# cydifflib stateless (fresh matcher each pair)
t0 = time.perf_counter()
for i, j in pairs:
    CySM(None, sub[i], sub[j], autojunk=False).ratio()
r_cy_naive = np_ / (time.perf_counter() - t0)

# cydifflib seq2-reuse (its fastest exact path): group pairs by i, set_seq2 once
byi: dict[int, list[int]] = {}
for i, j in pairs:
    byi.setdefault(i, []).append(j)
m = CySM(autojunk=False)
t0 = time.perf_counter()
for i, js in byi.items():
    m.set_seq2(sub[i])
    for j in js:
        m.set_seq1(sub[j]); m.ratio()
r_cy_reuse = np_ / (time.perf_counter() - t0)

print(f"{name:13} N={n} mean_len={mean_len:5} pairs={np_:5} byte-for-byte_mismatches={mism}")
print(f"  difflib-fast  ratio (stateless)   : {r_df:9.0f} pairs/s")
print(f"  cydifflib     ratio (stateless)   : {r_cy_naive:9.0f} pairs/s   -> df {r_df/r_cy_naive:5.1f}x")
print(f"  cydifflib     ratio (seq2-reuse)  : {r_cy_reuse:9.0f} pairs/s   -> df {r_df/r_cy_reuse:5.1f}x  (df no-reuse vs cy best)")
