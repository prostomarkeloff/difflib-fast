#!/usr/bin/env bash
# Benchmark runner for difflib-fast. Each line reports pairs/s; comparison lines also report the
# speedup (Nx = difflib-fast / competitor). Every measurement command is wrapped in `timeout 30`.
#
# Sections:
#   self       difflib-fast itself: raw ratio (1T), threshold join (1T & all cores), clustering
#   crates     vs other exact-RO Rust crates, 1 thread, same process (benchmarks/compare)
#   cydifflib  vs CyDifflib (Python), 1 thread (needs difflib_fast + cydifflib in $PYTHON)
#   parallel   CyDifflib multiprocessing (its best) vs difflib-fast rayon — same qualifying-pairs task
#   pydifflib  vs Python stdlib difflib (best-effort, time-bounded) + difflib-fast raw 1T & MT
#
# Usage:  benchmarks/run_all.sh [self|crates|cydifflib|parallel|pydifflib|all]   (default: all)
# Env:    PYTHON=/path/to/venv/bin/python   (cydifflib/pydifflib need difflib_fast in it; cydifflib too)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
BENCHDIR="$ROOT/benchmarks"
COR="$BENCHDIR/corpora"
PYTHON="${PYTHON:-python3}"
SECTION="${1:-all}"
BIN="$ROOT/target/release/bench"
COMPARE_MANIFEST="$BENCHDIR/compare/Cargo.toml"
COMPARE="$BENCHDIR/compare/target/release/compare"
REPOS=(mypy django transformers ha sympy)

# per-repo caps for the SLOW single-thread comparisons (long-function repos shrink so each stays <30s).
caps() { case "$1" in mypy|django|ha) echo "140 2500" ;; *) echo "100 800" ;; esac; }  # N P
pcap() { case "$1" in transformers) echo "100" ;; *) echo "300" ;; esac; }             # parallel N

build() {
  cargo build --release --manifest-path "$MANIFEST" --features bench --bin bench || exit 1
}

self() {
  echo "================= difflib-fast self (pairs/s) ================="
  for r in "${REPOS[@]}"; do
    echo "------ $r ------"
    timeout 30 "$BIN" "$COR/$r.canon.bin" 300 2>&1                     # raw full ratio, 1 thread
    timeout 30 "$BIN" "$COR/$r.canon.bin" 1000 0.5 2>&1                # threshold join, 1 thread
    timeout 30 env BENCH_MT=1 "$BIN" "$COR/$r.canon.bin" 2000 0.5 2>&1 # threshold join, all cores
    timeout 30 "$BIN" "$COR/$r.canon.bin" 2000 0.5 cluster 2>&1        # clustering, all cores
  done
}

crates() {
  echo "========== vs other exact-RO Rust crates (1 thread + all cores, same process) =========="
  cargo build --release --manifest-path "$COMPARE_MANIFEST" || exit 1
  for r in "${REPOS[@]}"; do
    read -r n p < <(caps "$r")
    timeout 30 "$COMPARE" "$COR/$r.canon.bin" "$n" "$p" 2>&1
  done
}

cydifflib() {
  echo "============ vs CyDifflib, 1 thread (Python, same process) ============"
  for r in "${REPOS[@]}"; do
    read -r n p < <(caps "$r")
    timeout 30 "$PYTHON" -u "$BENCHDIR/bench_vs_cydifflib.py" "$COR/$r.canon.bin" "$n" "$p" 2>&1
  done
}

parallel() {
  echo "==== parallel: CyDifflib multiprocessing vs difflib-fast rayon (qualifying@0.5) ===="
  for r in "${REPOS[@]}"; do
    n="$(pcap "$r")"
    echo "------ $r (N=$n) ------"
    timeout 30 "$PYTHON" -u "$BENCHDIR/bench_parallel.py" "$COR/$r.canon.bin" "$n" 0.5 2>&1
    timeout 30 env BENCH_MT=1 "$BIN" "$COR/$r.canon.bin" "$n" 0.5 2>&1
  done
}

pydifflib() {
  echo "============ vs Python stdlib difflib (best-effort, time-bounded) ============"
  for r in "${REPOS[@]}"; do
    echo "------ $r ------"
    timeout 30 "$PYTHON" -u "$BENCHDIR/bench_vs_pydifflib.py" "$COR/$r.canon.bin" 50 24 2>&1   # stdlib difflib
    timeout 30 "$BIN" "$COR/$r.canon.bin" 300 2>&1                                             # difflib-fast raw, 1T
    timeout 30 env BENCH_MT=1 "$BIN" "$COR/$r.canon.bin" 300 2>&1                              # difflib-fast raw, all cores
  done
}

build
case "$SECTION" in
  self)      self ;;
  crates)    crates ;;
  cydifflib) cydifflib ;;
  parallel)  parallel ;;
  pydifflib) pydifflib ;;
  all)       self; crates; cydifflib; parallel; pydifflib ;;
  *) echo "unknown section: $SECTION (use self|crates|cydifflib|parallel|pydifflib|all)"; exit 1 ;;
esac
echo "== done =="
