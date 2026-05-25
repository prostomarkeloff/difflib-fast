#!/usr/bin/env bash
# Record a samply CPU profile of the `bench` binary and print a self-time / inclusive report
# (via samply_selftime.py, which symbolicates through samply's own server so system frames resolve).
#
# Requires samply: `cargo install samply`.
# Usage:  benchmarks/profile.sh <corpus.bin> [N] [T] [mode]    # args forwarded to `bench`
#   e.g.  benchmarks/profile.sh benchmarks/corpora/sympy.canon.bin 2000 0.5 cluster
# Env:   SAMPLY=/path/to/samply   OUT=/tmp/prof.json.gz   RATE=4000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLY="${SAMPLY:-$(command -v samply || echo "$HOME/.cargo/bin/samply")}"
OUT="${OUT:-/tmp/difflib-fast-profile.json.gz}"
RATE="${RATE:-4000}"

cargo build --release --manifest-path "$ROOT/Cargo.toml" -p difflib-fast --features bench --bin bench
"$SAMPLY" record --save-only --no-open --unstable-presymbolicate -o "$OUT" --rate "$RATE" -- \
    "$ROOT/target/release/bench" "$@"
python3 "$ROOT/benchmarks/samply_selftime.py" "$OUT" 30
