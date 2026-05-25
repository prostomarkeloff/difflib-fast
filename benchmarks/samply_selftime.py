"""Headless self-time / inclusive report from a samply profile, symbolicated by samply's OWN symbol
server — so system libraries (dyld shared cache) resolve too, and names come back demangled and
de-monomorphized (`swtch_pri`, `_platform_memcmp`, `tokenize`) instead of `UNKNOWN` / 500-char
rayon types you get from the `.syms.json` sidecar alone.

How: `samply load` serves a Mozilla symbolication API under a per-session token printed on its stderr
(`…/<token>/symbolicate/v5`). We launch it, scrape the token, POST every (module, address) referenced
by the samples in one request, then aggregate. CPU-time weighting (`threadCPUDelta`) is the default so
a sampling profiler's coalesced idle/park samples don't bury the real work.

Usage:
    python benchmarks/samply_selftime.py <profile.json.gz> [N] [--only SUBSTR] [--by-samples]

Record a profile first, e.g. via benchmarks/profile.sh (samply record … target/release/bench …).
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter

# Prefer an explicit SAMPLY env var, else the one on PATH, else the default cargo install location.
SAMPLY = os.environ.get("SAMPLY") or shutil.which("samply") or os.path.expanduser("~/.cargo/bin/samply")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_symbol_server(profile_path: str) -> tuple[str, subprocess.Popen]:
    """Launch `samply load` and return (symbolication base URL, process). The base URL embeds the
    per-session token samply prints as `…?symbolServer=<url-encoded base>` on startup."""
    port = free_port()
    proc = subprocess.Popen(
        [SAMPLY, "load", "--no-open", "--port", str(port), profile_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            break
        if "symbolServer=" in line:
            return urllib.parse.unquote(line.split("symbolServer=", 1)[1].strip()), proc
    proc.terminate()
    raise RuntimeError("samply load did not advertise a symbol server")


def symbolicate(prof: dict, base: str) -> dict[tuple[int, int], str]:
    """(global lib index, module-relative address) → function name, via one /symbolicate/v5 call."""
    libs = prof["libs"]
    memmap = [[lib["debugName"], lib["breakpadId"]] for lib in libs]
    pairs: set[tuple[int, int]] = set()
    for th in prof["threads"]:
        res_lib, func_res = th["resourceTable"]["lib"], th["funcTable"]["resource"]
        frame_func, frame_addr = th["frameTable"]["func"], th["frameTable"]["address"]
        for si in th["samples"]["stack"]:
            node = si
            st = th["stackTable"]
            while node is not None:
                fr = st["frame"][node]
                pairs.add((res_lib[func_res[frame_func[fr]]], frame_addr[fr]))
                node = st["prefix"][node]
    ordered = sorted(pairs)
    req = {"jobs": [{"memoryMap": memmap, "stacks": [[[li, addr] for li, addr in ordered]]}]}
    with urllib.request.urlopen(base + "/symbolicate/v5", json.dumps(req).encode(), timeout=120) as r:
        frames = json.loads(r.read())["results"][0]["stacks"][0]
    out: dict[tuple[int, int], str] = {}
    for (li, addr), f in zip(ordered, frames):
        out[(li, addr)] = f.get("function") or f"{libs[li]['debugName']}+0x{addr:x}"
    return out


def main() -> None:
    profile_path = sys.argv[1]
    topn = next((int(a) for a in sys.argv[2:] if a.isdigit()), 25)
    only = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--only"), None)
    by_cpu = "--by-samples" not in sys.argv

    prof = json.loads(gzip.open(profile_path).read())
    base, proc = start_symbol_server(profile_path)
    try:
        sym = symbolicate(prof, base)
    finally:
        proc.terminate()

    libs = prof["libs"]
    self_t: Counter[str] = Counter()
    incl_t: Counter[str] = Counter()
    by_module: Counter[str] = Counter()
    by_thread: Counter[str] = Counter()
    total = 0
    for th in prof["threads"]:
        st = th["stackTable"]
        res_lib, func_res = th["resourceTable"]["lib"], th["funcTable"]["resource"]
        frame_func, frame_addr = th["frameTable"]["func"], th["frameTable"]["address"]

        def key(node: int) -> tuple[int, int]:
            fr = st["frame"][node]
            return res_lib[func_res[frame_func[fr]]], frame_addr[fr]

        samples = th["samples"]
        weights = (samples["threadCPUDelta"] if by_cpu else samples.get("weight")) or [1] * len(samples["stack"])
        for si, w in zip(samples["stack"], weights):
            if si is None:
                continue
            w = w or 0
            total += w
            by_thread[th["name"]] += w
            leaf = key(si)
            by_module[libs[leaf[0]]["debugName"]] += w
            self_t[sym[leaf]] += w
            seen, node = set(), si
            while node is not None:
                nm = sym[key(node)]
                if nm not in seen:
                    seen.add(nm)
                    incl_t[nm] += w
                node = st["prefix"][node]

    unit = "CPU-µs" if by_cpu else "samples"

    def show(title: str, counter: Counter, limit: int, flt: str | None = None) -> None:
        print(f"\n=== {title} (total {total} {unit}) ===")
        for name, c in [(n, c) for n, c in counter.most_common() if flt is None or flt in n][:limit]:
            print(f"  {c / total * 100:5.1f}%  {c:11}  {name}")

    show("by MODULE", by_module, 10)
    show("by THREAD (serial vs parallel)", by_thread, 14)
    show("SELF-TIME (leaf)", self_t, topn, only)
    show("INCLUSIVE", incl_t, topn, only)


if __name__ == "__main__":
    main()
