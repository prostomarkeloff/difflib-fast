"""difflib-fast — fast, byte-for-byte exact difflib Ratcliff-Obershelp similarity + clustering.

A drop-in for ``difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()``, computed with a suffix
automaton (Rust), plus exact single-linkage clustering of a corpus.

``ratio`` is overloaded: two strings → one float; a list of ``(a, b)`` pairs → a list of floats,
computed across all cores inside Rust (rayon, GIL released) — the contention-free way to score a batch.
"""

from typing import overload

from ._difflib_fast import (
    cluster_canonicals,
    cluster_canonicals_lsh,
    ratio as _ratio,
    ratio_many as _ratio_many,
)

__all__ = ["ratio", "cluster_canonicals", "cluster_canonicals_lsh"]


@overload
def ratio(a: str, b: str, /) -> float: ...
@overload
def ratio(pairs: list[tuple[str, str]], /, *, threads: int = 0) -> list[float]: ...
def ratio(a, b=None, /, *, threads=0):
    """Exact ``difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()`` — byte-for-byte.

    - ``ratio(a, b)`` → one float for the pair.
    - ``ratio(pairs)`` → one float per ``(a, b)`` pair, computed in parallel across all cores inside
      Rust (the GIL is released), so a batch saturates every core with no ``ThreadPoolExecutor`` and no
      per-call overhead. ``ratio(pairs)[i] == ratio(*pairs[i])``, in order. Pass ``threads=N`` to cap
      the pool to N workers for this call (``threads=0``, the default, uses every core — itself tunable
      process-wide via the ``RAYON_NUM_THREADS`` environment variable).
    """
    if b is None:
        return _ratio_many(a, threads)
    return _ratio(a, b)
