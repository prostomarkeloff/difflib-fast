"""Type stubs for difflib-fast — exact difflib Ratcliff-Obershelp similarity + clustering."""

from typing import overload

@overload
def ratio(a: str, b: str, /) -> float:
    """Exact ``difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()`` — byte-for-byte.

    The Ratcliff-Obershelp similarity ``2*M / (len(a) + len(b))``, identical to Python's ``difflib``
    (including its argument-order asymmetry), computed via a suffix automaton so it stays linear on
    long, repetitive inputs where ``difflib`` degrades.
    """

@overload
def ratio(pairs: list[tuple[str, str]], /, *, threads: int = 0) -> list[float]:
    """Exact :func:`ratio` for many ``(a, b)`` pairs at once, computed across all cores inside Rust.

    ``ratio(pairs)[i] == ratio(*pairs[i])``, bit-for-bit. The parallel fan-out happens in Rust with the
    GIL released — the contention-free way to score a batch from Python: no ``ThreadPoolExecutor``, no
    per-call overhead. Returns one float per input pair, in order. ``threads=N`` caps the pool to N
    workers for this call; ``threads=0`` (default) uses every core (tunable via ``RAYON_NUM_THREADS``).
    """

def cluster_canonicals(
    canonicals: list[str], threshold: float, threads: int = 0
) -> list[tuple[list[int], float]]:
    """Exact single-linkage clustering of ``canonicals`` by RO similarity.

    Two strings join a cluster when their exact ratio is ``>= threshold``. Returns one
    ``(member_indices, min_pairwise_ratio)`` per cluster of >= 2 members; ``member_indices`` index
    into ``canonicals`` (sorted), ``min_pairwise_ratio`` is the cluster's exact minimum pairwise ratio.
    The all-pairs join runs across all cores (GIL released); ``threads=N`` caps the pool to N for this
    call (``threads=0``, default, uses every core — tunable via ``RAYON_NUM_THREADS``).
    """

def cluster_canonicals_lsh(
    canonicals: list[str], threshold: float, num_perm: int, band_rows: int, threads: int = 0
) -> list[tuple[list[int], float]]:
    """Scalable MinHash-LSH variant of :func:`cluster_canonicals` for very large corpora.

    Generates candidate pairs via MinHash-LSH (``num_perm`` permutations, ``band_rows`` rows per band),
    then verifies each candidate with the exact ratio. Clusters match the exact path modulo LSH recall
    (tuned via ``band_rows``); use :func:`cluster_canonicals` when exact recall is required.
    ``threads=N`` caps the pool to N workers for this call (``threads=0``, default, uses every core).
    """

__all__: list[str]
