"""difflib-fast — fast, byte-for-byte exact difflib Ratcliff-Obershelp similarity + clustering.

A drop-in for ``difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()``, computed with a suffix
automaton (Rust), plus exact single-linkage clustering of a corpus.
"""

from ._difflib_fast import cluster_canonicals, cluster_canonicals_lsh, ratio

__all__ = ["ratio", "cluster_canonicals", "cluster_canonicals_lsh"]
