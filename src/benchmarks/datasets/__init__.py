"""Dataset loaders for the case studies.

Every loader returns (X, y, meta): a float array of shape (n_samples,
n_features), a pandas Series of labels, and a dict recording provenance and
the exact preprocessing applied. Each dataset's preprocessing lives only in
its loader.
"""

from ._cusanovich import load_cusanovich
from ._heca import load_heca
from ._klein import DATA_ROOT, load_klein, resolve_data_dir
from ._levine import load_levine32

__all__ = [
    "DATA_ROOT",
    "load_cusanovich",
    "load_heca",
    "load_klein",
    "load_levine32",
    "resolve_data_dir",
]
