"""Dataset loaders for the case studies.

Every loader returns (X, y, meta): a float array of shape (n_samples,
n_features), a pandas Series of labels, and a dict recording provenance and
the exact preprocessing applied. The loaders are the single home for
preprocessing that previously appeared in two notebooks each.
"""

from ._klein import DATA_ROOT, load_klein, resolve_data_dir
from ._levine import load_levine32

__all__ = ["DATA_ROOT", "load_klein", "load_levine32", "resolve_data_dir"]
