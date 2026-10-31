"""Private bridge between CARVE and :class:`~anndata.AnnData`.

This module owns every detail of moving data in and out of an ``AnnData``:
choosing which representation to cluster, and writing fitted results back in a
form that survives :meth:`~anndata.AnnData.write_h5ad`. It imports nothing
from :mod:`carve.api`, so ``api`` can import it without a cycle.

Notes
-----
:func:`results_to_uns` coerces every non-numeric column to ``str``. A
heterogeneous estimator grid, such as KMeans (no ``linkage``) next to
agglomerative clustering (no ``affinity``), makes ``DataFrame.from_records``
fill the gaps with ``NaN``. Under pandas 2.3, the oldest version anndata 0.13
supports, those columns are ``object`` dtype mixing ``str`` and
``float('nan')``, and anndata's variable-length string writer raises::

    TypeError: Can't implicitly convert non-string objects to strings

pandas 3.0 stores the same columns in its ``str`` dtype and writes them. The
coercion also keeps ``NaN`` out of columns that end up in plot labels.
"""

import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

__all__ = [
    "get_representation",
    "is_anndata",
    "resolve_basis",
]

#: Embeddings searched, in order, when a plotting call does not name a basis.
_BASIS_PREFERENCE = ("X_umap", "X_tsne", "X_draw_graph_fa", "X_pca")

#: Longest string persisted into ``uns[key]["results"]``; guards against a
#: pathological ``repr`` of a custom estimator bloating the h5ad file.
_MAX_UNS_STR = 256


def is_anndata(X: Any) -> bool:
    """Return whether ``X`` is an :class:`~anndata.AnnData` instance."""
    return isinstance(X, AnnData)


def _carve_version() -> str:
    """Return the installed CARVE version, or ``"unknown"`` if undetermined."""
    try:
        return version("carve-validate")
    except PackageNotFoundError:  # pragma: no cover - editable/source trees
        return "unknown"


def get_representation(
    adata: AnnData,
    *,
    use_rep: str | None = None,
    layer: str | None = None,
    n_pcs: int | None = None,
) -> np.ndarray:
    """Select the matrix CARVE should cluster.

    Follows scanpy's representation handling, except that it never computes
    a PCA: reducing the data would change what is being validated.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_rep : str, optional
        Key in ``adata.obsm`` to use, or ``"X"`` for ``adata.X``. If None and
        ``layer`` is None, ``"X_pca"`` is used when present, else ``adata.X``.
    layer : str, optional
        Key in ``adata.layers`` to use. Mutually exclusive with ``use_rep``.
    n_pcs : int, optional
        Restrict the representation to its first ``n_pcs`` columns.

    Returns
    -------
    rep : ndarray of shape (n_obs, n_features)
        Dense 2D representation.

    Raises
    ------
    ValueError
        If both ``use_rep`` and ``layer`` are given, if a named key is
        absent, or if ``n_pcs`` exceeds the available width.
    """
    if use_rep is not None and layer is not None:
        raise ValueError(
            "Pass at most one of use_rep= and layer=; they select the same "
            "thing from different places."
        )

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(
                f"layer={layer!r} not found. Available layers: {_sorted_keys(adata.layers)}"
            )
        rep = adata.layers[layer]
    elif use_rep == "X":
        rep = adata.X
    elif use_rep is not None:
        if use_rep not in adata.obsm:
            raise ValueError(
                f"use_rep={use_rep!r} not found in adata.obsm. Available "
                f"keys: {_sorted_keys(adata.obsm)}. Pass use_rep='X' to cluster "
                "adata.X itself, or layer=... for a layer."
            )
        rep = adata.obsm[use_rep]
    elif "X_pca" in adata.obsm:
        rep = adata.obsm["X_pca"]
    else:
        rep = adata.X
        if adata.n_vars > 50:
            warnings.warn(
                f"No representation given and adata.obsm['X_pca'] is absent, "
                f"so CARVE is clustering adata.X directly "
                f"({adata.n_vars} features). Clustering validation on "
                "high-dimensional raw data is slow and usually not what you "
                "want; run `sc.pp.pca(adata)` first and pass "
                "use_rep='X_pca'.",
                UserWarning,
                stacklevel=3,
            )

    rep = _densify(rep)

    if n_pcs is not None:
        if n_pcs > rep.shape[1]:
            raise ValueError(
                f"n_pcs={n_pcs} exceeds the {rep.shape[1]} available "
                "components in the selected representation."
            )
        rep = rep[:, :n_pcs]

    return rep


def _sorted_keys(mapping: Any) -> list[str]:
    """Return the string keys of an AnnData mapping, sorted.

    ``adata.layers`` yields a ``None`` key standing for ``.X`` itself, so a
    bare ``sorted()`` over it raises ``TypeError``.
    """
    return sorted(k for k in mapping if k is not None)


def _densify(rep: Any) -> np.ndarray:
    """Return ``rep`` as a dense 2D float array."""
    if sparse.issparse(rep):
        return np.asarray(rep.todense())
    return np.asarray(rep)


def resolve_basis(
    adata: AnnData,
    basis: str | None = None,
    *,
    fallback: str | None = None,
) -> np.ndarray | None:
    """Resolve a 2D embedding for scatter plots.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    basis : str, optional
        Embedding name, with or without the ``X_`` prefix (``"umap"`` and
        ``"X_umap"`` are equivalent). If None, the first available of
        ``X_umap``, ``X_tsne``, ``X_draw_graph_fa``, ``X_pca`` is used.
    fallback : str, optional
        ``obsm`` key to fall back on when no preferred basis exists,
        typically the representation recorded at fit time.

    Returns
    -------
    embedding : ndarray of shape (n_obs, 2) or None
        The first two columns of the chosen embedding, or None when nothing
        suitable exists (the caller then lets the plotting layer reduce).

    Raises
    ------
    ValueError
        If a named basis is missing, or has fewer than two columns.
    """
    if basis is not None:
        key = basis if basis.startswith("X_") else f"X_{basis}"
        if key not in adata.obsm:
            raise ValueError(
                f"basis={basis!r} not found in adata.obsm (looked for "
                f"{key!r}). Available keys: {_sorted_keys(adata.obsm)}"
            )
        return _as_2d_embedding(adata.obsm[key], key)

    for key in _BASIS_PREFERENCE:
        if key in adata.obsm:
            return _as_2d_embedding(adata.obsm[key], key)

    if fallback is not None and fallback in adata.obsm:
        return _as_2d_embedding(adata.obsm[fallback], fallback)

    return None


def _as_2d_embedding(arr: Any, key: str) -> np.ndarray:
    """Return the first two columns of ``arr`` as a dense array."""
    dense = _densify(arr)
    if dense.ndim != 2 or dense.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{key!r}] has shape {dense.shape}; a basis needs at "
            "least two columns."
        )
    return dense[:, :2]


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy scalars and 0-d arrays to plain Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def params_to_uns(params: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parameter mapping into an h5ad-writable ``uns`` dict.

    Keys whose value is None are dropped.

    Parameters
    ----------
    params : dict
        Raw parameter mapping.

    Returns
    -------
    clean : dict
        Flat, string-keyed mapping of scalars, strings and arrays.
    """
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        value = _to_python_scalar(value)
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.dtype == object:
                arr = np.array([_stringify(v) for v in arr], dtype=object)
            clean[key] = arr
        elif isinstance(value, (str, bool, int, float)):
            clean[key] = value
        else:
            clean[key] = _stringify(value)
    clean["carve_version"] = _carve_version()
    return clean


def _stringify(value: Any) -> str:
    """Render ``value`` as a bounded string, mapping missing values to ""."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    try:
        if value is pd.NA or (value is not pd.NaT and pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)[:_MAX_UNS_STR]


def results_to_uns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize a results table so it survives an h5ad round trip.

    Used for both ``estimator_results_`` and ``preprocessing_results_``.

    Numeric and boolean columns pass through untouched. Every other column is
    coerced to pure ``str``: heterogeneous estimator grids leave columns mixing
    ``str`` with ``float('nan')``, which anndata's variable-length string
    writer rejects with ``TypeError`` under pandas 2.3 (see the module notes).
    The columns stay plain strings because ``groupby`` on a categorical
    ``method_id`` yields empty groups for unused categories, which the
    plotting layer cannot index.

    Parameters
    ----------
    df : pandas.DataFrame
        A fitted results table, ``estimator_results_`` or
        ``preprocessing_results_``.

    Returns
    -------
    clean : pandas.DataFrame
        Copy safe to assign into ``adata.uns``.
    """
    clean = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for col in df.columns:
        series = df[col].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            clean[col] = series.to_numpy()
        else:
            clean[col] = np.array([_stringify(v) for v in series], dtype=object)
    return clean


def results_from_uns(stored: Any) -> pd.DataFrame:
    """Rebuild a results frame read back from ``uns``.

    An h5ad round trip turns the original ``RangeIndex`` into string labels and
    may widen integer columns, so the index is reset and the identity and
    count columns are re-cast.

    Parameters
    ----------
    stored : pandas.DataFrame or dict
        Whatever ``adata.uns[key]["results"]`` holds.

    Returns
    -------
    df : pandas.DataFrame
        Frame with a clean ``RangeIndex`` and integral identity columns.
    """
    df = pd.DataFrame(stored).reset_index(drop=True)
    for col in ("config_id", "sweep_rank", "n_resamples"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def labels_to_categorical(labels: np.ndarray) -> pd.Categorical:
    """Render integer cluster labels as a scanpy-style categorical.

    Categories are ordered numerically rather than lexicographically, so that
    ``"10"`` follows ``"9"``. This matches ``sc.tl.leiden`` and keeps
    ``sc.pl.umap(color=...)`` on a discrete palette.

    Parameters
    ----------
    labels : ndarray of shape (n_obs,)
        Integer cluster assignments.

    Returns
    -------
    categorical : pandas.Categorical
        String-valued categorical with numerically sorted categories.
    """
    values = np.asarray(labels)
    categories = [str(v) for v in sorted(np.unique(values))]
    return pd.Categorical(
        [str(v) for v in values], categories=categories, ordered=False
    )
