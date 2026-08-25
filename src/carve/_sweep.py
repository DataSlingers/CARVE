"""Sweep-axis abstraction for CARVE.

CARVE evaluates clustering robustness along a one-dimensional axis of
increasing granularity. Historically that axis was always ``n_clusters``.
This module generalizes it so that resolution-style parameters (Leiden and
Louvain ``resolution``, HDBSCAN ``min_cluster_size``) drive the same
machinery.

A run uses exactly one sweep parameter. Comparing k-based estimators with
resolution-based estimators inside a single run is not supported: their
metrics are not commensurable across axes.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._types import GridSpec
from ._utils import _coerce_n_clusters

# Bookkeeping columns CARVE adds to describe the sweep axis. These are not
# estimator hyperparameters and must be excluded from estimator labels and
# from plot grouping keys.
SWEEP_META_COLS = frozenset(
    {
        "sweep_param",
        "sweep_value",
        "sweep_rank",
        "n_clusters_observed",
        "n_clusters_observed_se",
        "noise_fraction",
    }
)

# Columns that identify a row. ``config_id`` keys the per-configuration
# artifact containers; ``method_id`` groups rows into curves.
# Like SWEEP_META_COLS these are not estimator hyperparameters.
IDENTITY_COLS = frozenset({"config_id", "method_id", "method_label"})

# param -> (finer_is_larger, fixes_k, axis label)
SWEEP_REGISTRY: dict[str, tuple[bool, bool, str]] = {
    "n_clusters": (True, True, "Number of Clusters (k)"),
    "resolution": (True, False, "Resolution"),
    "min_cluster_size": (False, False, "Minimum Cluster Size"),
}


@dataclass(frozen=True)
class SweepSpec:
    """Description of the hyperparameter swept by a CARVE run.

    Attributes
    ----------
    param : str
        Name of the estimator hyperparameter being swept, e.g.
        ``"n_clusters"``, ``"resolution"``, or ``"min_cluster_size"``.
    values : ndarray of shape (n_values,)
        Sorted values to evaluate.
    finer_is_larger : bool
        Whether larger values of ``param`` yield more clusters. True for
        ``n_clusters`` and ``resolution``; False for ``min_cluster_size``.
    fixes_k : bool
        Whether ``param`` pins the number of clusters exactly. Only true
        for ``n_clusters``.
    label : str
        Axis label used by plots and console output.
    """

    param: str
    values: np.ndarray
    finer_is_larger: bool
    fixes_k: bool
    label: str

    @property
    def is_k_mode(self) -> bool:
        """Whether this run sweeps ``n_clusters`` (original mode)."""
        return self.param == "n_clusters"

    def ranks(self) -> np.ndarray:
        """Return the coarse-to-fine rank of a single sweep value."""
        keys = self.values if self.finer_is_larger else -self.values
        order = np.argsort(keys)
        ranks = np.empty(self.values.size, dtype=int)
        ranks[order] = np.arange(self.values.size)
        return ranks

    def rank_of(self, value: Any) -> int:
        """Return the coarse-to-fine rank of a single sweep value."""
        matches = np.flatnonzero(np.isclose(self.values.astype(float), float(value)))

        if matches.size == 0:
            raise ValueError(
                f"{value!r} is not among the swept {self.param} values "
                f"{self.values.tolist()}."
            )
        return int(self.ranks()[matches[0]])


def coerce_sweep_values(value: Any, param: str) -> np.ndarray:
    """Coerce a sweep specification to a sorted 1-D array.

    Parameters
    ----------
    value : scalar or array-like
        Values to evaluate along the sweep axis.
    param : str
        Name of the sweep parameter, used to pick validation rules.

    Returns
    -------
    values : ndarray of shape (n_values,)
        Sorted, validated sweep values.
    """
    if param == "n_clusters":
        return _coerce_n_clusters(value)

    if param == "min_cluster_size":
        arr = np.atleast_1d(np.asarray(value))

        # Catch non-integer types and values < 2.
        # (HDBSCAN's min_cluster_size must be an integer >= 2.)
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("min_cluster_size values must be integers.")
        if np.any(arr < 2):
            raise ValueError("All min_cluster_size values must be >= 2.")

        return np.sort(arr.astype(int, copy=False))

    # Case: param == "resolution" or any other float-valued sweep parameter.
    arr = np.atleast_1d(np.asarray(value, dtype=float))

    # Catch non-scalar, non-1D arrays and values <= 0.
    if arr.ndim != 1:
        raise ValueError(f"{param} must be a scalar or a 1D array.")
    if np.any(arr <= 0):
        raise ValueError(f"All {param} values must be > 0.")

    return np.sort(arr)


def resolve_sweep(
    *,
    n_clusters: Any = None,
    resolution: Any = None,
    sweep: str | None = None,
    sweep_values: Any = None,
    finer_is_larger: bool | None = None,
) -> SweepSpec:
    """Resolve CARVE constructor arguments into a single :class:`SweepSpec`.

    Parameters
    ----------
    n_clusters : int or ndarray, optional
        Cluster counts, used when the sweep parameter is ``"n_clusters"``.
    resolution : float or ndarray, optional
        Resolution values. Supplying this switches the run to resolution
        mode unless ``sweep`` says otherwise.
    sweep : str, optional
        Explicit sweep parameter name. Overrides inference.
    sweep_values : array-like, optional
        Explicit sweep values, required for parameters other than
        ``"n_clusters"`` and ``"resolution"``.
    finer_is_larger : bool, optional
        Whether larger values yield more clusters. Required for sweep
        parameters that are not in :data:`SWEEP_REGISTRY`.

    Returns
    -------
    spec : SweepSpec
        The resolved sweep axis.
    """
    if resolution is not None and sweep is not None and sweep != "resolution":
        raise ValueError(
            f"resolution= was given but sweep={sweep!r}. Pass sweep_values= "
            "instead, or drop resolution=."
        )

    if sweep is None:
        sweep = "resolution" if resolution is not None else "n_clusters"

    if sweep_values is None:
        if sweep == "n_clusters":
            sweep_values = n_clusters
        elif sweep == "resolution":
            sweep_values = resolution

    if sweep_values is None:
        raise ValueError(
            f"No values supplied for sweep parameter {sweep!r}. Pass "
            "sweep_values=, or supply estimator grids that contain it."
        )

    values = coerce_sweep_values(sweep_values, sweep)
    known = SWEEP_REGISTRY.get(sweep)

    if known is None:
        if finer_is_larger is None:
            raise ValueError(
                f"Unknown sweep parameter {sweep!r}. Pass finer_is_larger= to "
                "declare whether larger values yield more clusters."
            )

        default_finer, fixes_k, label = (
            bool(finer_is_larger),
            False,
            sweep.replace("_", " ").title(),
        )

    else:
        default_finer, fixes_k, label = known

    return SweepSpec(
        param=sweep,
        values=values,
        finer_is_larger=(
            default_finer if finer_is_larger is None else bool(finer_is_larger)
        ),
        fixes_k=fixes_k,
        label=label,
    )


def infer_sweep_param(estimator_grids: list[GridSpec]) -> str | None:
    """Infer the sweep parameter from user-supplied grids, if unambiguous.

    Returns
    -------
    param : str or None
        The single known sweep parameter present in the grids, or None if
        none is present.

    Raises
    ------
    ValueError
        If the grids reference more than one known sweep parameter.
    """
    present = [
        {p for p in set(grid) & set(SWEEP_REGISTRY) if len(grid[p]) > 1}
        for _, grid in estimator_grids
    ]
    union: set[str] = set().union(*present) if present else set()

    if not union:
        return None
    if len(union) > 1:
        raise ValueError(
            f"Estimator grids sweep more than one parameter ({sorted(union)}). "
            "CARVE evaluates exactly one sweep axis per run; k-based and "
            "resolution-based estimators cannot be compared in the same run."
        )
    return union.pop()


def grid_sweep_values(
    estimator_grids: list[GridSpec],
    param: str,
) -> list[Any] | None:
    """Return the values of param from the first grid that carries it."""
    for _, grid in estimator_grids:
        if param in grid:
            return list(grid[param])
    return None


def validate_grids(
    estimator_grids: list[GridSpec],
    sweep: SweepSpec,
) -> np.ndarray:
    """Check user-supplied grids against the resolved sweep axis.

    Every grid must sweep the same parameter over the same values, and no
    grid may reference a competing sweep parameter.

    Returns
    -------
    values : ndarray
        The sweep values actually present in the grids.
    """
    if not estimator_grids:
        raise ValueError("estimator_param_grids must not be empty.")

    other_params = set(SWEEP_REGISTRY) - {sweep.param}

    for est_class, grid in estimator_grids:
        if sweep.param not in grid:
            raise ValueError(
                f"Estimator grid for {est_class.__name__} does not contain the "
                f"sweep parameter {sweep.param!r}. All grids in a run must "
                "sweep the same parameter."
            )

        clash = {p for p in other_params.intersection(grid) if len(grid[p]) > 1}
        if clash:
            raise ValueError(
                f"Estimator grid for {est_class.__name__} mixes sweep "
                f"parameters {sweep.param!r} and {sorted(clash)}. CARVE "
                "evaluates exactly one sweep axis per run; k-based and "
                "resolution-based estimators cannot be compared in the same "
                "run."
            )

    reference = np.asarray(estimator_grids[0][1][sweep.param])
    for est_class, grid in estimator_grids[1:]:
        if not np.array_equal(reference, np.asarray(grid[sweep.param])):
            raise ValueError(
                f"All estimator parameter grids must contain the same "
                f"{sweep.param} values."
            )

    return coerce_sweep_values(reference, sweep.param)


# ---------------------------------------------------------------------- #
#  Results-table helpers                                                 #
# ---------------------------------------------------------------------- #
def sweep_param_name(results_df: pd.DataFrame) -> str:
    """Return the swept hyperparameter name recorded in a results table."""
    return str(results_df["sweep_param"].iloc[0])


# def sweep_axis_col(results_df: pd.DataFrame) -> str:
#     """Return the column holding the sweep value (the plot x-axis)."""
#     return "sweep_value"

# def sweep_order_col(results_df: pd.DataFrame) -> str:
#     """Return the column ordering configurations coarse -> fine."""
#     return "sweep_rank"


def sweep_exclude_cols(results_df: pd.DataFrame) -> set[str]:
    """Columns describing the sweep axis or row identity, not the estimator."""
    return set(SWEEP_META_COLS) | set(IDENTITY_COLS) | {sweep_param_name(results_df)}


def sweep_axis_label(results_df: pd.DataFrame) -> str:
    """Human-readable axis label for the sweep parameter."""
    param = sweep_param_name(results_df)
    known = SWEEP_REGISTRY.get(param)
    return known[2] if known else param.replace("_", " ").title()


def observed_k_series(results_df: pd.DataFrame) -> pd.Series:
    """Per-row cluster count from ``n_clusters_observed``."""
    return results_df["n_clusters_observed"].round()


def observed_k(row: pd.Series) -> int:
    """Return the number of clusters associated with a single results row."""
    return int(round(float(row["n_clusters_observed"])))


# ---------------------------------------------------------------------- #
#  Row identity                                                          #
#                                                                        #
#  config_id joins a results row to its consensus matrix and score       #
#  arrays; method_id groups rows into curves.                            #
# ---------------------------------------------------------------------- #
def _format_param_value(val: Any) -> str:
    """Format one hyperparameter value for a method label.

    Mirrors the formatting in ``_plotting._build_estimator_label``.

    Parameters
    ----------
    val : Any
        Hyperparameter value.

    Returns
    -------
    text : str
        Formatted value.
    """
    if isinstance(val, bool):
        return f"{val}"
    if isinstance(val, (int, np.integer)):
        return f"{val}"
    if isinstance(val, (float, np.floating)):
        return f"{int(val)}" if val == int(val) else f"{val:.3g}"
    return f"{val}"


def format_method_label(est_name: str, params: dict, sweep_param: str) -> str:
    """Build a human-readable label for one method.

    A method is an estimator together with every hyperparameter except
    the swept one, i.e. exactly one curve in a metric-over-sweep plot.

    Parameters
    ----------
    est_name : str
        Class name of the estimator.
    params : dict
        Full hyperparameter combination for one configuration.
    sweep_param : str
        Name of the swept hyperparameter, omitted from the label.

    Returns
    -------
    label : str
        Comma-separated label, e.g. ``"AgglomerativeClustering, linkage=ward"``.
    """
    parts = [est_name]
    parts += [
        f"{key}={_format_param_value(params[key])}"
        for key in sorted(params)
        if key != sweep_param
    ]

    return ", ".join(parts)


class MethodIds:
    """Assign method identifiers during a run.

    Two configurations that differ only in the swept hyperparameter share a
    method id; they are two points on one curve.

    Parameters
    ----------
    sweep_param : str
        Name of the swept hyperparameter, excluded from the method key.

    Examples
    --------
    >>> ids = MethodIds("n_clusters")
    >>> ids.assign("KMeans", {"n_clusters": 2})[0]
    'm0'
    >>> ids.assign("KMeans", {"n_clusters": 3})[0]
    'm0'
    """

    def __init__(self, sweep_param: str):
        self._sweep_param = sweep_param
        self._seen: dict[tuple, tuple[str, str]] = {}

    def assign(self, est_name: str, params: dict) -> tuple[str, str]:
        """Return the ``(method_id, method_label)`` pair for a configuration.

        Parameters
        ----------
        est_name : str
            Class name of the estimator.
        params : dict
            Full hyperparameter combination for one configuration.

        Returns
        -------
        method_id : str
            ``"m<n>"`` identifier, numbered in first-seen order.
        method_label : str
            Human-readable label for the same method.
        """
        key = (
            est_name,
            tuple(
                sorted(
                    # repr(): user-supplied grids may hold unhashable values.
                    (k, repr(v))
                    for k, v in params.items()
                    if k != self._sweep_param
                )
            ),
        )

        if key not in self._seen:
            self._seen[key] = (
                f"m{len(self._seen)}",
                format_method_label(est_name, params, self._sweep_param),
            )

        return self._seen[key]


def config_id_of(row: pd.Series) -> int:
    """Return the artifact-container key for a results row.

    ``config_id`` is a column, so it travels with the row through
    ``sort_values``, ``reset_index``, ``query`` and ``merge``. Never use the
    row's index label for this.

    Parameters
    ----------
    row : pandas.Series
        A row of ``estimator_results_``.

    Returns
    -------
    config_id : int
        Index into ``consensus_matrices_`` and the sample-level score
        arrays.
    """
    return int(row["config_id"])


# def method_group_cols(results_df: pd.DataFrame) -> list[str]:
#     """Return the columns that identify one curve in a results table.

#     Parameters
#     ----------
#     results_df : pandas.DataFrame
#         Results table.

#     Returns
#     -------
#     group_cols : list of str
#         ``["method_id"]``.
#     """
#     return ["method_id"]
