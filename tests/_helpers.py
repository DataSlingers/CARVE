"""Helpers shared across the carve test files.

Not a conftest: conftest.py holds fixtures and hooks, and pytest discourages
importing it as a module. Anything a test file imports by name lives here.
"""

import pandas as pd

from carve._sweep import resolve_sweep


def with_sweep_cols(
    df: pd.DataFrame,
    *,
    param: str,
    method_label: str,
    observed: list[float],
    observed_se: list[float] | None = None,
    noise_fraction: list[float] | None = None,
    finer_is_larger: bool | None = None,
) -> pd.DataFrame:
    """Attach the bookkeeping columns the runner writes onto every record.

    Adds the three identity columns (``config_id``, ``method_id``,
    ``method_label``) and the six sweep columns (``sweep_param``,
    ``sweep_value``, ``sweep_rank``, ``n_clusters_observed``,
    ``n_clusters_observed_se``, ``noise_fraction``).

    ``sweep_rank`` is taken from a real :class:`~carve._sweep.SweepSpec`
    rather than from an argsort of the frame: rank is a property of the
    swept axis, not of row order.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame carrying at least the column named by *param*.
    param : str
        Name of the swept hyperparameter.
    method_label : str
        Human-readable label shared by every row (one curve).
    observed : list of float
        Mean number of clusters observed per row.
    observed_se : list of float, optional
        Standard error of *observed*. Defaults to zeros.
    noise_fraction : list of float, optional
        Mean noise fraction per row. Defaults to zeros.
    finer_is_larger : bool, optional
        Passed through to ``resolve_sweep`` for parameters outside the
        registry.

    Returns
    -------
    df : pandas.DataFrame
        A copy of *df* with the nine columns prepended.
    """
    df = df.copy()
    values = df[param].to_numpy()

    spec = resolve_sweep(
        sweep=param, sweep_values=values, finer_is_larger=finer_is_larger
    )

    n = len(df)
    df.insert(0, "config_id", list(range(n)))
    df.insert(1, "method_id", ["m0"] * n)
    df.insert(2, "method_label", [method_label] * n)
    df["sweep_param"] = param
    df["sweep_value"] = values
    df["sweep_rank"] = [spec.rank_of(v) for v in values]
    df["n_clusters_observed"] = [float(v) for v in observed]
    df["n_clusters_observed_se"] = (
        [0.0] * n if observed_se is None else list(observed_se)
    )
    df["noise_fraction"] = [0.0] * n if noise_fraction is None else list(noise_fraction)
    return df
