"""Turn a Scenario and one axis point into simulated data.

This replaces parse_difficulty_and_simulate and parse_range_and_simulate,
which differed only in how they resolved the swept parameter. Scenario.
sim_kwargs now does that, so one function covers both.
"""

from typing import Any

import numpy as np

from ._types import Scenario


def simulate(
    scenario: Scenario, axis_value: Any, axis_label: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one dataset at one point on a scenario's axis.

    Parameters
    ----------
    scenario : Scenario
        The experiment definition.
    axis_value : Any
        The axis value for this cell. Used as a simulator argument when the
        axis name is a simulator keyword, and only to pick the anchor
        otherwise.
    axis_label : str
        The axis label for this cell, selecting the anchor.
    seed : int
        The fully derived benchmark seed, not the raw loop index.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    from carve.sim import simulate_clusters

    kwargs = scenario.sim_kwargs(axis_value=axis_value, axis_label=axis_label)
    X, y = simulate_clusters(
        k=scenario.k_star,
        plotting=False,
        random_state=int(seed),
        **kwargs,
    )
    return np.asarray(X), np.asarray(y)
