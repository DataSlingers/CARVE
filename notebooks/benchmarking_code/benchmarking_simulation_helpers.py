"""Simulation helpers for benchmarking experiments.

Selects one of three calibrated anchor settings (easy/medium/hard for
difficulty experiments, start/middle/end for scaling experiments) and
calls ``simulate_clusters``.
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Any, Dict, Tuple

import numpy as np

from carve.sim import simulate_clusters

from benchmarking_config import SCALING_CONSTANTS


# =============================================================================
# Anchor label constants
# =============================================================================
DIFFICULTY_LABELS: Tuple[str, str, str] = ("easy", "medium", "hard")
STAGE_LABELS: Tuple[str, str, str] = ("start", "middle", "end")


# =============================================================================
# Public simulation entry points
# =============================================================================
def parse_difficulty_and_simulate(
    anchor_settings: Dict[str, Dict[str, Any]],
    other_settings: Dict,
    difficulty: str,
    true_cluster_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate clustered data at the given difficulty anchor.

    Args:
        anchor_settings: Mapping ``{"easy": {...}, "medium": {...}, "hard": {...}}``
            of calibrated parameter dicts.
        other_settings: Additional keyword arguments forwarded to ``simulate_clusters``.
        difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``.
        true_cluster_count: Number of clusters to simulate.
        seed: Random seed.

    Returns:
        Tuple of (X, y) — data matrix and ground-truth labels.
    """
    if difficulty not in DIFFICULTY_LABELS:
        raise ValueError(
            f"difficulty must be one of {DIFFICULTY_LABELS}, got {difficulty!r}"
        )
    settings = dict(anchor_settings[difficulty])

    X, y = simulate_clusters(
        k=true_cluster_count,
        plotting=False,
        random_state=seed,
        **settings,
        **other_settings,
    )
    return X, y


def parse_range_and_simulate(
    *,
    anchor_settings: Dict[str, Dict[str, Any]],
    other_settings: Dict,
    stage_label: str,
    true_cluster_count: int,
    axis_name: str,
    axis_value: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate data for a scaling benchmark at the given stage anchor.

    Args:
        anchor_settings: Mapping ``{"start": {...}, "middle": {...}, "end": {...}}``
            of calibrated parameter dicts.
        other_settings: Additional keyword arguments forwarded to ``simulate_clusters``.
        stage_label: One of ``"start"``, ``"middle"``, ``"end"``.
        true_cluster_count: Number of clusters to simulate.
        axis_name: Scaling axis (``"n_total"``, ``"p"``, or ``"embed_dim"``).
        axis_value: Value for the scaling axis at this stage.
        random_state: Random seed.

    Returns:
        Tuple of (X, y) — data matrix and ground-truth labels.
    """
    if axis_name not in {"n_total", "p", "embed_dim"}:
        raise ValueError("axis_name must be 'n_total', 'p', or 'embed_dim'")
    if stage_label not in STAGE_LABELS:
        raise ValueError(
            f"stage_label must be one of {STAGE_LABELS}, got {stage_label!r}"
        )
    settings = dict(anchor_settings[stage_label])

    n_total = SCALING_CONSTANTS["n_total"]
    p = SCALING_CONSTANTS["p"]
    embed_dim = SCALING_CONSTANTS["embed_dim"]

    if axis_name == "n_total":
        n_total = int(axis_value)
    elif axis_name == "p":
        p = int(axis_value)
    elif axis_name == "embed_dim":
        embed_dim = int(axis_value)

    filtered_other = {
        k: v
        for k, v in other_settings.items()
        if k not in ("n_total", "p", "embed_dim")
    }

    return simulate_clusters(
        n_total=n_total,
        p=p,
        embed_dim=embed_dim,
        k=int(true_cluster_count),
        plotting=False,
        random_state=random_state,
        **settings,
        **filtered_other,
    )
