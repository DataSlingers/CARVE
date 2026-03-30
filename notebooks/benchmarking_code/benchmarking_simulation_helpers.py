"""Simulation helpers for benchmarking experiments.

Provides functions that interpolate between calibrated anchor settings
and call ``simulate_clusters`` to generate synthetic data.
"""

from typing import Any, Dict, Tuple

import numpy as np

from carve.sim import simulate_clusters


# ── Generic 3-anchor interpolation ────────────────────────────────────────────


def _interpolate_settings(
    anchors: Dict[str, Dict[str, Any]],
    anchor_order: Tuple[str, str, str],
    stage: int,
    total_stages: int,
) -> dict:
    """Piecewise-linear interpolation between three anchor settings dicts.

    Given three ordered anchors (e.g. easy/medium/hard or start/middle/end),
    returns the interpolated settings dict for a given *stage* index.

    * ``stage == 0``                  → first anchor (verbatim)
    * ``stage == total_stages // 2``  → second anchor (verbatim)
    * ``stage == total_stages - 1``   → third anchor (verbatim)
    * intermediate stages             → linear blend of the surrounding anchors

    Args:
        anchors: Mapping from anchor name to parameter dict.
        anchor_order: 3-tuple of anchor names in order (first, middle, last).
        stage: Current stage index (0-based).
        total_stages: Total number of stages.

    Returns:
        Parameter dict for the given stage.
    """
    if total_stages < 3:
        raise ValueError(f"total_stages must be >= 3, got {total_stages}")
    if not (0 <= stage < total_stages):
        raise ValueError(
            f"stage must be in [0, {total_stages - 1}], got {stage}"
        )

    mid = total_stages // 2
    a_name, b_name, c_name = anchor_order

    # Exact anchor positions
    if stage == 0:
        return dict(anchors[a_name])
    if stage == mid:
        return dict(anchors[b_name])
    if stage == total_stages - 1:
        return dict(anchors[c_name])

    # Interpolation between two surrounding anchors
    if stage < mid:
        frac = stage / mid
        lo, hi = anchors[a_name], anchors[b_name]
    else:
        frac = (stage - mid) / (total_stages - 1 - mid)
        lo, hi = anchors[b_name], anchors[c_name]

    interpolated = {}
    for key in lo:
        v_lo = np.array(lo[key])
        v_hi = np.array(hi[key])
        interpolated[key] = ((1 - frac) * v_lo + frac * v_hi).tolist()
    return interpolated


# ── Public simulation wrappers ────────────────────────────────────────────────


def parse_difficulty_and_simulate(
    settings_by_k: Dict[str, Dict[str, Any]],
    other_settings: Dict,
    difficulty_levels: int,
    difficulty_level: int,
    true_cluster_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate clustered data at a given difficulty level.

    Interpolates between the ``easy`` / ``medium`` / ``hard`` anchors in
    *settings_by_k* and calls :func:`simulate_clusters`.

    Args:
        settings_by_k: Mapping from difficulty anchor name to parameter dict.
        other_settings: Additional keyword arguments forwarded to ``simulate_clusters``.
        difficulty_levels: Total number of difficulty levels.
        difficulty_level: Current difficulty index (0 = easy, last = hard).
        true_cluster_count: Number of clusters to simulate.
        seed: Random seed.

    Returns:
        Tuple of (X, y) — data matrix and ground-truth labels.
    """
    settings = _interpolate_settings(
        settings_by_k,
        ("easy", "medium", "hard"),
        difficulty_level,
        difficulty_levels,
    )

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
    settings_by_k: Dict,
    other_settings: Dict,
    total_stages: int,
    stage: int,
    true_cluster_count: int,
    axis_name: str,
    axis_value: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate data for a scaling benchmark at a given axis value.

    Interpolates between the ``start`` / ``middle`` / ``end`` anchors in
    *settings_by_k*, overrides the scaling axis with *axis_value*, and
    calls :func:`simulate_clusters`.

    Args:
        settings_by_k: Mapping from stage anchor name to parameter dict.
        other_settings: Additional keyword arguments forwarded to ``simulate_clusters``.
        total_stages: Total number of stages in the benchmark.
        stage: Current stage index (0-based).
        true_cluster_count: Number of clusters to simulate.
        axis_name: Scaling axis ('n_total', 'p', or 'embed_dim').
        axis_value: Value for the scaling axis at this stage.
        random_state: Random seed.

    Returns:
        Tuple of (X, y) — data matrix and ground-truth labels.
    """
    if axis_name not in {"n_total", "p", "embed_dim"}:
        raise ValueError("axis_name must be 'n_total', 'p', or 'embed_dim'")

    settings = _interpolate_settings(
        settings_by_k,
        ("start", "middle", "end"),
        stage,
        total_stages,
    )

    # Read defaults from other_settings, then override the scaling axis
    n_total = int(other_settings.get("n_total", 500))
    p = int(other_settings.get("p", 50))
    embed_dim = int(other_settings.get("embed_dim", 64))

    if axis_name == "n_total":
        n_total = int(axis_value)
    elif axis_name == "p":
        p = int(axis_value)
    elif axis_name == "embed_dim":
        embed_dim = int(axis_value)

    # Filter axis keys from other_settings to prevent duplicate keyword args
    filtered_other = {
        k: v for k, v in other_settings.items() if k not in ("n_total", "p", "embed_dim")
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
