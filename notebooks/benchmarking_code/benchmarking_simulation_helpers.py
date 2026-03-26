from typing import Any, Dict, List, Tuple

import numpy as np
from torch import seed

from carve.sim import simulate_clusters


def parse_difficulty_and_simulate(
    settings_by_k: Dict[int, Dict[str, List[float]]],
    other_settings: Dict,
    difficulty_levels: int,
    difficulty_level: int,
    true_cluster_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates clustered datasets based on difficulty level by interpolating settings.

    Args:
        - settings_by_k (Dict[int, Dict[str, List[float]]]): Dictionary mapping difficulty levels to simulation settings.
        - other_settings (Dict): Additional simulation settings.
        - difficulty_levels (int): Total number of difficulty levels/datasets.
        - difficulty_level (int): Difficulty level (0=easy, middle=medium, last=difficult).
        - true_cluster_count (int): True number of clusters to simulate.
        - seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated data matrix X and cluster labels y.
    """
    if (
        difficulty_level == 0
        or difficulty_level == difficulty_levels - 1
        or difficulty_level == int(round(difficulty_levels // 2))
    ):
        if difficulty_level == 0:
            level = "easy"
        elif difficulty_level == difficulty_levels - 1:
            level = "hard"
        elif difficulty_level == int(round(difficulty_levels // 2)):
            level = "medium"

        X, y = simulate_clusters(
            k=true_cluster_count,
            plotting=False,
            random_state=seed,
            **settings_by_k[level],
            **other_settings,
        )

    else:
        if difficulty_level > 0 and difficulty_level < int(
            round(difficulty_levels // 2)
        ):
            # Linear interpolation between 'easy' and 'medium' settings
            frac = difficulty_level / int(round(difficulty_levels // 2))
            easy_settings = settings_by_k["easy"]
            medium_settings = settings_by_k["medium"]

            interpolated_settings = {}
            for key in easy_settings:
                v_easy = np.array(easy_settings[key])
                v_medium = np.array(medium_settings[key])
                interpolated = (1 - frac) * v_easy + frac * v_medium
                interpolated_settings[key] = interpolated.tolist()

        elif difficulty_level > int(
            round(difficulty_levels // 2)
        ) and difficulty_level < (difficulty_levels - 1):
            # Linear interpolation between 'medium' and 'difficult' settings
            frac = (difficulty_level - int(round(difficulty_levels // 2))) / (
                difficulty_levels - 1 - int(round(difficulty_levels // 2))
            )
            medium_settings = settings_by_k["medium"]
            difficult_settings = settings_by_k["hard"]

            interpolated_settings = {}
            for key in medium_settings:
                v_medium = np.array(medium_settings[key])
                v_difficult = np.array(difficult_settings[key])
                interpolated = (1 - frac) * v_medium + frac * v_difficult
                interpolated_settings[key] = interpolated.tolist()

        else:
            raise ValueError(
                f"Got difficulty_level={difficulty_level} but expected 0, "
                f"{difficulty_levels - 1}, {int(round(difficulty_levels // 2))}, "
                "or a value in between those."
            )

        X, y = simulate_clusters(
            k=true_cluster_count,
            plotting=False,
            random_state=seed,
            **interpolated_settings,
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
    """
    Simulates a dataset for scaling experiments by varying one axis (n_total, p, or embed_dim).

    Args:
        - settings_by_k (Dict): Simulation regime settings keyed by true_k.
        - other_settings (Dict): Other simulation settings.
        - total_stages (int): Total number of stages in the benchmark.
        - stage (int): Current stage index (0-based).
        - true_cluster_count (int): True number of clusters.
        - axis_name (str): Scaling axis name ('n_total', 'p', or 'embed_dim').
        - axis_value (int): Value to set for the chosen scaling axis.
        - base_random_state (int): Base seed for reproducibility (default: 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated data matrix X and cluster labels y.
    """
    if axis_name not in {"n_total", "p", "embed_dim"}:
        raise ValueError("axis_name must be 'n_total', 'p', or 'embed_dim'")

    n_total = int(other_settings.get("n_total", 500))
    p = int(other_settings.get("p", 50))
    embed_dim = int(other_settings.get("embed_dim", 64))

    if axis_name == "n_total":
        n_total = int(axis_value)
    elif axis_name == "p":
        p = int(axis_value)
    elif axis_name == "embed_dim":
        embed_dim = int(axis_value)
    else:
        raise ValueError("axis_name must be 'n_total', 'p', or 'embed_dim'")

    if (
        stage == 0
        or stage == total_stages - 1
        or stage == int(round(total_stages // 2))
    ):
        if stage == 0:
            level = "start"
        elif stage == total_stages - 1:
            level = "end"
        elif stage == int(round(total_stages // 2)):
            level = "middle"

        return simulate_clusters(
            n_total=n_total,
            p=p,
            embed_dim=embed_dim,
            k=int(true_cluster_count),
            plotting=False,
            random_state=random_state,
            **settings_by_k[level],
            **other_settings,
        )

    else:
        if stage > 0 and stage < int(round(total_stages // 2)):
            # Linear interpolation between 'start' and 'middle' settings
            frac = stage / int(round(total_stages // 2))
            start_settings = settings_by_k["start"]
            middle_settings = settings_by_k["middle"]

            interpolated_settings = {}
            for key in start_settings:
                v_start = np.array(start_settings[key])
                v_middle = np.array(middle_settings[key])
                interpolated = (1 - frac) * v_start + frac * v_middle
                interpolated_settings[key] = interpolated.tolist()

        elif stage > int(round(total_stages // 2)) and stage < (total_stages - 1):
            # Linear interpolation between 'middle' and 'end' settings
            frac = (stage - int(round(total_stages // 2))) / (
                total_stages - 1 - int(round(total_stages // 2))
            )
            middle_settings = settings_by_k["middle"]
            end_settings = settings_by_k["end"]

            interpolated_settings = {}
            for key in middle_settings:
                v_middle = np.array(middle_settings[key])
                v_end = np.array(end_settings[key])
                interpolated = (1 - frac) * v_middle + frac * v_end
                interpolated_settings[key] = interpolated.tolist()

        else:
            raise ValueError(
                f"Got stage={stage} but expected 0, "
                f"{total_stages - 1}, {int(round(total_stages // 2))}, "
                "or a value in between those."
            )

        return simulate_clusters(
            n_total=n_total,
            p=p,
            embed_dim=embed_dim,
            k=int(true_cluster_count),
            plotting=False,
            random_state=random_state,
            **interpolated_settings,
            **other_settings,
        )
