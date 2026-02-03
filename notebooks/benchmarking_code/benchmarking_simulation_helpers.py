from typing import Any, Dict, List, Tuple

import numpy as np

from carve.sim import simulate_clusters
    
    
def parse_difficulty_and_simulate(
    settings_by_k: Dict[int, Dict[str, List[float]]],
    other_settings: Dict,
    difficulty_levels: int,
    difficulty_index: int,
    true_cluster_count: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates clustered datasets based on difficulty level by interpolating settings.

    Args:
        - settings_by_k (Dict[int, Dict[str, List[float]]]): Dictionary mapping difficulty levels to simulation settings.
        - other_settings (Dict): Additional simulation settings.
        - difficulty_levels (int): Total number of difficulty levels/datasets.
        - difficulty_index (int): Difficulty index (0=easy, middle=medium, last=difficult).
        - true_cluster_count (int): True number of clusters to simulate.
        - seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated data matrix X and cluster labels y.
    """
    if (
        difficulty_index == 0
        or difficulty_index == difficulty_levels - 1
        or difficulty_index == int(round(difficulty_levels // 2))
    ):
        if difficulty_index == 0:
            level = 'easy'
        elif difficulty_index == difficulty_levels - 1:
            level = 'hard'
        elif difficulty_index == int(round(difficulty_levels // 2)):
            level = 'medium'
            
        X, y = simulate_clusters(
            k=true_cluster_count,
            plotting=False,
            random_state=seed,
            **{k: v for k, v in settings_by_k[level].items()},
            **{k: v for k, v in other_settings.items()}
        )
        
    else:
        if difficulty_index > 0 and difficulty_index < int(round(difficulty_levels // 2)):
            # Linear interpolation between 'easy' and 'medium' settings
            frac = difficulty_index / int(round(difficulty_levels // 2))
            easy_settings = settings_by_k['easy']
            medium_settings = settings_by_k['medium']
            
            interpolated_settings = {}
            for key in easy_settings:
                v_easy = np.array(easy_settings[key])
                v_medium = np.array(medium_settings[key])
                interpolated = (1 - frac) * v_easy + frac * v_medium
                interpolated_settings[key] = interpolated.tolist()
                
        elif (
            difficulty_index > int(round(difficulty_levels // 2))
            and difficulty_index < (difficulty_levels - 1)
        ):
            # Linear interpolation between 'medium' and 'difficult' settings
            frac = (
                (difficulty_index - int(round(difficulty_levels // 2)))
                / (difficulty_levels - 1 - int(round(difficulty_levels // 2)))
            )
            medium_settings = settings_by_k['medium']
            difficult_settings = settings_by_k['hard']
            
            interpolated_settings = {}
            for key in medium_settings:
                v_medium = np.array(medium_settings[key])
                v_difficult = np.array(difficult_settings[key])
                interpolated = (1 - frac) * v_medium + frac * v_difficult
                interpolated_settings[key] = interpolated.tolist()
                
        else:
            raise ValueError(
                f"Got difficulty_index={difficulty_index} but expected 0, "
                f"{difficulty_levels - 1}, {int(round(difficulty_levels // 2))}, "
                "or a value in between those."
            )
        
        if "noise_dims" in interpolated_settings:
            interpolated_settings["noise_dims"] = int(np.floor(interpolated_settings["noise_dims"] + 0.5))
            
        X, y = simulate_clusters(
            k=true_cluster_count,
            plotting=False,
            random_state=seed,
            **{k: v for k, v in interpolated_settings.items()},
            **{k: v for k, v in other_settings.items()}
        )
    
    return X, y

