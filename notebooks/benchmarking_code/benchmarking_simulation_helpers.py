from typing import Any, Dict, List, Tuple

import numpy as np

from carve.sim import simulate_clusters

    
def simulate_scaling(
    *,
    regime: Dict[str, Any],
    true_k: int,
    x_name: str,
    x_value: int,
    seed: int,
    base_random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a dataset for scaling experiments by varying one axis (n_total, p, or embed_dim).

    Args:
        - regime (Dict[str, Any]): Simulation regime parameters for a given k.
        - true_k (int): True number of clusters.
        - x_name (str): Scaling axis name ('n_total', 'p', or 'embed_dim').
        - x_value (int): Value to set for the chosen scaling axis.
        - seed (int): Seed offset for simulation randomness.
        - base_random_state (int): Base seed for reproducibility (default: 0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated data matrix X and cluster labels y.
    """
    if x_name not in {"n_total", "p", "embed_dim"}:
        raise ValueError("x_name must be 'n_total', 'p', or 'embed_dim'")

    n_total = int(regime.get("n_total", 500))
    p = int(regime.get("p", 100))

    if x_name == "n_total":
        n_total = int(x_value)
    elif x_name == "embed_dim":
        regime["embed_dim"] = int(x_value)
    else:
        p = int(x_value)

    X, y, _ = simulate_clusters(
        n_total=n_total,
        p=p,
        k=int(true_k),
        plotting=False,
        random_state=int(base_random_state + seed),
        **{k: v for k, v in regime.items() if k not in {"n_total", "p"}},
    )
    return X, y
    
    
def parse_difficulty_and_simulate(
    settings_by_k: Dict[int, Dict[str, List[float]]],
    other_settings: Dict,
    n_datasets: int,
    difficulty: int,
    true_k: int,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates clustered datasets based on difficulty level by interpolating settings.

    Args:
        - settings_by_k (Dict[int, Dict[str, List[float]]]): Dictionary mapping difficulty levels to simulation settings.
        - other_settings (Dict): Additional simulation settings.
        - n_datasets (int): Total number of difficulty levels/datasets.
        - difficulty (int): Difficulty index (0=easy, middle=medium, last=difficult).
        - true_k (int): True number of clusters to simulate.
        - random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated data matrix X and cluster labels y.
    """
    if difficulty == 0 or difficulty == n_datasets - 1 or difficulty == int(round(n_datasets // 2)):
        if difficulty == 0:
            level = 'easy'
        elif difficulty == n_datasets - 1:
            level = 'difficult'
        elif difficulty == int(round(n_datasets // 2)):
            level = 'medium'
            
        X, y, _ = simulate_clusters(
            k=true_k,
            plotting=False,
            random_state=random_state,
            **{k: v for k, v in settings_by_k[level].items()},
            **{k: v for k, v in other_settings.items()}
        )
        
    else:
        if difficulty > 0 and difficulty < int(round(n_datasets // 2)):
            # Linear interpolation between 'easy' and 'medium' settings
            frac = difficulty / int(round(n_datasets // 2))
            easy_settings = settings_by_k['easy']
            medium_settings = settings_by_k['medium']
            
            interpolated_settings = {}
            for key in easy_settings:
                v_easy = np.array(easy_settings[key])
                v_medium = np.array(medium_settings[key])
                interpolated = (1 - frac) * v_easy + frac * v_medium
                interpolated_settings[key] = interpolated.tolist()
                
        elif difficulty > int(round(n_datasets // 2)) and difficulty < (n_datasets - 1):
            # Linear interpolation between 'medium' and 'difficult' settings
            frac = (difficulty - int(round(n_datasets // 2))) / (n_datasets - 1 - int(round(n_datasets // 2)))
            medium_settings = settings_by_k['medium']
            difficult_settings = settings_by_k['difficult']
            
            interpolated_settings = {}
            for key in medium_settings:
                v_medium = np.array(medium_settings[key])
                v_difficult = np.array(difficult_settings[key])
                interpolated = (1 - frac) * v_medium + frac * v_difficult
                interpolated_settings[key] = interpolated.tolist()
                
        else:
            raise ValueError(f"Got difficulty={difficulty} but expected 0, {n_datasets - 1}, {int(round(n_datasets // 2))}, or a value in between those.")
        
        if "noise_dims" in interpolated_settings:
            noise_dims = interpolated_settings["noise_dims"]
            interpolated_settings["noise_dims"] = int(np.floor(noise_dims + 0.5))
            
        X, y, _ = simulate_clusters(
            k=true_k,
            plotting=False,
            random_state=random_state,
            **{k: v for k, v in interpolated_settings.items()},
            **{k: v for k, v in other_settings.items()}
        )
    
    return X, y

