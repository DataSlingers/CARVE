import random
from typing import Any, Callable, Dict, List, Tuple, Union
from sklearn.base import TransformerMixin
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

PreprocSpec = Tuple[Callable[..., TransformerMixin], Dict[str, List[Any]]]
PreprocSpecWithName = Tuple[Callable[..., TransformerMixin], str, Dict[str, List[Any]]]
PreprocOption = Union[PreprocSpec, PreprocSpecWithName]


def create_pipeline(
    random_preprocess: bool, 
    norm_options: List[PreprocOption], 
    dr_options: List[PreprocOption], 
    seed: int
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any], str, str]:
    if random_preprocess:
        pipeline, norm_params, dr_params, norm_name, dr_name = draw_random_pipeline(norm_options, dr_options, seed)
    else:
        pipeline = Pipeline([('id', FunctionTransformer(lambda x: x))])
        norm_params = dr_params = {}
        norm_name = dr_name = 'Identity'
        
    return pipeline, norm_params, dr_params, norm_name, dr_name

def draw_random_pipeline(
    norm_options: List[PreprocOption],
    dr_options: List[PreprocOption],
    seed: int
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any], str, str]:
    rnd = random.Random(seed)

    norm, norm_params, norm_name = _choose_one(rnd, norm_options)
    dr, dr_params, dr_name = _choose_one(rnd, dr_options)
    
    pipeline = Pipeline([('norm', norm), ('dr', dr)])
    return pipeline, norm_params, dr_params, norm_name, dr_name

def _choose_one(
    rnd: random.Random, 
    options: List[PreprocOption]
) -> Tuple[TransformerMixin, Dict[str, Any], str]:
    """
    Supports:
      - (cls, params)
      - (cls, name, params)
      - {'cls': cls, 'params': {...}, 'name': optional}
    Returns instantiated transformer, chosen params, and resolved name.
    """
    opt = rnd.choice(options)
    name = None
    params: Dict[str, List[Any]] = {}

    if isinstance(opt, tuple):
        if len(opt) == 3:
            cls, name, params = opt
        elif len(opt) == 2:
            cls, params = opt
        else:
            raise ValueError("Option tuples must be (cls, params) or (cls, name, params).")
        
    elif isinstance(opt, dict):
        # be tolerant with keys
        cls = opt.get("cls") or opt.get("estimator") or opt.get("transformer")
        
        if cls is None:
            raise ValueError("Dict option must contain key 'cls' (or 'estimator'/'transformer').")
        
        params = opt.get("params") or opt.get("grid") or {}
        name = opt.get("name")
    
    else:
        raise TypeError("Unsupported option type. Use (cls, params), (cls, name, params), or {'cls','params','name'}.")

    # sample a concrete parameter setting
    chosen_params = {k: rnd.choice(v) for k, v in (params or {}).items()}
    inst = cls(**chosen_params)
    resolved_name = name or inst.__class__.__name__
    return inst, chosen_params, resolved_name