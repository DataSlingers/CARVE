import random
from typing import Any, Callable, Dict, List, Tuple
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

PreprocSpec = Tuple[Callable[..., TransformerMixin], Dict[str, List[Any]]]

def create_pipeline(
    random_preproc: bool, 
    norm_options: List[PreprocSpec], 
    dr_options: List[PreprocSpec], 
    seed: int
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any], str, str]:
    if random_preproc:
        pipeline, norm_params, dr_params = draw_random_pipeline(norm_options, dr_options, seed)
        norm_name = pipeline.named_steps['norm'].__class__.__name__
        dr_name = pipeline.named_steps['dr'].__class__.__name__
    else:
        pipeline = Pipeline([('id', FunctionTransformer(lambda x: x))])
        norm_params = dr_params = {}
        norm_name = dr_name = 'Identity'
        
    return pipeline, norm_params, dr_params, norm_name, dr_name

def draw_random_pipeline(
    norm_options: List[PreprocSpec],
    dr_options: List[PreprocSpec],
    seed: int = 0
) -> PreprocSpec:
    rnd = random.Random(seed)
        
    norm_cls, norm_grid = rnd.choice(norm_options)
    dr_cls, dr_grid = rnd.choice(dr_options)
    
    norm_params = {k: rnd.choice(v) for k, v in norm_grid.items()}
    dr_params = {k: rnd.choice(v) for k, v in dr_grid.items()}
    
    norm = norm_cls(**norm_params)
    dr = dr_cls(**dr_params)
    
    pipeline = Pipeline([('norm', norm), ('dr', dr)])
    return pipeline, norm_params, dr_params