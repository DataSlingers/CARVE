# ClusteringValidation (carve)

A small library for stability-based clustering validation via repeated subsampling, consensus matrices, and ARI-based scores with optional randomized preprocessing.

## Installation
- Python 3.9+
- Dependencies used in code: numpy, pandas, scipy, scikit-learn, joblib, tqdm, umap-learn
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas scipy scikit-learn joblib tqdm umap-learn
```

## Quick start
```python
import numpy as np
from carve.api import CARVE
from carve.config import ValidatorConfig

# X must be 2D (n_samples, n_features)
X = np.random.RandomState(0).randn(200, 10)

cfg = ValidatorConfig(
    X=X,
    K=10,           # max K or iterable of K
    B=50,           # bootstrap/subsample iterations
    rho=0.6,        # subsample ratio
    random_state=42
)

cv = CARVE(cfg)
method_df = cv.validate(random_preprocess=False, prog_bar=True, random_state=42)
print(method_df.head())  # columns include estimator, n_clusters, ari_stability, ari_generalizability, *_se

# Get labels from the best model by chosen measure (validate() must be called first)
labels = cv.get_optimal_labels(measure="stability", k=None, one_se=False)
```

## What validate() does
- For each estimator/parameter combination in the model grid:
  - Repeats B subsampling iterations at ratio rho.
  - Builds a reference clustering (or uses cfg.ref_labels if provided) and aligns subsample labels to it.
  - Computes:
    - ari_stability: ARI between two aligned subsamples.
    - ari_generalizability: train RF on one subsample to predict labels on held-out data and compare by ARI.
  - Aggregates mean and standard error for both ARIs.
- Also builds:
  - consensus_mats (reordered) and consensus_mats_raw (pairwise co-clustering fractions with NaN where never co-sampled).
  - mis_arrs: per-point misclassification rates from the predictive step.
- If random_preprocess=True, randomly samples a normalization and dimensionality-reduction pipeline per iteration and exposes a summarized cv.pipeline_df.

## Defaults used (from carve.grids)
- Model grid per K in 2..K (or your iterable K):
  - sklearn.cluster.KMeans(n_clusters=K)
  - sklearn.cluster.AgglomerativeClustering(n_clusters=K, linkage in ["ward","complete","average","single"])
  - sklearn.cluster.SpectralClustering(n_clusters=K, gamma auto-grid from data)
- Normalization options:
  - Identity (FunctionTransformer), StandardScaler(), log1p (FunctionTransformer(func=np.log1p))
- Dimensionality reduction options (parameter ranges depend on n and rho):
  - Identity (FunctionTransformer), PCA, TSNE (2 or 3 dims), UMAP

You can override cfg.model_grids, cfg.norm_options, and cfg.dr_options to supply your own options.

## Configuration (carve.config.ValidatorConfig)
- X: np.ndarray
- K: int or array-like of cluster counts
- B: int (iterations)
- rho: float in (0,1)
- model_grids, norm_options, dr_options: optional overrides (see carve.grids)
- ref_labels: optional np.ndarray to align against
- n_jobs: parallel workers (default: CPU count - 1)
- random_state: optional int

cv.get_params() returns a dict view of the config; cv.set_params(**kwargs) returns a new CARVE with updated config.

## Outputs (after validate)
- cv.method_df: DataFrame with one row per estimator/param combo
  - columns: estimator, params (e.g., n_clusters, linkage, gamma), ari_stability, ari_stability_se, ari_generalizability, ari_generalizability_se, consensus_pac_stability
- cv.pipeline_df: DataFrame summary by preprocessing and n_clusters (only when random_preprocess=True)
- cv.consensus_mats, cv.consensus_mats_raw: list of n×n matrices (one per estimator config)
- cv.mis_arrs: list of length-n arrays (one per estimator config)

## Notes
- X must be numeric and 2D; pandas DataFrames are accepted but converted to NumPy.
- UMAP requires umap-learn installed; TSNE/UMAP can be slow on large n.
- one_se selection in get_optimal_labels is present but experimental; use the default path.
- No CLI; use the Python API.
