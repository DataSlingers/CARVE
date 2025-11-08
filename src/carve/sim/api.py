from typing import Callable, Literal, NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix

from rpy2.robjects.methods import RS4
from rpy2 import robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

from ._outliers import _parse_outliers, _sample_outliers
from ._centers import _sample_centers
from ._sizes import _compute_cluster_sizes, _get_cluster_scales, _post_embed_scaling
from ._covariance import _build_correlation_matrix, _cluster_covariances
from ._distributions import _sample_cluster_points
from ._embed import _apply_embedding
from ._plot import _plot_simulation
from ._noise import _sample_noise

class SimulationMeta(NamedTuple):
    centers: np.ndarray
    cluster_sizes: np.ndarray
    cluster_scales: list[float]
    correlation: np.ndarray
    covariances: list[np.ndarray]
    outliers: int
    signal_dims: int
    noise_dims: int
    noise_mask: np.ndarray

def simulate_clusters(
    n_total: int,
    p: int,
    k: int,
    cluster_scale: float | list[float] | Callable[[], float] = 1.0,
    balanced: bool = True,
    cluster_sizes_frac: list[float] | None = None,
    min_cluster_size_abs: int = 5,
    min_cluster_size_frac: float = 0.1,
    cluster_size_dirichlet_alpha: float | np.ndarray = 0.3,
    corr_type: Literal["none", "ar1", "block"] = "none",
    corr_strength: float = 1.0,
    block_size: int | None = None,
    outliers: int | float = 0,
    outlier_scale: float = 5.0,
    outlier_mode: Literal["far_gaussian", "uniform_box"] = "far_gaussian",
    distribution: str = "gaussian",
    t_df: int = 3,
    nonlinear: bool = False,
    embed_dim: int | None = None,
    embed_method: Literal["random_fourier", "poly", "rbf"] = "random_fourier",
    embed_param: float = 2.0,
    center_box: float = 3.0,
    centroid_method: Literal["none", "lhs", "best_candidate", "min_dist"] = "best_candidate",
    n_candidates: int = 64,
    min_center_dist: float | None = None,
    post_embed_standardize: bool = True,
    preserve_global_scale: bool = False,
    compactness: float = 0.65,
    embed_scale_by_dim: bool = True,
    noise_dims: int = 0,
    noise_dist: Literal["gaussian", "uniform", "laplace", "t"] = "gaussian",
    noise_scale: float | Literal["match"] = "match",
    plotting: bool = True,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, SimulationMeta]:
    """
    Generate synthetic clustered data with optional correlation, outliers, and embedding.

    Returns: (X, y, meta)
      - X: (n_total, p) array
      - y: (n_total,) labels, 0..k-1 for clusters, -1 for outliers
      - meta: SimulationMeta with centers, sizes, scales, correlation, covariances, outliers
    """
    rng = np.random.default_rng(seed=random_state)
    p = int(np.floor(p + 0.5))
    n_total = int(np.floor(n_total + 0.5))

    # Outliers
    n_outliers = _parse_outliers(outliers, n_total)
    n_total_clusters = n_total - n_outliers
    if n_total_clusters < 1:
        raise ValueError("`outliers` too large, no points left for clusters.")

    # Sizes
    cluster_sizes = _compute_cluster_sizes(
        n_total_clusters=n_total_clusters,
        k=k,
        balanced=balanced,
        cluster_sizes_frac=cluster_sizes_frac,
        rng=rng,
        min_abs=min_cluster_size_abs,
        min_frac=min_cluster_size_frac,
        alpha=cluster_size_dirichlet_alpha
    )
    assert int(cluster_sizes.sum()) == n_total_clusters

    # Centers and covariances
    centers = _sample_centers(
        k=k, p=p, center_box=center_box, rng=rng,
        method=centroid_method, n_candidates=n_candidates, min_center_dist=min_center_dist
    )
    scales = _get_cluster_scales(cluster_scale, k)
    R = _build_correlation_matrix(p=p, corr_type=corr_type, corr_strength=corr_strength, block_size=block_size)
    covs = _cluster_covariances(scales, R)

    # Sample clusters
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for c in range(k):
        size = int(cluster_sizes[c])
        mean = centers[c]
        cov = covs[c]
        X_c = _sample_cluster_points(
            rng=rng, size=size, mean=mean, cov=cov,
            distribution=distribution, t_df=t_df
        )
        X_parts.append(X_c)
        y_parts.append(np.full(size, c, dtype=int))

    # Outliers
    if n_outliers > 0:
        X_out = _sample_outliers(
            rng=rng, n_outliers=int(n_outliers), p=p,
            centers=centers, cluster_sizes=cluster_sizes, covs=covs,
            center_box=center_box, outlier_mode=outlier_mode, outlier_scale=outlier_scale
        )
        X_parts.append(X_out)
        y_parts.append(np.full(int(n_outliers), -1, dtype=int))

    # Concatenate
    X = np.vstack(X_parts) if X_parts else np.empty((0, p), dtype=float)
    y = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=int)

    # Optional embedding
    X_pre_embed = X
    if nonlinear:
        X = _apply_embedding(
            X, method=embed_method, embed_dim=embed_dim,
            embed_param=embed_param, scale_by_dim=embed_scale_by_dim, rng=rng
        )
        
        if post_embed_standardize or preserve_global_scale or compactness != 1.0:
            X = _post_embed_scaling(
                X,
                X_pre_embed=X_pre_embed,
                preserve_global_scale=preserve_global_scale,
                post_embed_standardize=post_embed_standardize,
                compactness=compactness
            )

    # Add noise dimensions
    if noise_dims > 0:
        if noise_scale == "match":
            stds = X.std(axis=0, ddof=1)
            base = float(np.nanmean(np.where(stds > 0, stds, np.nan))) if X.shape[1] > 0 else 1.0
            if not np.isfinite(base) or base == 0.0:
                base = 1.0
            scale_val = base
        else:
            scale_val = float(noise_scale)

        Z = _sample_noise(
            rng=rng, n=X.shape[0], q=noise_dims,
            dist=noise_dist, scale=scale_val, t_df=t_df
        )
        X = np.hstack([X, Z])

    # Shuffle
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # Optional plotting
    if plotting:
        _plot_simulation(X, y, random_state=random_state)

    # build meta data
    total_dims = X.shape[1]
    sig_dims = (X_pre_embed.shape[1] if nonlinear else p)
    noise_dims_int = int(np.floor(noise_dims + 0.5))
    
    if noise_dims > 0:
        sig_dims = total_dims - noise_dims
        
    noise_mask = np.zeros(total_dims, dtype=bool)
    
    if noise_dims > 0:
        noise_mask[sig_dims: total_dims] = True

    meta = SimulationMeta(
        centers=centers, cluster_sizes=cluster_sizes, cluster_scales=scales,
        correlation=R, covariances=covs, outliers=int(n_outliers),
        signal_dims=sig_dims, noise_dims=int(noise_dims), noise_mask=noise_mask
    )
    
    return X, y, meta



class ScDesign3Simulator:
    def __init__(
        self, 
        reference_sce_rds: str, 
        assay_use: str = "counts", 
        label_col: Optional[str] = "cell_type", 
        pseudotime_col: Optional[str] = None,
        family: str = "nb",
        copula: str = "gaussian",
        prefit: bool = False, 
        n_genes_fit: int = 3000,
        min_detect_rate: float = 0.01,
        n_cores: int = 1, 
    ):
        self.scdesign3 = importr("scDesign3")
        self.base = importr("base")
        self.sce = self.base.readRDS(reference_sce_rds)
        self.assay_use = assay_use
        self.label_col = label_col
        self.pseudotime_col = pseudotime_col
        self.family = family
        self.copula = copula
        self.n_genes_fit = int(n_genes_fit)
        self.min_detect_rate = float(min_detect_rate)
        self.n_cores = n_cores
        self.prefit = prefit
        self._fit_obj = None
        self._subset_reference_genes()
        
        if prefit:  # placeholder, may implement later
            pass  # see scDesign3 introduction vignette. :contentReference[oaicite:2]{index=2}
        
    def _subset_reference_genes(self):
        Matrix = importr("Matrix")
        SummExp = importr("SummarizedExperiment")
        # try to use sparse rowVars if available (best for dgCMatrix)
        try:
            sparseMatrixStats = importr("sparseMatrixStats")
            have_sparse = True
        except Exception:
            have_sparse = False
            MatrixGenerics = importr("MatrixGenerics")

        # R code to select genes
        r = ro.r
        r("set.seed(1)")
        r.assign("sce", self.sce)
        r.assign("assay_name", self.assay_use)
        r.assign("min_rate", self.min_detect_rate)
        r.assign("n_keep", self.n_genes_fit)

        r("""
        counts <- SummarizedExperiment::assay(sce, assay_name)
        nc <- ncol(counts)
        det <- Matrix::rowSums(counts > 0) / nc
        keep <- det >= min_rate
        # if too many, pick the top by variance when possible, otherwise random
        if (sum(keep) > n_keep) {
        idx_keep <- which(keep)
        if ("sparseMatrixStats" %in% rownames(installed.packages())) {
            vars <- sparseMatrixStats::rowVars(counts)
            ord <- order(vars[idx_keep], decreasing = TRUE)
            sel <- idx_keep[ ord[ seq_len(n_keep) ] ]
        } else {
            set.seed(1)
            sel <- sample(idx_keep, n_keep)
        }
        sce <- sce[sel, ]
        } else {
        sce <- sce[keep, ]
        }
        """)
        self.sce = r("sce")
        
    def _r_matrix_to_numpy(self, r_obj):
        with localconverter(default_converter + numpy2ri.converter):
            return ro.conversion.rpy2py(r_obj)

    def _r_df_to_pandas(self, r_df):
        with localconverter(default_converter + pandas2ri.converter):
            return ro.conversion.rpy2py(r_df)
        
    def _rnull(self, x):
        return ro.NULL if x is None else x
    
    def _counts_to_numpy(self, r_counts):
        if isinstance(r_counts, RS4):
            classes = set(map(str, r_counts.rclass))
            if "dgCMatrix" in classes or "dCsparseMatrix" in classes:
                i = np.asarray(r_counts.do_slot("i"),   dtype=np.int32)
                p = np.asarray(r_counts.do_slot("p"),   dtype=np.int32)
                x = np.asarray(r_counts.do_slot("x"),   dtype=np.float64)
                dim = tuple(np.asarray(r_counts.do_slot("Dim"), dtype=np.int32))
                return csc_matrix((x, i, p), shape=dim).toarray()
        
        with localconverter(default_converter + numpy2ri.converter): 
            return ro.conversion.rpy2py(r_counts)
        
    def simulate(
        self, 
        *, 
        k: int, 
        seed: int,
        n_total: int, 
        p: int, 
        noise_genes: float, 
        plotting: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        ro.r("set.seed")(seed)

        res = self.scdesign3.scdesign3(
            sce=self.sce,
            assay_use=self.assay_use,
            celltype=self._rnull(self.label_col),
            pseudotime=self._rnull(self.pseudotime_col),
            family_use=self.family,
            ncell=int(n_total),

            copula="gaussian",
            correlation_function="coop",
            if_sparse=True,    
            fastmvn=True,   
            important_feature=0.95, 
            n_cores=int(self.n_cores), 
            parallelization="pbmcmapply", 
            DT=True,
            usebam=True,

            mu_formula="1", sigma_formula="1",
            corr_formula="1",
            other_covariates=ro.NULL,
            pseudo_obs=False,
            return_model=False
        )

        # counts: genes x cells (R matrix / dgCMatrix -> numpy)
        r_counts = res.rx2("new_count")
        counts = self._counts_to_numpy(r_counts)
        X = counts.T.astype(float)
        
        lib = X.sum(axis=1, keepdims=True)
        X = np.log1p(1e4 * X / np.maximum(lib, 1))

        # covariates / labels
        cov = None
        if "new_covariate" in res.names:
            r_cov = res.rx2("new_covariate")
            
            if not bool(ro.r("is.null")(r_cov)[0]):
                cov = self._r_df_to_pandas(r_cov)

        if cov is None:
            SE = importr("SummarizedExperiment")
            r_cold = SE.colData(self.sce)
            
            r_df = ro.r("as.data.frame")(r_cold)
            cov = self._r_df_to_pandas(r_df)

        if self.label_col and self.label_col in cov.columns:
            y_raw = cov[self.label_col].astype(str).to_numpy()
            # enforce exactly k types (optional)
            top = pd.Series(y_raw).value_counts().index[:k]
            keep = np.isin(y_raw, top)
            X, y_raw = X[keep], y_raw[keep]
            _, y = np.unique(y_raw, return_inverse=True)
        elif self.pseudotime_col and self.pseudotime_col in cov.columns:
            pt = cov[self.pseudotime_col].to_numpy().astype(float)
            bins = np.quantile(pt, np.linspace(0, 1, k + 1))
            y = np.clip(np.digitize(pt, bins[1:-1]), 0, k - 1)
        else:
            y = np.zeros(X.shape[0], dtype=int)

        rng = np.random.default_rng(seed)
        q = int(round(noise_genes * X.shape[1]))
        if q > 0:
            Z = rng.normal(0, X.std() * 0.2, size=(X.shape[0], q))
            X = np.hstack([X, Z])

        if X.shape[1] > p:
            var = X.var(axis=0)
            X = X[:, np.argsort(var)[-p:]]

        if plotting:
            _plot_simulation(X, y, random_state=None)

        return X, y