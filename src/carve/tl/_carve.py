"""AnnData-native entry point for CARVE validation."""

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.base import ClassifierMixin

from .. import _anndata
from .._types import GridSpec, NoisePolicy, PreprocOption, RunMode
from ..api import CARVE

__all__ = ["attach_results", "carve"]

#: Densifying a consensus matrix this large is worth warning about: a
#: 10k x 10k float32 block is ~400 MB in .obsp, and again on disk.
_CONSENSUS_WARN_OBS = 10_000


def carve(
    adata: AnnData,
    *,
    # --- representation ---
    use_rep: str | None = None,
    layer: str | None = None,
    n_pcs: int | None = None,
    # --- sweep axis ---
    n_clusters: int | Sequence[int] | np.ndarray | None = None,
    resolution: float | Sequence[float] | np.ndarray | None = None,
    sweep: str | None = None,
    sweep_values: Sequence[float] | np.ndarray | None = None,
    finer_is_larger: bool | None = None,
    # --- resampling ---
    n_resamples: int = 100,
    subsample_ratio: float = 0.618,
    anchor_threshold: int = 5000,
    consensus_anchors: int | float | None = None,
    noise_policy: NoisePolicy = "drop",
    # --- estimators and preprocessing ---
    estimator_param_grids: list[GridSpec] | Literal["light", "full"] = "light",
    normalization_options: list[PreprocOption] | None = None,
    dim_reduction_options: list[PreprocOption] | None = None,
    randomize_preprocessing: bool = False,
    classifier: ClassifierMixin | None = None,
    n_trees: int = 100,
    # --- which configuration is written back ---
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    k: int | None = None,
    sweep_value: float | None = None,
    consensus_k: int | None = None,
    reference_key: str | None = None,
    # --- what is written ---
    key_added: str = "carve",
    store_consensus: bool = True,
    store_results: bool = True,
    # --- execution ---
    mode: RunMode = "default",
    n_jobs: int = 1,
    random_state: int | None = 0,
    show_progress: bool = False,
    verbose: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """Validate clustering solutions and write the results into ``adata``.

    Repeatedly subsamples the chosen representation, clusters each subsample
    across a grid of algorithms and granularities, and scores every
    configuration for stability (do the same groups reappear under
    perturbation?) and generalizability (do held-out cells get predicted into
    the groups they were assigned to?). The single best configuration under
    ``measure`` and ``rule`` is then written back to ``adata``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_rep : str, optional
        Key in ``adata.obsm`` to cluster, or ``"X"`` for ``adata.X``. If None
        and ``layer`` is None, ``"X_pca"`` is used when present, else
        ``adata.X``.
    layer : str, optional
        Key in ``adata.layers`` to cluster. Mutually exclusive with
        ``use_rep``.
    n_pcs : int, optional
        Restrict the chosen representation to its first ``n_pcs`` columns.
    n_clusters : int or sequence of int, optional
        Numbers of clusters to evaluate. An integer *K* expands to 2..*K*.
        Defaults to ``range(2, 11)`` unless ``resolution`` or ``sweep`` puts
        the run on a different axis.
    resolution : float or sequence of float, optional
        Resolution values for graph-community estimators. Supplying this
        switches the run to resolution mode, where the number of clusters
        becomes an observed outcome rather than an input.
    sweep : str, optional
        Name of the hyperparameter to sweep. Inferred from the other
        arguments when omitted.
    sweep_values : sequence of float, optional
        Values for ``sweep`` when it is neither ``"n_clusters"`` nor
        ``"resolution"``.
    finer_is_larger : bool, optional
        Whether larger values of ``sweep`` yield more clusters. Inferred for
        known parameters; required for unknown ones.
    n_resamples : int, default=100
        Number of resampling iterations per configuration.
    subsample_ratio : float, default=0.618
        Fraction of cells drawn without replacement per resample.
    anchor_threshold : int, default=5000
        Runs with ``n_samples <= anchor_threshold`` build full consensus
        matrices exactly as before. Above it, consensus quantities are
        computed over a fixed anchor subset, because a dense n-by-n matrix
        per configuration is not feasible at large n. The comparison is
        inclusive.
    consensus_anchors : int, float, or None, default=None
        Number of anchors, or a fraction of ``n_samples`` as a float in
        (0, 1]. None resolves to ``min(n_samples, anchor_threshold)``, which
        keeps the effective anchor count continuous across the threshold.
        Lower it to reduce the memory the retained blocks occupy: each block
        is 4 * m**2 bytes and there are two per configuration.
    noise_policy : {"drop", "as_cluster", "singleton"}, default="drop"
        How to treat the ``-1`` labels emitted by density-based methods.
    estimator_param_grids : list of tuple or {"light", "full"}, default="light"
        Clustering estimators and their parameter grids.
    normalization_options : list of preprocessing specs, optional
        Normalization options used when ``randomize_preprocessing=True``.
    dim_reduction_options : list of preprocessing specs, optional
        Dimensionality reduction options used when
        ``randomize_preprocessing=True``.
    randomize_preprocessing : bool, default=False
        Sample a random preprocessing pipeline per resample.
    classifier : sklearn classifier, optional
        Classifier used to score generalizability. Defaults to a random
        forest with ``n_trees`` trees.
    n_trees : int, default=100
        Trees in the default random forest. Ignored when ``classifier`` is
        given.
    measure : str, default="stability"
        Metric used to pick the configuration written back. Aliases include
        ``"stability"``, ``"generalizability"``, ``"average"``, ``"pac"``,
        ``"gini"``, ``"ce"`` and ``"accuracy"``.
    rule : {"max", "1se", "quantile"}, default="1se"
        Selection rule applied to ``measure``.
    not_two : bool, default=False
        Exclude two-cluster solutions during selection.
    k : int, optional
        Pin the number of clusters instead of selecting it. Only valid when
        the run swept ``n_clusters``.
    sweep_value : float, optional
        Pin the swept hyperparameter instead of selecting it.
    consensus_k : int, optional
        Number of clusters used to cut the consensus matrix.
    reference_key : str, optional
        Column in ``adata.obs`` holding reference labels, used to keep
        cluster numbering stable across calls.
    key_added : str, default="carve"
        Base name for every key written; see Notes.
    store_consensus : bool, default=True
        Write the selected consensus matrix into ``adata.obsp``.
    store_results : bool, default=True
        Write the per-configuration metrics table into ``adata.uns``.
    mode : {"default", "stability", "generalizability"}, default="default"
        Which analyses to run.
    n_jobs : int, default=1
        Parallel jobs for resampling. ``-1`` uses all cores.
    random_state : int, optional
        Seed for reproducibility. Defaults to 0, following scanpy.
    show_progress : bool, default=False
        Display a progress bar over grid configurations.
    verbose : int, default=0
        Console verbosity during fitting.
    copy : bool, default=False
        Return a modified copy instead of writing into ``adata``.

    Returns
    -------
    adata : AnnData or None
        A modified copy when ``copy=True``, otherwise None and ``adata`` is
        updated in place.

    Notes
    -----
    With the default ``key_added="carve"``, the following are written:

    ``adata.obs["carve"]``
        Consensus cluster labels, as a categorical of strings.
    ``adata.obs["carve_stability"]``
        Per-cell Gini stability.
    ``adata.obs["carve_stability_ce"]``
        Per-cell cross-entropy stability.
    ``adata.obs["carve_generalizability"]``
        Per-cell held-out prediction accuracy.
    ``adata.obsp["carve_consensus"]``
        The selected consensus matrix. It is ``float32`` and contains NaN for
        pairs of cells never drawn into the same subsample.
    ``adata.uns["carve"]["params"]``
        Run and selection parameters.
    ``adata.uns["carve"]["results"]``
        Per-configuration metrics, one row per configuration.

    Everything written survives :meth:`~anndata.AnnData.write_h5ad`. The
    fitted :class:`~carve.CARVE` object itself is not retained; use
    :func:`attach_results` if you need it.

    Examples
    --------
    >>> import scanpy as sc, carve
    >>> adata = sc.datasets.pbmc3k_processed()
    >>> carve.tl.carve(adata, use_rep="X_pca", n_clusters=range(2, 11))
    >>> adata.obs["carve"].cat.categories
    >>> sc.pl.umap(adata, color=["carve", "carve_stability"])

    See Also
    --------
    attach_results : Write an already-fitted CARVE model into an AnnData.
    carve.CARVE : The underlying scikit-learn compatible estimator.
    """
    if copy:
        adata = adata.copy()

    reference_labels = None
    if reference_key is not None:
        if reference_key not in adata.obs:
            raise ValueError(
                f"reference_key={reference_key!r} not found in adata.obs. "
                f"Available columns: {sorted(adata.obs.columns)}"
            )
        reference_labels, _ = pd.factorize(adata.obs[reference_key])

    model_kwargs: dict = {
        "resolution": resolution,
        "sweep": sweep,
        "sweep_values": sweep_values,
        "finer_is_larger": finer_is_larger,
        "noise_policy": noise_policy,
        "n_resamples": n_resamples,
        "subsample_ratio": subsample_ratio,
        "anchor_threshold": anchor_threshold,
        "consensus_anchors": consensus_anchors,
        "estimator_param_grids": estimator_param_grids,
        "normalization_options": normalization_options,
        "dim_reduction_options": dim_reduction_options,
        "classifier": classifier,
        "n_trees": n_trees,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "verbose": verbose,
    }
    # CARVE defaults n_clusters via default_factory, so only pass it through
    # when the caller asked for it -- otherwise a resolution sweep would
    # collide with a k grid nobody requested.
    if n_clusters is not None:
        model_kwargs["n_clusters"] = (
            n_clusters if isinstance(n_clusters, int) else np.asarray(list(n_clusters))
        )

    model = CARVE(**model_kwargs)
    model.fit(
        adata,
        use_rep=use_rep,
        layer=layer,
        n_pcs=n_pcs,
        reference_labels=reference_labels,
        randomize_preprocessing=randomize_preprocessing,
        show_progress=show_progress,
        mode=mode,
    )

    attach_results(
        adata,
        model,
        key_added=key_added,
        measure=measure,
        rule=rule,
        not_two=not_two,
        k=k,
        sweep_value=sweep_value,
        consensus_k=consensus_k,
        store_consensus=store_consensus,
        store_results=store_results,
        use_rep=use_rep,
        layer=layer,
        n_pcs=n_pcs,
        mode=mode,
    )

    return adata if copy else None


def attach_results(
    adata: AnnData,
    model: CARVE,
    *,
    key_added: str = "carve",
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    k: int | None = None,
    sweep_value: float | None = None,
    consensus_k: int | None = None,
    store_consensus: bool = True,
    store_results: bool = True,
    use_rep: str | None = None,
    layer: str | None = None,
    n_pcs: int | None = None,
    mode: RunMode = "default",
) -> None:
    """Write a fitted :class:`~carve.CARVE` model into an ``AnnData``.

    Use this when you want to keep the fitted model -- to save it, or to
    inspect configurations other than the selected one -- while still getting
    the same keys :func:`carve` writes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in place.
    model : CARVE
        A fitted CARVE instance whose observations correspond, in order, to
        those of ``adata``.
    key_added : str, default="carve"
        Base name for every key written. See :func:`carve` for the full list.
    measure : str, default="stability"
        Metric used to pick the configuration written back.
    rule : {"max", "1se", "quantile"}, default="1se"
        Selection rule applied to ``measure``.
    not_two : bool, default=False
        Exclude two-cluster solutions during selection.
    k : int, optional
        Pin the number of clusters instead of selecting it.
    sweep_value : float, optional
        Pin the swept hyperparameter instead of selecting it.
    consensus_k : int, optional
        Number of clusters used to cut the consensus matrix.
    store_consensus : bool, default=True
        Write the selected consensus matrix into ``adata.obsp``.
    store_results : bool, default=True
        Write the per-configuration metrics table into ``adata.uns``.
    use_rep : str, optional
        Representation the model was fitted on, recorded in ``params`` so
        that plotting can fall back to it when choosing an embedding.
    layer : str, optional
        Layer the model was fitted on, recorded in ``params``.
    n_pcs : int, optional
        Component count the model was fitted with, recorded in ``params``.
    mode : {"default", "stability", "generalizability"}, default="default"
        The mode the model was fitted with, recorded in ``params``.

    Raises
    ------
    RuntimeError
        If ``model`` has not been fitted.
    ValueError
        If the model's observation count does not match ``adata.n_obs``.

    Examples
    --------
    >>> model = carve.CARVE(n_clusters=range(2, 11), n_jobs=8)
    >>> model.fit(adata, use_rep="X_pca")
    >>> carve.tl.attach_results(adata, model)
    >>> model.save("run.carve")
    """
    if model.estimator_results_ is None:
        raise RuntimeError("The CARVE model is not fitted; call fit() first.")

    row, config_id, n_clusters_sel, pinned = model._select_row(
        measure=measure,
        rule=rule,
        not_two=not_two,
        k=k,
        sweep_value=sweep_value,
    )

    # A generalizability-only run builds no stability consensus, so both the
    # label cut and the stored matrix have to come from the generalizability
    # consensus instead. CARVE.get_labels makes the same distinction.
    labels_mode = "generalizability" if mode == "generalizability" else "default"
    consensus_source = (
        model.consensus_generalizability_matrices_
        if labels_mode == "generalizability"
        else model.consensus_matrices_
    )

    labels = model.get_labels(
        measure=measure,
        rule=rule,
        k=k,
        sweep_value=sweep_value,
        consensus_k=consensus_k,
        not_two=not_two,
        mode=labels_mode,
    )
    _check_length(labels, adata.n_obs, "consensus labels")

    adata.obs[key_added] = _anndata.labels_to_categorical(labels)

    # Per-cell diagnostics. Which of these exist depends on `mode`. A skipped
    # analysis leaves either the container itself None or, for the list-valued
    # ones, a list of None -- both mean "not computed".
    for suffix, container in (
        ("stability", model.stability_gini_scores_),
        ("stability_ce", model.stability_ce_scores_),
        ("generalizability", model.generalizability_scores_),
    ):
        entry = _artifact(container, config_id)
        if entry is None:
            continue
        scores = np.asarray(entry, dtype=float)
        _check_length(scores, adata.n_obs, f"{suffix} scores")
        adata.obs[f"{key_added}_{suffix}"] = scores

    consensus = _artifact(consensus_source, config_id)
    if store_consensus and consensus is not None:
        matrix = np.asarray(consensus)
        if matrix.shape != (adata.n_obs, adata.n_obs):
            raise ValueError(
                f"Consensus matrix has shape {matrix.shape}, but adata has "
                f"{adata.n_obs} observations. The model was fitted on "
                "different data."
            )
        if adata.n_obs > _CONSENSUS_WARN_OBS:
            warnings.warn(
                f"Storing a dense {adata.n_obs}x{adata.n_obs} consensus "
                f"matrix in adata.obsp[{key_added + '_consensus'!r}] "
                f"(~{matrix.nbytes / 1e9:.1f} GB). Pass store_consensus=False "
                "if you do not need it.",
                UserWarning,
                stacklevel=2,
            )
        adata.obsp[f"{key_added}_consensus"] = matrix

    params = _build_params(
        model=model,
        row=row,
        config_id=config_id,
        n_clusters_sel=n_clusters_sel,
        pinned=pinned,
        measure=measure,
        rule=rule,
        not_two=not_two,
        consensus_k=consensus_k,
        use_rep=use_rep,
        layer=layer,
        n_pcs=n_pcs,
        mode=mode,
        n_obs=adata.n_obs,
    )

    entry: dict = {"params": _anndata.params_to_uns(params)}
    if store_results:
        entry["results"] = _anndata.results_to_uns(model.estimator_results_)
    adata.uns[key_added] = entry


def _artifact(container, config_id: int):
    """Return one per-configuration artifact, or None when it was not computed.

    A skipped analysis (``mode="stability"`` or ``mode="generalizability"``)
    leaves the container either as None or as a list whose entries are None,
    so both cases have to be treated as absent.
    """
    if container is None:
        return None
    try:
        entry = container[config_id]
    except (IndexError, KeyError):
        return None
    return entry


def _check_length(values: np.ndarray, n_obs: int, what: str) -> None:
    """Raise if a per-observation array does not match ``adata``."""
    if len(values) != n_obs:
        raise ValueError(
            f"The model produced {len(values)} {what} but adata has {n_obs} "
            "observations. The model was fitted on different data."
        )


def _build_params(
    *,
    model: CARVE,
    row: pd.Series,
    config_id: int,
    n_clusters_sel: int,
    pinned: bool,
    measure: str,
    rule: str,
    not_two: bool,
    consensus_k: int | None,
    use_rep: str | None,
    layer: str | None,
    n_pcs: int | None,
    mode: RunMode,
    n_obs: int,
) -> dict:
    """Assemble the parameter record stored under ``uns[key]["params"]``."""
    sweep = model.sweep_
    sweep_param = sweep.param if sweep is not None else "n_clusters"

    grids = model.estimator_param_grids
    grid_desc = grids if isinstance(grids, str) else "custom"

    params: dict = {
        "measure": measure,
        "rule": rule,
        "not_two": bool(not_two),
        "pinned": bool(pinned),
        "sweep_param": sweep_param,
        "n_resamples": int(model.n_resamples),
        "subsample_ratio": float(model.subsample_ratio),
        "n_consensus_anchors": (
            int(model.consensus_anchors_.size)
            if model.consensus_anchors_ is not None
            else None
        ),
        "noise_policy": str(model.noise_policy),
        "mode": str(mode),
        "random_state": model.random_state,
        "use_rep": use_rep,
        "layer": layer,
        "n_pcs": n_pcs,
        "consensus_k": consensus_k,
        "estimator_param_grids": grid_desc,
        "selected_config_id": int(config_id),
        "selected_k": int(n_clusters_sel),
        "n_obs": int(n_obs),
    }

    if sweep is not None and sweep.values is not None:
        params["sweep_values"] = np.asarray(sweep.values, dtype=float)

    if sweep_param in row.index:
        params[f"selected_{sweep_param}"] = row[sweep_param]
    if "estimator" in row.index:
        params["selected_estimator"] = row["estimator"]
    if "method_label" in row.index:
        params["selected_method_label"] = row["method_label"]

    return params
