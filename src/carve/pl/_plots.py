"""Plotting wrappers that read CARVE results out of an :class:`~anndata.AnnData`.

Each function mirrors one method on :class:`~carve.CARVE`, but sources its
inputs from the keys :func:`carve.tl.carve` wrote rather than from a fitted
model. The fitted model is deliberately not retained by ``tl.carve`` -- it
cannot be stored in ``.uns`` and survive an h5ad round trip -- so these
functions reconstruct only what the plots actually need: the per-cell scores in
``.obs``, the consensus matrix in ``.obsp``, and the metrics table in ``.uns``.

No model selection is re-run here. The score columns in ``.obs`` belong to the
configuration chosen when ``tl.carve`` ran, so re-selecting would silently pair
one configuration's labels with another's scores.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes

from .. import _anndata
from .._plotting import _get_annotation
from .._plotting import plot_cluster_boxplot as _plot_cluster_boxplot
from .._plotting import plot_cluster_scatter as _plot_cluster_scatter
from .._plotting import plot_cluster_violin as _plot_cluster_violin
from .._plotting import plot_consensus_matrix as _plot_consensus_matrix
from .._plotting import plot_diagnostic_scatter as _plot_diagnostic_scatter
from .._plotting import (
    plot_metric_by_pipeline as _plot_metric_by_pipeline,
)
from .._plotting import (
    plot_metric_over_n_clusters as _plot_metric_over_n_clusters,
)

__all__ = [
    "cluster_boxplot",
    "cluster_scatter",
    "cluster_violin",
    "consensus_matrix",
    "diagnostic_scatter",
    "metric_by_pipeline",
    "metric_over_n_clusters",
]

Source = Literal["gini", "ce", "accuracy"]

#: Maps a score source to its ``.obs`` suffix, default axis label, and the
#: ``mode=`` that produces it.
_SOURCES: dict[str, tuple[str, str, str]] = {
    "gini": ("stability", "Cluster Stability (Gini)", "stability"),
    "ce": ("stability_ce", "Cluster Stability (CE)", "stability"),
    "accuracy": (
        "generalizability",
        "Cluster Generalizability",
        "generalizability",
    ),
}


def _entry(adata: AnnData, key: str) -> dict:
    """Return the ``uns`` entry written by ``tl.carve``, or explain why not."""
    if key not in adata.uns:
        raise KeyError(
            f"adata.uns[{key!r}] not found. Run "
            f"`carve.tl.carve(adata, key_added={key!r})` first, or pass the "
            "key you used."
        )
    entry = adata.uns[key]
    if not isinstance(entry, dict) or "params" not in entry:
        raise KeyError(
            f"adata.uns[{key!r}] does not look like a CARVE result; expected "
            "a mapping with a 'params' entry."
        )
    return entry


def _labels(adata: AnnData, key: str) -> np.ndarray:
    """Return the stored consensus labels as integer codes."""
    if key not in adata.obs:
        raise KeyError(
            f"adata.obs[{key!r}] not found. Run "
            f"`carve.tl.carve(adata, key_added={key!r})` first."
        )
    column = adata.obs[key]
    if isinstance(column.dtype, pd.CategoricalDtype):
        return np.asarray(column.cat.codes, dtype=np.int32)
    return np.asarray(column, dtype=np.int32)


def _scores(adata: AnnData, key: str, source: str) -> tuple[np.ndarray, str]:
    """Return the per-cell scores for ``source`` and their default label."""
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {sorted(_SOURCES)}; got {source!r}.")
    suffix, default_label, producing_mode = _SOURCES[source]
    column = f"{key}_{suffix}"
    if column not in adata.obs:
        raise KeyError(
            f"adata.obs[{column!r}] not found, so source={source!r} is "
            f"unavailable. It is written when tl.carve runs with "
            f"mode='default' or mode={producing_mode!r}."
        )
    return np.asarray(adata.obs[column], dtype=float), default_label


def _results(adata: AnnData, key: str) -> pd.DataFrame:
    """Return the per-configuration metrics table."""
    entry = _entry(adata, key)
    if "results" not in entry:
        raise KeyError(
            f"adata.uns[{key!r}]['results'] not found. It is omitted when "
            "tl.carve runs with store_results=False."
        )
    return _anndata.results_from_uns(entry["results"])


def _preprocessing_results(adata: AnnData, key: str) -> pd.DataFrame:
    """Return the per-pipeline metrics table, or explain why it is absent."""
    entry = _entry(adata, key)
    if "preprocessing_results" not in entry:
        raise KeyError(
            f"adata.uns[{key!r}]['preprocessing_results'] not found. It is "
            "written only for a randomized fit, so either tl.carve ran without "
            "randomize_preprocessing=True or it ran with store_results=False."
        )
    return _anndata.results_from_uns(entry["preprocessing_results"])


def _annotation_text(
    adata: AnnData,
    key: str,
    annotation: bool | str,
) -> str | None:
    """Resolve the annotation argument to literal text, as CARVE's methods do."""
    if isinstance(annotation, str):
        return annotation
    if annotation is not True:
        return None
    entry = _entry(adata, key)
    params = dict(entry["params"])
    try:
        results = _results(adata, key)
    except KeyError:
        return None
    config_id = int(params.get("selected_config_id", 0))
    matches = results.index[results["config_id"] == config_id]
    if len(matches) == 0:
        return None
    return _get_annotation(
        measure=str(params.get("measure", "stability")),
        rule=str(params.get("rule", "1se")),
        estimator_results=results,
        row=results.loc[matches[0]],
        selected_k=int(params.get("selected_k", 0)) or None,
        pinned=bool(params.get("pinned", False)),
    )


def _representation(adata: AnnData, key: str) -> np.ndarray:
    """Recover the representation CARVE was fitted on, for scatter plots."""
    params = dict(_entry(adata, key)["params"])
    use_rep = params.get("use_rep") or None
    layer = params.get("layer") or None
    n_pcs = params.get("n_pcs")
    return _anndata.get_representation(
        adata,
        use_rep=use_rep,
        layer=layer,
        n_pcs=int(n_pcs) if n_pcs is not None else None,
    )


def metric_over_n_clusters(
    adata: AnnData,
    *,
    key: str = "carve",
    measure: str | None = None,
    rule: str | None = None,
    not_two: bool | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes | None:
    """Plot a validation metric across the swept granularity axis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from :func:`carve.tl.carve`.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    measure : str, optional
        Metric to plot. Defaults to the measure recorded at fit time, so the
        selection marker matches the labels in ``adata.obs``.
    rule : str, optional
        Selection rule to mark. Defaults to the recorded rule.
    not_two : bool, optional
        Whether two-cluster solutions were excluded. Defaults to the
        recorded value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    title, xlabel, ylabel : str, optional
        Axis text overrides.
    legend : bool, default=True
        Draw the legend.
    legend_loc : str, default="best"
        Legend location.
    palette : str, default="Accent"
        Matplotlib colormap name used for the per-method colors.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.
    **kwargs
        Forwarded to the underlying line plot.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.

    Notes
    -----
    In resolution mode the x-axis is the swept resolution rather than a
    cluster count, despite the function name.
    """
    params = dict(_entry(adata, key)["params"])
    return _plot_metric_over_n_clusters(
        _results(adata, key),
        measure=(
            measure if measure is not None else str(params.get("measure", "stability"))
        ),
        rule=rule if rule is not None else str(params.get("rule", "1se")),
        not_two=(
            not_two if not_two is not None else bool(params.get("not_two", False))
        ),
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def metric_by_pipeline(
    adata: AnnData,
    *,
    key: str = "carve",
    method_id: str | None = None,
    measure: str | None = None,
    rule: str | None = None,
    not_two: bool | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    legend_loc: str = "best",
    palette: str = "Accent",
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
    **kwargs,
) -> Axes | None:
    """Plot a validation metric across the sweep axis, one line per pipeline.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from a
        :func:`carve.tl.carve` run with ``randomize_preprocessing=True``.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    method_id : str, optional
        Configuration whose pipelines are drawn. Defaults to the
        configuration selected when the results were written, so the lines
        belong to the labels in ``adata.obs``.
    measure : str, optional
        ``"stability"`` or ``"generalizability"``. Defaults to the measure
        recorded at fit time.
    rule : str, optional
        Selection rule for the marked sweep value. Defaults to the recorded
        rule.
    not_two : bool, optional
        Whether two-cluster solutions are excluded when marking the sweep
        value. Defaults to the recorded value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    title, xlabel, ylabel : str, optional
        Axis text overrides.
    legend : bool, default=True
        Draw the legend.
    legend_loc : str, default="best"
        Legend location.
    palette : str, default="Accent"
        Matplotlib colormap name used for the per-pipeline colors.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.
    **kwargs
        Forwarded to the underlying line plot.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.

    Raises
    ------
    KeyError
        If the per-pipeline table is absent: the fit was not randomized, or
        the results were written with ``store_results=False``.
    """
    params = dict(_entry(adata, key)["params"])
    table = _preprocessing_results(adata, key)
    return _plot_metric_by_pipeline(
        table,
        method_id=(
            method_id if method_id is not None else str(params["selected_method_id"])
        ),
        measure=(
            measure if measure is not None else str(params.get("measure", "stability"))
        ),
        rule=rule if rule is not None else str(params.get("rule", "1se")),
        not_two=(
            not_two if not_two is not None else bool(params.get("not_two", False))
        ),
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        legend_loc=legend_loc,
        palette=palette,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def consensus_matrix(
    adata: AnnData,
    *,
    key: str = "carve",
    ax: Axes | None = None,
    figsize: tuple | None = None,
    cmap: str = "viridis",
    palette: str = "Accent",
    colorbar: bool = True,
    colorbar_label: str = "Consensus",
    title: str | None = None,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | None:
    """Plot the selected consensus matrix, ordered by cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from :func:`carve.tl.carve`.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    cmap : str, default="viridis"
        Colormap for the matrix.
    palette : str, default="Accent"
        Colormap for the cluster sidebar.
    colorbar : bool, default=True
        Draw the colorbar.
    colorbar_label : str, default="Consensus"
        Colorbar label.
    title : str, optional
        Axis title.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.

    Raises
    ------
    KeyError
        If no consensus matrix was stored in ``adata.obsp``. This happens
        when ``tl.carve`` ran with ``store_consensus=False``, or when
        anchored consensus was active: an m-by-m anchor block cannot be
        written to ``adata.obsp``, which requires an n_obs-by-n_obs matrix.
    """
    obsp_key = f"{key}_consensus"
    if obsp_key not in adata.obsp:
        raise KeyError(
            f"adata.obsp[{obsp_key!r}] not found. It is omitted when "
            "tl.carve runs with store_consensus=False, or when anchored "
            "consensus was active: an anchor block is m-by-m rather than "
            "n_obs-by-n_obs and cannot be written to adata.obsp."
        )
    return _plot_consensus_matrix(
        np.asarray(adata.obsp[obsp_key]),
        _labels(adata, key),
        ax=ax,
        figsize=figsize,
        cmap=cmap,
        palette=palette,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        title=title,
        show=show,
        save=save,
        dpi=dpi,
    )


def cluster_boxplot(
    adata: AnnData,
    *,
    key: str = "carve",
    source: Source = "gini",
    ax: Axes | None = None,
    figsize: tuple | None = None,
    order: list[int | str] | None = None,
    palette: str = "Accent",
    showfliers: bool = False,
    width: float = 0.75,
    title: str | None = None,
    xlabel: str = "Cluster",
    ylabel: str | None = None,
    annotation: bool | str = True,
    rotation: float | None = None,
    ylim: tuple[float, float] = (-0.02, 1.02),
    fit_ylim: bool = True,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | None:
    """Box plot of per-cell diagnostic scores, grouped by cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from :func:`carve.tl.carve`.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    source : {"gini", "ce", "accuracy"}, default="gini"
        Which per-cell score to plot: Gini stability, cross-entropy
        stability, or held-out prediction accuracy.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    order : list, optional
        Explicit cluster ordering along the x-axis.
    palette : str, default="Accent"
        Colormap name for the per-cluster colors.
    showfliers : bool, default=False
        Draw outlier points.
    width : float, default=0.75
        Box width.
    title : str, optional
        Axis title.
    xlabel : str, default="Cluster"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Defaults to a label describing ``source``.
    annotation : bool or str, default=True
        True renders a description of the selected model; a string is used
        verbatim; False disables it.
    rotation : float, optional
        Rotation applied to the x tick labels.
    ylim : tuple of float, default=(-0.02, 1.02)
        Y-axis limits.
    fit_ylim : bool, default=True
        Shrink ``ylim`` to the observed data range.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.
    """
    scores, default_ylabel = _scores(adata, key, source)
    return _plot_cluster_boxplot(
        scores,
        _labels(adata, key),
        ax=ax,
        figsize=figsize,
        order=order,
        palette=palette,
        showfliers=showfliers,
        width=width,
        title=title,
        xlabel=xlabel,
        ylabel=default_ylabel if ylabel is None else ylabel,
        annotation=_annotation_text(adata, key, annotation),
        rotation=rotation,
        ylim=ylim,
        fit_ylim=fit_ylim,
        show=show,
        save=save,
        dpi=dpi,
    )


def cluster_violin(
    adata: AnnData,
    *,
    key: str = "carve",
    source: Source = "gini",
    ax: Axes | None = None,
    figsize: tuple | None = None,
    order: list[int | str] | None = None,
    palette: str = "Accent",
    density_norm: Literal["width", "area", "count"] = "width",
    stripplot: bool = True,
    jitter: bool | float = True,
    size: float = 8.0,
    alpha: float = 0.22,
    inner: Literal["box", "quartile", "none"] = "box",
    title: str | None = None,
    xlabel: str = "Cluster",
    ylabel: str | None = None,
    annotation: bool | str = True,
    rotation: float | None = None,
    ylim: tuple[float, float] = (-0.02, 1.02),
    fit_ylim: bool = True,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | None:
    """Violin plot of per-cell diagnostic scores, grouped by cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from :func:`carve.tl.carve`.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    source : {"gini", "ce", "accuracy"}, default="gini"
        Which per-cell score to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    order : list, optional
        Explicit cluster ordering along the x-axis.
    palette : str, default="Accent"
        Colormap name for the per-cluster colors.
    density_norm : {"width", "area", "count"}, default="width"
        How each violin is scaled.
    stripplot : bool, default=True
        Overlay the individual observations.
    jitter : bool or float, default=True
        Jitter applied to the overlaid points.
    size : float, default=8.0
        Marker size for the overlaid points.
    alpha : float, default=0.22
        Marker opacity for the overlaid points.
    inner : {"box", "quartile", "none"}, default="box"
        Inner annotation drawn within each violin.
    title : str, optional
        Axis title.
    xlabel : str, default="Cluster"
        X-axis label.
    ylabel : str, optional
        Y-axis label. Defaults to a label describing ``source``.
    annotation : bool or str, default=True
        True renders a description of the selected model; a string is used
        verbatim; False disables it.
    rotation : float, optional
        Rotation applied to the x tick labels.
    ylim : tuple of float, default=(-0.02, 1.02)
        Y-axis limits.
    fit_ylim : bool, default=True
        Shrink ``ylim`` to the observed data range.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.
    """
    scores, default_ylabel = _scores(adata, key, source)
    return _plot_cluster_violin(
        scores,
        _labels(adata, key),
        ax=ax,
        figsize=figsize,
        order=order,
        palette=palette,
        density_norm=density_norm,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        alpha=alpha,
        inner=inner,
        title=title,
        xlabel=xlabel,
        ylabel=default_ylabel if ylabel is None else ylabel,
        annotation=_annotation_text(adata, key, annotation),
        rotation=rotation,
        ylim=ylim,
        fit_ylim=fit_ylim,
        show=show,
        save=save,
        dpi=dpi,
    )


def cluster_scatter(
    adata: AnnData,
    *,
    key: str = "carve",
    source: Source = "gini",
    basis: str | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    palette: str = "Accent",
    alpha_range: tuple[float, float] = (0.45, 0.9),
    size_range: tuple[float, float] = (15.0, 60.0),
    sort_order: bool = True,
    legend: bool = True,
    legend_loc: str = "right margin",
    annotation: bool | str = True,
    annotation_style: Literal["legend", "box"] = "legend",
    title: str | None = None,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    show_ticks: bool = False,
    frameon: bool = False,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | None:
    """Embedding scatter coloured by cluster, with score encoded as size and alpha.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from :func:`carve.tl.carve`.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    source : {"gini", "ce", "accuracy"}, default="gini"
        Which per-cell score drives point size and opacity.
    basis : str, optional
        Embedding to plot on, with or without the ``X_`` prefix. If None,
        the first available of ``X_umap``, ``X_tsne``, ``X_draw_graph_fa``,
        ``X_pca`` is used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    palette : str, default="Accent"
        Colormap name for the per-cluster colors.
    alpha_range : tuple of float, default=(0.45, 0.9)
        Opacity range mapped from the score.
    size_range : tuple of float, default=(15.0, 60.0)
        Marker size range mapped from the score.
    sort_order : bool, default=True
        Draw lower-scoring points first, so they sit underneath.
    legend : bool, default=True
        Draw the cluster legend.
    legend_loc : str, default="right margin"
        Legend placement.
    annotation : bool or str, default=True
        True renders a description of the selected model; a string is used
        verbatim; False disables it.
    annotation_style : {"legend", "box"}, default="legend"
        Where the annotation is drawn.
    title : str, optional
        Axis title.
    xlabel, ylabel : str
        Axis labels.
    show_ticks : bool, default=False
        Draw axis ticks.
    frameon : bool, default=False
        Draw the axes frame.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.
    """
    scores, score_label = _scores(adata, key, source)
    params = dict(_entry(adata, key)["params"])
    embedding = _anndata.resolve_basis(
        adata, basis, fallback=params.get("use_rep") or None
    )
    return _plot_cluster_scatter(
        _representation(adata, key),
        _labels(adata, key),
        scores,
        embedding=embedding,
        ax=ax,
        figsize=figsize,
        palette=palette,
        alpha_range=alpha_range,
        size_range=size_range,
        sort_order=sort_order,
        legend=legend,
        legend_loc=legend_loc,
        annotation=_annotation_text(adata, key, annotation),
        annotation_style=annotation_style,
        title=title,
        scores_name=score_label,
        xlabel=xlabel,
        ylabel=ylabel,
        show_ticks=show_ticks,
        frameon=frameon,
        show=show,
        save=save,
        dpi=dpi,
    )


def diagnostic_scatter(
    adata: AnnData,
    *,
    key: str = "carve",
    source: Source = "gini",
    basis: str | None = None,
    ax: Axes | None = None,
    figsize: tuple | None = None,
    cmap: str = "Greens_r",
    alpha_encoding: bool = True,
    alpha_range: tuple[float, float] = (0.3, 1.0),
    marker_size: float = 30.0,
    marker_linewidth: float = 0.2,
    markers: list[str] | None = None,
    sort_order: bool = True,
    legend: bool = True,
    legend_loc: str = "right margin",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    annotation: bool | str = True,
    annotation_style: Literal["legend", "box"] = "legend",
    title: str | None = None,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    show_ticks: bool = False,
    frameon: bool = False,
    show: bool = False,
    save: str | Path | None = None,
    dpi: int = 300,
) -> Axes | None:
    """Embedding scatter coloured by diagnostic score, with clusters as markers.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix carrying results from :func:`carve.tl.carve`.
    key : str, default="carve"
        The ``key_added`` used when the results were written.
    source : {"gini", "ce", "accuracy"}, default="gini"
        Which per-cell score drives the color scale.
    basis : str, optional
        Embedding to plot on, with or without the ``X_`` prefix.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.
    figsize : tuple, optional
        Figure size, used only when ``ax`` is None.
    cmap : str, default="Greens_r"
        Colormap for the score.
    alpha_encoding : bool, default=True
        Also encode the score as opacity.
    alpha_range : tuple of float, default=(0.3, 1.0)
        Opacity range used when ``alpha_encoding`` is True.
    marker_size : float, default=30.0
        Marker size.
    marker_linewidth : float, default=0.2
        Marker edge width.
    markers : list of str, optional
        Marker shapes cycled across clusters.
    sort_order : bool, default=True
        Draw lower-scoring points first, so they sit underneath.
    legend : bool, default=True
        Draw the cluster legend.
    legend_loc : str, default="right margin"
        Legend placement.
    colorbar : bool, default=True
        Draw the score colorbar.
    colorbar_label : str, optional
        Colorbar label. Defaults to a label describing ``source``.
    annotation : bool or str, default=True
        True renders a description of the selected model; a string is used
        verbatim; False disables it.
    annotation_style : {"legend", "box"}, default="legend"
        Where the annotation is drawn.
    title : str, optional
        Axis title.
    xlabel, ylabel : str
        Axis labels.
    show_ticks : bool, default=False
        Draw axis ticks.
    frameon : bool, default=False
        Draw the axes frame.
    show : bool, default=False
        Call ``plt.show()`` before returning.
    save : str or pathlib.Path, optional
        Write the figure to this path and return None.
    dpi : int, default=300
        Resolution used when saving.

    Returns
    -------
    ax : matplotlib.axes.Axes or None
        The Axes drawn on, or None when ``save`` is given.
    """
    scores, score_label = _scores(adata, key, source)
    params = dict(_entry(adata, key)["params"])
    embedding = _anndata.resolve_basis(
        adata, basis, fallback=params.get("use_rep") or None
    )
    return _plot_diagnostic_scatter(
        _representation(adata, key),
        _labels(adata, key),
        scores,
        embedding=embedding,
        ax=ax,
        figsize=figsize,
        cmap=cmap,
        alpha_encoding=alpha_encoding,
        alpha_range=alpha_range,
        marker_size=marker_size,
        marker_linewidth=marker_linewidth,
        markers=markers,
        sort_order=sort_order,
        legend=legend,
        legend_loc=legend_loc,
        colorbar=colorbar,
        colorbar_label=(score_label if colorbar_label is None else colorbar_label),
        annotation=_annotation_text(adata, key, annotation),
        annotation_style=annotation_style,
        title=title,
        scores_name=score_label,
        xlabel=xlabel,
        ylabel=ylabel,
        show_ticks=show_ticks,
        frameon=frameon,
        show=show,
        save=save,
        dpi=dpi,
    )
