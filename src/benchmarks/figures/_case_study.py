"""The composite layout shared by Fig 5 and Fig 6.

The two notebook cells this replaces were 178 lines each and differed in four
things: marker size, the embedding, the axis labels, and panel F. Those are
parameters here; the remaining 174 lines are shared.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .._panels import (
    aligned_color_maps,
    axis_arrows,
    carve_lines,
    cvi_lines,
    panel_letter,
    scatter_clusters,
)
from .._theme import FONT_SIZES, save_figure, theme_context
from .._types import EstimatorSpec
from ._paths import CASE_STUDY_DIR, figure_path


@dataclass(frozen=True)
class CompositeInputs:
    """Everything the composite draws, already computed.

    Assembling this is compute; drawing it is reporting. Keeping them apart is
    what lets the figure be tested without fitting anything.

    measure, rule and not_two are the caller's own selection -- whatever was
    passed to prepare_composite, defaulting to the same ("stability", "1se",
    False) that prepare_composite itself defaults to. carve_output_figure
    reads them to decide which measure drives its detail panels (B, D, E, F)
    and which not_two setting its fixed-measure overview panels (A and C)
    honor; composite_figure reads not_two to place carve_lines' selected-k
    markers consistently with the same call.
    """

    X: np.ndarray
    y: np.ndarray
    Z: np.ndarray
    carve: Any
    carve_labels: np.ndarray
    comparison_labels: np.ndarray
    comparison_name: str
    comparison_k: int
    curves_df: pd.DataFrame
    best_df: pd.DataFrame
    measure: str = "stability"
    rule: str = "1se"
    not_two: bool = False


def _estimator_spec_from_model_label(model: str) -> EstimatorSpec:
    """Map a cvi_sweep ``model`` label back to the estimator it names.

    ``curves_df``/``best_df`` carry the estimator only as the human-readable
    string ``_studies._model_label`` renders (``"KMeans"``, or
    ``"AgglomerativeClustering (linkage=ward)"`` once a fixed parameter is
    present) -- there is no separate machine-usable column, and the sweep
    never varies a parameter beyond one estimator's ``ESTIMATOR_DEFAULTS``,
    so matching the label's leading class name back to the registered
    estimator is exact, not a guess.
    """
    from .._estimators import ESTIMATOR_CLASSES

    for name, cls in ESTIMATOR_CLASSES.items():
        if model == cls.__name__ or model.startswith(f"{cls.__name__} ("):
            return EstimatorSpec(name=name)
    raise ValueError(
        f"Cannot map model label {model!r} to a known estimator; expected "
        f"one of {sorted(cls.__name__ for cls in ESTIMATOR_CLASSES.values())}."
    )


def reference_codes(y: np.ndarray) -> np.ndarray:
    """The reported labels as integer codes, in sorted category order.

    One definition, used both by _align_to_reference and by
    carve_labels_aligned, so the ids the composite draws and the ids CARVE's
    own figure draws are relabeled against the same thing.
    """
    return np.asarray(pd.Categorical(y).codes, dtype=np.int64)


@contextmanager
def carve_labels_aligned(carve: Any, y: np.ndarray) -> Iterator[None]:
    """Make CARVE's own plotting draw the ids the composite draws.

    CARVE's plot_* methods take no ``labels`` argument -- each resolves its
    own through ``get_labels``, which relabels onto ``self.reference_labels``
    whenever the two carry the same number of classes. Pointing that at
    reference_codes(y) for the duration of a figure is therefore the only
    hook available, and it is enough: the labels those methods draw come out
    identical to CompositeInputs.carve_labels, so a cluster keeps both its
    color and its number across the two figures.

    When the clustering and the reported labels disagree on class count
    there is nothing to align and CARVE falls back to its own ids. It then
    anchors every later call to the first one's labels, so the figure stays
    self-consistent; it just cannot agree with the composite.

    The attribute is restored on exit. The fitted object is shared with the
    composite figure and is usually a loaded cache the caller draws from
    more than once.
    """
    previous = getattr(carve, "reference_labels", None)
    carve.reference_labels = reference_codes(y)
    try:
        yield
    finally:
        carve.reference_labels = previous


def _align_to_reference(labels: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Relabel a clustering onto the reported labels' own codes.

    Cluster ids are arbitrary: the same partition can come back as 0,1,2,3 or
    3,0,2,1 from one estimator to the next. Every panel that colors by
    cluster id, and the alluvial's C1..Ck names, only mean something once
    those ids are matched to the reported labels they correspond to.

    align_cluster_labels is the Hungarian matcher CARVE.get_labels already
    uses internally. It is reused here rather than reimplemented so the two
    cannot drift apart, even though it sits in carve._utils rather than on
    the public surface. Codes come from pandas' sorted category order, which
    is the same order aligned_color_maps assigns the palette in.
    """
    from carve._utils import align_cluster_labels

    return np.asarray(align_cluster_labels(reference_codes(y), np.asarray(labels)))


def composite_color_maps(
    inputs: CompositeInputs,
) -> tuple[dict[Any, str], dict[Any, str], dict[Any, str]]:
    """The (reported, CARVE, comparison) color maps the composite draws with.

    One call, one shared palette, so the scatter panels and the alluvial in
    the same figure cannot disagree about which color a cluster is.
    """
    true_cmap, carve_cmap, comparison_cmap = aligned_color_maps(
        inputs.y, inputs.carve_labels, inputs.comparison_labels
    )
    return true_cmap, carve_cmap, comparison_cmap


def prepare_composite(
    X: np.ndarray,
    y: np.ndarray,
    carve: Any,
    *,
    curves_df: pd.DataFrame,
    best_df: pd.DataFrame,
    comparison_metric: str = "silhouette",
    embedding: np.ndarray | None = None,
    measure: str = "stability",
    rule: str = "1se",
    not_two: bool = False,
    random_state: int = 42,
) -> CompositeInputs:
    """Assemble the composite's inputs, computing a PCA embedding if needed.

    Fits a PCA when no embedding is supplied, and always fits the comparison
    estimator named by best_df's winning row for comparison_metric (not a
    fixed choice) at that row's k. This is compute, not reporting -- call it
    once per study and reuse the returned CompositeInputs across both of that
    study's figures, rather than calling it again inside a render loop.

    Both clusterings come back relabeled onto the reported labels' own codes,
    so cluster *i* means "the cluster best matching reported label *i*" in
    every panel that draws them. See _align_to_reference.
    """
    from sklearn.decomposition import PCA

    from .._estimators import build_estimator

    X = np.asarray(X)
    y = np.asarray(y)

    Z = (
        np.asarray(embedding)
        if embedding is not None
        else PCA(n_components=2, random_state=random_state).fit_transform(X)
    )

    carve_labels = np.asarray(
        carve.get_labels(measure=measure, rule=rule, not_two=not_two)
    )

    best_row = best_df.loc[best_df["metric"] == comparison_metric]
    if best_row.empty:
        raise ValueError(
            f"No row for {comparison_metric!r} in best_df; "
            f"available metrics are {sorted(best_df['metric'].unique())}."
        )
    comparison_k = int(best_row["k"].iloc[0])
    comparison_spec = _estimator_spec_from_model_label(str(best_row["model"].iloc[0]))
    estimator = build_estimator(
        comparison_spec,
        n_clusters=comparison_k,
        random_state=random_state,
    )
    comparison_labels = np.asarray(estimator.fit_predict(X))

    carve_labels = _align_to_reference(carve_labels, y)
    comparison_labels = _align_to_reference(comparison_labels, y)

    return CompositeInputs(
        X=X,
        y=y,
        Z=Z,
        carve=carve,
        carve_labels=carve_labels,
        comparison_labels=comparison_labels,
        comparison_name=comparison_metric.title(),
        comparison_k=comparison_k,
        curves_df=curves_df,
        best_df=best_df,
        measure=measure,
        rule=rule,
        not_two=not_two,
    )


def composite_figure(
    inputs: CompositeInputs,
    *,
    bottom_panel: Callable[[Axes, CompositeInputs], Axes],
    marker_size: float,
    axis_labels: Sequence[str],
    save_name: str,
    save: bool = True,
    out_dir: Path | None = None,
) -> Figure:
    """Draw the six-panel case-study composite.

    Panels A, B and C are scatters of the reported labels, the CARVE
    clustering, and the CVI clustering; D and E are the CARVE and CVI curves
    over k; F is supplied by the caller.

    ``axis_labels`` names the embedding's two directions. Only panel A shows
    them, as a corner arrow pair rather than as xlabel/ylabel: all three
    scatters share one embedding, so repeating the names under B and C says
    nothing the reader does not already have from A.
    """
    true_cmap, carve_cmap, comparison_cmap = composite_color_maps(inputs)

    with theme_context():
        fig = plt.figure(figsize=(16, 16), constrained_layout=False)
        gs = fig.add_gridspec(
            4, 6, height_ratios=[1.0, 1.0, 0.3, 1.0], hspace=0.2, wspace=0.8
        )
        ax_a = fig.add_subplot(gs[0, 0:2])
        ax_b = fig.add_subplot(gs[0, 2:4])
        ax_c = fig.add_subplot(gs[0, 4:6])
        ax_d = fig.add_subplot(gs[1, 0:3])
        ax_e = fig.add_subplot(gs[1, 3:6])
        ax_f = fig.add_subplot(gs[3, 1:5])

        legend_ax_d = fig.add_subplot(gs[2, 0:3])
        legend_ax_d.set_axis_off()
        legend_ax_e = fig.add_subplot(gs[2, 3:6])
        legend_ax_e.set_axis_off()

        scatter_clusters(
            ax_a,
            inputs.Z,
            inputs.y,
            color_map=true_cmap,
            s=marker_size,
            title="Reported Labels",
        )
        axis_arrows(ax_a, axis_labels)
        scatter_clusters(
            ax_b,
            inputs.Z,
            inputs.carve_labels,
            color_map=carve_cmap,
            s=marker_size,
            title="CARVE clustering",
        )
        scatter_clusters(
            ax_c,
            inputs.Z,
            inputs.comparison_labels,
            color_map=comparison_cmap,
            s=marker_size,
            title=f"CVI ({inputs.comparison_name}, k={inputs.comparison_k})",
        )

        carve_lines(
            ax_d, inputs.carve, not_two=inputs.not_two, title="CARVE ARI over k"
        )
        cvi_lines(ax_e, inputs.curves_df, inputs.best_df, title="CVIs over k")

        bottom_panel(ax_f, inputs)

        # Move the two panel legends into their own strip. Building each one
        # exactly once; the cell this replaces assigned leg_D twice and threw
        # the first away.
        for source_ax, target_ax, title in (
            (ax_d, legend_ax_d, "CARVE"),
            (ax_e, legend_ax_e, "CVIs"),
        ):
            handles, labels = source_ax.get_legend_handles_labels()
            existing = source_ax.get_legend()
            if existing is not None:
                existing.remove()
            target_ax.legend(
                handles,
                labels,
                title=title,
                loc="center",
                ncol=1,
                frameon=False,
                fontsize=FONT_SIZES["legend"],
            )

        for letter, ax in zip("ABCDEF", (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f)):
            panel_letter(ax, letter)

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
    return fig
