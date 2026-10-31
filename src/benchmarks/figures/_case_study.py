"""The composite layout shared by Fig 5 and Fig 6.

Marker size, the embedding, the axis labels and panel F are parameters; the
rest is shared.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from .._panels import (
    aligned_color_maps,
    ari_lollipop,
    axis_arrows,
    carve_lines,
    compact_setting,
    cvi_lines,
    panel_letter,
    scatter_clusters,
)
from .._theme import (
    PRINT_FONT_SIZES,
    PRINT_WIDTH_IN,
    print_axes_text,
    print_free_text,
    print_theme_context,
    save_figure,
    save_plos_tiff,
    thin_lines,
)
from .._types import EstimatorSpec
from ._heca_carve import resolution_axis
from ._paths import CASE_STUDY_DIR, figure_path, plos_figure_path

if TYPE_CHECKING:
    from .._cusanovich_compare import CusanovichInputs

# The composite is drawn square at the print width. Its curves are drawn at
# a width that suits a figure twice that size, so at print they take this
# fraction; scatter marker areas are given for a 16-inch figure and scale
# with the area.
_SCREEN_WIDTH_IN = 16.0
_PRINT_CURVE_SCALE = 0.75
_PRINT_MARKER_SCALE = (PRINT_WIDTH_IN / _SCREEN_WIDTH_IN) ** 2

# At the print width the legend strip needs 0.42 of a panel row for the four
# CVI entries and their title in the print theme's compact legend, and the
# rows need more space between them for the axis labels and titles than at
# the screen width. The figure is that much taller than wide and stays
# under PRINT_MAX_HEIGHT_IN.
_PRINT_HEIGHT_IN = 8.0
_LEGEND_ROW_RATIO = 0.42
_ROW_SPACE = 0.3

# The estimator reprs CARVE and the CVI sweep write into legends, as the
# printed figures name them. At the print width the full reprs run across
# the curves or past the legend's column.
_SHORT_ESTIMATOR_NAMES: dict[str, str] = {
    "AgglomerativeClustering, linkage=ward": "Ward agglomerative",
    "AgglomerativeClustering (linkage=ward)": "Ward agglomerative",
    "SpectralClustering, affinity=self_tuning": "Spectral, self-tuning",
    "SpectralClustering (affinity=self_tuning)": "Spectral, self-tuning",
}


def short_estimator_names(text: str) -> str:
    """A legend entry with the known estimator reprs in it shortened."""
    for full, short in _SHORT_ESTIMATOR_NAMES.items():
        text = text.replace(full, short)
    return text


@dataclass(frozen=True)
class CompositeInputs:
    """Everything the composite draws, already computed.

    Kept apart from drawing, so the figure can be tested without fitting
    anything.

    measure, rule and not_two are the caller's own selection: whatever was
    passed to prepare_composite, defaulting to the same ("stability", "1se",
    False) that prepare_composite itself defaults to. carve_output_figure
    reads them to decide which measure drives its detail panels (C to F),
    which of its curve panels (A, B) carries the legend, and which not_two
    setting both curves honor; composite_figure reads not_two to place carve_lines' selected-k
    markers consistently with the same call.

    comparison_resolution is the resolution the CVI winner was fit at when
    the CVIs swept resolution (cvi_resolution_sweep), and None for a k sweep.
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
    comparison_resolution: float | None = None


def _estimator_spec_from_model_label(model: str) -> EstimatorSpec:
    """Map a cvi_sweep ``model`` label back to the estimator it names.

    ``curves_df``/``best_df`` carry the estimator only as the human-readable
    string ``_studies._model_label`` renders (``"KMeans (n_init=10)"``,
    ``"AgglomerativeClustering (linkage=ward)"``); there is no separate
    machine-usable column. Every registered estimator renders to exactly one
    such label from its ``ESTIMATOR_DEFAULTS``, so the match is on the whole
    label.
    """
    from .._estimators import ESTIMATOR_CLASSES, ESTIMATOR_DEFAULTS
    from .._studies import _model_label

    labels = {
        _model_label(cls, ESTIMATOR_DEFAULTS[name]): name
        for name, cls in ESTIMATOR_CLASSES.items()
    }
    if model in labels:
        return EstimatorSpec(name=labels[model])
    raise ValueError(
        f"Cannot map model label {model!r} to a known estimator; expected "
        f"one of {sorted(labels)}."
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

    CARVE's plot_* methods take no ``labels`` argument; each resolves its
    own through ``get_labels``, which relabels onto ``self.reference_labels``
    whenever the two carry the same number of classes. Pointing that at
    reference_codes(y) for the duration of a figure makes those methods draw
    CompositeInputs.carve_labels, so a cluster keeps its color and number
    across the two figures.

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

    align_cluster_labels (carve._utils) is the Hungarian matcher
    CARVE.get_labels uses. Codes come from pandas' sorted category order,
    which is the order aligned_color_maps assigns the palette in.
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


def carve_cluster_numbers(
    inputs: "CompositeInputs | CusanovichInputs",
) -> dict[int, int]:
    """The number CARVE's own figure prints for each id in inputs.carve_labels.

    carve_output_figure draws CARVE's own labels inside carve_labels_aligned
    and numbers each cluster by its id plus one. When the selected k differs
    from the number of reported labels, as on Levine, those ids are CARVE's
    own and not the aligned ids the composite colors by, so the two
    labelings are matched through the cells they share. The Cusanovich
    inputs carry the same fields and are matched the same way.
    """
    with carve_labels_aligned(inputs.carve, inputs.y):
        own = np.asarray(
            inputs.carve.get_labels(
                measure=inputs.measure, rule=inputs.rule, not_two=inputs.not_two
            )
        )
    pairs = pd.crosstab(pd.Series(np.asarray(inputs.carve_labels)), pd.Series(own))
    numbers: dict[int, int] = {}
    for aligned_id, row in pairs.iterrows():
        shared = row[row > 0]
        if len(shared) != 1:
            break
        numbers[int(aligned_id)] = int(shared.index[0]) + 1
    else:
        if len(set(numbers.values())) == len(numbers):
            return numbers
    raise ValueError(
        "inputs.carve_labels and CARVE's own labels are not the same partition, "
        "so CARVE's rows cannot carry its own figure's cluster numbers."
    )


def carve_output_cmap(inputs: CompositeInputs) -> ListedColormap:
    """The composite's CARVE colors, in the order CARVE's own panels use them.

    CARVE's plotting methods color a cluster by its position among CARVE's
    own ids. Entry j is the composite's color for the cluster CARVE numbers
    j + 1, so a cluster keeps its color across the composite, the
    composition figure and carve_output_figure. Where CARVE's ids already
    are the aligned ones, as on Klein, this is the palette's first k colors.
    """
    _, carve_cmap, _ = composite_color_maps(inputs)
    return carve_order_cmap(carve_cmap, carve_cluster_numbers(inputs))


def carve_order_cmap(
    colors: Mapping[int, str], numbers: Mapping[int, int]
) -> ListedColormap:
    """Colors keyed by aligned id, as the map CARVE's own panels index.

    ``numbers`` is carve_cluster_numbers' result. Entry j of the map is the
    color of the aligned id CARVE numbers j + 1.
    """
    color_of = {number: colors[aligned_id] for aligned_id, number in numbers.items()}
    return ListedColormap([color_of[number] for number in sorted(color_of)])


_ARI_ROW_COLUMNS: tuple[str, ...] = ("method", "metric", "ari", "k")


#: How METRIC_COLORS spells each selection rule after the measure: the 1-SE
#: rule as "_1se", the quantile rule as "_quant", and the maximum bare.
_RULE_SUFFIX = {"1se": "_1se", "quantile": "_quant", "max": ""}


def ari_table(
    inputs: CompositeInputs,
    *,
    extra_rows: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """ARI of each selection against the reported labels.

    Shared by the Levine figure (through ari_panel, where it is the bottom
    panel's lollipop chart) and by figure_comparators.

    extra_rows appends comparators that do not come out of best_df, which is
    the CVI sweep's own output and has no room for a tool that is not a CVI.
    figure_comparators passes one M3C row and one SC3 row here; ari_panel
    passes none.
    """
    rows = [
        {
            "method": "CARVE",
            "metric": f"ari_{inputs.measure}{_RULE_SUFFIX[inputs.rule]}",
            "ari": float(adjusted_rand_score(inputs.y, inputs.carve_labels)),
            "k": int(len(set(np.asarray(inputs.carve_labels).tolist()))),
        }
    ]
    for _, row in inputs.best_df.iterrows():
        rows.append(
            {
                "method": str(row["metric"]).replace("_", " ").title(),
                "metric": str(row["metric"]),
                "ari": float(row["ari"]),
                "k": int(row["k"]),
            }
        )
    for extra in extra_rows or ():
        missing = [name for name in _ARI_ROW_COLUMNS if name not in extra]
        if missing:
            raise ValueError(
                f"An extra ARI row is missing {missing}. Every row needs "
                f"{list(_ARI_ROW_COLUMNS)}, so the lollipop can color it by "
                "metric and annotate it with k."
            )
        rows.append({name: extra[name] for name in _ARI_ROW_COLUMNS})
    return pd.DataFrame(rows)


def ari_panel(ax: Axes, inputs: CompositeInputs) -> Axes:
    return ari_lollipop(
        ax, ari_table(inputs), title="Agreement with Reported Labels (ARI)"
    )


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
    model_grids: list[tuple[type, dict[str, list[Any]]]] | None = None,
) -> CompositeInputs:
    """Assemble the composite's inputs, computing a PCA embedding if needed.

    Fits a PCA when no embedding is supplied, and always fits the comparison
    estimator named by best_df's winning row for comparison_metric at that
    row's k. Call it once per study and reuse the result for both of that
    study's figures.

    When best_df comes from cvi_resolution_sweep (it has a resolution
    column), the winner is refit at its resolution from model_grids, the
    grids the CVIs swept: its model label is matched against each grid's
    estimator and fixed parameters. A Leiden on a non-default graph has no
    registered spec to rebuild it from by label alone.

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
    comparison_resolution = None
    if "resolution" in best_df.columns:
        comparison_resolution = float(best_row["resolution"].iloc[0])
        estimator = _estimator_from_swept_grids(
            str(best_row["model"].iloc[0]),
            comparison_resolution,
            model_grids,
            random_state=random_state,
        )
        comparison_labels = np.asarray(estimator.fit_predict(X))
        comparison_k = int(np.unique(comparison_labels).size)
    else:
        comparison_k = int(best_row["k"].iloc[0])
        comparison_spec = _estimator_spec_from_model_label(
            str(best_row["model"].iloc[0])
        )
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
        comparison_resolution=comparison_resolution,
    )


def _swept_resolutions(carve: Any) -> list[float] | None:
    """The resolutions a CARVE fit swept, or None when it swept k."""
    sweep = getattr(carve, "sweep_", None)
    if sweep is None or sweep.param != "resolution":
        return None
    return sorted(float(r) for r in carve.estimator_results_["sweep_value"].unique())


def _estimator_from_swept_grids(
    model: str,
    resolution: float,
    model_grids: list[tuple[type, dict[str, list[Any]]]] | None,
    *,
    random_state: int,
) -> Any:
    """The estimator a cvi_resolution_sweep model label names, at one resolution."""
    from itertools import product

    from .._estimators import apply_random_state
    from .._studies import _model_label

    if model_grids is None:
        raise ValueError(
            "best_df comes from a resolution sweep; pass the model_grids it "
            "swept so the winner can be refit at its resolution."
        )
    rendered = []
    for estimator_cls, grid in model_grids:
        keys = [key for key in grid if key != "resolution"]
        for combo in product(*(grid[key] for key in keys)):
            fixed = dict(zip(keys, combo))
            label = _model_label(estimator_cls, fixed)
            if label == model:
                params = {**fixed, "resolution": float(resolution)}
                apply_random_state(estimator_cls, params, random_state)
                return estimator_cls(**params)
            rendered.append(label)
    raise ValueError(
        f"No swept grid renders the model label {model!r}; the grids render {rendered}."
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
    curves_first: bool = False,
    plos_number: int | None = None,
) -> Figure:
    """Draw the six-panel case-study composite.

    Panels A, B and C are scatters of the reported labels, the CARVE
    clustering, and the CVI clustering; D and E are the CARVE and CVI curves
    over k, or over resolution on a log axis for a resolution run; F is
    supplied by the caller. With ``curves_first`` the curves
    take the top row as A and B, their legends the strip below them, and
    the scatters the row after that as C, D and E.

    ``axis_labels`` names the embedding's two directions. Only panel A shows
    them, as a corner arrow pair; all three scatters share one embedding.

    The figure is laid out at PRINT_WIDTH_IN under print_theme_context, and
    the panel helpers' screen text sizes are reset to the print sizes.
    ``plos_number`` names the PLOS figure file saved beside the PNG; an SI
    figure, which PLOS takes in any format, leaves it None.
    """
    true_cmap, carve_cmap, comparison_cmap = composite_color_maps(inputs)
    carve_resolutions = _swept_resolutions(inputs.carve)
    cvi_over_resolution = "resolution" in inputs.curves_df.columns
    # A resolution axis stands its tick labels upright, and they hang into
    # the legend strip under the curves; the strip grows to hold both.
    legend_height = _LEGEND_ROW_RATIO + (
        0.05 if carve_resolutions is not None or cvi_over_resolution else 0.0
    )
    marker_size = marker_size * _PRINT_MARKER_SCALE

    with print_theme_context():
        fig = plt.figure(
            figsize=(PRINT_WIDTH_IN, _PRINT_HEIGHT_IN), constrained_layout=False
        )
        if curves_first:
            curve_row, legend_row, scatter_row = 0, 1, 2
            height_ratios = [1.0, legend_height, 1.0, 1.0]
        else:
            scatter_row, curve_row, legend_row = 0, 1, 2
            height_ratios = [1.0, 1.0, legend_height, 1.0]
        gs = fig.add_gridspec(
            4,
            6,
            height_ratios=height_ratios,
            hspace=_ROW_SPACE,
            wspace=0.8,
            left=0.08,
            right=0.98,
            top=0.97,
            bottom=0.03,
        )
        ax_true = fig.add_subplot(gs[scatter_row, 0:2])
        ax_carve = fig.add_subplot(gs[scatter_row, 2:4])
        ax_cvi = fig.add_subplot(gs[scatter_row, 4:6])
        ax_carve_curves = fig.add_subplot(gs[curve_row, 0:3])
        ax_cvi_curves = fig.add_subplot(gs[curve_row, 3:6])
        ax_bottom = fig.add_subplot(gs[3, 1:5])

        legend_ax_carve = fig.add_subplot(gs[legend_row, 0:3])
        legend_ax_carve.set_axis_off()
        legend_ax_cvi = fig.add_subplot(gs[legend_row, 3:6])
        legend_ax_cvi.set_axis_off()

        scatter_clusters(
            ax_true,
            inputs.Z,
            inputs.y,
            color_map=true_cmap,
            s=marker_size,
            title="Reported Labels",
        )
        axis_arrows(ax_true, axis_labels)
        scatter_clusters(
            ax_carve,
            inputs.Z,
            inputs.carve_labels,
            color_map=carve_cmap,
            s=marker_size,
            title="CARVE clustering",
        )
        scatter_clusters(
            ax_cvi,
            inputs.Z,
            inputs.comparison_labels,
            color_map=comparison_cmap,
            s=marker_size,
            title=(
                f"CVI ({inputs.comparison_name}, k={inputs.comparison_k})"
                if inputs.comparison_resolution is None
                else f"CVI ({inputs.comparison_name}, resolution "
                f"{inputs.comparison_resolution:g}, k={inputs.comparison_k})"
            ),
        )

        carve_lines(
            ax_carve_curves,
            inputs.carve,
            not_two=inputs.not_two,
            title=f"CARVE ARI over {'k' if carve_resolutions is None else 'resolution'}",
        )
        cvi_lines(
            ax_cvi_curves,
            inputs.curves_df,
            inputs.best_df,
            title=f"CVIs over {'resolution' if cvi_over_resolution else 'k'}",
        )
        # A log-spaced resolution grid crowds its low end into one tick on a
        # linear axis, and that is where the coarse selections sit.
        if carve_resolutions is not None:
            resolution_axis(ax_carve_curves, carve_resolutions)
        if cvi_over_resolution:
            resolution_axis(
                ax_cvi_curves, sorted(inputs.curves_df["resolution"].unique())
            )

        bottom_panel(ax_bottom, inputs)
        for ax in (ax_carve_curves, ax_cvi_curves):
            thin_lines(ax, _PRINT_CURVE_SCALE)
        # Before the legend strip: a resolution run's legends are placed
        # under the curve panels' tick and axis labels at their print size.
        for ax in fig.get_axes():
            print_axes_text(ax)
            print_free_text(ax)

        # Move the two panel legends into their own strip.
        for source_ax, target_ax, title in (
            (ax_carve_curves, legend_ax_carve, "CARVE"),
            (ax_cvi_curves, legend_ax_cvi, "CVIs"),
        ):
            handles, labels = source_ax.get_legend_handles_labels()
            existing = source_ax.get_legend()
            if existing is not None:
                existing.remove()
            # A resolution run's upright tick labels and axis label hang into
            # the strip, so its legends start just below them.
            over_resolution = (
                carve_resolutions is not None
                if source_ax is ax_carve_curves
                else cvi_over_resolution
            )
            placement: dict[str, Any] = {"loc": "center"}
            if over_resolution:
                labels_bottom = (
                    source_ax.get_tightbbox().transformed(fig.transFigure.inverted()).y0
                )
                strip = target_ax.get_position()
                placement = {
                    "loc": "upper center",
                    "bbox_to_anchor": (0.5, (labels_bottom - strip.y0) / strip.height),
                }
            target_ax.legend(
                handles,
                [compact_setting(short_estimator_names(label)) for label in labels],
                title=title,
                ncol=1,
                frameon=False,
                fontsize=PRINT_FONT_SIZES["legend"],
                **placement,
            )

        scatters = (ax_true, ax_carve, ax_cvi)
        curves = (ax_carve_curves, ax_cvi_curves)
        ordered = (*curves, *scatters) if curves_first else (*scatters, *curves)
        for letter, ax in zip("ABCDEF", (*ordered, ax_bottom)):
            panel_letter(ax, letter, fontsize=PRINT_FONT_SIZES["panel_letter"])

        if save:
            save_figure(
                fig,
                figure_path(save_name, subdir=CASE_STUDY_DIR, out_dir=out_dir),
            )
            if plos_number is not None:
                save_plos_tiff(fig, plos_figure_path(plos_number, out_dir=out_dir))
    return fig
