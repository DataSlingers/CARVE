import numpy as np
import pandas as pd
import pytest

from matplotlib.collections import PathCollection
from sklearn.metrics import adjusted_rand_score

from benchmarks.figures import figure_heca_results
from benchmarks.figures._heca_results import AXIS_LABELS, subsample_inputs


@pytest.fixture
def inputs():
    from benchmarks.figures._case_study import CompositeInputs

    rng = np.random.default_rng(0)
    n = 600
    y = rng.choice(["Lung", "Brain", "Kidney"], n)
    curves = pd.DataFrame(
        {
            "metric": np.repeat(["silhouette", "gap"], 5),
            "model": "MiniBatchKMeans",
            "k": list(range(3, 8)) * 2,
            "score": rng.random(10),
            "ari": rng.random(10),
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["MiniBatchKMeans"] * 2,
            "k": [3, 5],
            "score": [0.4, 0.3],
            "ari": [0.5, 0.4],
        }
    )

    class _Carve:
        # composite_figure's panel D draws through _panels.carve_lines,
        # which reads estimator_results_/_select_row/get_k -- not a
        # plot_metric_over_n_clusters method. Mirrors the _StubCarve in
        # tests/benchmarks/figures/test_case_study.py.
        def __init__(self):
            ks = [3, 4, 5]
            self.estimator_results_ = pd.DataFrame(
                {
                    "n_clusters": ks,
                    "method_id": ["m0"] * len(ks),
                    "method_label": ["MiniBatchKMeans"] * len(ks),
                    "ari_stability": [0.1, 0.2, 0.3],
                    "ari_generalizability": [0.15, 0.25, 0.35],
                }
            )

        def _select_row(self, *, measure, rule="1se", not_two=False):
            row = self.estimator_results_.iloc[1]
            return row, 0, int(row["n_clusters"]), False

        def get_k(self, *, measure="stability", rule="1se", not_two=False):
            return 4

    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=rng.normal(size=(n, 2)),
        carve=_Carve(),
        carve_labels=rng.integers(0, 3, n),
        comparison_labels=rng.integers(0, 3, n),
        comparison_name="Silhouette",
        comparison_k=3,
        curves_df=curves,
        best_df=best,
    )


class TestSubsampleInputs:
    def test_reduces_every_per_cell_array_together(self, inputs):
        out = subsample_inputs(inputs, size=100)
        assert out.Z.shape[0] == 100
        assert out.y.shape[0] == 100
        assert out.carve_labels.shape[0] == 100
        assert out.comparison_labels.shape[0] == 100
        assert out.X.shape[0] == 100

    def test_rows_stay_aligned_across_arrays(self, inputs):
        # Independently subsampling each array would silently decouple a
        # cell's label from its position, which no shape assertion catches.
        #
        # y and carve_labels alone are too coarse a fingerprint to catch
        # that: with only 3 tissues and 3 cluster ids, all 9 combinations
        # already occur among the fixture's 600 rows, so a decoupled (y,
        # carve_labels) pair still lands on some real row's combination by
        # chance -- confirmed by mutating subsample_inputs to draw each
        # array's index independently: that broken version still passed a
        # membership check on (y, carve_labels) alone. X's continuous,
        # effectively-unique rows are folded into the fingerprint so a
        # decoupled row cannot coincidentally match an unrelated one.
        out = subsample_inputs(inputs, size=50, random_state=1)
        again = subsample_inputs(inputs, size=50, random_state=1)
        assert np.array_equal(out.y, again.y)
        assert np.array_equal(out.carve_labels, again.carve_labels)

        def _fingerprint(y, label, x_row):
            return (str(y), int(label), tuple(np.round(x_row, 8)))

        rows = {
            _fingerprint(a, b, x)
            for a, b, x in zip(inputs.y, inputs.carve_labels, inputs.X)
        }
        for a, b, x in zip(out.y, out.carve_labels, out.X):
            assert _fingerprint(a, b, x) in rows

    def test_a_size_at_or_above_n_is_a_passthrough(self, inputs):
        out = subsample_inputs(inputs, size=10_000)
        assert out.y.shape[0] == inputs.y.shape[0]

    def test_non_per_cell_fields_are_preserved(self, inputs):
        out = subsample_inputs(inputs, size=100)
        assert out.comparison_k == inputs.comparison_k
        assert out.comparison_name == inputs.comparison_name
        assert out.curves_df.equals(inputs.curves_df)


def test_axis_labels_name_the_umap():
    # The source ships no coordinates, so the notebook computes a UMAP on
    # the loader's principal components, the way the source publication
    # visualizes, and passes it to prepare_composite; the overview scatter
    # that opens the notebook draws the same embedding. Labeling the panels
    # PC1/PC2 would misreport what is drawn.
    assert AXIS_LABELS == ("UMAP 1", "UMAP 2")


def test_figure_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_heca_results(inputs, scatter_subsample=200, save=True, out_dir=tmp_path)
    assert (tmp_path / "heca_results.png").is_file()


def test_figure_does_not_write_when_save_is_false(inputs, tmp_path):
    fig = figure_heca_results(inputs, scatter_subsample=200, save=False,
                              out_dir=tmp_path)
    assert len([ax for ax in fig.axes if ax.has_data() or ax.get_title()]) >= 6
    assert list(tmp_path.iterdir()) == []


class _CarveForAriPanel:
    """Minimal stand-in for panel D; irrelevant to panel F's ARI value.

    See TestSubsampleInputs._Carve/the fixture above for the same shape --
    duplicated rather than shared because these two test files predate this
    one (a Minor finding already deferred to final review).
    """

    def __init__(self):
        ks = [3, 4, 5]
        self.estimator_results_ = pd.DataFrame(
            {
                "n_clusters": ks,
                "method_id": ["m0"] * len(ks),
                "method_label": ["MiniBatchKMeans"] * len(ks),
                "ari_stability": [0.1, 0.2, 0.3],
                "ari_generalizability": [0.15, 0.25, 0.35],
            }
        )

    def _select_row(self, *, measure, rule="1se", not_two=False):
        row = self.estimator_results_.iloc[1]
        return row, 0, int(row["n_clusters"]), False

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return 4


def _plotted_ari_by_method(fig):
    """Read panel F's actual plotted (method -> ARI) values off its axes.

    ari_lollipop draws both an hlines LineCollection and a scatter
    PathCollection; LineCollection also exposes get_offsets(), so the
    PathCollection must be selected by isinstance, not by hasattr -- an
    earlier ad hoc version of this check picked the wrong collection and
    silently read a degenerate (0, 0) point instead of the real data.
    """
    ari_ax = next(ax for ax in fig.axes if "ARI" in (ax.get_xlabel() or ""))
    labels = [t.get_text() for t in ari_ax.get_yticklabels()]
    scatter = next(c for c in ari_ax.collections if isinstance(c, PathCollection))
    return dict(zip(labels, (point[0] for point in scatter.get_offsets())))


class TestPanelFUsesTheFullPopulation:
    """Regression guard for the fix in figure_heca_results.

    The brief wired composite_figure(subsample_inputs(inputs, ...),
    bottom_panel=_ari_panel, ...), which would score CARVE's ARI against
    the scatter-legibility subsample while best_df's CVI rows are already
    computed at full scale -- silently comparing the two methods on
    different populations. figure_heca_results now closes panel F over the
    full, unsubsampled inputs instead. Nothing else in the committed suite
    inspects panel F's actual plotted value, so a future revert of
    bottom_panel back to _ari_panel would pass every other test here.

    The fixture below corrupts the first 400 of 1000 "a"-labeled cells so
    the full-population ARI is well below 1.0, then confirms (rather than
    assumes) that a subsample drawn with the module's own default
    random_state produces a numerically different ARI -- if it did not,
    the assertion below would not be able to tell the fix from the revert
    it guards against.
    """

    @pytest.fixture
    def lopsided_inputs(self):
        from benchmarks.figures._case_study import CompositeInputs

        rng = np.random.default_rng(7)
        n = 2000
        y = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        carve_labels = np.where(y == "a", 0, 1).astype(np.int64)
        # Mislabel the first fifth of the population. Concentrating the
        # corruption in one contiguous block, rather than spreading it
        # evenly, is what makes a random subsample's corrupted fraction
        # diverge from the full population's.
        carve_labels[:400] = 1
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["MiniBatchKMeans"],
                "k": [3],
                "score": [0.4],
                "ari": [0.5],
            }
        )
        return CompositeInputs(
            X=rng.normal(size=(n, 5)),
            y=y,
            Z=rng.normal(size=(n, 2)),
            carve=_CarveForAriPanel(),
            carve_labels=carve_labels,
            comparison_labels=rng.integers(0, 2, n),
            comparison_name="Silhouette",
            comparison_k=3,
            curves_df=best.copy(),
            best_df=best,
        )

    def test_carve_ari_matches_the_full_population_not_the_scatter_subsample(
        self, lopsided_inputs
    ):
        subsample_size = 200
        full_ari = adjusted_rand_score(
            lopsided_inputs.y, lopsided_inputs.carve_labels
        )
        subsampled = subsample_inputs(lopsided_inputs, size=subsample_size)
        subsample_ari = adjusted_rand_score(subsampled.y, subsampled.carve_labels)

        # Confirm the fixture actually distinguishes the two populations --
        # if this ever failed, the assertion below would prove nothing.
        assert full_ari != pytest.approx(subsample_ari, abs=1e-6)

        fig = figure_heca_results(
            lopsided_inputs, scatter_subsample=subsample_size, save=False
        )
        plotted = _plotted_ari_by_method(fig)

        assert plotted["CARVE"] == pytest.approx(full_ari, abs=1e-9)
        assert plotted["CARVE"] != pytest.approx(subsample_ari, abs=1e-6)
