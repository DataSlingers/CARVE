"""Tests for the Levine composition figure and its tables."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._estimators import param_grids
from benchmarks._types import EstimatorSpec
from benchmarks.datasets._levine import _POPULATION_NAMES
from benchmarks.figures._case_study import (
    CompositeInputs,
    _align_to_reference,
    carve_labels_aligned,
)
from benchmarks.figures._levine_composition import (
    POPULATION_ORDER,
    carve_cluster_numbers,
    cluster_composition,
    composition_tables,
)
from carve import CARVE
from tests.benchmarks._helpers import StubCarve

# The 5,000-cell subsample's population sizes, in POPULATION_ORDER (the
# notebook's population table).
SIZES = [1265, 965, 187, 108, 25, 294, 793, 16, 1013, 59, 58, 44, 158, 15]


def _inputs(*, X, y, carve, carve_labels, comparison_labels):
    """CompositeInputs with the fields these tests never read left minimal."""
    best = pd.DataFrame(
        {"metric": ["silhouette"], "model": ["KMeans (n_init=100)"], "k": [7]}
    )
    return CompositeInputs(
        X=np.asarray(X),
        y=np.asarray(y),
        Z=np.zeros((len(y), 2)),
        carve=carve,
        carve_labels=np.asarray(carve_labels),
        comparison_labels=np.asarray(comparison_labels),
        comparison_name="Silhouette",
        comparison_k=int(len(np.unique(comparison_labels))),
        curves_df=best,
        best_df=best,
    )


def _stub(own_labels):
    """A StubCarve whose get_labels returns own_labels, selecting KMeans."""
    k = int(len(np.unique(own_labels)))
    results = pd.DataFrame(
        {"method_id": ["m0"], "method_label": ["KMeans, n_init=100"], "n_clusters": [k]}
    )
    return StubCarve(
        results, select=lambda measure, not_two: ("m0", k), labels=np.asarray(own_labels)
    )


def _levine_like_inputs():
    """Inputs at Levine's sizes: 14 populations, 10 CARVE and 7 CVI clusters.

    CARVE's own ids are a permutation of the aligned ones, as on Levine,
    where k=10 differs from the 14 populations. The CVI partition pools CD4 T,
    CD8 T and CD16- NK cells into one cluster. A plain function, so a script
    can draw the figure from it too.
    """
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(14), SIZES)
    y = np.array(POPULATION_ORDER, dtype=object)[codes]
    X = rng.normal(size=(len(y), 32)) + 3.0 * (codes[:, None] % 5 == np.arange(32) % 5)
    carve_of = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9])
    comparison_of = np.array([0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6])
    own = (carve_of[codes] * 3) % 10
    return _inputs(
        X=X,
        y=y,
        carve=_stub(own),
        carve_labels=_align_to_reference(own, y),
        comparison_labels=_align_to_reference(comparison_of[codes], y),
    )


@pytest.fixture
def levine_like():
    return _levine_like_inputs()


@pytest.fixture
def fitted():
    """A small real fit whose selected k differs from the reported label count.

    Three blobs carry four reported labels, so inside carve_labels_aligned
    CARVE keeps its own ids, as it does on Levine (k=10 against 14
    populations), while the aligned ids follow the reported codes.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=6.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=(0.0, 6.0, 0.0), scale=0.4, size=(40, 3)),
    ])
    y = np.array(["d"] * 40 + ["c"] * 40 + ["b"] * 35 + ["a"] * 5)
    carve = CARVE(
        estimator_param_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
        n_resamples=3,
        random_state=0,
    )
    carve.fit(X, reference_labels=y)
    aligned = _align_to_reference(carve.get_labels(measure="stability", rule="1se"), y)
    return _inputs(X=X, y=y, carve=carve, carve_labels=aligned, comparison_labels=aligned)


class TestPopulationOrder:
    def test_lists_every_gated_population_the_loader_names_once(self):
        assert len(POPULATION_ORDER) == 14
        assert set(POPULATION_ORDER) == set(_POPULATION_NAMES.values())


class TestClusterComposition:
    def test_a_merged_cluster_has_one_count_per_population_it_holds(self):
        y = ["CD4_T_cells"] * 5 + ["CD8_T_cells"] * 3 + ["Monocytes"] * 4
        counts = cluster_composition([0] * 8 + [1] * 4, y, order=POPULATION_ORDER)
        assert counts.loc[0, "CD4_T_cells"] == 5
        assert counts.loc[0, "CD8_T_cells"] == 3
        assert (counts.loc[0] > 0).sum() == 2
        assert counts.loc[1, "Monocytes"] == 4
        assert (counts.loc[1] > 0).sum() == 1

    def test_every_cell_is_counted_once_in_its_own_cluster(self):
        rng = np.random.default_rng(0)
        y = rng.choice(np.array(POPULATION_ORDER), size=300)
        labels = rng.integers(0, 6, size=300)
        counts = cluster_composition(labels, y, order=POPULATION_ORDER)
        ids, sizes = np.unique(labels, return_counts=True)
        assert counts.to_numpy().sum() == 300
        assert counts.sum(axis=1).loc[ids].tolist() == sizes.tolist()
        fractions = counts.div(counts.sum(axis=1), axis=0)
        np.testing.assert_allclose(fractions.sum(axis=1), 1.0)

    def test_columns_are_every_population_in_lineage_order(self):
        counts = cluster_composition([0, 0], ["Monocytes", "pDCs"], order=POPULATION_ORDER)
        assert list(counts.columns) == list(POPULATION_ORDER)
        assert counts.loc[0, "CD4_T_cells"] == 0

    def test_rows_follow_the_populations_their_cells_come_from(self):
        # Cluster 0 holds monocytes (lineage position 8), cluster 1 CD4 T
        # cells (position 0), so cluster 1 comes first.
        y = ["Monocytes"] * 4 + ["CD4_T_cells"] * 4
        counts = cluster_composition([0] * 4 + [1] * 4, y, order=POPULATION_ORDER)
        assert list(counts.index) == [1, 0]

    def test_a_population_the_order_does_not_list_raises(self):
        with pytest.raises(ValueError, match="unassigned"):
            cluster_composition(
                [0, 1], ["CD4_T_cells", "unassigned"], order=POPULATION_ORDER
            )


class TestCarveClusterNumbers:
    def test_each_cluster_gets_the_number_s8_fig_prints(self, fitted):
        with carve_labels_aligned(fitted.carve, fitted.y):
            own = np.asarray(fitted.carve.get_labels(measure="stability", rule="1se"))
        aligned = fitted.carve_labels
        # Only meaningful while the two labelings differ.
        assert set(np.unique(aligned).tolist()) != set(np.unique(own).tolist())
        numbers = carve_cluster_numbers(fitted)
        assert sorted(numbers) == sorted(np.unique(aligned).tolist())
        for aligned_id, number in numbers.items():
            assert set((own[aligned == aligned_id] + 1).tolist()) == {number}

    def test_a_different_partition_raises(self):
        y = np.array(["a", "a", "b", "b"])
        aligned = np.array([0, 0, 1, 1])
        inputs = _inputs(
            X=np.zeros((4, 2)),
            y=y,
            carve=_stub(np.array([0, 1, 1, 1])),
            carve_labels=aligned,
            comparison_labels=aligned,
        )
        with pytest.raises(ValueError, match="same partition"):
            carve_cluster_numbers(inputs)


class TestCompositionTables:
    def test_carve_rows_carry_the_numbers_s8_fig_prints(self, levine_like):
        carve, _ = composition_tables(levine_like)
        counts = cluster_composition(
            levine_like.carve_labels, levine_like.y, order=POPULATION_ORDER
        )
        own = levine_like.carve.get_labels()
        expected = []
        for cluster in counts.index:
            (own_id,) = np.unique(own[levine_like.carve_labels == cluster])
            expected.append(f"C{own_id + 1}")
        assert list(carve.index) == expected
        np.testing.assert_array_equal(carve.to_numpy(), counts.to_numpy())

    def test_comparison_rows_are_numbered_in_display_order(self, levine_like):
        _, comparison = composition_tables(levine_like)
        assert list(comparison.index) == [f"C{i}" for i in range(1, 8)]

    def test_the_comparison_pools_the_three_lymphocyte_populations(self, levine_like):
        _, comparison = composition_tables(levine_like)
        pooled = comparison.loc[
            :, ["CD4_T_cells", "CD8_T_cells", "CD16-_NK_cells"]
        ].gt(0).all(axis=1)
        assert pooled.sum() == 1
