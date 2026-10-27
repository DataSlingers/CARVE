"""Tests for the case-study comparison with M3C and SC3."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from sklearn.metrics import adjusted_rand_score

from benchmarks._m3c import M3CResult
from benchmarks._sc3 import SC3Result
from benchmarks._theme import PRINT_MAX_HEIGHT_IN, PRINT_WIDTH_IN
from benchmarks.figures._case_study import CompositeInputs
from benchmarks.figures._comparators import figure_comparators, m3c_ari_rows, sc3_ari_rows
from tests.benchmarks._helpers import StubCarve, simple_results

N = 40


@pytest.fixture
def m3c_result():
    rng = np.random.default_rng(0)
    return M3CResult(
        scores=pd.DataFrame(
            {
                "K": [2, 3, 4, 5],
                "ENTROPY_REAL": [0.62, 0.48, 0.31, 0.35],
                "ENTROPY_REF": [0.65, 0.63, 0.60, 0.58],
                "RCSI": [0.66, 0.27, 0.20, 0.11],
                "RCSI_SE": [0.01, 0.02, 0.03, 0.03],
                "MONTECARLO_P": [0.01, 0.08, 0.12, 0.20],
                "NORM_P": [0.01, 0.07, 0.11, 0.19],
                "P_SCORE": [2.0, 1.15, 0.96, 0.72],
            }
        ),
        labels={k: rng.integers(0, k, size=N) for k in (2, 3, 4, 5)},
        selected_k=2,
        p_value=0.01,
        runtime_s=480.0,
        config={"maxK": 5},
        m3c_version="1.30.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )


@pytest.fixture
def subsampled(m3c_result):
    """The same result as if M3C had clustered 25 of the 40 rows."""
    rng = np.random.default_rng(2)
    rows = np.sort(rng.choice(N, size=25, replace=False))
    return dataclasses.replace(
        m3c_result,
        labels={k: rng.integers(0, k, size=25) for k in (2, 3, 4, 5)},
        rows=rows,
    )


@pytest.fixture
def sc3_result():
    rng = np.random.default_rng(3)
    return SC3Result(
        labels=rng.integers(0, 3, size=N),
        k=3,
        spectrum=pd.DataFrame(
            {
                "index": np.arange(1, 9),
                "eigenvalue": [5000.0, 2000.0, 900.0, 300.0, 120.0, 80.0, 40.0, 1e-12],
            }
        ),
        bound=500.0,
        n_features_used=5,
        n_dim=(2, 3),
        kmeans_nstart=1000,
        n_cores=1,
        runtime_s=60.0,
        config={"gene_filter": True, "seed": 1},
        sc3_version="1.36.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )


@pytest.fixture
def inputs():
    rng = np.random.default_rng(1)
    y = rng.choice(["a", "b", "c", "d"], N)
    frame = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["AgglomerativeClustering (linkage=ward)"] * 2,
            "k": [4, 4],
            "score": [0.5, 0.4],
            "ari": [0.5, 0.4],
        }
    )
    return CompositeInputs(
        X=rng.normal(size=(N, 5)),
        y=y,
        Z=rng.normal(size=(N, 2)),
        carve=StubCarve(
            simple_results([3, 4, 5], "AgglomerativeClustering (linkage=ward)"),
            select=lambda measure, not_two: ("m0", 4),
        ),
        carve_labels=rng.integers(0, 4, N),
        comparison_labels=rng.integers(0, 4, N),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=frame,
        best_df=frame.copy(),
    )


def _draw(inputs, m3c, sc3, **kwargs):
    kwargs.setdefault("save", False)
    return figure_comparators(
        inputs, m3c, sc3, axis_labels=("PC1", "PC2"), marker_size=15.0,
        save_name="klein_carve_m3c_sc3.png", **kwargs,
    )


def _offsets(ax):
    return np.concatenate(
        [c.get_offsets() for c in ax.collections if isinstance(c, PathCollection)]
    )


class TestAriRows:
    def test_m3c_gives_one_row_at_its_own_k(self, inputs, m3c_result):
        rows = m3c_ari_rows(inputs, m3c_result)
        assert [(row["method"], row["metric"], row["k"]) for row in rows] == [("M3C", "m3c_rcsi", 2)]

    def test_m3c_on_a_subsample_is_scored_on_its_own_cells_and_says_so(self, inputs, subsampled):
        (row,) = m3c_ari_rows(inputs, subsampled)
        y_sub = np.asarray(inputs.y)[subsampled.rows]
        assert row["ari"] == pytest.approx(adjusted_rand_score(y_sub, subsampled.labels[2]))
        assert row["method"] == "M3C (25 cells)"

    def test_sc3_gives_one_row_at_its_own_k(self, inputs, sc3_result):
        (row,) = sc3_ari_rows(inputs, sc3_result)
        assert (row["method"], row["metric"], row["k"]) == ("SC3", "sc3", 3)
        assert row["ari"] == pytest.approx(adjusted_rand_score(inputs.y, sc3_result.labels))

    def test_sc3_refuses_labels_for_other_cells(self, inputs, sc3_result):
        short = dataclasses.replace(sc3_result, labels=sc3_result.labels[:-1])
        with pytest.raises(ValueError, match="every cell"):
            sc3_ari_rows(inputs, short)


class TestFigure:
    def test_returns_a_figure_with_five_panels(self, inputs, m3c_result, sc3_result):
        fig = _draw(inputs, m3c_result, sc3_result)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 5

    def test_is_laid_out_at_the_print_width(self, inputs, m3c_result, sc3_result):
        width, height = _draw(inputs, m3c_result, sc3_result).get_size_inches()
        assert width == pytest.approx(PRINT_WIDTH_IN)
        assert height <= PRINT_MAX_HEIGHT_IN

    def test_saves_under_the_name_it_is_given(self, inputs, m3c_result, sc3_result, tmp_path):
        _draw(inputs, m3c_result, sc3_result, save=True, out_dir=tmp_path)
        assert (tmp_path / "klein_carve_m3c_sc3.png").is_file()

    def test_does_not_save_when_asked_not_to(self, inputs, m3c_result, sc3_result, tmp_path):
        _draw(inputs, m3c_result, sc3_result, save=False, out_dir=tmp_path)
        assert not (tmp_path / "klein_carve_m3c_sc3.png").exists()

    def test_the_ari_panel_holds_each_tool_once_at_its_own_k(self, inputs, m3c_result, sc3_result):
        fig = _draw(inputs, m3c_result, sc3_result)
        labels = [label.get_text() for label in fig.axes[4].get_yticklabels()]
        assert sum(label.startswith("M3C") for label in labels) == 1
        assert labels.count("SC3") == 1
        assert not any("at $k" in label for label in labels)

    def test_panel_c_fills_exactly_sc3_s_k_points(self, inputs, m3c_result, sc3_result):
        fig = _draw(inputs, m3c_result, sc3_result)
        above = next(c for c in fig.axes[2].collections if c.get_label() == "Above the bound")
        assert len(above.get_offsets()) == sc3_result.k

    def test_the_m3c_scatter_draws_only_the_rows_m3c_clustered(self, inputs, subsampled, sc3_result):
        fig = _draw(inputs, subsampled, sc3_result)
        drawn = _offsets(fig.axes[1])
        assert len(drawn) == len(subsampled.rows)
        expected = np.asarray(inputs.Z)[subsampled.rows]
        assert sorted(map(tuple, np.round(drawn, 9))) == sorted(map(tuple, np.round(expected, 9)))

    def test_the_m3c_scatter_title_names_the_subsample(self, inputs, subsampled, sc3_result):
        fig = _draw(inputs, subsampled, sc3_result)
        assert "25 cells" in fig.axes[1].get_title()

    def test_the_sc3_scatter_draws_every_cell(self, inputs, m3c_result, sc3_result):
        fig = _draw(inputs, m3c_result, sc3_result)
        assert len(_offsets(fig.axes[3])) == N
        assert f"k={sc3_result.k}" in fig.axes[3].get_title()

    def test_panel_c_annotates_sc3_s_k(self, inputs, m3c_result, sc3_result):
        fig = _draw(inputs, m3c_result, sc3_result)
        texts = [t.get_text() for t in fig.axes[2].texts]
        assert any(f"k = {sc3_result.k} eigenvalues" in t for t in texts)
