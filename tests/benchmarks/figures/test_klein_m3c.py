"""Tests for the Klein M3C comparison figure."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from benchmarks._m3c import M3CResult
from benchmarks.figures._case_study import CompositeInputs
from benchmarks.figures._klein_m3c import figure_klein_m3c, m3c_ari_rows
from tests.benchmarks._helpers import StubCarve, simple_results


@pytest.fixture
def m3c_result():
    rng = np.random.default_rng(0)
    n = 40
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
        labels={k: rng.integers(0, k, size=n) for k in (2, 3, 4, 5)},
        selected_k=2,
        p_value=0.01,
        runtime_s=480.0,
        config={"maxK": 10},
        m3c_version="1.34.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )


@pytest.fixture
def inputs():
    # n must match m3c_result's label length (40).
    rng = np.random.default_rng(1)
    n = 40
    y = rng.choice(["a", "b", "c", "d"], n)
    curves = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["AgglomerativeClustering (linkage=ward)"] * 2,
            "k": [4, 4],
            "score": [0.5, 0.4],
            "ari": [0.5, 0.4],
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["AgglomerativeClustering (linkage=ward)"] * 2,
            "k": [4, 4],
            "score": [0.5, 0.4],
            "ari": [0.5, 0.4],
        }
    )
    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=rng.normal(size=(n, 2)),
        carve=StubCarve(
            simple_results([3, 4, 5], "AgglomerativeClustering (linkage=ward)"),
            select=lambda measure, not_two: ("m0", 4),
        ),
        carve_labels=rng.integers(0, 4, n),
        comparison_labels=rng.integers(0, 4, n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=curves,
        best_df=best,
    )


class TestAriRows:
    def test_returns_one_row_at_the_selected_k_and_one_at_four(
        self, inputs, m3c_result
    ):
        rows = m3c_ari_rows(inputs, m3c_result)
        assert [row["k"] for row in rows] == [2, 4]
        assert rows[0]["method"] == "M3C"
        assert rows[1]["method"] == "M3C (at $k=4$)"

    def test_every_row_carries_the_columns_ari_table_requires(
        self, inputs, m3c_result
    ):
        for row in m3c_ari_rows(inputs, m3c_result):
            assert set(row) == {"method", "metric", "ari", "k"}

    def test_returns_one_row_when_m3c_also_selects_four(self, inputs, m3c_result):
        # Panel C must not show the same partition twice under a second name.
        at_four = dataclasses.replace(m3c_result, selected_k=4)
        rows = m3c_ari_rows(inputs, at_four)
        assert [row["k"] for row in rows] == [4]


class TestFigure:
    def test_returns_a_figure_with_three_panels(self, inputs, m3c_result, tmp_path):
        fig = figure_klein_m3c(inputs, m3c_result, save=False, out_dir=tmp_path)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3

    def test_saves_under_its_manuscript_filename(self, inputs, m3c_result, tmp_path):
        figure_klein_m3c(inputs, m3c_result, save=True, out_dir=tmp_path)
        assert (tmp_path / "klein_m3c.png").is_file()

    def test_does_not_save_when_asked_not_to(self, inputs, m3c_result, tmp_path):
        figure_klein_m3c(inputs, m3c_result, save=False, out_dir=tmp_path)
        assert not (tmp_path / "klein_m3c.png").exists()
