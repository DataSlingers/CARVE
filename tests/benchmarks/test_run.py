"""Tests for the unified runner."""

import numpy as np
import pytest

from benchmarks._artifacts import SCHEMA, read_run
from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS
from benchmarks._run import run_cell, run_scenario
from benchmarks._types import Axis, EstimatorSpec, Scenario

N_METRICS = len(CARVE_METRICS_ALL) + len(CVI_METRICS)


@pytest.fixture
def tiny_scenario():
    """A scenario small enough to fit and score in a couple of seconds."""
    return Scenario(
        name="tiny",
        axis=Axis(name="difficulty_level", values=(0, 1), labels=("easy", "hard")),
        anchors={"easy": {"cluster_scale": 0.6}, "hard": {"cluster_scale": 2.5}},
        shared={"n_total": 120, "p": 4, "distribution": "gaussian"},
        estimator=EstimatorSpec(name="kmeans"),
        candidate_k=(3, 4, 5),
        n_seeds=2,
    )


class TestRunCell:
    def test_emits_one_row_per_metric_and_k(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        assert len(rows) == N_METRICS * len(tiny_scenario.candidate_k)

    def test_rows_carry_exactly_the_schema(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        assert set(rows[0]) == set(SCHEMA)

    def test_exactly_one_k_is_selected_per_metric(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        for metric in CARVE_METRICS_ALL + CVI_METRICS:
            selected = [r for r in rows if r["metric_name"] == metric and r["is_selected"]]
            assert len(selected) == 1, metric

    def test_selects_true_k_marks_k_star(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        for row in rows:
            assert row["selects_true_k"] == (row["k"] == tiny_scenario.k_star)

    def test_oracle_ari_is_constant_within_a_cell(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        assert len({row["oracle_ari"] for row in rows}) == 1

    def test_generalizability_metrics_use_the_generalizability_matrix(
        self, tiny_scenario, monkeypatch
    ):
        """The old runner never passed mode=, so every metric got stability.

        Asserted on the calls rather than on the values, because the two
        consensus matrices can legitimately agree at a given k on easy data.
        What must hold is that generalizability metrics are cut from the
        generalizability matrix at all.
        """
        import benchmarks._run as run_module

        seen_modes = []
        original = run_module.CARVE

        class RecordingCarve(original):
            def get_labels(self, **kwargs):
                seen_modes.append(kwargs.get("mode", "default"))
                return super().get_labels(**kwargs)

        monkeypatch.setattr(run_module, "CARVE", RecordingCarve)
        run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        assert "generalizability" in seen_modes
        assert "default" in seen_modes

    def test_labels_mode_routes_each_metric_family(self):
        from benchmarks._registry import GENERALIZABILITY_METRICS
        from benchmarks._run import _labels_mode

        assert _labels_mode("ari_stability_1se") == "default"
        assert _labels_mode("consensus_gini_stability") == "default"
        for metric in GENERALIZABILITY_METRICS:
            assert _labels_mode(metric) == "generalizability"

    def test_is_deterministic_for_a_fixed_seed(self, tiny_scenario):
        kwargs = dict(
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        first = run_cell(tiny_scenario, **kwargs)
        second = run_cell(tiny_scenario, **kwargs)
        assert [r["metric_value"] for r in first] == [r["metric_value"] for r in second]

    def test_provenance_columns_record_the_actual_estimator(self, tiny_scenario):
        rows = run_cell(
            tiny_scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        assert {row["estimator"] for row in rows} == {"kmeans"}
        assert {row["axis_label"] for row in rows} == {"easy"}


class TestRunScenario:
    def test_writes_a_complete_run(self, tiny_scenario, tmp_path):
        rd = run_scenario(
            tiny_scenario, root=tmp_path, n_jobs=1, random_state=0, n_resamples=20
        )
        df = read_run(rd)
        expected = (
            len(tiny_scenario.axis)
            * tiny_scenario.n_seeds
            * N_METRICS
            * len(tiny_scenario.candidate_k)
        )
        assert len(df) == expected

    def test_writes_a_manifest(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        assert (rd / "manifest.json").exists()

    def test_resumes_without_recomputing_completed_cells(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        before = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        run_scenario(tiny_scenario, root=tmp_path, n_resamples=20, resume=True)
        after = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        assert before == after

    def test_the_same_config_reuses_one_directory(self, tiny_scenario, tmp_path):
        first = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        second = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        assert first == second

    def test_a_changed_config_gets_a_new_directory(self, tiny_scenario, tmp_path):
        first = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        second = run_scenario(tiny_scenario, root=tmp_path, n_resamples=21)
        assert first != second

    def test_n_seeds_override_shortens_the_run(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_seeds=1, n_resamples=20)
        assert len(read_run(rd)["seed"].unique()) == 1


def test_the_runner_does_not_import_matplotlib():
    """Removing and reimporting matplotlib modules in place, permanently,
    corrupts class identity for any code elsewhere in the session that
    already holds references to the pre-removal modules (observed as
    spurious AttributeErrors in test_plotting.py when this test ran
    first). The removed entries are saved and restored in a finally block
    so the check is still exercised without leaking state across tests.
    """
    import sys

    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("matplotlib")
    }
    try:
        import benchmarks._run  # noqa: F401

        assert not any(m.startswith("matplotlib") for m in sys.modules)
    finally:
        sys.modules.update(removed)
