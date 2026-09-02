"""Tests for the unified runner."""

import dataclasses
import json
from pathlib import Path

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

    def test_carve_receives_the_scenario_n_trees(self, tiny_scenario, monkeypatch):
        """n_trees=100 is also Scenario's dataclass default, so a regression
        that hardcoded n_trees=100 in run_cell instead of threading
        scenario.n_trees through would pass every other test in this file.
        Built a scenario at n_trees=500 -- the published value for
        circles/moons/swiss_rolls -- and checked CARVE actually received it.
        """
        import benchmarks._run as run_module

        scenario = dataclasses.replace(tiny_scenario, n_trees=500)

        seen_n_trees = []
        original = run_module.CARVE

        class RecordingCarve(original):
            def __init__(self, *args, **kwargs):
                seen_n_trees.append(kwargs.get("n_trees"))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(run_module, "CARVE", RecordingCarve)
        run_cell(
            scenario,
            axis_idx=0,
            axis_value=0,
            axis_label="easy",
            seed=0,
            run_id="r1",
            random_state=0,
            n_resamples=20,
        )
        assert seen_n_trees == [500]

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

    def test_resuming_a_partial_run_keeps_one_run_id(self, tiny_scenario, tmp_path):
        """Simulates an interrupted run by deleting one cell's checkpoint
        after a complete run, then resuming. Every row on disk -- whether
        recomputed by the resume or left over from the first invocation --
        must carry the same run_id the manifest records, not a mix of the
        original invocation's run_id and a freshly minted one.
        """
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        checkpoints = sorted(rd.glob("cell__*.parquet"))
        checkpoints[0].unlink()

        run_scenario(tiny_scenario, root=tmp_path, n_resamples=20, resume=True)

        manifest = json.loads((rd / "manifest.json").read_text())
        df = read_run(rd)
        assert set(df["run_id"].unique()) == {manifest["run_id"]}

    def test_wall_clock_accumulates_across_a_resume(self, tiny_scenario, tmp_path):
        """wall_clock_s must represent total work across resumed
        invocations, not only the most recent increment, so a resume that
        recomputes one missing cell should not report less wall-clock time
        than the original complete run already recorded.
        """
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=20)
        first_manifest = json.loads((rd / "manifest.json").read_text())

        checkpoints = sorted(rd.glob("cell__*.parquet"))
        checkpoints[0].unlink()
        run_scenario(tiny_scenario, root=tmp_path, n_resamples=20, resume=True)

        second_manifest = json.loads((rd / "manifest.json").read_text())
        assert second_manifest["wall_clock_s"] >= first_manifest["wall_clock_s"]


def test_compute_modules_do_not_import_matplotlib_directly():
    """Checks each compute module's source for a direct matplotlib import,
    rather than importing the module and inspecting sys.modules.

    An import-based check cannot express this constraint: benchmarks._run
    imports carve, and carve/__init__.py does `from . import pl`, which
    pulls in matplotlib transitively no matter what this package itself
    imports. So "matplotlib ends up in sys.modules after importing
    benchmarks._run" is true regardless of whether _run.py imports it
    directly, and a test asserting the negation would simply fail -- that
    was verified separately, which is why this checks source text via ast
    instead of import behavior. It also avoids a plain substring search,
    which would be tripped by a mention in a comment or docstring rather
    than an actual import statement.
    """
    import ast

    import benchmarks._run as run_module

    compute_dir = Path(run_module.__file__).parent
    module_files = (
        "_types.py",
        "_registry.py",
        "_estimators.py",
        "_simulate.py",
        "_cvi.py",
        "_artifacts.py",
        "_run.py",
    )

    for filename in module_files:
        path = compute_dir / filename
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                assert name != "matplotlib" and not name.startswith("matplotlib."), (
                    f"{filename} imports matplotlib directly: {ast.dump(node)}"
                )
