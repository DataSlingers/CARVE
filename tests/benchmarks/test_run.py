"""Tests for the unified runner."""

import dataclasses
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from joblib import Parallel, cpu_count, delayed

from benchmarks._artifacts import SCHEMA, read_run
from benchmarks._estimators import param_grids
from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS, SCENARIOS
from benchmarks._run import (
    CLUSTER_COUNT_WARNING,
    benchmark_seed,
    cpu_cap,
    fit_carve,
    labels_by_mode,
    rare_cluster_recall,
    run_cell,
    run_scenario,
    scenario_n_jobs,
    smallest_cluster,
    thread_cap_for,
)
from benchmarks._simulate import simulate
from benchmarks._types import Axis, EstimatorSpec, Scenario

# The tiny scenario's hard axis point leaves no configuration inside the
# best score's quantile band for some seeds, so carve's quantile rule warns
# and falls back to max. That is the rule working as documented, not a
# defect in the runner, and it is data-dependent: pin the message, not the
# category, so any other RuntimeWarning still fails the test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:No estimators within quantile thresholds:RuntimeWarning"
)

N_METRICS = len(CARVE_METRICS_ALL) + len(CVI_METRICS)


@pytest.fixture(scope="module")
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


CELL_KWARGS = dict(
    axis_idx=0, axis_value=0, axis_label="easy", seed=0, run_id="r1", random_state=0
)


@pytest.fixture(scope="module")
def cell(tiny_scenario):
    """One easy-axis cell at n_resamples=20; every TestRunCell test reads it."""
    return run_cell(tiny_scenario, n_resamples=20, **CELL_KWARGS)


@pytest.fixture(scope="module")
def completed(tiny_scenario, tmp_path_factory):
    """A finished run_scenario at n_resamples=20: (root, run directory).

    Shared read-only. Tests that delete checkpoints or corrupt the manifest
    take run_copy instead.
    """
    root = tmp_path_factory.mktemp("completed")
    rd = run_scenario(tiny_scenario, root=root, n_resamples=20)
    return root, rd


@pytest.fixture
def run_copy(completed, tmp_path):
    """A private copy of the completed run under this test's tmp_path."""
    root, rd = completed
    new_root = tmp_path / "runs"
    shutil.copytree(root, new_root)
    return new_root, new_root / rd.relative_to(root)


@pytest.fixture
def recording_carve(monkeypatch):
    """Patch benchmarks._run.CARVE with a subclass that records constructor
    kwargs and get_labels calls, then delegates to the real class."""
    import benchmarks._run as run_module

    original = run_module.CARVE
    record = SimpleNamespace(init_kwargs=[], label_calls=[])

    class RecordingCarve(original):
        def __init__(self, *args, **kwargs):
            record.init_kwargs.append(kwargs)
            super().__init__(*args, **kwargs)

        def get_labels(self, **kwargs):
            record.label_calls.append(kwargs)
            return super().get_labels(**kwargs)

    monkeypatch.setattr(run_module, "CARVE", RecordingCarve)
    return record


class TestRunCell:
    def test_emits_one_row_per_metric_and_k(self, tiny_scenario, cell):
        rows, _ = cell
        assert len(rows) == N_METRICS * len(tiny_scenario.candidate_k)

    def test_rows_carry_exactly_the_schema(self, cell):
        rows, _ = cell
        assert set(rows[0]) == set(SCHEMA)

    def test_exactly_one_k_is_selected_per_metric(self, cell):
        rows, _ = cell
        for metric in CARVE_METRICS_ALL + CVI_METRICS:
            selected = [r for r in rows if r["metric_name"] == metric and r["is_selected"]]
            assert len(selected) == 1, metric

    def test_selects_true_k_marks_k_star(self, tiny_scenario, cell):
        rows, _ = cell
        for row in rows:
            assert row["selects_true_k"] == (row["k"] == tiny_scenario.k_star)

    def test_oracle_ari_is_constant_within_a_cell(self, cell):
        rows, _ = cell
        assert len({row["oracle_ari"] for row in rows}) == 1

    def test_generalizability_metrics_use_the_generalizability_matrix(
        self, tiny_scenario, recording_carve
    ):
        """The old runner never passed mode=, so every metric got stability.

        Asserted on the calls rather than on the values, because the two
        consensus matrices can legitimately agree at a given k on easy data.
        What must hold is that generalizability metrics are cut from the
        generalizability matrix at all.
        """
        run_cell(tiny_scenario, n_resamples=20, **CELL_KWARGS)
        seen_modes = [c.get("mode", "default") for c in recording_carve.label_calls]
        assert "generalizability" in seen_modes
        assert "default" in seen_modes

    def test_labels_mode_routes_each_metric_family(self):
        from benchmarks._registry import GENERALIZABILITY_METRICS
        from benchmarks._run import _labels_mode

        assert _labels_mode("ari_stability_1se") == "default"
        assert _labels_mode("consensus_gini_stability") == "default"
        for metric in GENERALIZABILITY_METRICS:
            assert _labels_mode(metric) == "generalizability"

    def test_is_deterministic_for_a_fixed_seed(self, tiny_scenario, cell):
        first, _ = cell
        second, _ = run_cell(tiny_scenario, n_resamples=20, **CELL_KWARGS)
        assert [r["metric_value"] for r in first] == [r["metric_value"] for r in second]

    def test_carve_receives_the_scenario_n_trees(self, tiny_scenario, recording_carve):
        """n_trees=100 is also Scenario's dataclass default, so a regression
        that hardcoded n_trees=100 in run_cell instead of threading
        scenario.n_trees through would pass every other test in this file.
        Built a scenario at n_trees=500 -- the published value for
        circles/moons/swiss_rolls -- and checked CARVE actually received it.
        """
        scenario = dataclasses.replace(tiny_scenario, n_trees=500)
        run_cell(scenario, n_resamples=20, **CELL_KWARGS)
        assert [k.get("n_trees") for k in recording_carve.init_kwargs] == [500]

    def test_provenance_columns_record_the_actual_estimator(self, cell):
        rows, _ = cell
        assert {row["estimator"] for row in rows} == {"kmeans"}
        assert {row["axis_label"] for row in rows} == {"easy"}


class TestRunScenario:
    def test_writes_a_complete_run(self, tiny_scenario, completed):
        _, rd = completed
        df = read_run(rd)
        expected = (
            len(tiny_scenario.axis)
            * tiny_scenario.n_seeds
            * N_METRICS
            * len(tiny_scenario.candidate_k)
        )
        assert len(df) == expected

    def test_writes_a_manifest(self, completed):
        _, rd = completed
        assert (rd / "manifest.json").exists()

    def test_resumes_without_recomputing_completed_cells(self, tiny_scenario, run_copy):
        root, rd = run_copy
        before = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        after = {p: p.stat().st_mtime_ns for p in rd.glob("cell__*.parquet")}
        assert before == after

    def test_the_same_config_reuses_one_directory(self, tiny_scenario, run_copy):
        root, rd = run_copy
        assert run_scenario(tiny_scenario, root=root, n_resamples=20) == rd

    def test_a_changed_config_gets_a_new_directory(self, tiny_scenario, run_copy):
        root, rd = run_copy
        other = run_scenario(tiny_scenario, root=root, n_seeds=1, n_resamples=21)
        assert other != rd

    def test_n_seeds_override_shortens_the_run(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_seeds=1, n_resamples=20)
        assert len(read_run(rd)["seed"].unique()) == 1

    def test_resuming_a_partial_run_keeps_one_run_id(self, tiny_scenario, run_copy):
        """Simulates an interrupted run by deleting one cell's checkpoint
        after a complete run, then resuming. Every row on disk -- whether
        recomputed by the resume or left over from the first invocation --
        must carry the same run_id the manifest records, not a mix of the
        original invocation's run_id and a freshly minted one.
        """
        root, rd = run_copy
        sorted(rd.glob("cell__*.parquet"))[0].unlink()
        run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert set(read_run(rd)["run_id"].unique()) == {manifest["run_id"]}

    def test_resume_with_a_corrupt_manifest_warns_and_completes(
        self, tiny_scenario, run_copy
    ):
        """A truncated or otherwise unparseable manifest.json must not be
        fatal. Corrupts the manifest of a real, fully completed run -- so
        the cell checkpoints are genuinely resumable and this invocation has
        nothing left to compute -- then resumes. run_scenario must warn and
        fall back to fresh provenance rather than raising
        json.JSONDecodeError, so a run can still resume unattended after a
        process was killed mid-write of the manifest.
        """
        root, rd = run_copy
        (rd / "manifest.json").write_text("{not valid json")
        with pytest.warns(UserWarning, match="manifest"):
            run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        assert "run_id" in json.loads((rd / "manifest.json").read_text())

    def test_wall_clock_accumulates_across_a_resume(self, tiny_scenario, run_copy):
        """wall_clock_s must represent total work across resumed
        invocations, not only the most recent increment, so a resume that
        recomputes one missing cell should not report less wall-clock time
        than the original complete run already recorded.
        """
        root, rd = run_copy
        first = json.loads((rd / "manifest.json").read_text())
        sorted(rd.glob("cell__*.parquet"))[0].unlink()
        run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)
        second = json.loads((rd / "manifest.json").read_text())
        assert second["wall_clock_s"] >= first["wall_clock_s"]

    def test_manifest_exists_before_the_pool_finishes(self, tiny_scenario, tmp_path):
        """An interrupted scenario must still leave a readable manifest.

        Run directories are located by their manifest, so a scenario killed
        mid-pool otherwise leaves checkpoints the notebook cannot open. The
        ablation runner was fixed for this in e720073; the scenario runner
        was not.
        """
        seen = {}

        def _explode(*args, **kwargs):
            rd = tmp_path / tiny_scenario.name
            found = list(rd.glob("*/manifest.json"))
            seen["manifest"] = json.loads(found[0].read_text()) if found else None
            raise KeyboardInterrupt

        with mock.patch("benchmarks._run.Parallel", side_effect=_explode):
            with pytest.raises(KeyboardInterrupt):
                run_scenario(tiny_scenario, root=tmp_path, n_resamples=2, n_seeds=1)

        assert seen["manifest"] is not None
        assert seen["manifest"]["status"] == "running"

    def test_manifest_is_complete_after_the_pool(self, tiny_scenario, tmp_path):
        # n_resamples=10, not the brief's 2: below roughly 7 resamples the
        # stability measures come back all-NaN and CARVE.get_k raises (see
        # _cell's comment above), which run_cell does not catch. This test
        # exercises a full, real pool, so it needs a safe n_resamples.
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["status"] == "complete"

    def test_a_resumed_run_keeps_its_run_id_across_the_running_manifest(
        self, tiny_scenario, run_copy
    ):
        """The manifest a resume writes before its pool starts must carry
        the same run_id as the completed run it is resuming, not a freshly
        minted one -- so a reader that opens the running manifest mid-resume
        gets the same provenance the finished manifest will report.

        Deletes one cell checkpoint so the resume has pending work, then
        patches Parallel to capture the on-disk manifest and abort before
        any cell runs. (The brief's version of this test compared two runs
        at different n_seeds, which always land in different content-
        addressed directories -- config_hash includes n_seeds -- so
        `second != first` was trivially true and the assertion could never
        fail. This version resumes the *same* configuration and inspects
        the running manifest directly.)
        """
        root, rd = run_copy
        first_run_id = json.loads((rd / "manifest.json").read_text())["run_id"]
        sorted(rd.glob("cell__*.parquet"))[0].unlink()

        seen = {}

        def _explode(*args, **kwargs):
            seen["manifest"] = json.loads((rd / "manifest.json").read_text())
            raise KeyboardInterrupt

        with mock.patch("benchmarks._run.Parallel", side_effect=_explode):
            with pytest.raises(KeyboardInterrupt):
                run_scenario(tiny_scenario, root=root, n_resamples=20, resume=True)

        assert seen["manifest"]["status"] == "running"
        assert seen["manifest"]["run_id"] == first_run_id

    def test_resuming_onto_different_code_raises(self, tiny_scenario, tmp_path):
        """read_run concatenates whatever checkpoints it finds.

        A run directory is content-addressed on configuration only, so
        re-running the same config on newer code resumes into the old
        directory and mixes two code versions' rows in one frame with
        nothing marking it. git_sha was already recorded and never checked;
        this is what left the ablation dev run unreadable.

        n_resamples=10, not the brief's 2: see the comment on
        test_manifest_is_complete_after_the_pool -- these tests run a full,
        real pool to completion rather than a mocked one.
        """
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["git_sha"] = "0" * 40
        (rd / "manifest.json").write_text(json.dumps(manifest))

        # The brief's own match string, "different code version", never
        # occurs in its own verbatim error message below (no "different" at
        # all) and could never pass; matching a phrase the message actually
        # contains instead.
        with pytest.raises(RuntimeError, match="concatenate two code versions"):
            run_scenario(tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1)

    def test_the_override_allows_the_resume(self, tiny_scenario, tmp_path):
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["git_sha"] = "0" * 40
        (rd / "manifest.json").write_text(json.dumps(manifest))

        again = run_scenario(
            tiny_scenario,
            root=tmp_path,
            n_resamples=10,
            n_seeds=1,
            allow_code_change=True,
        )
        assert again == rd

    def test_no_resume_does_not_check_the_code_version(self, tiny_scenario, tmp_path):
        """resume=False recomputes every cell, so there is nothing to mix."""
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        manifest["git_sha"] = "0" * 40
        (rd / "manifest.json").write_text(json.dumps(manifest))

        again = run_scenario(
            tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1, resume=False
        )
        assert again == rd


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

    The module list is derived by globbing the package directory rather
    than hardcoded, so a new compute module is covered automatically
    instead of silently falling outside the guard. Two names are
    subtracted from that glob: _theme.py and _panels.py. Those are the
    reporting layer -- palette/rcParams/figure-saving and the ax-in/
    ax-out drawing primitives -- and importing matplotlib is their entire
    job. The guard exists to keep the *compute* layer free of matplotlib,
    not to ban matplotlib from the package outright, so a new reporting
    module needs to be added to this exclusion set explicitly; a new
    compute module does not, and keeps tripping the guard automatically
    if it ever imports matplotlib.
    """
    import ast

    import benchmarks._run as run_module

    compute_dir = Path(run_module.__file__).parent
    reporting_modules = {"_theme.py", "_panels.py"}
    module_files = sorted(
        path.name
        for path in compute_dir.glob("_*.py")
        if path.name != "__init__.py" and path.name not in reporting_modules
    )
    assert module_files, "glob found no compute modules -- check compute_dir/pattern"
    assert {"_run.py", "_registry.py"}.issubset(module_files), (
        "glob is missing known compute modules; it should not be this narrow"
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


def _cell(scenario, **overrides):
    kwargs = dict(
        axis_idx=0,
        axis_value=0,
        axis_label="easy",
        seed=0,
        run_id="r1",
        random_state=0,
        # Below roughly 7 resamples, consensus_gini_stability and
        # consensus_ce_stability come back all-NaN and CARVE.get_k raises
        # ValueError: Encountered all NA values (see MIN_SAFE_N_RESAMPLES in
        # test_ci_config.py). 10 keeps a margin above that measured boundary
        # while staying small enough to fit quickly.
        n_resamples=10,
    )
    kwargs.update(overrides)
    return run_cell(scenario, **kwargs)


@pytest.fixture(scope="module")
def cell10(tiny_scenario):
    return _cell(tiny_scenario)


@pytest.fixture(scope="module")
def cell10_timed(tiny_scenario):
    return _cell(tiny_scenario, timing_fits=True)


class TestRuntimeCapture:
    def test_run_cell_returns_metric_rows_and_a_runtime_row(self, cell10):
        rows, runtime = cell10
        assert isinstance(rows, list)
        assert isinstance(runtime, dict)
        assert runtime["t_default_s"] > 0.0

    def test_runtime_records_the_actual_data_shape(self, cell10):
        _, runtime = cell10
        assert runtime["n_samples"] == 120
        assert runtime["n_features"] == 4

    def test_timing_fits_are_off_by_default(self, cell10):
        _, runtime = cell10
        assert np.isnan(runtime["t_stability_s"])
        assert np.isnan(runtime["t_generalizability_s"])

    def test_timing_fits_populate_both_modes_when_requested(self, cell10_timed):
        _, runtime = cell10_timed
        assert runtime["t_stability_s"] > 0.0
        assert runtime["t_generalizability_s"] > 0.0

    def test_timed_fits_receive_the_scenario_n_trees(self, tiny_scenario, recording_carve):
        """The mode-specific timed fits must use the scenario's forest size,
        not CARVE's dataclass default of 100 -- otherwise t_stability_s and
        t_generalizability_s are timed against a different-sized random
        forest than t_default_s, silently, for any scenario with a
        non-default n_trees (circles/moons/swiss_rolls run at 500). Mirrors
        test_carve_receives_the_scenario_n_trees, but with timing_fits=True
        so it actually reaches the timed-fit branch.
        """
        scenario = dataclasses.replace(tiny_scenario, n_trees=500)
        _cell(scenario, timing_fits=True)

        # One CARVE for the default-mode fit, plus one per timed mode.
        assert [k.get("n_trees") for k in recording_carve.init_kwargs] == [500, 500, 500]

    def test_per_k_runtimes_divide_by_the_candidate_count(self, tiny_scenario, cell10_timed):
        _, runtime = cell10_timed
        n_k = len(tiny_scenario.candidate_k)
        assert runtime["t_per_k_stability_s"] == pytest.approx(
            runtime["t_stability_s"] / n_k
        )
        assert runtime["t_per_k_generalizability_s"] == pytest.approx(
            runtime["t_generalizability_s"] / n_k
        )

    def test_timing_fits_do_not_change_the_metric_rows(self, cell10, cell10_timed):
        """The timed fits are discarded; only the default fit feeds metrics."""
        without, _ = cell10
        with_timing, _ = cell10_timed
        assert [r["metric_value"] for r in without] == [
            r["metric_value"] for r in with_timing
        ]
        assert [r["ari_at_k"] for r in without] == [
            r["ari_at_k"] for r in with_timing
        ]

    def test_the_experimental_mode_warning_is_suppressed(self, tiny_scenario, recwarn):
        _cell(tiny_scenario, timing_fits=True)
        messages = [str(w.message) for w in recwarn]
        assert not any("Non-default mode is experimental" in m for m in messages)

    def test_run_scenario_writes_a_runtime_row_per_cell(self, tiny_scenario, completed):
        from benchmarks._artifacts import read_runtimes

        _, rd = completed
        runtimes = read_runtimes(rd)
        assert len(runtimes) == len(tiny_scenario.axis) * tiny_scenario.n_seeds

    def test_scaling_scenarios_are_timed_by_default(self):
        from benchmarks._registry import SCENARIOS, TIMED_SCENARIOS

        assert TIMED_SCENARIOS == {"gaussians_samples", "gaussians_dimensionality"}
        assert TIMED_SCENARIOS <= set(SCENARIOS)

    def test_difficulty_scenarios_are_not_timed_by_default(self):
        from benchmarks._registry import TIMED_SCENARIOS

        assert "gaussians" not in TIMED_SCENARIOS


class TestBenchmarkSeed:
    def test_matches_the_published_derivation(self):
        assert benchmark_seed(3, 2, 42) == 3 + 20000 + 42


class TestSmallestCluster:
    def test_returns_the_smallest_label_and_its_fraction(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        assert smallest_cluster(y) == (2, 1 / 6)

    def test_ties_go_to_the_lowest_label(self):
        y = np.array([0, 1, 1, 2, 3, 3])
        assert smallest_cluster(y)[0] == 0


class TestRareClusterRecall:
    def test_perfect_recovery_is_one(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        labels = np.array([5, 5, 5, 7, 7, 9])
        assert rare_cluster_recall(y, labels, rare_label=2) == 1.0

    def test_a_rare_cluster_merged_into_another_is_zero(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        labels = np.array([0, 0, 0, 1, 1, 1])
        assert rare_cluster_recall(y, labels, rare_label=2) == 0.0

    def test_fewer_clusters_than_truth_leaves_the_rare_one_unmatched(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2])
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        assert rare_cluster_recall(y, labels, rare_label=2) == 0.0

    def test_partial_recovery_is_the_matched_fraction(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 1, 1])
        assert rare_cluster_recall(y, labels, rare_label=2) == pytest.approx(0.5)

    def test_rejects_a_label_absent_from_the_truth(self):
        with pytest.raises(ValueError, match="rare_label"):
            rare_cluster_recall(np.array([0, 1]), np.array([0, 1]), rare_label=7)


def _budget_under_cap(cap):
    from carve._utils import resolve_core_budget

    with cpu_cap(cap):
        inside = resolve_core_budget(1, n_resamples=100)
    after = resolve_core_budget(1, n_resamples=100)
    return inside, after, os.environ.get("LOKY_MAX_CPU_COUNT")


class TestCpuCap:
    def test_reaches_carves_core_budget_inside_loky_workers(self):
        cores = cpu_count()
        cap = max(1, cores // 2)
        results = Parallel(n_jobs=2)(delayed(_budget_under_cap)(cap) for _ in range(4))
        for inside, after, env in results:
            assert inside == (1, cap)
            assert after == (1, cores)
            assert env is None

    def test_none_leaves_the_environment_alone(self, monkeypatch):
        monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
        with cpu_cap(None):
            assert "LOKY_MAX_CPU_COUNT" not in os.environ

    def test_restores_a_previous_value(self, monkeypatch):
        monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "3")
        with cpu_cap(1):
            assert os.environ["LOKY_MAX_CPU_COUNT"] == "1"
        assert os.environ["LOKY_MAX_CPU_COUNT"] == "3"

    def test_thread_cap_for_splits_the_machine(self):
        workers, cap = thread_cap_for(1)
        assert (workers, cap) == (1, cpu_count())
        workers, cap = thread_cap_for(-1)
        assert workers == cpu_count()
        assert cap == 1


class TestScenarioNJobs:
    def test_a_timed_scenario_is_forced_serial(self):
        """The scaling scenarios time their fits. Concurrent workers change
        those timings, so the runtime curves they feed are only meaningful
        at one worker."""
        for name in ("gaussians_samples", "gaussians_dimensionality"):
            assert scenario_n_jobs(SCENARIOS[name], None) == 1

    def test_a_difficulty_scenario_takes_every_worker(self):
        """--all previously defaulted to 1 for every scenario, which turns
        a seven-hour job into a three-day one."""
        assert scenario_n_jobs(SCENARIOS["gaussians"], None) == -1

    def test_an_explicit_request_overrides_both(self):
        assert scenario_n_jobs(SCENARIOS["gaussians"], 4) == 4
        assert scenario_n_jobs(SCENARIOS["gaussians_samples"], 4) == 4

    def test_run_scenario_resolves_it(self, tiny_scenario, tmp_path):
        # n_resamples=10, not the brief's 2: below roughly 7 resamples the
        # stability measures come back all-NaN and CARVE.get_k raises (see
        # TestRunScenario.test_manifest_is_complete_after_the_pool above).
        # This test runs a full, real pool, so it needs a safe n_resamples.
        rd = run_scenario(tiny_scenario, root=tmp_path, n_resamples=10, n_seeds=1)
        manifest = json.loads((rd / "manifest.json").read_text())
        assert manifest["n_jobs"] == -1


class TestFitCarve:
    def test_regex_matches_carves_own_message(self):
        # The literal format string lives in carve/_runner.py:
        # f"labels_1 has {k_1} clusters, expected {expected}".
        assert re.match(CLUSTER_COUNT_WARNING, "labels_1 has 3 clusters, expected 4")
        assert re.match(CLUSTER_COUNT_WARNING, "labels_test has 2 clusters, expected 10")
        assert not re.match(CLUSTER_COUNT_WARNING, "Non-default mode is experimental")

    def test_counts_cluster_count_warnings_without_raising(self, monkeypatch):
        # filterwarnings=error is on, so if fit_carve did not intercept these
        # they would raise here instead of being counted.
        import benchmarks._run as run_module

        class ShortCarve(run_module.CARVE):
            def fit(self, X, **kwargs):
                warnings.warn("labels_1 has 3 clusters, expected 4", stacklevel=2)
                warnings.warn("labels_test has 2 clusters, expected 4", stacklevel=2)
                return super().fit(X, **kwargs)

        monkeypatch.setattr(run_module, "CARVE", ShortCarve)
        X = np.random.default_rng(0).normal(size=(60, 3))
        fit = fit_carve(
            X,
            grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
            n_resamples=4,
            n_trees=10,
            random_state=0,
            subsample_ratio=0.5,
        )
        assert fit.n_cluster_count_warnings == 2
        assert fit.fit_seconds > 0
        assert fit.carve.estimator_results_ is not None

    def test_re_emits_other_warnings(self, tiny_scenario, monkeypatch):
        import benchmarks._run as run_module

        class NoisyCarve(run_module.CARVE):
            def fit(self, X, **kwargs):
                warnings.warn("something else entirely", stacklevel=2)
                return super().fit(X, **kwargs)

        monkeypatch.setattr(run_module, "CARVE", NoisyCarve)
        X = np.random.default_rng(0).normal(size=(60, 3))
        with pytest.warns(UserWarning, match="something else entirely"):
            fit_carve(
                X,
                grids=param_grids(tiny_scenario.estimator, (2, 3)),
                n_resamples=4,
                n_trees=10,
                random_state=0,
            )

    def test_subsample_ratio_none_keeps_the_constructor_kwargs(self, recording_carve):
        X = np.random.default_rng(0).normal(size=(60, 3))
        grids = param_grids(EstimatorSpec(name="kmeans"), (2, 3))
        fit_carve(X, grids=grids, n_resamples=4, n_trees=10, random_state=0)
        assert "subsample_ratio" not in recording_carve.init_kwargs[0]
        fit_carve(X, grids=grids, n_resamples=4, n_trees=10, random_state=0, subsample_ratio=0.5)
        assert recording_carve.init_kwargs[1]["subsample_ratio"] == 0.5


class TestLabelsByMode:
    def test_scores_every_mode_and_k(self, tiny_scenario):
        X, y = simulate(tiny_scenario, axis_value=0, axis_label="easy", seed=0)
        fit = fit_carve(
            X,
            grids=param_grids(tiny_scenario.estimator, tiny_scenario.candidate_k),
            n_resamples=8,
            n_trees=10,
            random_state=0,
        )
        rare_label, _ = smallest_cluster(y)
        scores = labels_by_mode(
            fit.carve, y, candidate_k=tiny_scenario.candidate_k, rare_label=rare_label
        )
        assert set(scores) == {"default", "generalizability"}
        for mode in scores:
            assert set(scores[mode]) == set(tiny_scenario.candidate_k)
            for k in scores[mode]:
                assert -1.0 <= scores[mode][k]["ari"] <= 1.0
                assert 0.0 <= scores[mode][k]["rare_recall"] <= 1.0

    def test_omits_rare_recall_without_a_rare_label(self, tiny_scenario):
        X, y = simulate(tiny_scenario, axis_value=0, axis_label="easy", seed=0)
        fit = fit_carve(
            X,
            grids=param_grids(tiny_scenario.estimator, (3,)),
            n_resamples=8,
            n_trees=10,
            random_state=0,
        )
        scores = labels_by_mode(fit.carve, y, candidate_k=(3,))
        assert set(scores["default"][3]) == {"ari"}
