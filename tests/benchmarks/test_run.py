"""Tests for the unified runner."""

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks._artifacts import SCHEMA, read_run
from benchmarks._registry import CARVE_METRICS_ALL, CVI_METRICS
from benchmarks._run import run_cell, run_scenario
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
