"""Helpers shared across the benchmarks test files."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _fake_run(root, scenario, cfg, *, n_seeds, n_resamples, status="complete"):
    """Write a run directory's manifest.json without running anything.

    Used by test_artifacts.py (widest_run) and test_tables_cli.py (the
    --tables CLI), both of which only need a manifest to resolve, never a
    real checkpoint.
    """
    rd = root / scenario / cfg
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": cfg,
                "status": status,
                "scenario": scenario,
                "n_seeds": n_seeds,
                "n_resamples": n_resamples,
            }
        )
    )
    return rd


class StubCarve:
    """Minimal stand-in for a fitted CARVE object.

    The panels and figure code read a few members off a fitted model:
    estimator_results_, sweep_, _select_row(), get_k() and get_sweep_value()
    (and prepare_composite also calls get_labels()). This implements exactly
    those instead of fitting a real model, and records the selection
    arguments each call received in selection_calls.

    Column names are the canonical estimator_results_ names a real fitted
    model uses (ari_stability and ari_generalizability plus their _se
    variants, method_id, method_label, n_clusters). "stability" and
    "generalizability" are only measure aliases, never column names; a stub
    that named its columns after the aliases would let carve_lines index the
    alias directly and still pass.

    Parameters
    ----------
    results : DataFrame
        The table, with at least method_id and the sweep column: n_clusters
        for a k-based stub, sweep_value (and n_clusters_observed) otherwise.
    select : callable (measure, not_two) -> (method_id, sweep value)
        Which row _select_row, get_k and get_sweep_value resolve to. Making it
        a function lets a test decide whether measure or not_two changes the
        answer, so the test can tell whether the code under test forwarded
        them.
    labels : ndarray, optional
        What get_labels returns. Left None by tests that never call it.
    sweep_param : str, default="n_clusters"
        The swept parameter sweep_.param reports.
    """

    def __init__(
        self,
        results: pd.DataFrame,
        select: Callable[[str, bool], tuple[str, float]],
        labels: np.ndarray | None = None,
        *,
        sweep_param: str = "n_clusters",
    ):
        self.estimator_results_ = results
        self._select = select
        self._labels = labels
        self.sweep_ = SimpleNamespace(param=sweep_param)
        self.get_labels_calls: list[dict] = []
        self.selection_calls: list[dict] = []

    def _record(self, measure, rule, not_two):
        self.selection_calls.append(
            {"measure": measure, "rule": rule, "not_two": not_two}
        )
        return self._select(measure, not_two)

    def _select_row(self, *, measure, rule, not_two=False):
        method_id, value = self._record(measure, rule, not_two)
        results = self.estimator_results_
        if self.sweep_.param == "n_clusters":
            row = results.loc[
                (results["method_id"] == method_id) & (results["n_clusters"] == value)
            ].iloc[0]
            return row, 0, int(value), False
        row = results.loc[
            (results["method_id"] == method_id)
            & np.isclose(results["sweep_value"], value)
        ].iloc[0]
        return row, 0, int(round(float(row["n_clusters_observed"]))), False

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return int(self._record(measure, rule, not_two)[1])

    def get_sweep_value(self, *, measure="stability", rule="1se", not_two=False):
        return float(self._record(measure, rule, not_two)[1])

    def get_labels(self, *, measure="stability", rule="1se", not_two=False):
        self.get_labels_calls.append(
            {"measure": measure, "rule": rule, "not_two": not_two}
        )
        if self._labels is None:
            raise AssertionError("get_labels called on a StubCarve built without labels")
        return self._labels


def simple_results(ks, method_label: str) -> pd.DataFrame:
    """One method swept over ks with monotone metrics, for composite figures."""
    ks = list(ks)
    n = len(ks)
    return pd.DataFrame(
        {
            "n_clusters": ks,
            "method_id": ["m0"] * n,
            "method_label": [method_label] * n,
            "ari_stability": list(np.linspace(0.1, 0.3, n)),
            "ari_generalizability": list(np.linspace(0.15, 0.35, n)),
        }
    )


def resolution_results(
    resolutions, method_label: str, *, method_id: str = "m0"
) -> pd.DataFrame:
    """One Leiden configuration swept over resolutions, as estimator_results_.

    Observed cluster counts rise with resolution, as they do on real data.
    """
    resolutions = [float(r) for r in resolutions]
    n = len(resolutions)
    return pd.DataFrame(
        {
            "method_id": [method_id] * n,
            "method_label": [method_label] * n,
            "estimator": ["LeidenClustering"] * n,
            "resolution": resolutions,
            "sweep_param": ["resolution"] * n,
            "sweep_value": resolutions,
            "sweep_rank": list(range(n)),
            "n_clusters_observed": [4.0 + 4.0 * i for i in range(n)],
            "ari_stability": list(np.linspace(0.8, 0.4, n)),
            "ari_stability_se": [0.02] * n,
            "ari_generalizability": list(np.linspace(0.7, 0.3, n)),
            "ari_generalizability_se": [0.03] * n,
        }
    )


def pipeline_results(
    pipelines, resolutions, *, method_ids=("m0",)
) -> pd.DataFrame:
    """A preprocessing_results_ table: every pipeline at every resolution.

    Pipeline i at resolution rank r of method j scores
    0.5 + 0.1 * i - 0.05 * r + 0.3 * j on stability, and 0.1 less on
    generalizability, so every line is distinguishable and a test can check
    which configuration a panel drew.
    """
    rows = []
    for j, method_id in enumerate(method_ids):
        for i, pipeline in enumerate(pipelines):
            for r, resolution in enumerate(resolutions):
                stability = 0.5 + 0.1 * i - 0.05 * r + 0.3 * j
                rows.append(
                    {
                        "method_id": method_id,
                        "method_label": "LeidenClustering",
                        "pipeline": pipeline,
                        "resolution": float(resolution),
                        "n_resamples": 5,
                        "ari_stability": stability,
                        "ari_stability_se": 0.02,
                        "ari_generalizability": stability - 0.1,
                        "ari_generalizability_se": 0.03,
                        "n_clusters_observed": 4.0 + 4.0 * r,
                        "sweep_param": "resolution",
                        "sweep_value": float(resolution),
                        "sweep_rank": r,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_run_frame(scenario_name, *, n_seeds=2, metrics=None):
    """A schema-complete artifact frame for one scenario, cheap to build.

    Covers every axis point, every candidate k and every requested metric,
    with a selection per (metric, dataset) so is_selected is never all
    False. Figure tests need shape and completeness, not real numbers, and
    running the real pipeline for a layout assertion costs minutes per test.
    """
    import numpy as np
    import pandas as pd

    from benchmarks._registry import SCENARIOS
    from benchmarks.figures._benchmarking_results import DEFAULT_METRICS

    scenario = SCENARIOS[scenario_name]
    metrics = [
        m
        for m in (DEFAULT_METRICS if metrics is None else metrics)
        if m != "baseline_oracle"
    ]
    rng = np.random.default_rng(0)
    rows = []
    for axis_idx, axis_value, axis_label in scenario.axis:
        for seed in range(n_seeds):
            oracle = 0.9 - 0.1 * axis_idx
            for metric in metrics:
                chosen = int(rng.choice(scenario.candidate_k))
                for k in scenario.candidate_k:
                    rows.append(
                        {
                            "run_id": "synthetic",
                            "scenario": scenario_name,
                            "axis_name": scenario.axis.name,
                            "axis_value": axis_value,
                            "axis_label": axis_label,
                            "seed": seed,
                            "k_star": scenario.k_star,
                            "estimator": scenario.estimator.name,
                            "oracle_ari": oracle,
                            "metric_name": metric,
                            "k": int(k),
                            "metric_value": float(rng.uniform(0.1, 0.9)),
                            "is_selected": k == chosen,
                            "selects_true_k": k == scenario.k_star,
                            "ari_at_k": float(rng.uniform(0.3, 0.95)),
                            "consensus_ari_at_k": float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def make_carve_spy() -> type:
    """A CARVE stand-in class that records its constructor and fit kwargs.

    Used where a test needs to see what a call site passed to CARVE(...)
    without paying for a fit. A new class per call keeps the record private
    to the test. Instances expose enough of a fitted model for
    fit_or_load_carve to run to completion.
    """

    class SpyCARVE:
        captured_kwargs: dict | None = None
        captured_fit_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).captured_kwargs = kwargs
            self.estimator_results_ = pd.DataFrame({"config_id": [0]})
            self._n = 0

        def fit(self, X, *args, **kwargs):
            type(self).captured_fit_kwargs = kwargs
            self._n = int(np.asarray(X).shape[0])
            return self

        def save(self, path):
            Path(path).write_text("stub")

        def get_labels(self, **kwargs):
            return np.zeros(self._n, dtype=int)

        def get_k(self, **kwargs):
            return 2

    return SpyCARVE


def small_ablation(ablation, *, study=None):
    """The ablation with a single small "test" scale in place of its own.

    Figure and table tests render synthetic frames at this scale, since a
    publication run's units make every such test slow. The registry
    declares no small scale, so no real run can use it. study adds a case
    study (two replicates per setting) so the study panels and columns,
    which the registry's ablation no longer runs, stay covered.
    """
    import dataclasses

    from benchmarks._types import AblationScale, ArmScale

    study_replicates = 0 if study is None else 2
    scale = AblationScale(
        rho_arm=ArmScale(("medium",), (0, 1), 1, study_replicates),
        b_arm=ArmScale(("medium",), (0, 1), 2, study_replicates),
        similarity_draws=5,
        study_scale=None if study is None else "publication",
        n_total=500,
    )
    return dataclasses.replace(
        ablation, study=study, scales={"test": scale}, default_scale="test"
    )


HECA_N_PEAKS = 300


def write_heca_organ(path, name, n_cells, seed, *, study_id="10.1000/x"):
    """Write one synthetic hECA organ file in the real layout.

    CSR raw counts, cells by cPeaks, annotations in obs, an empty obsm. An
    organ-specific block plus a shared background gives the embedding
    structure that clustering can recover. The random draws are made in the
    order the loader tests were written against, so their numbers do not
    move.
    """
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(seed)
    types = rng.choice(["T cell", "Epithelial cell", "Unclassified"], n_cells)
    M = np.zeros((n_cells, HECA_N_PEAKS))
    offset = {"Lung": 0, "Brain": 100}.get(name, 200)
    M[:, offset : offset + 100] = rng.random((n_cells, 100)) < 0.5
    M[:, 250:] = rng.random((n_cells, 50)) < 0.4

    adata = ad.AnnData(
        X=sparse.csr_matrix(M.astype(np.int32)),
        obs=pd.DataFrame(
            {
                "cell_type": types,
                "organ": name,
                "donor_id": rng.choice(["d1", "d2"], n_cells),
                "study_id": study_id,
            },
            index=[f"{name}_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"peak{i}" for i in range(HECA_N_PEAKS)]),
    )
    adata.write_h5ad(path / f"ATAC-{name}.h5ad")


def synthetic_ablation_frames(
    ablation, scale: str, *, seed: int = 0, tmp_path: Path | None = None
) -> dict:
    """Every ablation frame for every unit at a scale, with random values,
    written as the runner's checkpoint files and read back through
    read_frames.

    Shapes follow the runner exactly (one row per cell, metric, estimator
    and k; at_k for simulations only, so a study cell writes an empty at_k
    file; similarity per rho-arm dataset, including the refit reference).
    Each unit's rows go through write_frame to the paths unit_paths gives
    it, and the frames come back through read_frames, so they carry what a
    real run directory produces rather than what an in-memory DataFrame
    would: an empty at_k part per study cell, and one undefined selection
    (the all-NaN guard's row shape, on a non-headline metric of a B-arm
    cell, the study's when there is one) that makes selected_k float64 for the whole run. Figures
    and tables are then exercised against the frames they will meet.

    tmp_path is the directory the checkpoints are written to; None uses a
    temporary directory that is removed once the frames are in memory.
    """
    import tempfile

    from benchmarks._ablation_cells import (
        REFERENCE_RATIO,
        STUDY_DATASET,
        STUDY_DIFFICULTY,
        Cell,
        Unit,
        carve_seed,
        enumerate_units,
        unit_paths,
    )
    from benchmarks._ablation_summary import HEADLINE_METRICS
    from benchmarks._artifacts import ABLATION_SCHEMAS, read_frames, write_frame
    from benchmarks._registry import CARVE_METRICS_ALL

    rng = np.random.default_rng(seed)
    draws = ablation.scales[scale].similarity_draws
    undefined_metric = "consensus_gini_stability"
    assert undefined_metric in CARVE_METRICS_ALL
    assert undefined_metric not in HEADLINE_METRICS
    other_b = int(next(b for b in ablation.b_grid if b != ablation.b_default))
    if ablation.study is not None:
        undefined_cell = Cell(
            ablation.study,
            STUDY_DIFFICULTY,
            STUDY_DATASET,
            float(ablation.rho_default),
            other_b,
            0,
        )
    else:
        b_arm = ablation.scales[scale].b_arm
        undefined_cell = Cell(
            ablation.scenarios[0],
            b_arm.difficulties[0],
            int(b_arm.datasets[0]),
            float(ablation.rho_default),
            other_b,
            0,
        )

    def _estimators(cell):
        if cell.study == ablation.study:
            return ("AgglomerativeClustering", "SpectralClustering"), tuple(range(2, 11))
        return ("KMeans",), (3, 4, 5, 6, 7)

    def _dataset(cell, is_study):
        return {
            "study": cell.study, "difficulty": cell.difficulty, "dataset": cell.dataset,
            "n_samples": 500,
            "k_star": np.nan if is_study else 5.0,
            "oracle_ari": np.nan if is_study else 0.9,
            "rare_label": np.nan if is_study else 0.0,
            "rare_fraction": np.nan if is_study else 0.1,
        }

    def _similarity(cell):
        estimators, ks = _estimators(cell)
        return [
            {
                "study": cell.study, "difficulty": cell.difficulty, "dataset": cell.dataset,
                "subsample_ratio": float(cell.subsample_ratio), "estimator": estimator,
                "k": k, "draw": draw, "ari": float(rng.uniform(0.5, 1.0)),
            }
            for estimator in estimators
            for k in ks
            for draw in range(draws)
        ]

    def _cell(cell, is_study):
        key = cell.key()
        estimators, ks = _estimators(cell)
        curves, selection, at_k = [], [], []
        for metric in CARVE_METRICS_ALL:
            for estimator in estimators:
                for k in ks:
                    curves.append(
                        {
                            **key, "metric_name": metric, "estimator": estimator, "k": k,
                            "metric_value": float(rng.uniform(0, 1)),
                            "metric_se": float(rng.uniform(0.01, 0.05)) if metric.startswith("ari_") else np.nan,
                        }
                    )
            row = {
                **key, "metric_name": metric,
                "selected_estimator": estimators[int(rng.integers(len(estimators)))],
                "selected_k": int(rng.choice(ks)),
                "k_star": np.nan if is_study else 5.0,
                "ari_selected": float(rng.uniform(0, 1)),
            }
            if cell == undefined_cell and metric == undefined_metric:
                row.update(selected_estimator=None, selected_k=np.nan, ari_selected=np.nan)
            selection.append(row)
        if not is_study:
            for mode in ("default", "generalizability"):
                for k in ks:
                    at_k.append(
                        {
                            **key, "mode": mode, "k": k,
                            "ari_at_k": float(rng.uniform(0, 1)),
                            "rare_recall_at_k": float(rng.uniform(0, 1)),
                        }
                    )
        cells = [
            {
                **key, "carve_random_state": carve_seed(cell, ablation=ablation),
                "n_samples": 500, "fit_seconds": float(rng.uniform(1, 5)),
                "consensus_nan_fraction": float(rng.uniform(0, 0.05)),
                "consensus_generalizability_nan_fraction": float(rng.uniform(0, 0.4)),
                "n_cluster_count_warnings": int(rng.integers(0, 3)),
            }
        ]
        return {"curves": curves, "selection": selection, "at_k": at_k, "cells": cells}

    def _unit_frames(unit: Unit) -> dict:
        is_study = unit.cell.study == ablation.study
        if unit.kind == "dataset":
            return {"datasets": [_dataset(unit.cell, is_study)]}
        if unit.kind == "similarity":
            return {"similarity": _similarity(unit.cell)}
        return _cell(unit.cell, is_study)

    def _write_and_read(rd: Path) -> dict:
        units = enumerate_units(ablation, scale)
        assert Unit("cell", undefined_cell) in units
        assert REFERENCE_RATIO in {
            u.cell.subsample_ratio for u in units if u.kind == "similarity"
        }
        for unit in units:
            rows = _unit_frames(unit)
            for name, path in unit_paths(rd, unit).items():
                write_frame(path, rows[name], ABLATION_SCHEMAS[name])
        return read_frames(rd)

    if tmp_path is not None:
        return _write_and_read(Path(tmp_path))
    with tempfile.TemporaryDirectory() as tmp_dir:
        return _write_and_read(Path(tmp_dir))


def small_heca_study(root):
    """STUDIES["heca"] on two synthetic organs, sized for a test.

    Lung and Brain from different studies, the real 15- and 50-neighbor
    settings, a two-value grid that splits the organs into at least two
    clusters on every subsample (a single cluster warns, and warnings are
    errors here), two resamples, and no anchors: the fit is far below
    anchor_threshold.
    """
    from dataclasses import replace

    from benchmarks._studies import STUDIES
    from benchmarks.datasets import load_heca

    root = Path(root)
    directory = root / "hECA"
    directory.mkdir(parents=True, exist_ok=True)
    write_heca_organ(directory, "Lung", 90, 0, study_id="10.1000/lung")
    write_heca_organ(directory, "Brain", 80, 1, study_id="10.1000/brain")

    def loader(subsample):
        return load_heca(
            root=root,
            organs=("Lung", "Brain"),
            subsample=subsample,
            random_state=42,
            label_column="organ",
            n_top_peaks=50,
            n_components=5,
        )

    return replace(
        STUDIES["heca"],
        loader=loader,
        resolutions=(1.0, 2.0),
        n_resamples=2,
        consensus_anchors=None,
    )
