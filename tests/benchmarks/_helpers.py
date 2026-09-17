"""Helpers shared across the benchmarks test files."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


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


def make_carve_spy() -> type:
    """A CARVE stand-in class that records its constructor and fit kwargs.

    Used where a test needs to see what a call site passed to CARVE(...)
    without paying for a fit. A new class per call keeps the record private
    to the test. Instances expose enough of a fitted model for
    fit_or_load_carve and study_scaling_sweep to run to completion.
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


def synthetic_ablation_frames(ablation, scale: str, *, seed: int = 0) -> dict:
    """Every ablation frame for every cell at a scale, with random values.

    Shapes follow the runner exactly (one row per cell, metric, estimator
    and k; at_k for simulations only; similarity per rho-arm dataset,
    including the refit reference), so figures and tables can be exercised
    without a fit.
    """
    from benchmarks._ablation_cells import (
        REFERENCE_RATIO,
        enumerate_cells,
        carve_seed,
    )
    from benchmarks._artifacts import ABLATION_SCHEMAS
    from benchmarks._registry import CARVE_METRICS_ALL

    rng = np.random.default_rng(seed)
    cells = enumerate_cells(ablation, scale)
    rho_cells = enumerate_cells(ablation, scale, arm="rho")
    draws = ablation.scales[scale].similarity_draws

    def _estimators(cell):
        if cell.study == ablation.study:
            return ("AgglomerativeClustering", "SpectralClustering"), tuple(range(2, 11))
        return ("KMeans",), (3, 4, 5, 6, 7)

    datasets, cells_rows, curves, selection, at_k, similarity = [], [], [], [], [], []
    seen = set()
    for cell in cells:
        key = cell.key()
        is_study = cell.study == ablation.study
        estimators, ks = _estimators(cell)
        dataset_id = (cell.study, cell.difficulty, cell.dataset)
        if dataset_id not in seen:
            seen.add(dataset_id)
            datasets.append(
                {
                    "study": cell.study, "difficulty": cell.difficulty, "dataset": cell.dataset,
                    "n_samples": 500,
                    "k_star": np.nan if is_study else 5.0,
                    "oracle_ari": np.nan if is_study else 0.9,
                    "rare_label": np.nan if is_study else 0.0,
                    "rare_fraction": np.nan if is_study else 0.1,
                }
            )
        cells_rows.append(
            {
                **key, "carve_random_state": carve_seed(cell, ablation=ablation),
                "n_samples": 500, "fit_seconds": float(rng.uniform(1, 5)),
                "consensus_nan_fraction": float(rng.uniform(0, 0.05)),
                "n_cluster_count_warnings": int(rng.integers(0, 3)),
            }
        )
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
            selection.append(
                {
                    **key, "metric_name": metric,
                    "selected_estimator": estimators[int(rng.integers(len(estimators)))],
                    "selected_k": int(rng.choice(ks)),
                    "k_star": np.nan if is_study else 5.0,
                    "ari_selected": float(rng.uniform(0, 1)),
                }
            )
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
    for dataset_id in dict.fromkeys((c.study, c.difficulty, c.dataset) for c in rho_cells):
        study, difficulty, dataset = dataset_id
        estimators, ks = _estimators(next(c for c in rho_cells if (c.study, c.difficulty, c.dataset) == dataset_id))
        for rho in (*ablation.rho_grid, REFERENCE_RATIO):
            for estimator in estimators:
                for k in ks:
                    for draw in range(draws):
                        similarity.append(
                            {
                                "study": study, "difficulty": difficulty, "dataset": dataset,
                                "subsample_ratio": float(rho), "estimator": estimator, "k": k,
                                "draw": draw, "ari": float(rng.uniform(0.5, 1.0)),
                            }
                        )
    frames = {
        "curves": pd.DataFrame(curves), "selection": pd.DataFrame(selection),
        "at_k": pd.DataFrame(at_k), "cells": pd.DataFrame(cells_rows),
        "datasets": pd.DataFrame(datasets), "similarity": pd.DataFrame(similarity),
    }
    return {name: frame[list(ABLATION_SCHEMAS[name])] for name, frame in frames.items()}
