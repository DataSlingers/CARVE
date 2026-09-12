"""Helpers shared across the benchmarks test files."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


class StubCarve:
    """Minimal stand-in for a fitted CARVE object.

    The panels and figure code read three members off a fitted model:
    estimator_results_, _select_row() and get_k() (and prepare_composite
    also calls get_labels()). This implements exactly those instead of
    fitting a real model.

    Column names are the canonical estimator_results_ names a real fitted
    model uses (ari_stability and ari_generalizability plus their _se
    variants, method_id, method_label, n_clusters). "stability" and
    "generalizability" are only measure aliases, never column names; a stub
    that named its columns after the aliases would let carve_lines index the
    alias directly and still pass.

    Parameters
    ----------
    results : DataFrame
        The table, with at least method_id and n_clusters columns.
    select : callable (measure, not_two) -> (method_id, n_clusters)
        Which row _select_row and get_k resolve to. Making it a function lets
        a test decide whether measure or not_two changes the answer, so the
        test can tell whether the code under test forwarded them.
    labels : ndarray, optional
        What get_labels returns. Left None by tests that never call it.
    """

    def __init__(
        self,
        results: pd.DataFrame,
        select: Callable[[str, bool], tuple[str, int]],
        labels: np.ndarray | None = None,
    ):
        self.estimator_results_ = results
        self._select = select
        self._labels = labels
        self.get_labels_calls: list[dict] = []

    def _select_row(self, *, measure, rule, not_two=False):
        method_id, k = self._select(measure, not_two)
        results = self.estimator_results_
        row = results.loc[
            (results["method_id"] == method_id) & (results["n_clusters"] == k)
        ].iloc[0]
        return row, 0, int(k), False

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return int(self._select(measure, not_two)[1])

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


def make_carve_spy() -> type:
    """A CARVE stand-in class that records its constructor kwargs.

    Used where a test needs to see what a call site passed to CARVE(...)
    without paying for a fit. A new class per call keeps the record private
    to the test. Instances expose enough of a fitted model for
    fit_or_load_carve and study_scaling_sweep to run to completion.
    """

    class SpyCARVE:
        captured_kwargs: dict | None = None

        def __init__(self, **kwargs):
            type(self).captured_kwargs = kwargs
            self.estimator_results_ = pd.DataFrame({"config_id": [0]})
            self._n = 0

        def fit(self, X, *args, **kwargs):
            self._n = int(np.asarray(X).shape[0])
            return self

        def save(self, path):
            Path(path).write_text("stub")

        def get_labels(self, **kwargs):
            return np.zeros(self._n, dtype=int)

        def get_k(self, **kwargs):
            return 2

    return SpyCARVE
