"""Tests for carve._output module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from carve._output import _log_config_progress, _print_run_footer, _print_run_header


class TestPrintRunHeader:
    def test_verbose_zero_no_output(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            n_clusters=np.array([2, 3]),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=0,
        )
        assert capsys.readouterr().out == ""

    def test_verbose_one_prints(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            n_clusters=np.array([2, 3]),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=1,
        )
        out = capsys.readouterr().out
        assert "[CARVE]" in out
        assert "n_samples" in out
        assert "n_resamples" in out


class TestPrintRunFooter:
    def test_verbose_zero_no_output(self, capsys):
        df = pd.DataFrame({"a": [1, 2]})
        _print_run_footer(df, verbose=0)
        assert capsys.readouterr().out == ""

    def test_verbose_one_prints(self, capsys):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _print_run_footer(df, verbose=1)
        out = capsys.readouterr().out
        assert "3" in out
        assert "finished" in out


class TestLogConfigProgress:
    def test_verbose_zero_no_output(self, capsys):
        record = {
            "ari_stability": 0.9,
            "ari_stability_se": 0.02,
            "ari_generalizability": 0.85,
            "ari_generalizability_se": 0.03,
        }
        _log_config_progress(
            config_idx=1,
            total_configs=2,
            est_class=KMeans,
            params={"n_clusters": 2},
            record=record,
            pbar_obj=None,
            verbose=0,
        )
        assert capsys.readouterr().out == ""

    def test_verbose_one_prints(self, capsys):
        record = {
            "ari_stability": 0.9,
            "ari_stability_se": 0.02,
            "ari_generalizability": 0.85,
            "ari_generalizability_se": 0.03,
        }
        _log_config_progress(
            config_idx=1,
            total_configs=2,
            est_class=KMeans,
            params={"n_clusters": 2},
            record=record,
            pbar_obj=None,
            verbose=1,
        )
        out = capsys.readouterr().out
        assert "KMeans" in out
        assert "n_clusters=2" in out
        assert "ARI_stab" in out
