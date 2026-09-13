"""Tests for carve._output module."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from carve._output import _log_config_progress, _print_run_footer, _print_run_header
from carve._sweep import resolve_sweep


class TestPrintRunHeader:
    def test_verbose_zero_no_output(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=0,
        )
        assert capsys.readouterr().out == ""

    def test_verbose_one_no_output(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=1,
        )
        assert capsys.readouterr().out == ""

    def test_verbose_two_prints(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=2,
        )
        out = capsys.readouterr().out
        assert "[CARVE]" in out
        assert "n_samples" in out
        assert "n_resamples" in out
        assert "sweep parameter" in out
        assert "n_clusters" in out

    def test_verbose_two_prints_resolution_sweep(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(resolution=np.array([0.5, 1.0])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"resolution": [0.5, 1.0]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=2,
        )
        out = capsys.readouterr().out
        assert "sweep parameter" in out
        assert "resolution" in out
        assert "n_clusters" not in out

    def test_randomized_header_names_the_options(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=True,
            random_state=0,
            verbose=2,
            normalization_options=[
                (FunctionTransformer, {}),
                (FunctionTransformer, {"func": [np.log1p]}),
            ],
            dim_reduction_options=[
                (PCA, {"n_components": [2]}),
                (TSNE, "tsne", {"perplexity": [30]}),
            ],
        )
        out = capsys.readouterr().out
        assert "[CARVE] normalization      : identity, log1p\n" in out
        assert "[CARVE] dim_reduction      : PCA, tsne\n" in out

    def test_header_lists_no_options_without_randomization(self, capsys):
        _print_run_header(
            X=np.zeros((10, 3)),
            sweep=resolve_sweep(n_clusters=np.array([2, 3])),
            n_resamples=5,
            subsample_ratio=0.8,
            estimator_grids=[(KMeans, {"n_clusters": [2, 3]})],
            n_jobs=1,
            randomize_preprocessing=False,
            random_state=0,
            verbose=2,
            normalization_options=[(StandardScaler, {})],
            dim_reduction_options=[(PCA, {"n_components": [2]})],
        )
        out = capsys.readouterr().out
        assert "normalization" not in out
        assert "dim_reduction" not in out


class TestPrintRunFooter:
    def test_verbose_zero_no_output(self, capsys):
        df = pd.DataFrame({"a": [1, 2]})
        _print_run_footer(df, verbose=0)
        assert capsys.readouterr().out == ""

    def test_verbose_one_no_output(self, capsys):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _print_run_footer(df, verbose=1)
        assert capsys.readouterr().out == ""

    def test_verbose_two_prints(self, capsys):
        df = pd.DataFrame({"a": [1, 2, 3]})
        _print_run_footer(df, verbose=2)
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

    def test_uses_sweep_param(self, capsys):
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
            params={"resolution": 0.5, "n_neighbors": 15},
            record=record,
            pbar_obj=None,
            sweep_param="resolution",
            verbose=1,
        )
        out = capsys.readouterr().out
        assert "resolution=0.5" in out
        assert "n_clusters" not in out
