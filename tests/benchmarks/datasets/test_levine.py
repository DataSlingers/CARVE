"""Tests for the Levine 32-dimension CyTOF loader.

The R round trip is skipped unless CARVE_ALLOW_R=1, so the suite runs on a
machine without HDCytoData installed. The cache logic is tested with a
synthetic cache file, which needs no R at all.
"""

import os

import numpy as np
import pytest

from benchmarks.datasets import load_levine32
from benchmarks.datasets._levine import _cache_path, _read_cache, _write_cache

needs_r = pytest.mark.skipif(
    os.environ.get("CARVE_ALLOW_R") != "1",
    reason="set CARVE_ALLOW_R=1 to exercise the rpy2 path",
)


class TestCache:
    def test_round_trips_through_npz(self, tmp_path):
        X = np.arange(12, dtype=float).reshape(4, 3)
        y = np.array(["a", "b", "a", "c"])
        markers = ["m1", "m2", "m3"]
        _write_cache(tmp_path, X, y, markers)
        loaded_X, loaded_y, loaded_markers = _read_cache(tmp_path)
        np.testing.assert_array_equal(loaded_X, X)
        np.testing.assert_array_equal(loaded_y, y)
        assert loaded_markers == markers

    def test_reports_a_missing_cache_as_none(self, tmp_path):
        assert _read_cache(tmp_path) is None

    def test_cache_path_is_under_the_cache_dir(self, tmp_path):
        assert _cache_path(tmp_path).parent == tmp_path


class TestInstallGate:
    def test_refuses_to_install_without_explicit_opt_in(self, tmp_path):
        with pytest.raises(RuntimeError, match="allow_install"):
            load_levine32(cache_dir=tmp_path, allow_install=False)

    def test_error_explains_how_to_proceed(self, tmp_path):
        with pytest.raises(RuntimeError, match="HDCytoData"):
            load_levine32(cache_dir=tmp_path, allow_install=False)


class TestLoadFromCache:
    def test_uses_the_cache_without_touching_r(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = rng.choice(["1", "2", "3"], size=200)
        _write_cache(tmp_path, X, y, ["m1", "m2", "m3", "m4"])

        loaded_X, loaded_y, meta = load_levine32(cache_dir=tmp_path)
        assert loaded_X.shape == (200, 4)
        assert len(loaded_y) == 200
        assert meta["from_cache"] is True

    def test_subsampling_is_honored_and_deterministic(self, tmp_path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        y = rng.choice(["1", "2", "3"], size=300)
        _write_cache(tmp_path, X, y, ["m1", "m2", "m3", "m4"])

        first, _, _ = load_levine32(cache_dir=tmp_path, subsample=100, random_state=7)
        second, _, _ = load_levine32(cache_dir=tmp_path, subsample=100, random_state=7)
        assert first.shape[0] == 100
        np.testing.assert_array_equal(first, second)

    def test_metadata_records_the_scaler_ordering_quirk(self, tmp_path):
        rng = np.random.default_rng(0)
        _write_cache(tmp_path, rng.normal(size=(50, 3)), rng.choice(["1", "2"], 50), ["a", "b", "c"])
        _, _, meta = load_levine32(cache_dir=tmp_path)
        assert any("population 15" in step for step in meta["preprocessing"])


@needs_r
class TestLoadFromR:
    def test_populates_the_cache_on_first_load(self, tmp_path):
        X, y, meta = load_levine32(cache_dir=tmp_path, allow_install=True, subsample=500)
        assert X.shape[0] == 500
        assert meta["from_cache"] is False
        assert _cache_path(tmp_path).exists()

    def test_population_15_is_absent(self, tmp_path):
        _, y, _ = load_levine32(cache_dir=tmp_path, allow_install=True)
        assert "15" not in set(np.asarray(y).astype(str))
