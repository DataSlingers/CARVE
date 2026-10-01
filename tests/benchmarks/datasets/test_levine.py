"""Tests for the Levine 32-dimension CyTOF loader.

The R round trip is skipped unless CARVE_ALLOW_R=1, so the suite runs on a
machine without HDCytoData installed. The cache logic is tested with a
synthetic cache file, which needs no R at all.
"""

import os

import numpy as np
import pytest

from sklearn.model_selection import StratifiedShuffleSplit

from benchmarks.datasets import load_levine32
from benchmarks.datasets._levine import (
    _POPULATION_LEVELS,
    _cache_path,
    _check_levels,
    _read_cache,
    _write_cache,
)

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
        np.testing.assert_array_equal(loaded_X, X)
        names = {"1": "Basophils", "2": "CD16-_NK_cells", "3": "CD16+_NK_cells"}
        np.testing.assert_array_equal(np.asarray(loaded_y), [names[c] for c in y])
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


class TestLabels:
    """Codes are cached and stratified on; names are what a caller sees."""

    def test_returns_population_names_not_codes(self, tmp_path):
        y = np.array(["1", "10", "14", "10"])
        _write_cache(tmp_path, np.zeros((4, 2)), y, ["m1", "m2"])
        _, loaded_y, _ = load_levine32(cache_dir=tmp_path)
        assert list(loaded_y) == ["Basophils", "Monocytes", "Pro_B_cells", "Monocytes"]

    def test_refuses_a_code_with_no_gated_population(self, tmp_path):
        # 15 is the unassigned level, which the R path drops before caching.
        _write_cache(tmp_path, np.zeros((2, 2)), np.array(["1", "15"]), ["m1", "m2"])
        with pytest.raises(ValueError, match=r"\['15'\]"):
            load_levine32(cache_dir=tmp_path)

    def test_subsample_is_stratified_on_the_codes(self, tmp_path):
        # Codes 1, 10, 13 and 2 sort as "1" < "10" < "13" < "2", but their
        # names sort Basophils < CD16-_NK_cells < Monocytes < Pre_B_cells, so
        # stratifying on the names would walk the classes in another order
        # and draw other cells.
        rng = np.random.default_rng(0)
        codes = rng.choice(["1", "10", "13", "2"], size=400, p=[0.1, 0.2, 0.3, 0.4])
        X = rng.normal(size=(400, 3))
        _write_cache(tmp_path, X, codes, ["m1", "m2", "m3"])

        loaded_X, _, _ = load_levine32(cache_dir=tmp_path, subsample=60, random_state=3)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=60 / 400, random_state=3)
        _, idx = next(splitter.split(X, codes))
        np.testing.assert_array_equal(loaded_X, X[idx])


class TestCheckLevels:
    def test_accepts_the_factor_levels_the_codes_assume(self):
        _check_levels(list(_POPULATION_LEVELS))

    def test_rejects_reordered_levels(self):
        reordered = [_POPULATION_LEVELS[-1], *_POPULATION_LEVELS[:-1]]
        with pytest.raises(RuntimeError, match="wrong populations"):
            _check_levels(reordered)

    def test_unassigned_is_code_fifteen(self):
        assert _POPULATION_LEVELS.index("unassigned") + 1 == 15


@needs_r
class TestLoadFromR:
    def test_populates_the_cache_on_first_load(self, tmp_path):
        X, y, meta = load_levine32(cache_dir=tmp_path, allow_install=True, subsample=500)
        assert X.shape[0] == 500
        assert meta["from_cache"] is False
        assert _cache_path(tmp_path).exists()

    def test_keeps_the_gated_cells_and_drops_the_unassigned(self, tmp_path):
        _, y, meta = load_levine32(cache_dir=tmp_path, allow_install=True)
        assert meta["n_cells_full"] == 104_184
        assert set(y) == set(_POPULATION_LEVELS) - {"unassigned"}
        assert meta["n_features"] == 32
