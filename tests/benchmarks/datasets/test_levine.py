"""Tests for the Levine 32-dimension CyTOF loader.

The loader reads through R on every call. The R round trip is skipped unless
CARVE_ALLOW_R=1, so the suite runs on a machine without HDCytoData; everything
after the R step (the install gate, subsampling, labels, metadata) is tested
with _hdcytodata_is_installed and _load_from_r stubbed, which needs no R.
"""

import os

import numpy as np
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

import benchmarks.datasets._levine as levine_module
from benchmarks.datasets import load_levine32
from benchmarks.datasets._levine import _POPULATION_LEVELS, _check_levels

needs_r = pytest.mark.skipif(
    os.environ.get("CARVE_ALLOW_R") != "1",
    reason="set CARVE_ALLOW_R=1 to exercise the rpy2 path",
)

VERSIONS = {"hdcytodata_version": "1.0.0", "r_version": "R version 4.0.0"}


@pytest.fixture
def stub_r(monkeypatch):
    """Stand in for R: HDCytoData installed, _load_from_r returning given data.

    Returns a function taking (X, codes) and returning the list each R load
    appends to, so a test can count the loads.
    """

    def install(X, codes):
        loads = []

        def load():
            loads.append(1)
            markers = [f"m{i}" for i in range(X.shape[1])]
            return X, np.asarray(codes).astype(str), markers, dict(VERSIONS)

        monkeypatch.setattr(levine_module, "_hdcytodata_is_installed", lambda: True)
        monkeypatch.setattr(levine_module, "_load_from_r", load)
        return loads

    return install


class TestInstallGate:
    @pytest.fixture
    def missing(self, monkeypatch):
        """HDCytoData absent; records installs and fails on any R load."""
        installs = []

        def load():
            raise AssertionError("loaded through R without HDCytoData")

        monkeypatch.setattr(levine_module, "_hdcytodata_is_installed", lambda: False)
        monkeypatch.setattr(levine_module, "_install_hdcytodata", lambda: installs.append(1))
        monkeypatch.setattr(levine_module, "_load_from_r", load)
        return installs

    def test_refuses_to_install_without_explicit_opt_in(self, missing):
        with pytest.raises(RuntimeError, match="allow_install"):
            load_levine32()
        assert missing == []

    def test_error_explains_how_to_proceed(self, missing):
        with pytest.raises(RuntimeError, match=r"make levine-setup.*HDCytoData"):
            load_levine32()

    def test_installs_when_allowed(self, missing, monkeypatch):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        monkeypatch.setattr(
            levine_module,
            "_load_from_r",
            lambda: (X, np.full(20, "1"), ["a", "b", "c"], dict(VERSIONS)),
        )
        load_levine32(allow_install=True)
        assert missing == [1]

    def test_never_installs_what_is_already_there(self, stub_r, monkeypatch):
        installs = []
        monkeypatch.setattr(levine_module, "_install_hdcytodata", lambda: installs.append(1))
        stub_r(np.zeros((4, 2)), ["1", "2", "1", "2"])
        load_levine32(allow_install=True)
        assert installs == []


class TestLoad:
    def test_reads_through_r_on_every_call(self, stub_r):
        # No copy of the data is kept outside R, so a second load fetches
        # again rather than reading something written by the first.
        loads = stub_r(np.zeros((4, 2)), ["1", "2", "1", "2"])
        load_levine32()
        load_levine32()
        assert loads == [1, 1]

    def test_returns_the_matrix_r_produced(self, stub_r):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        codes = rng.choice(["1", "2", "3"], size=200)
        stub_r(X, codes)

        loaded_X, loaded_y, _ = load_levine32()
        np.testing.assert_array_equal(loaded_X, X)
        names = {"1": "Basophils", "2": "CD16-_NK_cells", "3": "CD16+_NK_cells"}
        np.testing.assert_array_equal(np.asarray(loaded_y), [names[c] for c in codes])

    def test_subsampling_is_honored_and_deterministic(self, stub_r):
        rng = np.random.default_rng(0)
        stub_r(rng.normal(size=(300, 4)), rng.choice(["1", "2", "3"], size=300))

        first, _, _ = load_levine32(subsample=100, random_state=7)
        second, _, _ = load_levine32(subsample=100, random_state=7)
        assert first.shape[0] == 100
        np.testing.assert_array_equal(first, second)

    def test_metadata_records_the_scaler_ordering_quirk(self, stub_r):
        rng = np.random.default_rng(0)
        stub_r(rng.normal(size=(50, 3)), rng.choice(["1", "2"], 50))
        _, _, meta = load_levine32()
        assert any("population 15" in step for step in meta["preprocessing"])

    def test_metadata_records_the_r_side_versions(self, stub_r):
        stub_r(np.zeros((4, 2)), ["1", "2", "1", "2"])
        _, _, meta = load_levine32()
        assert meta["hdcytodata_version"] == "1.0.0"
        assert meta["r_version"] == "R version 4.0.0"


class TestLabels:
    """Codes are what R returns and what is stratified on; names are what a
    caller sees."""

    def test_returns_population_names_not_codes(self, stub_r):
        stub_r(np.zeros((4, 2)), ["1", "10", "14", "10"])
        _, loaded_y, _ = load_levine32()
        assert list(loaded_y) == ["Basophils", "Monocytes", "Pro_B_cells", "Monocytes"]

    def test_refuses_a_code_with_no_gated_population(self, stub_r):
        # 15 is the unassigned level, which the R step drops.
        stub_r(np.zeros((2, 2)), ["1", "15"])
        with pytest.raises(ValueError, match=r"\['15'\]"):
            load_levine32()

    def test_subsample_is_stratified_on_the_codes(self, stub_r):
        # Codes 1, 10, 13 and 2 sort as "1" < "10" < "13" < "2", but their
        # names sort Basophils < CD16-_NK_cells < Monocytes < Pre_B_cells, so
        # stratifying on the names would walk the classes in another order
        # and draw other cells.
        rng = np.random.default_rng(0)
        codes = rng.choice(["1", "10", "13", "2"], size=400, p=[0.1, 0.2, 0.3, 0.4])
        X = rng.normal(size=(400, 3))
        stub_r(X, codes)

        loaded_X, _, _ = load_levine32(subsample=60, random_state=3)

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
    def test_keeps_the_gated_cells_and_drops_the_unassigned(self):
        _, y, meta = load_levine32()
        assert meta["n_cells_full"] == 104_184
        assert set(y) == set(_POPULATION_LEVELS) - {"unassigned"}
        assert meta["n_features"] == 32

    def test_subsample_and_versions(self):
        X, _, meta = load_levine32(subsample=500)
        assert X.shape == (500, 32)
        assert meta["hdcytodata_version"]
        assert meta["r_version"].startswith("R version")
