"""Tests for CARVE save / load persistence."""

import numpy as np
import pytest
from sklearn.cluster import KMeans

from carve import CARVE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_carve():
    """Return a small fitted CARVE instance (module-scoped for speed)."""
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(30, 5) + [3, 0, 0, 0, 0],
        rng.randn(30, 5) + [0, 3, 0, 0, 0],
    ])
    carve = CARVE(
        n_clusters=2,
        n_resamples=5,
        subsample_ratio=0.8,
        estimator_param_grids=[(KMeans, {"n_clusters": [2]})],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve


# ---------------------------------------------------------------------------
# save() guards
# ---------------------------------------------------------------------------

class TestSaveGuards:
    def test_save_unfitted_raises(self, tmp_path):
        carve = CARVE(verbose=0)
        with pytest.raises(RuntimeError, match="not been fitted"):
            carve.save(tmp_path / "should_not_exist.carve")

        assert not (tmp_path / "should_not_exist.carve").exists()


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_roundtrip_without_data(self, fitted_carve, tmp_path):
        """save(include_data=False) → load() preserves fitted results but not X_."""
        path = tmp_path / "no_data.carve"
        fitted_carve.save(path, include_data=False)

        loaded = CARVE.load(path)

        # Fitted results match
        assert loaded.estimator_results_ is not None
        assert loaded.estimator_results_.equals(fitted_carve.estimator_results_)

        # Consensus matrices match
        assert len(loaded.consensus_matrices_) == len(fitted_carve.consensus_matrices_)
        for orig, restored in zip(
            fitted_carve.consensus_matrices_, loaded.consensus_matrices_
        ):
            if orig is None:
                assert restored is None
            else:
                np.testing.assert_array_equal(orig, restored)

        # X_ was excluded
        assert loaded.X_ is None

        # Original instance is unaffected
        assert fitted_carve.X_ is not None

    def test_roundtrip_with_data(self, fitted_carve, tmp_path):
        """save(include_data=True) → load() preserves X_."""
        path = tmp_path / "with_data.carve"
        fitted_carve.save(path, include_data=True)

        loaded = CARVE.load(path)
        np.testing.assert_array_equal(loaded.X_, fitted_carve.X_)

    def test_labels_match_after_load(self, fitted_carve, tmp_path):
        """get_labels() on a loaded instance returns same labels."""
        path = tmp_path / "labels.carve"
        fitted_carve.save(path)

        loaded = CARVE.load(path)
        original_labels = fitted_carve.get_labels()
        loaded_labels = loaded.get_labels()

        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_config_preserved(self, fitted_carve, tmp_path):
        """Init parameters survive the round-trip."""
        path = tmp_path / "config.carve"
        fitted_carve.save(path)

        loaded = CARVE.load(path)
        assert loaded.n_clusters == fitted_carve.n_clusters
        assert loaded.n_resamples == fitted_carve.n_resamples
        assert loaded.subsample_ratio == fitted_carve.subsample_ratio
        assert loaded.random_state == fitted_carve.random_state

    def test_sample_level_scores_preserved(self, fitted_carve, tmp_path):
        """Stability / generalizability sample-level arrays survive."""
        path = tmp_path / "scores.carve"
        fitted_carve.save(path)
        loaded = CARVE.load(path)

        if fitted_carve.stability_gini_scores_ is not None:
            np.testing.assert_array_equal(
                loaded.stability_gini_scores_, fitted_carve.stability_gini_scores_
            )
        if fitted_carve.stability_ce_scores_ is not None:
            np.testing.assert_array_equal(
                loaded.stability_ce_scores_, fitted_carve.stability_ce_scores_
            )
        if fitted_carve.misclassification_rates_ is not None:
            np.testing.assert_array_equal(
                loaded.misclassification_rates_, fitted_carve.misclassification_rates_
            )


# ---------------------------------------------------------------------------
# load() guards
# ---------------------------------------------------------------------------

class TestLoadGuards:
    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CARVE.load(tmp_path / "no_such_file.carve")

    def test_load_non_carve_object_raises(self, tmp_path):
        """Loading a file that contains a non-CARVE object raises TypeError."""
        import joblib

        path = tmp_path / "not_carve.carve"
        joblib.dump({"hello": "world"}, path)

        with pytest.raises(TypeError, match="Expected a CARVE instance"):
            CARVE.load(path)


# ---------------------------------------------------------------------------
# Compression & file-size
# ---------------------------------------------------------------------------

class TestCompression:
    def test_compressed_smaller_than_uncompressed(self, fitted_carve, tmp_path):
        compressed = tmp_path / "compressed.carve"
        uncompressed = tmp_path / "uncompressed.carve"

        fitted_carve.save(compressed, compress=3)
        fitted_carve.save(uncompressed, compress=0)

        assert compressed.stat().st_size < uncompressed.stat().st_size

    def test_no_data_smaller_than_with_data(self, fitted_carve, tmp_path):
        no_data = tmp_path / "no_data.carve"
        with_data = tmp_path / "with_data.carve"

        fitted_carve.save(no_data, include_data=False)
        fitted_carve.save(with_data, include_data=True)

        assert no_data.stat().st_size < with_data.stat().st_size


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_save_creates_parent_dirs(self, fitted_carve, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "result.carve"
        fitted_carve.save(nested)

        assert nested.exists()
        loaded = CARVE.load(nested)
        assert loaded.estimator_results_ is not None
