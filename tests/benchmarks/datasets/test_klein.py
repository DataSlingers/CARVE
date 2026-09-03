"""Tests for the Klein loader.

The heavy path needs the raw GEO files under data/Klein and is skipped when
they are absent, so the suite still runs on a machine without the data.
"""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.datasets import load_klein, resolve_data_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
KLEIN_DIR = REPO_ROOT / "data" / "Klein"
needs_data = pytest.mark.skipif(
    not KLEIN_DIR.is_dir(), reason="data/Klein is not present in this checkout"
)


class TestResolveDataDir:
    def test_finds_a_directory_by_its_exact_name(self, tmp_path):
        (tmp_path / "Klein").mkdir()
        assert resolve_data_dir("Klein", root=tmp_path).name == "Klein"

    def test_finds_a_directory_ignoring_case(self, tmp_path):
        (tmp_path / "Klein").mkdir()
        assert resolve_data_dir("klein", root=tmp_path).name == "Klein"

    def test_raises_with_the_candidates_it_saw(self, tmp_path):
        (tmp_path / "Levine").mkdir()
        with pytest.raises(FileNotFoundError, match="Levine"):
            resolve_data_dir("klein", root=tmp_path)

    def test_error_names_what_was_requested(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="klein"):
            resolve_data_dir("klein", root=tmp_path)


@needs_data
class TestLoadKlein:
    def test_returns_x_y_and_metadata(self):
        X, y, meta = load_klein(subsample=200, random_state=42)
        assert X.ndim == 2
        assert len(y) == X.shape[0]
        assert meta["source"] == "Klein"

    def test_subsampling_is_honored(self):
        X, _, _ = load_klein(subsample=200, random_state=42)
        assert X.shape[0] == 200

    def test_subsampling_is_deterministic(self):
        first, _, _ = load_klein(subsample=200, random_state=42)
        second, _, _ = load_klein(subsample=200, random_state=42)
        np.testing.assert_array_equal(first, second)

    def test_labels_are_the_four_collection_days(self):
        _, y, _ = load_klein(subsample=200, random_state=42)
        assert set(y.unique()) <= {"d0", "d2", "d4", "d7"}

    def test_metadata_records_the_preprocessing_chain(self):
        _, _, meta = load_klein(subsample=200, random_state=42)
        assert any("normalize_total" in step for step in meta["preprocessing"])
        assert any("log1p" in step for step in meta["preprocessing"])
        assert meta["n_top_genes"] == 2000

    def test_hvg_selection_bounds_the_feature_count(self):
        X, _, _ = load_klein(subsample=200, random_state=42)
        assert X.shape[1] <= 2000
