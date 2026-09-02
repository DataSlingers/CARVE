"""Tests for the leaf dataclasses in benchmarks._types.

Named test_benchmarks_types.py rather than test_types.py to keep the basename
unique against tests/test_types.py: pytest's default prepend import mode
aborts the whole suite with an "import file mismatch" error when two test
modules share a basename and neither tests/ nor tests/benchmarks/ is a
package.
"""

import pytest

from benchmarks._types import KNOWN_ESTIMATORS, Axis, EstimatorSpec


class TestAxis:
    def test_iterates_index_value_label_triples(self):
        axis = Axis(name="difficulty", values=(0, 1, 2), labels=("easy", "medium", "hard"))
        assert list(axis) == [(0, 0, "easy"), (1, 1, "medium"), (2, 2, "hard")]

    def test_length_is_the_number_of_points(self):
        axis = Axis(name="n_total", values=(1000, 5500, 10000), labels=("start", "middle", "end"))
        assert len(axis) == 3

    def test_rejects_mismatched_values_and_labels(self):
        with pytest.raises(ValueError, match="same length"):
            Axis(name="p", values=(1, 2, 3), labels=("a", "b"))

    def test_rejects_an_empty_axis(self):
        with pytest.raises(ValueError, match="at least one point"):
            Axis(name="p", values=(), labels=())

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="duplicate"):
            Axis(name="p", values=(1, 2), labels=("a", "a"))

    def test_is_frozen(self):
        axis = Axis(name="p", values=(1,), labels=("a",))
        with pytest.raises(AttributeError):
            axis.name = "q"


class TestEstimatorSpec:
    @pytest.mark.parametrize("name", sorted(KNOWN_ESTIMATORS))
    def test_accepts_every_known_estimator(self, name):
        assert EstimatorSpec(name=name).name == name

    def test_raises_on_an_unknown_name(self):
        with pytest.raises(ValueError, match="aglomerative"):
            EstimatorSpec(name="aglomerative")

    def test_error_lists_the_valid_names(self):
        with pytest.raises(ValueError, match="kmeans"):
            EstimatorSpec(name="nope")
