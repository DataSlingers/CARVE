"""Tests for the M3C bridge.

The tests here never touch R. The live-R tests live in the same file behind
the requires_r marker, added in Task 3.
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks._m3c import (
    M3C_DEFAULTS,
    SCORES_COLUMNS,
    M3CResult,
    select_k_m3c,
    validate_scores,
)


def scores_frame() -> pd.DataFrame:
    """A scores frame shaped exactly as M3C returns one under objective="entropy".

    K runs 2..6, and the RCSI maximum sits at K=4, which is row 2. Both
    facts matter. A frame whose maximum sat in row 0 would let a selection
    that returns a row index instead of a K value pass, and a frame whose K
    started at 0 would let the same bug pass a second way.
    """
    return pd.DataFrame(
        {
            "K": [2, 3, 4, 5, 6],
            "ENTROPY_REAL": [0.62, 0.48, 0.31, 0.35, 0.41],
            "ENTROPY_REF": [0.65, 0.63, 0.60, 0.58, 0.57],
            "RCSI": [0.05, 0.27, 0.66, 0.51, 0.33],
            "RCSI_SE": [0.01, 0.02, 0.03, 0.03, 0.04],
            "MONTECARLO_P": [0.42, 0.08, 0.01, 0.02, 0.06],
            "NORM_P": [0.40, 0.07, 0.01, 0.02, 0.05],
            "P_SCORE": [0.40, 1.15, 2.00, 1.70, 1.30],
        }
    )


class TestSelectK:
    def test_selects_the_k_at_the_rcsi_maximum(self):
        assert select_k_m3c(scores_frame()) == 4

    def test_returns_a_k_value_not_a_row_index(self):
        # The RCSI maximum is at row 2 and at K=4. A selection that returned
        # the row index, or R's which.max + 1 transcribed to a 0-based
        # argmax, would give 2 or 3 here.
        assert select_k_m3c(scores_frame()) not in (2, 3)

    def test_works_when_the_maximum_is_the_last_row(self):
        scores = scores_frame()
        scores.loc[4, "RCSI"] = 0.99
        assert select_k_m3c(scores) == 6

    def test_raises_when_rcsi_is_absent(self):
        scores = scores_frame().drop(columns=["RCSI"])
        with pytest.raises(KeyError, match="RCSI"):
            select_k_m3c(scores)


class TestValidateScores:
    def test_accepts_the_entropy_schema(self):
        frame = scores_frame()
        assert list(validate_scores(frame).columns) == list(SCORES_COLUMNS)

    def test_rejects_the_pac_schema(self):
        # PAC_REAL and PAC_REF exist only under objective="PAC". Seeing them
        # means the run was not the one this module pins, so the numbers are
        # not the ones the spec describes.
        frame = scores_frame().rename(
            columns={"ENTROPY_REAL": "PAC_REAL", "ENTROPY_REF": "PAC_REF"}
        )
        with pytest.raises(ValueError, match="objective"):
            validate_scores(frame)

    def test_rejects_a_frame_missing_a_required_column(self):
        with pytest.raises(ValueError, match="MONTECARLO_P"):
            validate_scores(scores_frame().drop(columns=["MONTECARLO_P"]))


class TestDefaults:
    def test_pins_m3c_s_published_defaults(self):
        assert dict(M3C_DEFAULTS) == {
            "method": 1,
            "clusteralg": "pam",
            "objective": "entropy",
            "iters": 25,
            "repsref": 100,
            "repsreal": 100,
            "pItem": 0.8,
            "seed": 123,
        }

    def test_defaults_are_not_mutable(self):
        with pytest.raises(TypeError):
            M3C_DEFAULTS["iters"] = 5


class TestResult:
    def test_is_frozen(self):
        result = M3CResult(
            scores=scores_frame(),
            labels={4: np.zeros(10, dtype=int)},
            selected_k=4,
            p_value=0.01,
            runtime_s=1.5,
            config={"maxK": 10},
            m3c_version="1.30.0",
            r_version="R version 4.5.1",
        )
        with pytest.raises(AttributeError):
            result.selected_k = 5
