"""Tests for the M3C bridge.

The tests here never touch R. The live-R tests live in the same file behind
the requires_r marker.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

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


from benchmarks._m3c import align_assignment, check_own_selection, run_m3c


class TestAlignAssignment:
    """The orientation and naming guards, tested without R.

    These are the two ways M3C's labels can come back wrong, and they fail
    differently: the wrong count means the matrix reached M3C transposed,
    while the right count under the wrong names means its gsub mangled them.
    """

    def test_reorders_by_name(self):
        out = align_assignment(
            np.array([7, 8, 9]), ["c2", "c0", "c1"], ["c0", "c1", "c2"], k=3
        )
        np.testing.assert_array_equal(out, [8, 9, 7])

    def test_raises_when_the_count_does_not_match(self):
        # The transposed case: M3C clustered the features, so it returns one
        # label per feature instead of one per sample.
        with pytest.raises(RuntimeError, match="wrong orientation"):
            align_assignment(np.array([1, 2]), ["c0", "c1"], ["c0", "c1", "c2"], k=3)

    def test_raises_when_the_names_do_not_match(self):
        # Right count, wrong names: M3C's gsub("X", "", ...) mangled them.
        with pytest.raises(RuntimeError, match="names it returned"):
            align_assignment(
                np.array([1, 2, 3]), ["0", "1", "2"], ["c0", "c1", "c2"], k=3
            )

    def test_returns_integer_labels(self):
        out = align_assignment(
            np.array([1.0, 2.0]), ["c0", "c1"], ["c0", "c1"], k=2
        )
        assert out.dtype.kind == "i"


class TestCheckOwnSelection:
    """The spec's cross-check: M3C's own selection against select_k_m3c's.

    Pure, tested without R. The two vectors this receives are both meant to
    be the same original-column-order reading of res$realdataresults[[k]]
    for k = selected_k -- one via res$assignments (M3C's own which.max(RCSI)
    + 1), one via res$realdataresults[[selected_k]]$assignments directly.
    When Python's selected_k agrees with M3C's own choice, both calls read
    the identical R vector, so an exact element-wise match is the right bar,
    not merely matching cluster counts or a permutation-invariant score.
    """

    def test_silent_when_the_two_selections_agree(self):
        # Same values, same order -- the case where selected_k really is
        # M3C's own optk, so both reads pulled the identical vector.
        partition = np.array([1, 1, 2, 2, 3, 3])
        check_own_selection(partition, partition.copy(), selected_k=3)

    def test_raises_on_a_genuine_disagreement(self):
        # Two different partitions of the same 6 samples -- not a relabeling
        # of the same groups, an actually different grouping. This is the
        # shape a real disagreement would take: M3C's top-level assignments
        # came from a different K than the one select_k_m3c recovered from
        # the scores frame.
        top_level = np.array([1, 1, 1, 2, 2, 2])
        at_selected_k = np.array([1, 2, 1, 2, 1, 2])
        with pytest.raises(RuntimeError, match="disagrees"):
            check_own_selection(top_level, at_selected_k, selected_k=4)

    def test_raises_when_the_lengths_differ(self):
        # Can't happen if both really came from the same K's assignments,
        # but a shape mismatch is exactly as much a disagreement as a value
        # mismatch, and should not be waved through by a naive elementwise
        # comparison that only checks the overlapping prefix.
        with pytest.raises(RuntimeError, match="disagrees"):
            check_own_selection(np.array([1, 1, 2]), np.array([1, 1, 2, 2]), selected_k=2)

    def test_names_the_selected_k_in_the_message(self):
        with pytest.raises(RuntimeError, match=r"selected_k=7"):
            check_own_selection(
                np.array([1, 2, 3]), np.array([3, 2, 1]), selected_k=7
            )


class TestRunGuards:
    def test_refuses_to_install_unless_asked(self, monkeypatch):
        # Simulate M3C being absent so the install branch is reached on a
        # machine that has it. Without this the test would pass for the wrong
        # reason wherever M3C happens to be installed.
        monkeypatch.setattr("benchmarks._m3c._m3c_is_installed", lambda: False)
        X = np.zeros((10, 4))
        with pytest.raises(RuntimeError, match=r'BiocManager::install\("M3C"\)'):
            run_m3c(X, max_k=3, allow_install=False)

    def test_rejects_a_max_k_below_two(self):
        with pytest.raises(ValueError, match="max_k"):
            run_m3c(np.zeros((10, 4)), max_k=1)


@pytest.mark.requires_r
class TestRunLive:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        # n=80, p=50 sits inside M3C's own recommended 60-1000 band, and
        # iters=5 is the low end of the vignette's own speed advice, so this
        # runs in seconds rather than minutes.
        #
        # A class-scoped fixture defined as an instance method (`def
        # result(self)`) is deprecated as of pytest 9 (filterwarnings=error
        # turns that PytestRemovedIn10Warning into a collection failure);
        # @classmethod is pytest's own recommended replacement.
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.normal(0.0, 1.0, size=(40, 50)),
                rng.normal(3.0, 1.0, size=(40, 50)),
            ]
        )
        return run_m3c(X, max_k=4, iters=5)

    def test_scores_carry_the_entropy_schema(self, result):
        assert list(result.scores.columns) == list(SCORES_COLUMNS)

    def test_scores_cover_k_two_through_max_k(self, result):
        assert result.scores["K"].tolist() == [2, 3, 4]

    def test_labels_are_returned_for_every_k(self, result):
        assert sorted(result.labels) == [2, 3, 4]

    def test_every_label_vector_has_one_entry_per_row_of_x(self, result):
        # Cannot actually fail on a live run: any count mismatch would
        # already have made align_assignment raise inside the class-scoped
        # fixture, erroring every test in this class before this assertion
        # runs. It only documents the shape contract, not that
        # align_assignment fired correctly -- that guard is exercised, and
        # can actually fail, in TestAlignAssignment. Whether label i
        # actually belongs to row i of X (as opposed to some other row) is
        # checked separately by test_recovers_the_planted_partition below.
        assert all(labels.shape == (80,) for labels in result.labels.values())

    def test_labels_cover_exactly_k_groups(self, result):
        # Two well-separated blobs: M3C must not return a degenerate
        # partition at any swept K.
        for k, labels in result.labels.items():
            assert 1 < len(np.unique(labels)) <= k

    def test_recovers_the_planted_partition(self, result):
        # The one end-to-end check that label i actually belongs to row i
        # of X, not merely that the label vector has the right length and
        # group count -- both of which a silently permuted alignment would
        # still satisfy. The fixture's two blobs sit 3 sigma apart in 50
        # dimensions, well past anything M3C's consensus clustering could
        # confuse at K=2, so exact recovery (ARI == 1.0) is the right bar.
        planted = [0] * 40 + [1] * 40
        assert adjusted_rand_score(planted, result.labels[2]) == 1.0

    def test_selected_k_is_recomputed_from_the_scores_frame_by_construction(
        self, result
    ):
        # Restates the definition of selected_k (run_m3c sets it directly
        # from select_k_m3c(scores)) rather than checking anything new --
        # the same caveat as test_every_label_vector_has_one_entry_per_row_of_x
        # above. The actual cross-check against M3C's own selection is
        # check_own_selection, called inside run_m3c before this fixture
        # returns: a disagreement there would already have raised and
        # errored every test in this class, the same way an align_assignment
        # failure would have. check_own_selection's ability to fail is
        # exercised directly, without R, in TestCheckOwnSelection.
        assert result.selected_k == select_k_m3c(result.scores)

    def test_records_the_r_side_versions(self, result):
        assert result.m3c_version
        assert result.r_version.startswith("R version")

    def test_config_records_what_was_run(self, result):
        assert result.config["maxK"] == 4
        assert result.config["iters"] == 5
        assert result.config["clusteralg"] == "pam"


from pathlib import Path

from benchmarks._m3c import m3c_cache_path, run_or_load_m3c


def stub_result(config: dict | None = None) -> M3CResult:
    # dict(M3C_DEFAULTS) | {"maxK": 4, "cores": 1} is exactly what
    # run_or_load_m3c(X, cache_path=..., max_k=4) itself now expects
    # (_expected_config), so the TestRunOrLoad tests that call it with
    # max_k=4 and no other overrides get a config-guard match by default.
    # Callers exercising the mismatch path pass an explicit config instead.
    return M3CResult(
        scores=scores_frame(),
        labels={2: np.zeros(6, dtype=int), 3: np.arange(6) % 3, 4: np.arange(6) % 4},
        selected_k=4,
        p_value=0.01,
        runtime_s=2.0,
        config=config if config is not None else dict(M3C_DEFAULTS) | {"maxK": 4, "cores": 1},
        m3c_version="1.34.0",
        r_version="R version 4.5.1 (2025-06-13)",
    )


class TestCachePath:
    def test_encodes_study_and_scale(self):
        path = m3c_cache_path(
            study_name="klein", scale="publication", root=Path("/tmp"), config={"maxK": 10}
        )
        assert path.name.startswith("m3c_klein_publication_")
        assert path.suffix == ".parquet"

    def test_a_different_config_gives_a_different_path(self):
        base = dict(study_name="klein", scale="publication", root=Path("/tmp"))
        a = m3c_cache_path(**base, config={"maxK": 10})
        b = m3c_cache_path(**base, config={"maxK": 17})
        assert a != b


class TestRunOrLoad:
    def test_runs_once_then_serves_the_cache(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(X, **kwargs):
            calls.append(X)
            return stub_result()

        monkeypatch.setattr("benchmarks._m3c.run_m3c", fake_run)
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        first = run_or_load_m3c(X, cache_path=cache, max_k=4)
        second = run_or_load_m3c(X, cache_path=cache, max_k=4)

        assert len(calls) == 1
        assert second.selected_k == first.selected_k == 4
        assert second.p_value == pytest.approx(0.01)
        assert second.m3c_version == "1.34.0"
        pd.testing.assert_frame_equal(second.scores, first.scores)
        np.testing.assert_array_equal(second.labels[3], first.labels[3])

    def test_refuses_a_cache_computed_on_a_different_x(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks._m3c.run_m3c", lambda X, **kw: stub_result())
        X = np.arange(18, dtype=float).reshape(6, 3)
        # Same shape deliberately, so only the fingerprint can catch it.
        Y = X.copy()
        Y[0, 0] += 1.0
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        with pytest.raises(ValueError, match="fit on a different X"):
            run_or_load_m3c(Y, cache_path=cache, max_k=4)

    def test_force_recomputes(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "benchmarks._m3c.run_m3c",
            lambda X, **kw: (calls.append(X), stub_result())[1],
        )
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        run_or_load_m3c(X, cache_path=cache, max_k=4, force=True)
        assert len(calls) == 2

    def test_serves_the_cache_when_the_config_matches(self, tmp_path, monkeypatch):
        # dict(M3C_DEFAULTS) | {"maxK": 4, "cores": 1} -- stub_result's
        # default config -- is exactly what this call itself would produce,
        # so the config guard is a no-op and the cache is served.
        monkeypatch.setattr("benchmarks._m3c.run_m3c", lambda X, **kw: stub_result())
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        served = run_or_load_m3c(X, cache_path=cache, max_k=4)
        assert served.selected_k == 4

    def test_raises_when_the_cached_config_does_not_match_this_call(
        self, tmp_path, monkeypatch
    ):
        # Reachable in a single call, per the finding this guards against:
        # the cache is written under one config (here, iters overridden to
        # 5, as if M3C_DEFAULTS or the call's own kwargs had been different
        # when it was written), and a later call with unchanged kwargs now
        # expects the real M3C_DEFAULTS (iters=25). The fingerprint alone
        # cannot catch this, since X is identical both times -- only the
        # config guard can.
        stale = stub_result(config=dict(M3C_DEFAULTS) | {"maxK": 4, "cores": 1, "iters": 5})
        monkeypatch.setattr("benchmarks._m3c.run_m3c", lambda X, **kw: stale)
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        with pytest.raises(ValueError, match="different M3C configuration"):
            run_or_load_m3c(X, cache_path=cache, max_k=4)

    def test_config_mismatch_message_names_the_differing_key(
        self, tmp_path, monkeypatch
    ):
        stale = stub_result(config=dict(M3C_DEFAULTS) | {"maxK": 4, "cores": 1, "iters": 5})
        monkeypatch.setattr("benchmarks._m3c.run_m3c", lambda X, **kw: stale)
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        with pytest.raises(ValueError, match=r"iters.*force=True") as excinfo:
            run_or_load_m3c(X, cache_path=cache, max_k=4)
        assert "cached=5" in str(excinfo.value)
        assert "requested=25" in str(excinfo.value)

    def test_recomputes_when_a_cache_file_is_missing(self, tmp_path, monkeypatch):
        # A process killed mid-write can leave the main parquet in place
        # while _write_cache's other three files -- the labels parquet, the
        # JSON sidecar, or the fingerprint sidecar -- never got written.
        # Checking cache_path.is_file() alone would then hand _read_cache a
        # bare FileNotFoundError instead of recomputing.
        calls = []
        monkeypatch.setattr(
            "benchmarks._m3c.run_m3c",
            lambda X, **kw: (calls.append(X), stub_result())[1],
        )
        X = np.arange(18, dtype=float).reshape(6, 3)
        cache = tmp_path / "m3c.parquet"

        run_or_load_m3c(X, cache_path=cache, max_k=4)
        assert len(calls) == 1

        cache.with_name(cache.stem + ".labels.parquet").unlink()

        result = run_or_load_m3c(X, cache_path=cache, max_k=4)
        assert len(calls) == 2
        assert result.selected_k == 4


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m3c_klein_scores.csv"


class TestRecordedKleinScores:
    """Pin the published Klein numbers against M3C's real output.

    The synthetic frame above fixes the semantics; this fixes the result.
    Regenerating it means the manuscript numbers moved, which must be a
    deliberate act rather than a silent one.
    """

    @pytest.fixture
    def recorded(self):
        return pd.read_csv(FIXTURE)

    def test_carries_the_entropy_schema(self, recorded):
        assert list(validate_scores(recorded).columns) == list(SCORES_COLUMNS)

    def test_sweeps_k_two_through_ten(self, recorded):
        assert recorded["K"].tolist() == list(range(2, 11))

    def test_selection_is_the_k_the_run_reported(self, recorded):
        # Step 3 prints this line with the real k substituted. Paste it here.
        assert select_k_m3c(recorded) == 2
