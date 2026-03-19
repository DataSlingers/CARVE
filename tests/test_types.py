"""Tests for carve._types module."""

import pytest

from carve._types import ModePolicy, resolve_mode


class TestModePolicy:
    def test_frozen(self):
        policy = ModePolicy(
            mode="default",
            run_stability=True,
            run_generalizability=True,
            compute_average_ari=True,
        )
        with pytest.raises(AttributeError):
            policy.mode = "stability"

    def test_fields(self):
        policy = ModePolicy(
            mode="stability",
            run_stability=True,
            run_generalizability=False,
            compute_average_ari=False,
        )
        assert policy.mode == "stability"
        assert policy.run_stability is True
        assert policy.run_generalizability is False
        assert policy.compute_average_ari is False


class TestResolveMode:
    def test_default(self):
        policy = resolve_mode("default")
        assert policy.mode == "default"
        assert policy.run_stability is True
        assert policy.run_generalizability is True
        assert policy.compute_average_ari is True

    def test_stability(self):
        policy = resolve_mode("stability")
        assert policy.mode == "stability"
        assert policy.run_stability is True
        assert policy.run_generalizability is False
        assert policy.compute_average_ari is False

    def test_generalizability(self):
        policy = resolve_mode("generalizability")
        assert policy.mode == "generalizability"
        assert policy.run_stability is False
        assert policy.run_generalizability is True
        assert policy.compute_average_ari is False

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            resolve_mode("invalid")

    def test_invalid_mode_type(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            resolve_mode("foo")
