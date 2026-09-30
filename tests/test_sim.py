"""Tests for the interleaved-spirals distribution in carve.sim."""

import numpy as np
import pytest

from carve.sim import simulate_clusters
from carve.sim._distributions import SPIRAL_INNER_RADIUS, _sample_interleaved_spirals


def _spirals(**kwargs):
    params = {
        "n_total": 600,
        "p": 20,
        "k": 5,
        "distribution": "interleaved_spirals",
        "cluster_size_dirichlet_alpha": 0.9,
        "random_state": 0,
    }
    params.update(kwargs)
    return simulate_clusters(**params)


def _planar(*, turns=0.5, band=0.0, twist=0.0, seed=0, sizes=(40, 50, 60, 70, 80)):
    rng = np.random.default_rng(seed)
    return _sample_interleaved_spirals(
        rng=rng, sizes=np.asarray(sizes), turns=turns, band=band, twist=twist
    )


class TestInterleavedSpirals:
    def test_every_cluster_lies_in_one_shared_plane(self):
        """The arms share one plane and one center.

        swiss_roll draws each cluster in its own random plane around its own
        center, which is what let the geometric indices separate them; the
        whole point of this distribution is that they cannot.
        """
        X, y = _spirals(nonlinear=False)
        assert X.shape == (600, 20)
        assert set(np.unique(y)) == set(range(5))
        assert np.linalg.matrix_rank(X - X.mean(axis=0), tol=1e-8) == 2

    def test_the_embedding_maps_to_embed_dim(self):
        X, y = _spirals(nonlinear=True, embed_dim=64, embed_param=5.0)
        assert X.shape == (600, 64)
        assert y.shape == (600,)

    def test_without_band_or_twist_each_arm_is_an_exact_spiral(self):
        """r = t and theta = t + 2*pi*j/k, starting at SPIRAL_INNER_RADIUS."""
        Z2, y = _planar(turns=0.5)
        r = np.hypot(Z2[:, 0], Z2[:, 1])
        theta = np.arctan2(Z2[:, 1], Z2[:, 0])
        assert r.min() >= SPIRAL_INNER_RADIUS
        assert r.max() <= SPIRAL_INNER_RADIUS + 2 * np.pi * 0.5
        offset = np.angle(np.exp(1j * (theta - r - 2 * np.pi * y / 5)))
        np.testing.assert_allclose(offset, 0.0, atol=1e-9)

    def test_band_is_radial_noise_in_units_of_the_arm_gap(self):
        """band is the noise SD as a fraction of the 2*pi/k gap between arms."""
        Z2, y = _planar(turns=0.5, band=0.1, sizes=(4000,) * 5)
        r = np.hypot(Z2[:, 0], Z2[:, 1])
        theta = np.arctan2(Z2[:, 1], Z2[:, 0])
        # theta still equals t + 2*pi*j/k, so t is recoverable modulo 2*pi and
        # r - t is the noise; half a turn keeps t inside one branch.
        t = np.mod(theta - 2 * np.pi * y / 5 - SPIRAL_INNER_RADIUS, 2 * np.pi)
        noise = r - (t + SPIRAL_INNER_RADIUS)
        assert noise.std() == pytest.approx(0.1 * 2 * np.pi / 5, rel=0.03)

    def test_twist_rotates_each_ring_and_leaves_radii_alone(self):
        """The twist is a towel twist: the outer edge stays put, the center
        turns by twist full turns, rings in between rotate rigidly by a
        linearly interpolated angle, against the arms' winding."""
        flat, _ = _planar(turns=0.5, band=0.05, twist=0.0, seed=3)
        twisted, _ = _planar(turns=0.5, band=0.05, twist=0.2, seed=3)
        r_flat = np.hypot(flat[:, 0], flat[:, 1])
        r_twisted = np.hypot(twisted[:, 0], twisted[:, 1])
        np.testing.assert_allclose(r_twisted, r_flat)

        edge = SPIRAL_INNER_RADIUS + 2 * np.pi * 0.5
        expected = -2 * np.pi * 0.2 * np.clip(1 - r_flat / edge, 0, None)
        turned = np.angle(
            np.exp(
                1j
                * (
                    np.arctan2(twisted[:, 1], twisted[:, 0])
                    - np.arctan2(flat[:, 1], flat[:, 0])
                )
            )
        )
        np.testing.assert_allclose(turned, expected, atol=1e-9)

    def test_more_turns_lengthen_the_arms_outward(self):
        """The inner start is fixed; extra turns extend each arm outward."""
        short, _ = _planar(turns=0.4)
        long, _ = _planar(turns=0.7)
        r_short = np.hypot(short[:, 0], short[:, 1])
        r_long = np.hypot(long[:, 0], long[:, 1])
        assert r_long.max() > r_short.max() + 1.0
        assert r_long.min() == pytest.approx(r_short.min(), abs=0.2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"spiral_turns": 0.0}, "spiral_turns"),
            ({"spiral_band": -0.1}, "spiral_band"),
            ({"spiral_twist": np.inf}, "spiral_twist"),
        ],
    )
    def test_rejects_invalid_arm_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _spirals(**kwargs)

    def test_rejects_outliers(self):
        """Outliers are placed around per-cluster centers, which the shared
        spiral does not have."""
        with pytest.raises(ValueError, match="outliers"):
            _spirals(outliers=10)
