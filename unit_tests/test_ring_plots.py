"""The two nonlinear-ring plots."""

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.Modules.plotting.ring import (  # noqa: E402
    aperture_boundary,
    plot_amplitude_map,
    plot_dynamic_aperture,
    plot_frequency_map,
    resonance_lines,
)

TURNS = 512


@pytest.fixture
def footprint():
    return [
        (x, 1e-3, 0.28 + 0.09 * x * 50, 0.21 + 0.07 * x * 50, -14 + 600 * x)
        for x in np.linspace(0.001, 0.015, 12)
    ]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_rays_are_drawn_as_their_boundary():
    """One point per ray: the outermost survivor, with the next step lost."""
    rays = []
    for angle in np.linspace(0, np.pi, 7):
        for f in np.linspace(0.1, 1.0, 10):
            x, y = 0.02 * f * math.cos(angle), 0.01 * f * math.sin(angle)
            rays.append((x, y, TURNS - 1 if math.hypot(x / 0.016, y / 0.007) < 1 else 3))
    edge = aperture_boundary(rays, TURNS)
    assert len(edge) == 7
    for x, y in edge:
        assert math.hypot(x / 0.016, y / 0.007) < 1
        f = math.hypot(x / 0.02, y / 0.01)
        assert math.hypot(x * (f + 0.1) / f / 0.016, y * (f + 0.1) / f / 0.007) >= 1
    line = plot_dynamic_aperture(rays, TURNS).lines[0]
    assert np.allclose(line.get_xdata(), [x * 1e3 for x, _ in edge])


def test_an_island_past_a_loss_is_not_the_boundary():
    """As elegant's rays stop at their first loss."""
    ray = [(0.001, 0.001, TURNS - 1), (0.002, 0.002, 10), (0.003, 0.003, TURNS - 1)]
    assert aperture_boundary(ray, TURNS) == [(0.001, 0.001)]


def test_elegant_boundary_is_drawn_as_it_is():
    """``find_aperture`` survivors, ordered up the +x side and down the -x side."""
    boundary = [(-0.0057, 0.0, TURNS), (0.0041, 0.0, TURNS), (0.0, 0.0032, TURNS), (0.0022, 0.002, TURNS)]
    assert aperture_boundary(boundary, TURNS) == [(0.0041, 0.0), (0.0022, 0.002), (0.0, 0.0032), (-0.0057, 0.0)]


def test_an_empty_scan_does_not_raise():
    axes = plot_dynamic_aperture([], TURNS)
    assert "no data" in axes.get_title()


def test_the_footprint_is_drawn_in_tune_space(footprint):
    axes = plot_frequency_map(footprint, order=0)
    assert axes.get_xlabel() == r"$Q_x$"
    mapped = [c for c in axes.collections if c.get_array() is not None]
    assert len(mapped) == 1


def test_resonance_lines_are_drawn_and_recessive(footprint):
    axes = plot_frequency_map(footprint, order=3)
    assert axes.lines, "no resonance lines drawn"
    assert all(line.get_linewidth() < 1.0 for line in axes.lines)
    assert all(line.get_alpha() < 0.7 for line in axes.lines)


def test_no_resonance_lines_when_order_is_zero(footprint):
    assert not plot_frequency_map(footprint, order=0).lines


def test_resonance_lines_do_not_rescale_the_axes():
    """They are reference, not data."""
    _, axes = plt.subplots()
    axes.set_xlim(0.30, 0.32)
    axes.set_ylim(0.20, 0.22)
    resonance_lines(axes, order=4)
    assert axes.get_xlim() == pytest.approx((0.30, 0.32))
    assert axes.get_ylim() == pytest.approx((0.20, 0.22))


def test_a_lost_particle_with_nan_tunes_is_dropped(footprint):
    axes = plot_frequency_map(
        list(footprint) + [(0.02, 0.0, float("nan"), float("nan"), float("nan"))],
        order=0,
    )
    mapped = [c for c in axes.collections if c.get_array() is not None][0]
    assert len(mapped.get_array()) == len(footprint)


def test_an_empty_footprint_does_not_raise():
    assert "no data" in plot_frequency_map([]).get_title()


def test_the_amplitude_map_shares_the_diffusion_scale(footprint):
    axes = plot_amplitude_map(footprint)
    assert "x [mm]" in axes.get_xlabel()
    mapped = [c for c in axes.collections if c.get_array() is not None][0]
    assert np.asarray(mapped.get_array()).min() < 0


def test_an_empty_amplitude_map_does_not_raise():
    assert "no data" in plot_amplitude_map([]).get_title()
