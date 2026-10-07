"""The two nonlinear-ring plots."""

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simba.Modules.plotting.ring import (  # noqa: E402
    plot_amplitude_map,
    plot_dynamic_aperture,
    plot_frequency_map,
    resonance_lines,
)

TURNS = 512


@pytest.fixture
def aperture():
    """A grid straddling an elliptical aperture, so both states are present."""
    points = []
    for y in np.linspace(0.0005, 0.008, 6):
        for x in np.linspace(0.001, 0.02, 10):
            radius = math.hypot(x / 0.016, y / 0.007)
            survived = TURNS - 1 if radius < 1 else int(TURNS * 0.3 * (1.4 - radius))
            points.append((x, y, survived))
    return points


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


# --- the aperture plot --------------------------------------------------


def test_it_draws_both_states(aperture):
    axes = plot_dynamic_aperture(aperture, TURNS)
    assert len(axes.collections) == 2


def test_colour_is_mapped_to_the_losses_not_the_survivors(aperture):
    """The fix for the original error: the mapped collection must be the
    one whose values vary. Survivors are all `TURNS - 1`."""
    axes = plot_dynamic_aperture(aperture, TURNS)
    mapped = [c for c in axes.collections if c.get_array() is not None]
    assert len(mapped) == 1
    values = np.asarray(mapped[0].get_array())
    assert values.min() < values.max(), "colour must encode something that varies"
    assert values.max() < TURNS - 1, "survivors must not be in the colour mapping"


def test_the_two_states_differ_by_marker_not_only_colour(aperture):
    """A reader who cannot separate the hues must still see the boundary."""
    axes = plot_dynamic_aperture(aperture, TURNS)
    paths = [tuple(map(tuple, c.get_paths()[0].vertices[:4])) for c in axes.collections]
    assert paths[0] != paths[1]


def test_the_legend_does_not_imply_a_colour_means_lost(aperture):
    """The handles are built by hand; taking them from the mapped scatter
    would put an arbitrary colormap step beside the word 'lost'."""
    axes = plot_dynamic_aperture(aperture, TURNS)
    labels = [t.get_text() for t in axes.get_legend().get_texts()]
    assert any("lost" in label for label in labels)
    assert any(str(TURNS) in label for label in labels)


def test_an_all_surviving_grid_needs_no_colourbar(aperture):
    """Nothing was lost, so there is no magnitude and no legend to draw."""
    axes = plot_dynamic_aperture([(x, y, TURNS - 1) for x, y, _ in aperture], TURNS)
    assert axes.get_legend() is None
    assert all(c.get_array() is None for c in axes.collections)


def test_an_empty_scan_does_not_raise():
    axes = plot_dynamic_aperture([], TURNS)
    assert "no data" in axes.get_title()


# --- the frequency map --------------------------------------------------


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
    """They are reference, not data -- a line running off to infinity must
    not stretch the footprint into a corner."""
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


# --- the amplitude map --------------------------------------------------


def test_the_amplitude_map_shares_the_diffusion_scale(footprint):
    """Same quantity as the frequency map, different axes -- one says which
    resonance, the other says where in the aperture."""
    axes = plot_amplitude_map(footprint)
    assert "x [mm]" in axes.get_xlabel()
    mapped = [c for c in axes.collections if c.get_array() is not None][0]
    assert np.asarray(mapped.get_array()).min() < 0


def test_an_empty_amplitude_map_does_not_raise():
    assert "no data" in plot_amplitude_map([]).get_title()
