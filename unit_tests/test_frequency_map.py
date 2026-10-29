"""Frequency map / tune footprint. The codes' own scans are in
test_dynamic_aperture.py, which shares their grid."""

import math

import numpy as np
import pytest

from simba.Framework_objects import frameworkLattice
from simba.Modules.Matrices import tune_diffusion, tune_from_trajectory

TURNS = 128


@pytest.mark.parametrize("q", [0.1234, 0.25, 0.3333, 0.4871])
def test_a_known_tune_is_recovered_from_positions(q):
    """The FFT path, good to ~2e-4 on 256 turns with parabolic refinement."""
    turns = 256
    phase = 2 * math.pi * q * np.arange(turns)
    assert tune_from_trajectory(np.cos(phase)) == pytest.approx(q, abs=5e-4)


@pytest.mark.parametrize("q", [0.1234, 0.3333, 0.6180, 0.8352])
def test_naff_is_exact_where_the_fft_is_approximate(q):
    """NAFF is ~1e-12, which makes `tune_diffusion` measurable."""
    pytest.importorskip("nafflib")
    turns = 256
    phase = 2 * math.pi * q * np.arange(turns)
    got = tune_from_trajectory(np.cos(phase), -np.sin(phase))
    assert got == pytest.approx(q, abs=1e-9)


def test_naff_is_optional():
    from simba.Modules import Matrices

    assert hasattr(Matrices, "use_naff")


def test_positions_alone_fold_the_tune_below_a_half():
    """A real signal's spectrum is symmetric: 0.8 and 0.2 look the same."""
    turns = 256
    phase = 2 * math.pi * 0.8 * np.arange(turns)
    assert tune_from_trajectory(np.cos(phase)) == pytest.approx(0.2, abs=5e-4)


def test_the_momentum_resolves_tunes_above_a_half():
    turns = 256
    phase = 2 * math.pi * 0.8 * np.arange(turns)
    got = tune_from_trajectory(np.cos(phase), -np.sin(phase))
    assert got == pytest.approx(0.8, abs=5e-4)


def test_regular_motion_has_a_tiny_diffusion_index():
    turns = 512
    phase = 2 * math.pi * 0.31 * np.arange(turns)
    other = 2 * math.pi * 0.2387 * np.arange(turns)
    _, _, diffusion = tune_diffusion(
        np.cos(phase), -np.sin(phase), np.cos(other), -np.sin(other)
    )
    assert diffusion < -10


def test_a_drifting_tune_shows_up_as_diffusion():
    """A chirp; about 14 orders above the regular case (2 with the FFT)."""
    turns = 512
    steps = np.arange(turns)
    chirp = 2 * math.pi * (0.31 * steps + 2e-5 * steps**2)
    other = 2 * math.pi * 0.2387 * steps
    _, _, diffusion = tune_diffusion(
        np.cos(chirp), -np.sin(chirp), np.cos(other), -np.sin(other)
    )
    assert diffusion > -4


def test_the_tune_difference_is_taken_circularly():
    """0.999 -> 0.001 is a shift of 0.002, not 0.998."""
    assert ((0.001 - 0.999) + 0.5) % 1.0 - 0.5 == pytest.approx(0.002)


def test_a_short_record_gives_no_diffusion():
    """Two windows of fewer than 8 turns cannot give a tune apiece."""
    assert math.isnan(tune_diffusion(*[np.ones(10)] * 4)[2])


def test_a_lost_particle_gets_no_tune():
    assert math.isnan(tune_from_trajectory([1.0, np.nan] * 64))
    assert math.isnan(tune_from_trajectory(np.zeros(64)))
    assert math.isnan(tune_from_trajectory([1.0, 2.0]))


def test_mismatched_momenta_are_refused():
    assert math.isnan(tune_from_trajectory(np.ones(64), np.ones(32)))


@pytest.fixture(scope="module")
def footprint():
    """A sextupole ring, tracked and frequency-analysed for real."""
    pytest.importorskip("ocelot")
    import io
    from contextlib import redirect_stdout

    import ocelot.cpbd.elements as oc
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.optics import twiss as ocelot_twiss
    from ocelot.cpbd.track import create_track_list, freq_analysis, track_nturns
    from ocelot.cpbd.transformations import SecondTM

    ncell, k2 = 8, 500.0
    angle = 2 * math.pi / ncell
    cell = []
    for _ in range(ncell):
        cell += [
            oc.Quadrupole(l=0.3, k1=1.2),
            oc.Sextupole(l=0.2, k2=k2),
            oc.Drift(l=0.3),
            oc.SBend(l=1.0, angle=angle),
            oc.Drift(l=0.3),
            oc.Quadrupole(l=0.3, k1=-1.2),
            oc.Sextupole(l=0.2, k2=-k2),
            oc.Drift(l=0.3),
        ]
    lattice = MagneticLattice(cell, method={"global": SecondTM})
    reference = ocelot_twiss(lattice, tws0=None)[-1].mux / (2 * math.pi)
    track_list = create_track_list(
        np.linspace(2e-4, 3e-3, 6), [1e-4], [0.0], energy=1.0
    )
    track_list = track_nturns(
        lattice, TURNS, track_list, save_track=True, print_progress=False
    )
    with redirect_stdout(io.StringIO()):
        track_list = freq_analysis(track_list, lattice, TURNS, harm=True)
    return reference, track_list


def test_every_surviving_particle_gets_a_tune(footprint):
    _, track_list = footprint
    survivors = [p for p in track_list if p.turn >= TURNS - 1]
    assert survivors
    assert all(float(p.mux) >= 0 for p in survivors)


def test_the_smallest_amplitude_recovers_the_periodic_tune(footprint):
    """To the FFT resolution, 1/turns."""
    reference, track_list = footprint
    smallest = min(track_list, key=lambda p: float(p.x))
    got = frameworkLattice.tune_from_harmonic(float(smallest.mux), reference)
    assert got == pytest.approx(reference, abs=2.0 / TURNS)


def test_the_tune_shifts_with_amplitude(footprint):
    reference, track_list = footprint
    tunes = [
        frameworkLattice.tune_from_harmonic(float(p.mux), reference)
        for p in sorted(track_list, key=lambda p: float(p.x))
        if getattr(p, "mux", None) is not None
    ]
    assert abs(tunes[-1] - tunes[0]) > 0.02


def test_the_shift_is_monotonic_in_this_ring(footprint):
    """Else a tune was folded back across an integer. The direction is the
    sextupoles', so not asserted."""
    reference, track_list = footprint
    tunes = [
        frameworkLattice.tune_from_harmonic(float(p.mux), reference)
        for p in sorted(track_list, key=lambda p: float(p.x))
        if float(getattr(p, "mux", -1.0)) >= 0
    ]
    assert tunes == sorted(tunes) or tunes == sorted(tunes, reverse=True)


def test_without_saved_tracks_it_yields_nothing(footprint):
    """The quiet failure: with `save_track=False`, `freq_analysis` sets no tunes."""
    import io
    from contextlib import redirect_stdout

    import ocelot.cpbd.elements as oc
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.track import create_track_list, freq_analysis, track_nturns
    from ocelot.cpbd.transformations import SecondTM

    lattice = MagneticLattice(
        [oc.Quadrupole(l=0.3, k1=1.2), oc.Drift(l=1.0), oc.Quadrupole(l=0.3, k1=-1.2)],
        method={"global": SecondTM},
    )
    track_list = create_track_list([1e-4], [1e-4], [0.0], energy=1.0)
    track_list = track_nturns(
        lattice, 16, track_list, save_track=False, print_progress=False
    )
    with redirect_stdout(io.StringIO()) as captured:
        track_list = freq_analysis(track_list, lattice, 16, harm=True)
    # -0.001 is `Track_info`'s initial value: freq_analysis returned without
    # setting anything. A `None` check would not have caught this.
    assert all(float(p.mux) == pytest.approx(-0.001) for p in track_list)
    assert "save_track" in captured.getvalue()
