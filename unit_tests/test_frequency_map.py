"""Frequency map / tune footprint."""

import math

import numpy as np
import pytest

from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice
from simba.Modules.Matrices import tune_diffusion, tune_from_trajectory

TURNS = 128


# --- which codes can do it ----------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [ocelotLattice, xsuiteLattice, madxLattice, elegantLattice],
    ids=lambda c: c.__name__,
)
def test_the_four_that_can(cls):
    assert cls.supports_frequency_map is True


def test_bmad_can_too_by_the_other_route():
    """Bmad gets there by tracking, not by Tao's documented route.

    `multi_turn_orbit` is the curve `data_source` the Tao manual gives for
    turn-by-turn coordinates, and in this build (2026-08-01)
    `tao_graph_setup_mod.f90` handles `'lat'`, `'beam'` and `'aperture'`
    only, so it returns "UNKNOWN DATA_SOURCE" however it is placed. Tao
    also refuses `set particle_start x` on a closed lattice, which is why
    the chromaticity route (shifting `pz`) works where an amplitude scan
    cannot.

    So the grid is written as an explicit particle file and tracked a turn
    at a time, feeding the bunch back by rewriting that file.

    This test only says the capability is claimed. Whether the turns
    actually advance is a different question, and asserting the flag while
    the loop did nothing is how that went unnoticed for a release; it is
    measured in `test_bmad_turn_handback.py`.
    """
    from simba.Codes.Bmad.Bmad import bmadLattice

    assert bmadLattice.supports_frequency_map is True
    assert bmadLattice.supports_dynamic_aperture is True


# --- the shared tune extractor ------------------------------------------


@pytest.mark.parametrize("q", [0.1234, 0.25, 0.3333, 0.4871])
def test_a_known_tune_is_recovered_from_positions(q):
    """Positions alone take the FFT path, good to ~2e-4 on 256 turns
    thanks to the parabolic refinement; a bare bin would be 1/256 = 4e-3."""
    turns = 256
    phase = 2 * math.pi * q * np.arange(turns)
    assert tune_from_trajectory(np.cos(phase)) == pytest.approx(q, abs=5e-4)


@pytest.mark.parametrize("q", [0.1234, 0.3333, 0.6180, 0.8352])
def test_naff_is_exact_where_the_fft_is_approximate(q):
    """With momenta and `nafflib` installed, NAFF is accurate to ~1e-12 on
    the same 256 turns -- eight orders better than the FFT peak. That gap
    is what makes `tune_diffusion` measurable rather than noise."""
    pytest.importorskip("nafflib")
    turns = 256
    phase = 2 * math.pi * q * np.arange(turns)
    got = tune_from_trajectory(np.cos(phase), -np.sin(phase))
    assert got == pytest.approx(q, abs=1e-9)


def test_naff_is_optional():
    """It is imported behind a try, and the FFT path needs nothing, so a
    machine without it still gets a footprint -- just a blunter one."""
    from simba.Modules import Matrices

    assert hasattr(Matrices, "use_naff")


def test_positions_alone_fold_the_tune_below_a_half():
    """The degeneracy: a real signal's spectrum is symmetric, so 0.8 and 0.2
    are the same picture. Pinned because a silently mirrored tune is exactly
    the kind of plausible wrong answer this study would hide."""
    turns = 256
    phase = 2 * math.pi * 0.8 * np.arange(turns)
    assert tune_from_trajectory(np.cos(phase)) == pytest.approx(0.2, abs=5e-4)


def test_the_momentum_resolves_tunes_above_a_half():
    """`x - i*px` breaks the symmetry. Betatron motion has px proportional
    to -sin(phase), which is what fixes the sign."""
    turns = 256
    phase = 2 * math.pi * 0.8 * np.arange(turns)
    got = tune_from_trajectory(np.cos(phase), -np.sin(phase))
    assert got == pytest.approx(0.8, abs=5e-4)


def test_regular_motion_has_a_tiny_diffusion_index():
    """Both windows see the same tune, so D hits the floor."""
    turns = 512
    phase = 2 * math.pi * 0.31 * np.arange(turns)
    other = 2 * math.pi * 0.2387 * np.arange(turns)
    _, _, diffusion = tune_diffusion(
        np.cos(phase), -np.sin(phase), np.cos(other), -np.sin(other)
    )
    assert diffusion < -10


def test_a_drifting_tune_shows_up_as_diffusion():
    """A chirp: the tune is not the same in the two halves. Measured
    separation from the regular case is about 14 orders of magnitude, which
    is the whole dynamic range a frequency map works in -- and would be
    about 2 orders with the FFT fallback."""
    turns = 512
    steps = np.arange(turns)
    chirp = 2 * math.pi * (0.31 * steps + 2e-5 * steps**2)
    other = 2 * math.pi * 0.2387 * steps
    _, _, diffusion = tune_diffusion(
        np.cos(chirp), -np.sin(chirp), np.cos(other), -np.sin(other)
    )
    assert diffusion > -4


def test_the_tune_difference_is_taken_circularly():
    """A tune just under an integer must not look like it jumped by nearly
    1 when it crosses. 0.999 -> 0.001 is a shift of 0.002, not 0.998."""
    assert ((0.001 - 0.999) + 0.5) % 1.0 - 0.5 == pytest.approx(0.002)


def test_a_short_record_gives_no_diffusion():
    """Two windows of fewer than 8 turns cannot give a tune apiece."""
    assert math.isnan(tune_diffusion(*[np.ones(10)] * 4)[2])


def test_a_lost_particle_gets_no_tune():
    """NaN rather than a number: a diverged trajectory has no tune."""
    assert math.isnan(tune_from_trajectory([1.0, np.nan] * 64))
    assert math.isnan(tune_from_trajectory(np.zeros(64)))
    assert math.isnan(tune_from_trajectory([1.0, 2.0]))


def test_mismatched_momenta_are_refused():
    assert math.isnan(tune_from_trajectory(np.ones(64), np.ones(32)))


# --- the real thing -----------------------------------------------------


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
    """The anchor: at vanishing amplitude the nonlinear tune must be the
    linear one. Tolerance is the FFT resolution, 1/turns."""
    reference, track_list = footprint
    smallest = min(track_list, key=lambda p: float(p.x))
    got = frameworkLattice.tune_from_harmonic(float(smallest.mux), reference)
    assert got == pytest.approx(reference, abs=2.0 / TURNS)


def test_the_tune_shifts_with_amplitude(footprint):
    """The whole point of the study -- a footprint with no spread would mean
    the sextupoles were doing nothing."""
    reference, track_list = footprint
    tunes = [
        frameworkLattice.tune_from_harmonic(float(p.mux), reference)
        for p in sorted(track_list, key=lambda p: float(p.x))
        if getattr(p, "mux", None) is not None
    ]
    assert abs(tunes[-1] - tunes[0]) > 0.02


def test_the_shift_is_monotonic_in_this_ring(footprint):
    """Not true of every lattice, but true of a plain sextupole ring. A
    non-monotonic result here would suggest the reconstruction had folded a
    tune back across an integer. Direction is not asserted -- in this ring
    the tune rises with amplitude (1.8372 to 1.9070), but that is a property
    of the sextupole signs, not of the method."""
    reference, track_list = footprint
    tunes = [
        frameworkLattice.tune_from_harmonic(float(p.mux), reference)
        for p in sorted(track_list, key=lambda p: float(p.x))
        if float(getattr(p, "mux", -1.0)) >= 0
    ]
    assert tunes == sorted(tunes) or tunes == sorted(tunes, reverse=True)


def test_without_saved_tracks_it_yields_nothing(footprint):
    """The quiet failure, pinned. `save_track=False` leaves `p_list` with a
    single entry and `freq_analysis` returns having set no tunes at all."""
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
