"""Bmad's turn-by-turn hand-back (R24).

The frequency map tracks a grid of particles one turn at a time and feeds
each turn's output back in as the next turn's input. Whether the hand-back
*works* is the whole thing -- if it silently fails, every turn repeats turn
one, the tune series is a constant, and the scan reports nothing wrong. It
did silently fail: ``set beam beginning = END`` sets Tao's
``beam_at_start`` and reports no error, but the next lattice calculation
re-reads ``beam_init``, and ``beam_init%position_file`` is set by the
frequency map -- so Tao reproduced the grid every turn.

The test that was here asserted ``supports_frequency_map is True``. That is
an assertion about the source, and the source said yes while the code did
nothing; these drive Tao instead and ask whether the particles moved.

Two independent checks, because one of them on its own is weak:

* **the turns advance** -- turn *n* differs from turn 1 by much more than
  the beam's own size. Catches the frozen loop.
* **the turns are the right ones** -- the same ring, written out as one
  long open line and tracked in a single call, gives the same numbers.
  Catches a loop that moves but moves wrongly. Nothing is shared between
  the two routes but the physics, so agreement is worth something; they
  agree bit for bit.

``beam_init`` generates a fresh random distribution per Tao instance and
matches it to that lattice's Twiss, so every arm here is seeded from the
same explicit particle file. See the ``frameworkGenerator`` note in
``test_madx_native_turns.py`` -- this is the same confounder in another
code, and it would make the two routes look as though they disagreed.
"""

import math
import os
import tempfile
import warnings

import numpy as np
import pytest

from simba.Codes.Bmad.Bmad import bmadLattice

BMAD_SO = "/home/xkc85723/Documents/bmad-ecosystem/production/lib/libtao.so"

PC = 1e9
QUAD_L = 0.5
QUAD_K1 = 0.30
BEND_L = 1.0
DRIFT_L = 1.0
NCELL = 16
ANGLE = 2 * math.pi / (2 * NCELL)
TURNS = 12
NPART = 6

LATTICE = (
    "parameter[particle] = electron\n"
    "parameter[geometry] = {geometry}\n"
    f"parameter[p0c] = {PC}\n"
    "{twiss}"
    f"qf: quadrupole, l = {QUAD_L}, k1 = {QUAD_K1}\n"
    f"qd: quadrupole, l = {QUAD_L}, k1 = {-QUAD_K1}\n"
    f"b: sbend, l = {BEND_L}, angle = {ANGLE!r}\n"
    f"d: drift, l = {DRIFT_L}\n"
    "turnend: marker\n"
    "cell: line = (qf, d, b, d, qd, d)\n"
    f"oneturn: line = ({NCELL}*cell, turnend)\n"
    "lat: line = ({repeats}*oneturn)\n"
    "use, lat\n"
)


INIT = """
&tao_start
  n_universes = 1
/
&tao_design_lattice
  design_lattice(1)%file = "{lattice}"
/
&tao_params
  global%track_type = 'single'
  global%plot_on = F
  global%random_seed = 12345
/
&tao_beam_init
  beam_init%n_particle = {npart}
  beam_init%bunch_charge = 1e-12
/
"""


def _tao(path, **kwargs):
    """Tao on a lattice, through an init file as simba drives it.

    The init file is not decoration. Without a ``&tao_beam_init`` namelist
    Tao refuses to track a beam at all ("BEAM TRACKING CANNOT BE DONE
    UNLESS A TAO_BEAM_INIT NAMELIST HAS BEEN DEFINED"), and half-configuring
    it by command gets a lattice that tracks but saves the beam at only the
    first element of a repeated name -- which looks exactly like the
    unrolled arm being wrong.
    """
    pytest.importorskip("pytao")
    if not os.path.exists(BMAD_SO):
        pytest.skip("Bmad libtao not installed")
    from pytao import Tao

    init = f"{os.path.splitext(path)[0]}.init"
    with open(init, "w") as handle:
        handle.write(INIT.format(lattice=os.path.basename(path), npart=NPART))
    previous = os.getcwd()
    try:
        os.chdir(os.path.dirname(path))
        return Tao(init_file=init, so_lib=BMAD_SO, noplot=True, **kwargs)
    finally:
        os.chdir(previous)


def _write_lattice(directory, name, repeats, geometry, twiss=""):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(
            LATTICE.format(geometry=geometry, repeats=repeats, twiss=twiss)
        )
    return path


class FakeBmad:
    """The real turn loop, with only the lattice and beam faked.

    Everything the hand-back touches is `bmadLattice`'s own code; what is
    replaced is the surrounding framework -- the element list, the beam
    object, the settings dictionary -- which the loop only reads scalars
    from.
    """

    # `_ALIVE` is a pydantic private attribute, so the class holds a
    # descriptor rather than the value; `_GRID_CHARGE` and `_PHASE_SPACE`
    # are `ClassVar` and come through as themselves.
    _ALIVE = bmadLattice.__private_attributes__["_ALIVE"].default
    _GRID_CHARGE = bmadLattice._GRID_CHARGE
    _PHASE_SPACE = bmadLattice._PHASE_SPACE
    _POSITION_FILE_CMD = bmadLattice._POSITION_FILE_CMD
    _write_position_file = bmadLattice._write_position_file
    _write_grid_beam_file = bmadLattice._write_grid_beam_file
    _track_grid_turn_by_turn = bmadLattice._track_grid_turn_by_turn
    read_closed_orbit = bmadLattice.read_closed_orbit
    da_grid = bmadLattice.da_grid
    apply_programs = bmadLattice.apply_programs

    objectname = "ring"
    programs = ()

    def __init__(self, tao, lattice_file, turns=TURNS):
        self.tao = tao
        self.lattice_file = lattice_file
        self.turns = turns
        self.global_parameters = {"beam": type("B", (), {"species": "electron"})()}

    @property
    def da_settings(self):
        return {"nx": NPART, "ny": 1, "x_max": 1e-3, "y_max": 1e-4}

    def _reference_energy(self):
        return PC


@pytest.fixture(scope="module")
def directory():
    return tempfile.mkdtemp()


@pytest.fixture(scope="module")
def looped(directory):
    """`_track_grid_turn_by_turn` itself, on a closed one-turn lattice.

    An empty result is a failure, not a skip. `_track_grid_turn_by_turn`
    catches everything, warns, and returns ``{}``, so a broken Tao command
    leaves the frequency map reporting "no particle gave a tune" and
    nothing else -- which is how the hand-back stayed broken. Skipping here
    would reproduce that. Tao being absent is already a skip, in `_tao`.
    """
    path = _write_lattice(directory, "ring.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracks = lattice._track_grid_turn_by_turn()
    assert tracks, (
        "the turn loop returned nothing; Tao said: "
        + "; ".join(str(w.message) for w in caught)
    )
    return tracks


@pytest.fixture(scope="module")
def unrolled(directory, looped):
    """The same turns from one call, by writing the ring out `TURNS` times.

    An open geometry needs its Twiss given to it, so the closed ring's
    periodic solution is read off first and written in. Tracking does not
    use the Twiss, but Tao will not calculate the lattice without it.

    Order matters here, and not for a reason the code shows: **pytao `Tao`
    objects share one Fortran library**, so constructing a second one
    re-points the first at the new lattice. Building the long line's `Tao`
    before reading the closed ring would leave the long arm quietly
    tracking the one-turn ring -- which showed up as "Cannot locate
    element: TURNEND##2", the markers of turns 2 onwards having ceased to
    exist. Every `Tao` here is therefore finished with before the next is
    made.
    """
    closed = _write_lattice(directory, "probe.bmad", repeats=1, geometry="closed")

    # The looped arm overwrote its own grid file on the way round, so the
    # seed is regenerated -- from the *closed* lattice, because
    # `_write_grid_beam_file` offsets the grid from the closed orbit and
    # the open line has no closed orbit to read. Checked byte for byte
    # against the looped arm's, rather than argued to be the same.
    seed = os.path.join(directory, "seed.fma.beam")
    again = os.path.join(directory, "again.fma.beam")
    ring = os.path.join(directory, "ring.bmad")

    closed_tao = _tao(closed)
    twiss = closed_tao.ele_twiss("BEGINNING")
    FakeBmad(closed_tao, closed)._write_grid_beam_file(seed)
    FakeBmad(_tao(ring), ring)._write_grid_beam_file(again)
    assert open(seed).read() == open(again).read()

    written = "".join(
        f"beginning[{key}] = {twiss[key]!r}\n"
        for key in ("beta_a", "alpha_a", "beta_b", "alpha_b")
    )
    path = _write_lattice(
        directory, "long.bmad", repeats=TURNS, geometry="open", twiss=written
    )
    tao = _tao(path)
    tao.cmd(bmadLattice._POSITION_FILE_CMD.format(path=seed))
    tao.cmd("set beam add_saved_at = marker::*, END")
    tao.cmd("set global track_type = beam", raises=False)
    tao.track_beam("BEGINNING", "END", use_progress_bar=False)
    return {
        name: np.asarray(
            [
                list(
                    tao.bunch1(
                        f"TURNEND##{turn}",
                        coordinate=name,
                        which="model",
                        ix_bunch=1,
                    )
                )
                for turn in range(1, TURNS + 1)
            ]
        ).T
        for name in ("x", "px", "y", "py")
    }


# --- the turns have to actually happen ----------------------------------


def test_the_grid_is_tracked_for_every_turn(looped):
    """Shape first: one column per turn, one row per grid point."""
    for name, track in looped.items():
        assert track.shape == (NPART, TURNS), name


@pytest.mark.parametrize("name", ["x", "px", "y", "py"])
def test_every_turn_differs_from_the_first(looped, name):
    """The failure this replaces gave exactly 0.0 here, for all turns.

    The threshold is deliberately crude -- the point is to separate
    "moved" from "did not move at all", not to pin a number. A frozen
    loop gives 0; a working one moves by of order the beam size.
    """
    track = looped[name]
    moved = np.max(np.abs(track - track[:, [0]]), axis=0)
    assert np.all(moved[1:] > 0), f"{name} is frozen after turn 1"
    assert np.max(moved) > 0.1 * np.max(np.abs(track[:, 0]))


def test_the_motion_is_not_a_drift_in_one_direction(looped):
    """Betatron motion changes sign; a bookkeeping error usually does not.

    A loop that added a fixed offset every turn would pass the test above.
    Asked per particle, because the grid starts each one at a different
    amplitude and comparing across particles says nothing.
    """
    x = looped["x"]
    assert np.all(np.any(x > 0, axis=1) & np.any(x < 0, axis=1))


# --- and they have to be the right turns --------------------------------


@pytest.mark.parametrize("name", ["x", "px", "y", "py"])
def test_the_loop_agrees_with_the_ring_written_out_long(looped, unrolled, name):
    """Two routes, nothing shared but the lattice and the starting particles."""
    assert looped[name] == pytest.approx(unrolled[name], rel=1e-9, abs=1e-14)


def test_the_two_routes_are_not_trivially_equal(looped, unrolled):
    """Guards the test above: if both arms were frozen it would also pass."""
    reference = unrolled["x"]
    assert np.max(np.abs(reference - reference[:, [0]])) > 0


# --- the position file is what carries the beam -------------------------


def test_a_lost_particle_stays_lost(directory):
    """`_write_position_file` writes a per-particle `state` column.

    Without it every particle is written back alive, and one that left the
    aperture on turn 3 reappears on turn 4 at the coordinates it died with
    -- which is both wrong and silent, since the grid index still lines up.
    """
    path = _write_lattice(directory, "lost.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    rows = np.zeros((3, 6))
    rows[:, 0] = [1e-4, 2e-4, 3e-4]
    beam = os.path.join(directory, "lost.beam")
    lattice._write_position_file(beam, rows, states=[1, 2, 1])

    text = open(beam).read()
    assert "state" in text.splitlines()[4], "the #! line must name the column"
    states = [line.split()[-1] for line in text.splitlines()[5:] if line.strip()]
    assert states == ["Alive", "Lost", "Alive"]

    tao = lattice.tao
    tao.cmd(bmadLattice._POSITION_FILE_CMD.format(path=beam))
    tao.cmd("set beam add_saved_at = END")
    tao.cmd("set global track_type = beam", raises=False)
    tao.track_beam("BEGINNING", "END", use_progress_bar=False)
    out = np.asarray(tao.bunch1("END", coordinate="x", which="model", ix_bunch=1))

    assert len(out) == 3, "rows must stay in place or the grid mapping breaks"
    assert out[1] == pytest.approx(rows[1, 0]), "the dead particle was tracked"
    assert out[0] != pytest.approx(rows[0, 0]), "the live ones were not"


def test_the_grid_file_round_trips_through_tao(directory):
    """What is written is what Tao reads back, to full precision.

    The hand-back is only as good as this: the loop writes six coordinates
    and reads them back next turn, so a lossy format would show up as a
    slow drift rather than as an error.
    """
    path = _write_lattice(directory, "trip.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    rng = np.random.default_rng(3)
    rows = rng.normal(0, 1e-4, (5, 6))
    beam = os.path.join(directory, "trip.beam")
    lattice._write_position_file(beam, rows)

    tao = lattice.tao
    tao.cmd(bmadLattice._POSITION_FILE_CMD.format(path=beam))
    tao.cmd("set beam add_saved_at = BEGINNING, END")
    tao.cmd("set global track_type = beam", raises=False)
    tao.track_beam("BEGINNING", "END", use_progress_bar=False)
    back = np.column_stack(
        [
            tao.bunch1("BEGINNING", coordinate=name, which="model", ix_bunch=1)
            for name in bmadLattice._PHASE_SPACE
        ]
    )
    assert back == pytest.approx(rows, rel=1e-12, abs=1e-18)


# --- what Tao does not offer --------------------------------------------


def test_the_hand_back_cannot_go_through_set_beam_beginning(directory):
    """Why the loop writes a file rather than asking Tao to hand back.

    `set beam beginning = END` is the documented way and it reports no
    error, so this records the measurement rather than the manual: with a
    position file set, the beam at END is identical on every turn.

    If a later Tao fixes this, the test fails and the loop can be
    simplified -- which is the point of pinning it.
    """
    path = _write_lattice(directory, "handback.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    beam = os.path.join(directory, "handback.beam")
    lattice._write_grid_beam_file(beam)

    tao = lattice.tao
    tao.cmd(bmadLattice._POSITION_FILE_CMD.format(path=beam))
    tao.cmd("set beam add_saved_at = END")
    tao.cmd("set global track_type = beam", raises=False)
    seen = []
    for _ in range(3):
        tao.track_beam("BEGINNING", "END", use_progress_bar=False)
        seen.append(
            np.asarray(tao.bunch1("END", coordinate="x", which="model", ix_bunch=1))
        )
        tao.cmd("set beam beginning = END", raises=False)

    assert np.array_equal(seen[0], seen[1]) and np.array_equal(seen[1], seen[2])


def test_tao_has_no_multi_turn_beam_tracking(directory):
    """`track_beam` is one pass, whatever is asked of it.

    Tao's only turn counts are `da_param%n_turn` (the aperture scan, which
    `_tao_dynamic_aperture_namelist` already uses) and a `multi_turn_orbit`
    plot curve, which tracks `lat%particle_start` -- one particle, not a
    beam. `beam_init` has no turn count at all, so there is nothing to set.
    """
    path = _write_lattice(directory, "noturns.bmad", repeats=1, geometry="closed")
    tao = _tao(path)
    with pytest.raises(Exception):
        tao.cmd("set beam_init n_turn = 10")
