"""Bmad's turn-by-turn hand-back."""

import contextlib
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

    Without ``&tao_beam_init`` Tao will not track a beam, and configuring it
    by command saves the beam only at the first element of a repeated name.
    """
    pytest.importorskip("pytao")
    if not os.path.exists(BMAD_SO):
        pytest.skip("Bmad libtao not installed")
    from pytao import Tao

    init = f"{os.path.splitext(path)[0]}.init"
    with open(init, "w") as handle:
        handle.write(INIT.format(lattice=os.path.basename(path), npart=NPART))
    with contextlib.chdir(os.path.dirname(path)):
        return Tao(init_file=init, so_lib=BMAD_SO, noplot=True, **kwargs)


def _write_lattice(directory, name, repeats, geometry, twiss=""):
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write(
            LATTICE.format(geometry=geometry, repeats=repeats, twiss=twiss)
        )
    return path


def _load_beam(tao, beam, saved_at):
    """Point Tao's beam tracking at a position file, saving at ``saved_at``."""
    tao.cmd(bmadLattice._POSITION_FILE_CMD.format(path=beam))
    tao.cmd(f"set beam add_saved_at = {saved_at}")
    tao.cmd("set global track_type = beam", raises=False)


class FakeBmad:
    """The real turn loop; only the framework around it, read for scalars, is faked."""

    # `_ALIVE` is a pydantic private attribute, so the class holds a
    # descriptor rather than the value.
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

    def _reference_p0c(self):
        return PC


@pytest.fixture(scope="module")
def directory():
    return tempfile.mkdtemp()


@pytest.fixture(scope="module")
def looped(directory):
    """`_track_grid_turn_by_turn` itself, on a closed one-turn lattice.

    An empty result fails rather than skips: the loop catches everything and
    returns ``{}``, which is how the hand-back once stayed broken unnoticed.
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

    The open line needs the closed ring's Twiss written in before Tao will
    calculate it. pytao `Tao` objects share one Fortran library, so each is
    finished with before the next is made, or the first silently re-points.
    """
    closed = _write_lattice(directory, "probe.bmad", repeats=1, geometry="closed")

    # The looped arm overwrote its grid file, so the seed is regenerated from
    # the closed lattice (the grid is offset from the closed orbit) and
    # checked byte for byte against the looped arm's.
    seed = os.path.join(directory, "seed.fma.beam")
    again = os.path.join(directory, "again.fma.beam")
    ring = os.path.join(directory, "ring.bmad")

    closed_tao = _tao(closed)
    twiss = closed_tao.ele_twiss("BEGINNING")
    FakeBmad(closed_tao, closed)._write_grid_beam_file(seed)
    FakeBmad(_tao(ring), ring)._write_grid_beam_file(again)
    with open(seed) as a, open(again) as b:
        assert a.read() == b.read()

    written = "".join(
        f"beginning[{key}] = {twiss[key]!r}\n"
        for key in ("beta_a", "alpha_a", "beta_b", "alpha_b")
    )
    path = _write_lattice(
        directory, "long.bmad", repeats=TURNS, geometry="open", twiss=written
    )
    tao = _tao(path)
    _load_beam(tao, seed, "marker::*, END")
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


def test_the_grid_is_tracked_for_every_turn(looped):
    for name, track in looped.items():
        assert track.shape == (NPART, TURNS), name


@pytest.mark.parametrize("name", ["x", "px", "y", "py"])
def test_every_turn_differs_from_the_first(looped, name):
    """The failure this replaces gave exactly 0.0 here, for all turns."""
    track = looped[name]
    moved = np.max(np.abs(track - track[:, [0]]), axis=0)
    assert np.all(moved[1:] > 0), f"{name} is frozen after turn 1"
    assert np.max(moved) > 0.1 * np.max(np.abs(track[:, 0]))


def test_the_motion_is_not_a_drift_in_one_direction(looped):
    """Betatron motion changes sign per particle; a fixed offset per turn does not."""
    x = looped["x"]
    assert np.all(np.any(x > 0, axis=1) & np.any(x < 0, axis=1))


@pytest.mark.parametrize("name", ["x", "px", "y", "py"])
def test_the_loop_agrees_with_the_ring_written_out_long(looped, unrolled, name):
    """Two routes, nothing shared but the lattice and the starting particles."""
    assert looped[name] == pytest.approx(unrolled[name], rel=1e-9, abs=1e-14)


def test_the_two_routes_are_not_trivially_equal(looped, unrolled):
    """Guards the test above: if both arms were frozen it would also pass."""
    reference = unrolled["x"]
    assert np.max(np.abs(reference - reference[:, [0]])) > 0


def test_a_lost_particle_stays_lost(directory):
    """Without the `state` column a particle lost on turn 3 reappears on turn 4."""
    path = _write_lattice(directory, "lost.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    rows = np.zeros((3, 6))
    rows[:, 0] = [1e-4, 2e-4, 3e-4]
    beam = os.path.join(directory, "lost.beam")
    lattice._write_position_file(beam, rows, states=[1, 2, 1])

    with open(beam) as handle:
        text = handle.read()
    assert "state" in text.splitlines()[4], "the #! line must name the column"
    states = [line.split()[-1] for line in text.splitlines()[5:] if line.strip()]
    assert states == ["Alive", "Lost", "Alive"]

    tao = lattice.tao
    _load_beam(tao, beam, "END")
    tao.track_beam("BEGINNING", "END", use_progress_bar=False)
    out = np.asarray(tao.bunch1("END", coordinate="x", which="model", ix_bunch=1))

    assert len(out) == 3, "rows must stay in place or the grid mapping breaks"
    assert out[1] == pytest.approx(rows[1, 0]), "the dead particle was tracked"
    assert out[0] != pytest.approx(rows[0, 0]), "the live ones were not"


def test_the_grid_file_round_trips_through_tao(directory):
    """A lossy format would show up as a slow drift rather than an error."""
    path = _write_lattice(directory, "trip.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    rng = np.random.default_rng(3)
    rows = rng.normal(0, 1e-4, (5, 6))
    beam = os.path.join(directory, "trip.beam")
    lattice._write_position_file(beam, rows)

    tao = lattice.tao
    _load_beam(tao, beam, "BEGINNING, END")
    tao.track_beam("BEGINNING", "END", use_progress_bar=False)
    back = np.column_stack(
        [
            tao.bunch1("BEGINNING", coordinate=name, which="model", ix_bunch=1)
            for name in bmadLattice._PHASE_SPACE
        ]
    )
    assert back == pytest.approx(rows, rel=1e-12, abs=1e-18)


def test_the_hand_back_cannot_go_through_set_beam_beginning(directory):
    """Why the loop writes a file: `set beam beginning = END` reports no error
    but the beam at END is identical every turn. If Tao fixes this, simplify."""
    path = _write_lattice(directory, "handback.bmad", repeats=1, geometry="closed")
    lattice = FakeBmad(_tao(path), path)
    beam = os.path.join(directory, "handback.beam")
    lattice._write_grid_beam_file(beam)

    tao = lattice.tao
    _load_beam(tao, beam, "END")
    seen = []
    for _ in range(3):
        tao.track_beam("BEGINNING", "END", use_progress_bar=False)
        seen.append(
            np.asarray(tao.bunch1("END", coordinate="x", which="model", ix_bunch=1))
        )
        tao.cmd("set beam beginning = END", raises=False)

    assert np.array_equal(seen[0], seen[1]) and np.array_equal(seen[1], seen[2])


def test_tao_has_no_multi_turn_beam_tracking(directory):
    """`beam_init` has no turn count; Tao's only ones are for DA and one particle."""
    path = _write_lattice(directory, "noturns.bmad", repeats=1, geometry="closed")
    tao = _tao(path)
    with pytest.raises(Exception):
        tao.cmd("set beam_init n_turn = 10")
