"""MAD-X running its own turn loop, against simba running it."""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf
from helpers import fodo_machine, read_beam
from simba.Codes.Generators import frameworkGenerator
from simba.Codes.MADX.MADX import madxLattice

pytest.importorskip("cpymad")

TURNS = 4


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """One input beam for every run: ``frameworkGenerator`` is unseeded, and
    the first version of this file compared the generator's spread (8e-4)
    rather than the two paths."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=64,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def _build(tmp_path, tracking, seed_beam, cavity=False, closed=False):
    machine, names, section = fodo_machine(tmp_path, cavity=cavity, closed=closed)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": "madx",
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            "tracking": tracking,
        }
    }
    settings.layout = machine.layout
    # the geometry has to be repeated here: `settings.section` replaces the
    # machine's sections wholesale, and the plain-list form drops it
    settings.section = {"sections": {"FODO": section["sections"]["FODO"]}}
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(
        machine=machine, directory=str(tmp_path), clean=True, verbose=False,
    )
    framework.loadSettings(settings=settings)
    shutil.copy(
        seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5")
    )
    return framework


def _run(tmp_path, tracking, seed_beam, **kw):
    framework = _build(tmp_path, tracking, seed_beam, **kw)
    framework.track()
    return framework.subdirectory


def _lattice(tmp_path, tracking, seed_beam, **kw):
    """The MAD-X lattice object, its ``segments`` filled by ``writeElements``."""
    lattice = _build(tmp_path, tracking, seed_beam, **kw).latticeObjects["FODO"]
    lattice.writeElements()
    return lattice


def _run_watching_run_track(tmp_path, tracking, seed_beam, **kw):
    """Track, and report every ``run_track`` call's turn count and FFILE,
    which no output file shows."""
    framework = _build(tmp_path, tracking, seed_beam, **kw)
    calls = []
    original = madxLattice.run_track

    def spy(self, madx, coords, observe, numbers=None, turns=1, ffile=1):
        calls.append({"turns": turns, "ffile": ffile})
        return original(self, madx, coords, observe, numbers, turns, ffile)

    # on the class, not the instance: `track()` does not necessarily run the
    # lattice object `latticeObjects` is holding at this point
    madxLattice.run_track = spy
    try:
        framework.track()
    finally:
        madxLattice.run_track = original
    return calls


@pytest.mark.parametrize("coord", ["x", "px", "y", "py"])
def test_native_and_looped_agree_on_the_final_beam(tmp_path, coord, seed_beam):
    """Not equal only because the loop re-issues START as 15-digit text."""
    native = read_beam(_run(tmp_path / "n", {"turns": TURNS}, seed_beam), "M3")
    looped = read_beam(
        _run(tmp_path / "l", {"turns": TURNS, "native_turns": False}, seed_beam),
        "M3",
    )
    a = np.array(getattr(native, coord).val)
    b = np.array(getattr(looped, coord).val)
    assert len(a) == len(b) > 0
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12), np.abs(a - b).max()


def test_native_and_looped_agree_turn_by_turn(tmp_path, seed_beam):
    opts = {"turns": TURNS, "write_turns": True}
    n = _run(tmp_path / "n", opts, seed_beam)
    loop = _run(tmp_path / "l", {**opts, "native_turns": False}, seed_beam)
    for turn in range(1, TURNS + 1):
        a = np.array(read_beam(n, "M3", turn).x.val)
        b = np.array(read_beam(loop, "M3", turn).x.val)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12), f"turn {turn}"


def test_the_turns_are_actually_different(tmp_path, seed_beam):
    """Guards the test above."""
    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True}, seed_beam)
    first = np.array(read_beam(subdir, "M3", 1).x.val)
    last = np.array(read_beam(subdir, "M3", TURNS).x.val)
    assert not np.allclose(first, last)


def test_a_cavity_splits_a_line(tmp_path, seed_beam):
    lattice = _lattice(tmp_path, {"turns": 1}, seed_beam, cavity=True)
    assert lattice.segment_at_cavities
    assert len(lattice.segments) == 2


def test_the_same_cavity_does_not_split_a_ring(tmp_path, seed_beam):
    lattice = _lattice(tmp_path, {"turns": 1}, seed_beam, cavity=True, closed=True)
    assert not lattice.segment_at_cavities
    assert len(lattice.segments) == 1


def test_an_injection_study_on_a_ring_still_has_a_ring_cavity(tmp_path, seed_beam):
    """``periodic: false`` must not make a ring's cavities accelerating ones."""
    lattice = _lattice(
        tmp_path, {"turns": 1, "periodic": False}, seed_beam,
        cavity=True, closed=True,
    )
    assert lattice.periodic is False
    assert lattice.closed_geometry is True
    assert not lattice.segment_at_cavities


def test_a_ring_with_a_cavity_can_use_native_turns(tmp_path, seed_beam):
    lattice = _lattice(tmp_path, {"turns": TURNS}, seed_beam, cavity=True, closed=True)
    # whether the RF has to move pass by pass depends on the beam's speed, so
    # the question is only asked once the beam is in, as `run_segments` does
    rbf.openpmd.read_openpmd_beam_file(lattice.global_parameters["beam"], seed_beam)
    assert lattice.use_native_turns


# h = 10 on the 4 m cell at the seed's 5 MeV/c; 60 degrees off crest, so the
# bunch centroid really does gain and lose energy turn to turn
_BETA = 5e6 / np.hypot(5e6, 0.51099895e6)
LIVE_CAVITY = {
    "cavity": {"frequency": 10 * 299792458.0 * _BETA / 4.0, "phase": 60.0},
    "simulation": {"field_amplitude": 2.0e4},
}


@pytest.mark.parametrize("coord", ["t", "cp"])
def test_a_live_ring_cavity_agrees_between_the_two_paths(tmp_path, coord, seed_beam):
    """The loop re-centred T and p0c on the bunch every turn, erasing the
    synchrotron motion; the native path stamped every turn with turn 1's
    reference time. A cavity with no voltage hid both."""
    opts = {"turns": TURNS, "write_turns": True}
    kw = {"cavity": LIVE_CAVITY, "closed": True}
    n = _run(tmp_path / "n", opts, seed_beam, **kw)
    loop = _run(tmp_path / "l", {**opts, "native_turns": False}, seed_beam, **kw)
    for turn in range(1, TURNS + 1):
        a = np.array(getattr(read_beam(n, "M3", turn), coord).val)
        b = np.array(getattr(read_beam(loop, "M3", turn), coord).val)
        assert len(a) == len(b) > 0
        assert np.allclose(a, b, rtol=1e-9, atol=1e-15), (turn, np.abs(a - b).max())


def test_the_live_cavity_really_moves_the_centroid(tmp_path, seed_beam):
    """Guards the test above."""
    subdir = _run(
        tmp_path, {"turns": TURNS, "write_turns": True}, seed_beam,
        cavity=LIVE_CAVITY, closed=True,
    )
    first = np.mean(read_beam(subdir, "M3", 1).cp.val)
    last = np.mean(read_beam(subdir, "M3", TURNS).cp.val)
    assert abs(last - first) > 1e3


def test_a_split_line_cannot(tmp_path, seed_beam):
    """A segment boundary hands back to Python, so the loop has to be there."""
    lattice = _lattice(tmp_path, {"turns": TURNS}, seed_beam, cavity=True, closed=False)
    assert len(lattice.segments) > 1
    assert not lattice.use_native_turns


def test_a_single_turn_line_does_not_use_it(tmp_path, seed_beam):
    assert not _lattice(tmp_path, {"turns": 1}, seed_beam).use_native_turns


def test_a_plain_multi_turn_line_uses_it(tmp_path, seed_beam):
    assert _lattice(tmp_path, {"turns": TURNS}, seed_beam).use_native_turns


def test_the_override_turns_it_off(tmp_path, seed_beam):
    lattice = _lattice(tmp_path, {"turns": TURNS, "native_turns": False}, seed_beam)
    assert not lattice.use_native_turns


def test_single_particle_mode_does_not_use_it(tmp_path, seed_beam):
    lattice = _lattice(tmp_path, {"turns": TURNS, "single_particle": True}, seed_beam)
    assert lattice.single_particle
    assert not lattice.use_native_turns


def test_a_programmed_element_does_not_use_it(tmp_path, seed_beam):
    """A program is a MAD-X statement between turns."""
    tracking = {
        "turns": TURNS,
        "programs": [
            {"element": "QUAD1F", "parameter": "k1",
             "turns": [1, TURNS], "values": [0.0, 1.0]}
        ],
    }
    lattice = _lattice(tmp_path, tracking, seed_beam)
    assert lattice.programs
    assert not lattice.use_native_turns


def test_a_programmed_run_really_runs_the_loop(tmp_path, seed_beam):
    """One ``RUN`` per turn. Whether the program changes the beam is
    ``test_ring_outputs.test_madx_programs_a_sliced_quadrupole_as_xsuite_does``."""
    tracking = {
        "turns": 5,
        "programs": [
            {"element": "QUAD1F", "parameter": "k1",
             "turns": [1, 5], "values": [-1.0, -2.0]}
        ],
    }
    calls = _run_watching_run_track(tmp_path, tracking, seed_beam)
    assert len(calls) == 5
    assert all(c == {"turns": 1, "ffile": 1} for c in calls)


def test_the_default_run_asks_madx_for_only_the_last_turn(tmp_path, seed_beam):
    """So the table does not grow with the turn count."""
    calls = _run_watching_run_track(tmp_path, {"turns": 200}, seed_beam)
    assert calls == [{"turns": 200, "ffile": 200}]


def test_per_turn_output_asks_madx_for_every_turn(tmp_path, seed_beam):
    calls = _run_watching_run_track(
        tmp_path, {"turns": 5, "write_turns": True}, seed_beam
    )
    assert calls == [{"turns": 5, "ffile": 1}]


def test_one_run_not_one_per_turn(tmp_path, seed_beam):
    assert len(_run_watching_run_track(tmp_path, {"turns": 25}, seed_beam)) == 1


def test_the_loop_really_does_issue_one_run_per_turn(tmp_path, seed_beam):
    """The baseline for the test above."""
    calls = _run_watching_run_track(
        tmp_path, {"turns": 25, "native_turns": False}, seed_beam
    )
    assert len(calls) == 25
    assert all(c == {"turns": 1, "ffile": 1} for c in calls)


def test_a_superperiod_is_a_pass_and_not_a_turn(tmp_path, seed_beam):
    """4 turns of a 3-fold ring is 12 passes, a turn every 3rd."""
    calls = _run_watching_run_track(
        tmp_path,
        {"turns": 4, "nsuperperiods": 3, "write_turns": True},
        seed_beam,
    )
    assert calls == [{"turns": 12, "ffile": 3}]


def test_superperiods_agree_between_the_two_paths(tmp_path, seed_beam):
    opts = {"turns": 3, "nsuperperiods": 2}
    native = read_beam(_run(tmp_path / "n", opts, seed_beam), "M3")
    looped = read_beam(
        _run(tmp_path / "l", {**opts, "native_turns": False}, seed_beam), "M3"
    )
    assert np.allclose(
        np.array(native.x.val), np.array(looped.x.val), rtol=1e-9, atol=1e-12
    )
