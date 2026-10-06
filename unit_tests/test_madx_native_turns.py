"""MAD-X running its own turn loop, against simba running it.

``test_madx_turns.py`` showed bare ``RUN, TURNS=N`` and an N-times Python
loop agree to 1e-18 on a hand-built lattice. That measurement was taken as
licence to keep the Python loop, which was a mistake: nobody had measured
what the loop *cost*. It costs ~12 ms a turn at 512 particles, against 0.8 s
for all 20 000 turns of the same line natively, and the gap grows with the
turn count because the native side is nearly flat in it.

So the loop is now the fallback and MAD-X's own turn count is the default
(:func:`~simba.Codes.MADX.MADX.madxLattice.use_native_turns`). This file is
what makes that safe: the same framework run, down both paths, compared.
``native_turns: false`` in the ``tracking`` block selects the old one, which
is what lets a test hold everything else fixed.

Two things had to change for the native path to be reachable at all, and
both are tested here:

* A ring is no longer split at its RF cavities. The splitting is a linac
  measure -- an accelerating cavity moves the reference momentum, which is
  fixed per MAD-X ``BEAM`` statement -- and a storage-ring cavity does not
  accelerate. Splitting a ring bought nothing and cost everything, because
  a segment boundary is a hand-back to Python and so forbids ``TURNS=N``.
* ``FFILE`` carries :meth:`output_turns` down into MAD-X, so the
  ``trackone`` table holds the turns that were asked for rather than all of
  them. Without it the table is ~2M rows per 1000 turns at 512 particles and
  the native path trades a time problem for a memory one.
"""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole, RFCavity
from simba.Codes.Generators import frameworkGenerator
from simba.Codes.MADX.MADX import madxLattice
from simba.Framework_objects import OUTPUT_TURN_SEPARATOR as SEPARATOR

TURNS = 4


def _machine(tmp_path, cavity=False, closed=False):
    """A FODO cell, optionally with an RF cavity, optionally called a ring.

    ``closed`` sets LAURA's section geometry, which is what
    ``segment_at_cavities`` reads -- the point of the cavity/closed pair is
    that the *same* cavity splits the line or does not depending only on
    whether the machine is a ring.
    """
    middle = [
        Quadrupole(
            name="QUAD1F", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": -1},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
        ),
        Quadrupole(
            name="QUAD1D", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": 1.0},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
        ),
    ]
    if cavity:
        middle.append(
            RFCavity(
                name="CAV1", machine_area="FODO",
                physical={"length": 0.0,
                          "middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
            )
        )
    m1 = Marker(
        name="M1", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    last = middle[-1]
    end_z = last.physical.middle.z + (last.physical.length or 0.0)
    m3 = Marker(
        name="M3", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": end_z}},
    )
    names = ["M1"] + [e.name for e in middle] + ["M3"]
    section = (
        {"sections": {"FODO": {"elements": names, "geometry": "closed"}}}
        if closed
        else {"sections": {"FODO": names}}
    )
    machine = LAURA(
        element_list=[m1, *middle, m3],
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section=section,
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, names, section


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """One input beam, generated once and copied into every run.

    ``frameworkGenerator`` is unseeded: two calls with identical arguments
    give different particles (measured, 8 particles: max |dx| 2.4e-4 against
    a 1e-4 sigma). Generating per run would mean the two paths were handed
    different beams, and every comparison in this file would be measuring
    the generator. That is not hypothetical -- it is what the first version
    of this file did, and the 8e-4 "disagreement" it reported between the
    native and looped paths was exactly the generator's own spread.

    ``px`` happens to come out deterministic, which makes the trap worse
    rather than better: a confounded comparison looks half-right.
    """
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
    pytest.importorskip("cpymad")
    machine, names, section = _machine(tmp_path, cavity=cavity, closed=closed)
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
    """The MAD-X lattice object, with its sequences built.

    ``writeElements`` is what fills ``segments``, and most of what this file
    asks about is a question about those.
    """
    lattice = _build(tmp_path, tracking, seed_beam, **kw).latticeObjects["FODO"]
    lattice.writeElements()
    return lattice


def _run_watching_run_track(tmp_path, tracking, seed_beam, **kw):
    """Track, and report every ``run_track`` call's turn count and FFILE.

    What ``RUN`` MAD-X is actually given is the whole substance of the
    native path, and it is not visible in any output file: a run that
    looped 25 times and one that asked for 25 turns write the same beams.
    """
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


def _beam(subdir, name):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(
        beam, os.path.join(subdir, f"{name}.openpmd.hdf5")
    )
    return beam


# --- the two paths agree -------------------------------------------------
#
# The whole case for the switch. Everything else in this file is about when
# the native path is taken; this is about it being the same answer.


@pytest.mark.parametrize("coord", ["x", "px", "y", "py"])
def test_native_and_looped_agree_on_the_final_beam(tmp_path, coord, seed_beam):
    """Same lattice, same input beam, same turn count, both paths.

    Tracking is deterministic and the two are the identical sequence of
    thin-lens maps, so this is not a tolerance question in principle -- the
    only reason not to assert equality outright is that the loop re-issues
    the START coordinates as 15-significant-digit text every turn, and the
    native path does it once.
    """
    native = _beam(_run(tmp_path / "n", {"turns": TURNS}, seed_beam), "M3")
    looped = _beam(
        _run(tmp_path / "l", {"turns": TURNS, "native_turns": False}, seed_beam),
        "M3",
    )
    a = np.array(getattr(native, coord).val)
    b = np.array(getattr(looped, coord).val)
    assert len(a) == len(b) > 0
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12), np.abs(a - b).max()


def test_native_and_looped_agree_turn_by_turn(tmp_path, seed_beam):
    """Not just at the end: every recorded turn matches, so the agreement is
    not two different trajectories happening to meet."""
    opts = {"turns": TURNS, "write_turns": True}
    n = _run(tmp_path / "n", opts, seed_beam)
    loop = _run(tmp_path / "l", {**opts, "native_turns": False}, seed_beam)
    for turn in range(1, TURNS + 1):
        a = np.array(_beam(n, f"M3{SEPARATOR}{turn}").x.val)
        b = np.array(_beam(loop, f"M3{SEPARATOR}{turn}").x.val)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12), f"turn {turn}"


def test_the_turns_are_actually_different(tmp_path, seed_beam):
    """Guards the test above: if every turn wrote the same beam, comparing
    them turn by turn would pass without meaning anything."""
    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True}, seed_beam)
    first = np.array(_beam(subdir, f"M3{SEPARATOR}1").x.val)
    last = np.array(_beam(subdir, f"M3{SEPARATOR}{TURNS}").x.val)
    assert not np.allclose(first, last)


# --- a ring is not split at its cavities ---------------------------------


def test_a_cavity_splits_a_line(tmp_path, seed_beam):
    """The linac case, unchanged: the cavity moves the reference momentum,
    so the line is tracked either side of it."""
    lattice = _lattice(tmp_path, {"turns": 1}, seed_beam, cavity=True)
    assert lattice.segment_at_cavities
    assert len(lattice.segments) == 2


def test_the_same_cavity_does_not_split_a_ring(tmp_path, seed_beam):
    """The change. Nothing about the cavity differs -- only that the machine
    is a ring, where a cavity holds the energy rather than raising it."""
    lattice = _lattice(tmp_path, {"turns": 1}, seed_beam, cavity=True, closed=True)
    assert not lattice.segment_at_cavities
    assert len(lattice.segments) == 1


def test_an_injection_study_on_a_ring_still_has_a_ring_cavity(tmp_path, seed_beam):
    """``periodic: false`` asks for the open optics solution, which is a
    legitimate thing to want on a real ring. It must not thereby turn the
    ring's cavities into accelerating ones."""
    lattice = _lattice(
        tmp_path, {"turns": 1, "periodic": False}, seed_beam,
        cavity=True, closed=True,
    )
    assert lattice.periodic is False
    assert lattice.closed_geometry is True
    assert not lattice.segment_at_cavities


def test_a_ring_with_a_cavity_can_use_native_turns(tmp_path, seed_beam):
    """The two changes meeting: not splitting leaves one segment, and one
    segment is what ``use_native_turns`` needs."""
    lattice = _lattice(tmp_path, {"turns": TURNS}, seed_beam, cavity=True, closed=True)
    assert lattice.use_native_turns


def test_a_split_line_cannot(tmp_path, seed_beam):
    """And the converse, which is the condition that matters: a segment
    boundary is a hand-back to Python, so the turn loop has to be there."""
    lattice = _lattice(tmp_path, {"turns": TURNS}, seed_beam, cavity=True, closed=False)
    assert len(lattice.segments) > 1
    assert not lattice.use_native_turns


# --- when the native path is and is not used -----------------------------


def test_a_single_turn_line_does_not_use_it(tmp_path, seed_beam):
    """There is no turn loop to hand over."""
    assert not _lattice(tmp_path, {"turns": 1}, seed_beam).use_native_turns


def test_a_plain_multi_turn_line_uses_it(tmp_path, seed_beam):
    assert _lattice(tmp_path, {"turns": TURNS}, seed_beam).use_native_turns


def test_the_override_turns_it_off(tmp_path, seed_beam):
    """Which is what the agreement tests above rely on."""
    lattice = _lattice(tmp_path, {"turns": TURNS, "native_turns": False}, seed_beam)
    assert not lattice.use_native_turns


def test_single_particle_mode_does_not_use_it(tmp_path, seed_beam):
    """Not a tracking run at all: it builds each segment's map by finite
    differences and applies it to the distribution."""
    lattice = _lattice(tmp_path, {"turns": TURNS, "single_particle": True}, seed_beam)
    assert lattice.single_particle
    assert not lattice.use_native_turns


def test_a_programmed_element_does_not_use_it(tmp_path, seed_beam):
    """A device program is a MAD-X statement between one turn and the next,
    so the turns have to come back to Python. R18's kicker, in other words,
    still works the way R18 left it."""
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
    """And the fallback is not merely selected but taken: one ``RUN`` per
    turn, which is the only way ``apply_programs`` gets to run between them.

    Whether the program then changes the optics is R18/R19's question and
    is tested there. It is worth saying why this does not check it: MAD-X
    warns ``programs 'QUAD1F', which MAD-X does not have``, because
    ``program_attributes`` has no ``quadrupole`` entry and because
    ``apply_programs`` is called before the sequence exists on the first
    turn. Both are pre-existing and neither is touched here -- asserting
    around them would have buried them.
    """
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


# --- FFILE carries `output_turns` into MAD-X -----------------------------


def test_the_default_run_asks_madx_for_only_the_last_turn(tmp_path, seed_beam):
    """``write_turns`` off means one output beam, so the table should hold
    one turn -- this is what stops it growing with the turn count."""
    calls = _run_watching_run_track(tmp_path, {"turns": 200}, seed_beam)
    assert calls == [{"turns": 200, "ffile": 200}]


def test_per_turn_output_asks_madx_for_every_turn(tmp_path, seed_beam):
    calls = _run_watching_run_track(
        tmp_path, {"turns": 5, "write_turns": True}, seed_beam
    )
    assert calls == [{"turns": 5, "ffile": 1}]


def test_one_run_not_one_per_turn(tmp_path, seed_beam):
    """The point of the whole exercise: ``RUN`` is issued once, where the
    loop issued it ``turns`` times."""
    assert len(_run_watching_run_track(tmp_path, {"turns": 25}, seed_beam)) == 1


def test_the_loop_really_does_issue_one_run_per_turn(tmp_path, seed_beam):
    """The baseline the test above is measured against -- otherwise 'one
    call' is only a fact about this file, not a change."""
    calls = _run_watching_run_track(
        tmp_path, {"turns": 25, "native_turns": False}, seed_beam
    )
    assert len(calls) == 25
    assert all(c == {"turns": 1, "ffile": 1} for c in calls)


# --- superperiods -------------------------------------------------------


def test_a_superperiod_is_a_pass_and_not_a_turn(tmp_path, seed_beam):
    """``nsuperperiods`` multiplies the passes MAD-X is asked for, and the
    recorded turns stay turns: 4 turns of a 3-fold ring is 12 passes, and
    every 3rd of them is a turn boundary."""
    calls = _run_watching_run_track(
        tmp_path,
        {"turns": 4, "nsuperperiods": 3, "write_turns": True},
        seed_beam,
    )
    assert calls == [{"turns": 12, "ffile": 3}]


def test_superperiods_agree_between_the_two_paths(tmp_path, seed_beam):
    """The pass/turn arithmetic is the easiest thing here to get wrong by a
    factor of ``nsuperperiods``, and it would still run."""
    opts = {"turns": 3, "nsuperperiods": 2}
    native = _beam(_run(tmp_path / "n", opts, seed_beam), "M3")
    looped = _beam(
        _run(tmp_path / "l", {**opts, "native_turns": False}, seed_beam), "M3"
    )
    assert np.allclose(
        np.array(native.x.val), np.array(looped.x.val), rtol=1e-9, atol=1e-12
    )
