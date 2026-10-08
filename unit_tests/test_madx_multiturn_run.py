"""MAD-X multi-turn tracking, end to end through the framework."""

import os

import pytest

import simba.Framework as fw
from simba.Codes.Generators import frameworkGenerator
from laura.models.element import Marker, Quadrupole
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from simba.Framework_objects import OUTPUT_TURN_SEPARATOR as SEPARATOR

TURNS = 3


def _fodo_machine(tmp_path):
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
    m1 = Marker(
        name="M1", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    end_z = middle[-1].physical.middle.z + middle[-1].physical.length
    m3 = Marker(
        name="M3", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": end_z}},
    )
    names = ["M1"] + [e.name for e in middle] + ["M3"]
    machine = LAURA(
        element_list=[m1, *middle, m3],
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section={"sections": {"FODO": names}},
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, names


def _run(tmp_path, tracking):
    """Track the FODO line through MAD-X with the given ``tracking`` block."""
    pytest.importorskip("cpymad")
    machine, names = _fodo_machine(tmp_path)

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
    settings.section = {"sections": {"FODO": names}}
    settings.element_list = f"{tmp_path}/lattice"

    framework = fw.Framework(
        machine=machine, directory=str(tmp_path), clean=True, verbose=False,
    )
    framework.loadSettings(settings=settings)

    generator = frameworkGenerator(
        global_parameters={"master_subdir": framework.subdirectory},
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12,
    )
    generator.write()
    framework.track()
    return framework.subdirectory


def test_a_multi_turn_madx_run_completes(tmp_path):
    """The loop runs at all -- the sequence re-use across turns does not
    throw, which it would if an already-thin sequence were re-sliced."""
    subdir = _run(tmp_path, {"turns": TURNS})
    assert os.path.isfile(os.path.join(subdir, "M3.openpmd.hdf5"))


def _beam_files(subdir):
    return sorted(f for f in os.listdir(subdir) if f.endswith(".openpmd.hdf5"))


def _turns(subdir, name):
    """The turns held in ``name``'s file; ``[]`` for a single beam."""
    import simba.Modules.Beams as rbf

    return rbf.openpmd.openpmd_turns(os.path.join(subdir, f"{name}.openpmd.hdf5"))


def test_the_default_multi_turn_run_writes_one_beam_per_screen(tmp_path):
    """``write_turns`` is off by default (R20), so a multi-turn run writes
    what a single-turn run writes: the last turn, as a single beam."""
    subdir = _run(tmp_path, {"turns": TURNS})
    written = _beam_files(subdir)
    assert all(SEPARATOR not in f for f in written), written
    assert _turns(subdir, "M3") == []


def test_asking_for_per_turn_output_bundles_the_turns(tmp_path):
    """Every turn goes in the screen's one file, not a file per turn."""
    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True})
    written = _beam_files(subdir)
    assert all(SEPARATOR not in f for f in written), written
    assert _turns(subdir, "M3") == list(range(1, TURNS + 1))


# --- R21: the end of the line, per turn ----------------------------------
#
# It used to be written exactly once however many turns were tracked. For a
# ring that is the one place a per-turn record is most wanted -- the end of
# the line *is* the turn boundary. The start is never written: its file is
# the incoming beam, and in a ring its turns are the end's.


def test_the_start_of_the_line_is_not_written_over(tmp_path):
    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True})
    assert _turns(subdir, "M1") == []


def test_the_end_of_line_file_reads_as_its_last_turn(tmp_path):
    """What the next section reads by name, so the chain between two lines
    holds with `write_turns` on."""
    import simba.Modules.Beams as rbf

    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True})
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(subdir, "M3.openpmd.hdf5"))
    assert beam.turn == TURNS


def test_the_default_multi_turn_run_still_writes_the_end_once(tmp_path):
    written = _beam_files(_run(tmp_path, {"turns": TURNS}))
    assert written.count("M3.openpmd.hdf5") == 1


def test_each_end_of_line_turn_knows_which_turn_it_is(tmp_path):
    """R21. The turn used to live only in the filename."""
    import simba.Modules.Beams as rbf

    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True})
    for turn in range(1, TURNS + 1):
        beam = rbf.beam()
        rbf.openpmd.read_openpmd_beam_file(
            beam, os.path.join(subdir, "M3.openpmd.hdf5"), turn=turn
        )
        assert beam.turn == turn


def test_the_unsuffixed_file_says_it_is_the_last_turn(tmp_path):
    """The case with no filename to read it off: one file, and on its face
    indistinguishable from a single-turn run."""
    import simba.Modules.Beams as rbf

    subdir = _run(tmp_path, {"turns": TURNS})
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(subdir, "M3.openpmd.hdf5"))
    assert beam.turn == TURNS


def test_a_single_turn_run_says_turn_one(tmp_path):
    import simba.Modules.Beams as rbf

    subdir = _run(tmp_path, {"turns": 1})
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(subdir, "M3.openpmd.hdf5"))
    assert beam.turn == 1


def test_the_beam_is_carried_from_one_turn_to_the_next(tmp_path):
    """The point of a turn loop. Three turns through a FODO line must not
    land on the same beam as one turn -- if they do, the loop is tracking the
    input distribution three times rather than its own output."""
    import simba.Modules.Beams as rbf

    one = _run(tmp_path / "one", {"turns": 1})
    many = _run(tmp_path / "many", {"turns": TURNS})

    first, last = rbf.beam(), rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(first, os.path.join(one, "M3.openpmd.hdf5"))
    rbf.openpmd.read_openpmd_beam_file(last, os.path.join(many, "M3.openpmd.hdf5"))
    assert first.sigmas.sigma_x != pytest.approx(last.sigmas.sigma_x, rel=1e-9)


# --- R3: the same output in single-particle mode -------------------------
#
# The single-particle path once stored the beam at each observation point in
# `beam_data` and never wrote it out, so `write_turns` was a silent no-op
# there while the full-beam path wrote every turn. These assert on the files
# on disk rather than on which method is called, so they survive the two
# paths being merged.


def test_single_particle_mode_writes_every_turn(tmp_path):
    """R3. The beam at each screen is a linear reconstruction here rather
    than tracked particles, but it is reconstructed either way -- and the
    end-of-line beam has always been written -- so the turns are too."""
    subdir = _run(tmp_path, {"turns": TURNS, "write_turns": True,
                             "single_particle": True})
    assert _turns(subdir, "M3") == list(range(1, TURNS + 1))


def test_single_particle_and_full_beam_write_the_same_names(tmp_path):
    """The invariant R3 was really asking about: which files a run produces
    is a property of `turns` and `write_turns`, not of how the beam got
    there. Only the contents should differ between the two modes."""
    single_dir = _run(tmp_path / "single", {"turns": TURNS, "write_turns": True,
                                            "single_particle": True})
    full_dir = _run(tmp_path / "full", {"turns": TURNS, "write_turns": True})
    assert _beam_files(single_dir) == _beam_files(full_dir)
    for name in ("M1", "M3"):
        assert _turns(single_dir, name) == _turns(full_dir, name), name


def test_single_particle_still_honours_write_turns_being_off(tmp_path):
    """And the fix did not make the default chatty: with `write_turns` off a
    multi-turn single-particle run writes what a single-turn run writes."""
    written = _beam_files(
        _run(tmp_path, {"turns": TURNS, "single_particle": True})
    )
    assert all(SEPARATOR not in f for f in written), written
