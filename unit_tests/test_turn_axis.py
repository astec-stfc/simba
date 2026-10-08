"""Which turn a result came from, carried by the result rather than its name."""

import os

import h5py
import numpy as np
import pytest
import simba.Modules.Beams as rbf
import simba.Modules.Twiss as rtf
from simba.Codes.Generators import frameworkGenerator
from simba.Framework import Framework
from simba.Framework_objects import frameworkLattice
from simba.Modules.Twiss.hdf5 import twiss_file_version
from simba.Modules.units import UnitValue


class FakeLine:
    """A lattice stub carrying only what the turn-axis code reads."""

    def __init__(self, turns=1, write_turns=False, directory=None, end="M3"):
        self.file_block = {"tracking": {"turns": turns, "write_turns": write_turns}}
        self.start = "M0"
        self.end = end
        self.objectname = "RING"
        self.colliding_outputs = []
        self.global_parameters = (
            {"master_subdir": str(directory)} if directory is not None else {}
        )

    turns = frameworkLattice.turns
    write_turns = frameworkLattice.write_turns
    bundles_turns = frameworkLattice.bundles_turns
    output_turns = frameworkLattice.output_turns
    output_basename = frameworkLattice.output_basename
    output_beam_file = frameworkLattice.output_beam_file
    writes_output = frameworkLattice.writes_output
    write_beam_file = frameworkLattice.write_beam_file
    beam_turn = frameworkLattice.beam_turn


# --- `beam_turn`: resolving what `output_turns` leaves unsaid -------------


def test_a_single_turn_line_writes_turn_one():
    assert FakeLine(turns=1).beam_turn(None) == 1


def test_an_unsuffixed_multi_turn_file_is_the_last_turn():
    """The default multi-turn run keeps one file per screen. It is turn N."""
    assert FakeLine(turns=7).beam_turn(None) == 7


def test_a_suffixed_file_is_the_turn_it_says():
    assert FakeLine(turns=7).beam_turn(3) == 3


@pytest.mark.parametrize("turns", [1, 2, 10])
def test_every_output_turn_resolves_to_a_turn_in_range(turns):
    """Whichever half of the pair a backend kept, the answer is a real turn."""
    line = FakeLine(turns=turns, write_turns=True)
    for data_turn, name_turn in line.output_turns():
        assert line.beam_turn(data_turn) == line.beam_turn(name_turn)
        assert 1 <= line.beam_turn(name_turn) <= turns


def test_the_default_multi_turn_pair_is_unsuffixed_and_resolves_to_the_last():
    line = FakeLine(turns=4, write_turns=False)
    pairs = line.output_turns()
    assert [n for _, n in pairs] == [None]
    assert line.beam_turn(pairs[0][1]) == 4


# --- the turn survives the openPMD round trip ----------------------------


@pytest.fixture
def generated_beam(tmp_path):
    """A real generated distribution, read back off disk."""
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
        filename="seed.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12,
    ).write()
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(
        beam, os.path.join(str(tmp_path), "seed.openpmd.hdf5")
    )
    return beam


def written_beam(tmp_path, beam, turn, name="B"):
    """Stamp ``turn`` on ``beam``, write it to openPMD and read it back."""
    beam.turn = turn
    path = os.path.join(str(tmp_path), f"{name}.openpmd.hdf5")
    rbf.openpmd.write_openpmd_beam_file(beam, path)
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, path)
    return out, path


def species_group(h5file):
    """The one particle species in an openPMD beam file: `particles/electron`."""
    species = h5file["particles"]
    return species[list(species.keys())[0]]


def test_a_fresh_beam_has_no_turn():
    """Nobody tracked it, so nothing should claim a turn for it."""
    assert rbf.beam().turn is None


@pytest.mark.parametrize("turn", [1, 4, 1000])
def test_the_turn_survives_a_write_and_a_read(tmp_path, generated_beam, turn):
    out, _ = written_beam(tmp_path, generated_beam, turn)
    assert out.turn == turn


def test_the_turn_comes_back_as_an_int(tmp_path, generated_beam):
    """Not a numpy scalar out of HDF5, which compares equal but prints oddly."""
    out, _ = written_beam(tmp_path, generated_beam, 4)
    assert isinstance(out.turn, int)


def test_a_beam_with_no_turn_writes_no_turn(tmp_path, generated_beam):
    out, path = written_beam(tmp_path, generated_beam, None)
    assert out.turn is None
    with h5py.File(path, "r") as f:
        assert "turn" not in species_group(f)


def test_a_file_written_before_turns_existed_reads_as_none(tmp_path, generated_beam):
    """The key is simply absent; inventing a turn 1 for it would be a lie."""
    _, path = written_beam(tmp_path, generated_beam, 4)
    with h5py.File(path, "a") as f:
        del species_group(f)["turn"]
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, path)
    assert out.turn is None


# --- `write_beam_file`: one file a screen, every turn in it ---------------


def write_turns(line, beam, turns, name="M3"):
    """Write ``turns`` as a run would, each turn's beam shifted by turn mm."""
    x = np.array(beam.x.val)
    for turn in turns:
        beam._beam.x = UnitValue(x + 1e-3 * (turn or line.turns), units="m")
        beam.turn = line.beam_turn(turn)
        line.write_beam_file(beam, name, turn)
    beam._beam.x = UnitValue(x, units="m")


def read_turn(path, turn=None):
    out = rbf.beam()
    out.read_beam_file(str(path), turn=turn)
    return out


def test_every_turn_goes_in_the_one_file(tmp_path, generated_beam):
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [1, 2, 3, 4])
    assert sorted(p.name for p in tmp_path.glob("M3*")) == ["M3.openpmd.hdf5"]
    assert rbf.openpmd.openpmd_turns(tmp_path / "M3.openpmd.hdf5") == [1, 2, 3, 4]


def test_each_turn_reads_back_as_itself(tmp_path, generated_beam):
    line = FakeLine(turns=12, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, range(1, 13))
    x = np.mean(generated_beam.x.val)
    for turn in (1, 2, 10, 12):  # 10 sorts before 2 as a string
        out = read_turn(tmp_path / "M3.openpmd.hdf5", turn)
        assert out.turn == turn
        assert np.mean(out.x.val) == pytest.approx(x + 1e-3 * turn, abs=1e-12)


def test_the_file_reads_as_the_last_turn(tmp_path, generated_beam):
    """So the next line continues from where the run ended, not where it began."""
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [1, 2, 3, 4])
    assert read_turn(tmp_path / "M3.openpmd.hdf5").turn == 4


def test_a_beam_with_no_turn_is_the_last_turn_again(tmp_path, generated_beam):
    """A code that writes the end once more after the turns replaces turn N."""
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [1, 2, 3, 4, None])
    assert rbf.openpmd.openpmd_turns(tmp_path / "M3.openpmd.hdf5") == [1, 2, 3, 4]


def test_an_end_not_recorded_each_turn_still_gets_a_file(tmp_path, generated_beam):
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [None])
    assert read_turn(tmp_path / "M3.openpmd.hdf5").turn == 4


def test_the_default_multi_turn_run_writes_a_single_beam(tmp_path, generated_beam):
    line = FakeLine(turns=4, write_turns=False, directory=tmp_path)
    write_turns(line, generated_beam, [None])
    assert rbf.openpmd.openpmd_turns(tmp_path / "M3.openpmd.hdf5") == []
    assert read_turn(tmp_path / "M3.openpmd.hdf5").turn == 4


def test_a_rerun_starts_the_file_afresh(tmp_path, generated_beam):
    write_turns(FakeLine(turns=4, write_turns=True, directory=tmp_path),
                generated_beam, [1, 2, 3, 4])
    write_turns(FakeLine(turns=2, write_turns=True, directory=tmp_path),
                generated_beam, [1, 2])
    assert rbf.openpmd.openpmd_turns(tmp_path / "M3.openpmd.hdf5") == [1, 2]


def test_the_start_of_the_line_is_never_written(tmp_path, generated_beam):
    """Its file is the input beam; in a ring its turns are the end's."""
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [1, 2], name="M0")
    assert not (tmp_path / "M0.openpmd.hdf5").exists()


def test_a_turn_the_file_lacks_is_an_error(tmp_path, generated_beam):
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [1, 2])
    with pytest.raises(ValueError, match="turns 1 to 2"):
        read_turn(tmp_path / "M3.openpmd.hdf5", 3)


def test_the_beam_summary_links_a_multi_turn_file(tmp_path, generated_beam):
    line = FakeLine(turns=4, write_turns=True, directory=tmp_path)
    write_turns(line, generated_beam, [1, 2, 3, 4])
    summary = tmp_path / "Beam_Summary.hdf5"
    rbf.save_HDF5_summary_file(str(tmp_path), str(summary))
    with h5py.File(summary, "r") as f:
        assert "M3.openpmd" in f


# --- the twiss turn column -----------------------------------------------


def test_the_twiss_object_has_a_turn_column():
    assert "turn" in rtf.twiss().properties


def test_the_turn_column_is_an_integer_column():
    """A turn is counted, and `0` is the column's "nobody said"."""
    assert rtf.twiss().properties["turn"].dtype == "i"


def test_a_twiss_object_nobody_stamped_has_an_empty_turn_column():
    assert len(rtf.twiss().turn.val) == 0


def test_the_turn_column_round_trips_through_the_summary_file(tmp_path):
    t = rtf.twiss()
    t.z.val = np.array([0.0, 1.0, 2.0])
    t.s.val = np.array([0.0, 1.0, 2.0])
    t.turn.val = np.array([4, 4, 4])
    path = os.path.join(str(tmp_path), "Twiss_Summary.hdf5")
    t.save_HDF5_twiss_file(path)
    out = rtf.twiss()
    out.read_HDF5_twiss_file(path)
    assert list(np.array(out.turn.val)) == [4, 4, 4]


def test_a_twiss_file_rows_from_different_lines_keep_their_own_turn(tmp_path):
    """The summary merges every line in the directory, so it is per row."""
    t = rtf.twiss()
    t.z.val = np.array([0.0, 1.0, 2.0, 3.0])
    t.s.val = np.array([0.0, 1.0, 2.0, 3.0])
    t.turn.val = np.array([1, 1, 12, 12])
    path = os.path.join(str(tmp_path), "Twiss_Summary.hdf5")
    t.save_HDF5_twiss_file(path)
    out = rtf.twiss()
    out.read_HDF5_twiss_file(path)
    assert list(np.array(out.turn.val)) == [1, 1, 12, 12]


# --- the version check the round trip needed fixing first -----------------


def test_a_written_twiss_file_reports_its_version(tmp_path):
    t = rtf.twiss()
    t.z.val = np.array([0.0, 1.0])
    t.s.val = np.array([0.0, 1.0])
    path = os.path.join(str(tmp_path), "T.hdf5")
    t.save_HDF5_twiss_file(path)
    with h5py.File(path, "r") as f:
        assert twiss_file_version(f) == "2"


def test_a_file_with_no_version_is_version_one(tmp_path):
    """The old format, which has no `Parameters` group at all."""
    path = os.path.join(str(tmp_path), "old.hdf5")
    with h5py.File(path, "w") as f:
        f.create_group("twiss")
    with h5py.File(path, "r") as f:
        assert twiss_file_version(f) == "1"


def test_the_default_written_file_can_be_read_back(tmp_path):
    """It could not before: every file took the version-1 branch and died."""
    t = rtf.twiss()
    t.z.val = np.array([0.0, 1.0, 2.0])
    t.s.val = np.array([0.0, 1.0, 2.0])
    t.beta_x.val = np.array([3.0, 4.0, 5.0])
    path = os.path.join(str(tmp_path), "T.hdf5")
    t.save_HDF5_twiss_file(path)
    out = rtf.twiss()
    out.read_HDF5_twiss_file(path)
    assert list(np.array(out.beta_x.val)) == [3.0, 4.0, 5.0]


# --- `stamp_twiss_turns`: the framework knows what the reader cannot ------


class FakeFramework:
    """A framework stub holding only the line-name to turn-count mapping."""

    def __init__(self, turns_by_line):
        self.latticeObjects = {
            name: FakeLine(turns=turns) for name, turns in turns_by_line.items()
        }

    stamp_twiss_turns = Framework.stamp_twiss_turns


def stamped(turns_by_line, lattice_names):
    t = rtf.twiss()
    t.lattice_name.val = np.array(lattice_names, dtype=str)
    FakeFramework(turns_by_line).stamp_twiss_turns(t)
    return list(np.array(t.turn.val))


def test_a_row_is_stamped_with_its_own_lines_turn_count():
    assert stamped({"RING": 12}, ["RING", "RING"]) == [12, 12]


def test_rows_from_different_lines_get_different_turns():
    """The summary is one object spanning every line in the run."""
    assert stamped({"INJ": 1, "RING": 12}, ["INJ", "RING", "INJ"]) == [1, 12, 1]


def test_the_twiss_suffix_the_readers_leave_on_is_matched_too():
    """Ocelot and Xsuite name the row after the file: `RING_twiss`."""
    assert stamped({"RING": 12}, ["RING_twiss"]) == [12]


def test_an_unrecognised_line_is_left_unstamped():
    """`0` is the column's `None`: nobody said, rather than turn 1."""
    assert stamped({"RING": 12}, ["SOMETHING_ELSE"]) == [0]


def test_an_empty_twiss_object_is_not_an_error():
    """A run with no twiss files at all still writes a summary."""
    t = rtf.twiss()
    FakeFramework({"RING": 4}).stamp_twiss_turns(t)
    assert len(t.turn.val) == 0


def test_stamping_does_not_disturb_the_row_count():
    names = ["RING_twiss"] * 41
    assert len(stamped({"RING": 4}, names)) == 41
