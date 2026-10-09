"""Xsuite multi-turn tracking, end to end through the framework."""

import os

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf
from simba.Codes.Generators import frameworkGenerator

TURNS = 4


def _run(fodo_machine, tmp_path, tracking):
    """Track the FODO line through Xsuite with the given ``tracking`` block."""
    pytest.importorskip("xtrack")
    machine, names = fodo_machine(tmp_path)

    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": "xsuite",
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

    frameworkGenerator(
        global_parameters={"master_subdir": framework.subdirectory},
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12,
    ).write()
    framework.track()
    return framework.subdirectory


def _beam_files(subdir):
    return sorted(f for f in os.listdir(subdir) if f.endswith(".openpmd.hdf5"))


def _read(subdir, name, turn=None):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(subdir, name), turn=turn)
    return beam


# --- the crash ------------------------------------------------------------


def test_a_multi_turn_run_post_processes(tmp_path, fodo_machine):
    """The regression. It raised `IndexError` off an empty `beam_data`."""
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS})
    assert os.path.isfile(os.path.join(subdir, "M3.openpmd.hdf5"))


def test_a_multi_turn_run_writes_a_twiss_file(tmp_path, fodo_machine):
    """The second failure mode behind the first: guarding the index moved
    the crash into the twiss reader, which wants the bunch columns."""
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS})
    assert os.path.isfile(os.path.join(subdir, "FODO_twiss.csv"))


def test_the_twiss_file_has_the_bunch_statistic_columns(tmp_path, fodo_machine):
    """`momentum` is the one that raised `KeyError`; the rest come with it."""
    pd = pytest.importorskip("pandas")
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS})
    df = pd.read_csv(os.path.join(subdir, "FODO_twiss.csv"))
    for column in ("momentum", "sigma_x", "sigma_y", "emit_xn", "mean_x"):
        assert column in df.columns, sorted(df.columns)


def test_a_single_turn_run_still_works(tmp_path, fodo_machine):
    """The other code path, which was never broken and must stay that way."""
    subdir = _run(fodo_machine, tmp_path, {"turns": 1})
    assert os.path.isfile(os.path.join(subdir, "FODO_twiss.csv"))


# --- the diagnostic pass is not a tracking pass ---------------------------


def test_the_beam_that_leaves_the_line_is_the_tracked_one(tmp_path, fodo_machine):
    """`collect_beam_data` walks a *copy*. If it walked the real beam, the
    written beam would have gone round one extra time -- and would match a
    five-turn run rather than a four-turn one."""
    four = _read(_run(fodo_machine, tmp_path / "four", {"turns": 4}), "M3.openpmd.hdf5")
    five = _read(_run(fodo_machine, tmp_path / "five", {"turns": 5}), "M3.openpmd.hdf5")
    assert four.sigmas.sigma_x != pytest.approx(five.sigmas.sigma_x, rel=1e-12)


def test_multi_turn_tracking_actually_advances(tmp_path, fodo_machine):
    """One turn and four turns must not land on the same beam."""
    one = _read(_run(fodo_machine, tmp_path / "one", {"turns": 1}), "M3.openpmd.hdf5")
    many = _read(_run(fodo_machine, tmp_path / "many", {"turns": TURNS}), "M3.openpmd.hdf5")
    assert one.sigmas.sigma_x != pytest.approx(many.sigmas.sigma_x, rel=1e-9)


# --- R21: which turn the file is ------------------------------------------


def test_the_unsuffixed_file_says_it_is_the_last_turn(tmp_path, fodo_machine):
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS})
    assert _read(subdir, "M3.openpmd.hdf5").turn == TURNS


def test_a_single_turn_run_says_turn_one(tmp_path, fodo_machine):
    subdir = _run(fodo_machine, tmp_path, {"turns": 1})
    assert _read(subdir, "M3.openpmd.hdf5").turn == 1


def test_the_generated_input_beam_claims_no_turn(tmp_path, fodo_machine):
    """Nobody tracked it. `None` rather than a turn 1 it did not earn."""
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS})
    assert _read(subdir, "M1.openpmd.hdf5").turn is None


def test_each_bundled_turn_knows_its_turn(tmp_path, fodo_machine):
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS, "write_turns": True})
    path = os.path.join(subdir, "M3.openpmd.hdf5")
    assert rbf.openpmd.openpmd_turns(path) == list(range(1, TURNS + 1))
    for turn in range(1, TURNS + 1):
        assert _read(subdir, "M3.openpmd.hdf5", turn).turn == turn


def test_the_end_of_line_file_reads_as_its_last_turn(tmp_path, fodo_machine):
    """What the next section reads by name."""
    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS, "write_turns": True})
    assert _read(subdir, "M3.openpmd.hdf5").turn == TURNS


# --- the twiss turn column ------------------------------------------------


def test_the_twiss_summary_records_the_turn(tmp_path, fodo_machine):
    """One twiss file per *line* however many turns ran, so the only thing
    that can say which turn its bunch columns came from is the column."""
    import simba.Modules.Twiss as rtf

    subdir = _run(fodo_machine, tmp_path, {"turns": TURNS})
    t = rtf.twiss()
    t.read_HDF5_twiss_file(os.path.join(subdir, "Twiss_Summary.hdf5"))
    assert set(np.array(t.turn.val).tolist()) == {TURNS}


def test_a_single_turn_run_records_turn_one(tmp_path, fodo_machine):
    import simba.Modules.Twiss as rtf

    subdir = _run(fodo_machine, tmp_path, {"turns": 1})
    t = rtf.twiss()
    t.read_HDF5_twiss_file(os.path.join(subdir, "Twiss_Summary.hdf5"))
    assert set(np.array(t.turn.val).tolist()) == {1}
