"""What per-turn output does to the summary files."""

import h5py

from simba.Modules.Beams import save_HDF5_summary_file

TURNS = 8
SCREENS = ("SCREEN-01", "SCREEN-02")


def beam_file(directory, name):
    """The smallest thing the directory scan counts as a beam."""
    path = directory / f"{name}.openpmd.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("particles")
    return path


def summary_entries(directory):
    save_HDF5_summary_file(str(directory), str(directory / "Beam_Summary.hdf5"))
    with h5py.File(directory / "Beam_Summary.hdf5", "r") as handle:
        return list(handle.keys())


def test_a_single_turn_run_gets_one_entry_per_screen(tmp_path):
    for screen in SCREENS:
        beam_file(tmp_path, screen)
    assert len(summary_entries(tmp_path)) == len(SCREENS)


def test_a_multi_turn_run_gets_one_entry_per_screen_per_turn(tmp_path):
    """The whole of R20 in one assertion: the summary grows by the turn
    count, because the scan has no notion of a turn."""
    for screen in SCREENS:
        for turn in range(TURNS):
            beam_file(tmp_path, f"{screen}-t{turn:03d}")
    assert len(summary_entries(tmp_path)) == len(SCREENS) * TURNS


def test_the_growth_is_linear_not_quadratic(tmp_path):
    """Pinned because a quadratic scan would be the difference between
    twenty minutes and never finishing."""
    counts = []
    for block in range(3):
        for index in range(10):
            beam_file(tmp_path, f"BLOCK{block}-{index:03d}")
        counts.append(len(summary_entries(tmp_path)))
    assert counts == [10, 20, 30]


def test_a_file_without_particles_is_not_counted(tmp_path):
    """Why the scan opens every file rather than trusting the name -- and so
    why it costs an h5py open per turn."""
    beam_file(tmp_path, "REAL")
    with h5py.File(tmp_path / "NOTABEAM.openpmd.hdf5", "w") as handle:
        handle.create_group("something_else")
    # note the key keeps `.openpmd`: splitext only strips `.hdf5`
    assert summary_entries(tmp_path) == ["REAL.openpmd"]


def test_naming_the_screens_skips_the_scan(tmp_path):
    """The cheap way out if this ever does bite: passing `screens` takes the
    branch that builds paths directly instead of globbing and opening
    everything. `Framework.save_summary_files` does not currently do this."""
    for screen in SCREENS:
        beam_file(tmp_path, screen)
    beam_file(tmp_path, "UNWANTED")
    out = tmp_path / "Beam_Summary.hdf5"
    save_HDF5_summary_file(str(tmp_path), str(out), screens=list(SCREENS))
    with h5py.File(out, "r") as handle:
        assert sorted(handle.keys()) == [f"{s}.openpmd" for s in sorted(SCREENS)]
