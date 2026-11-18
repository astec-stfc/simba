"""``Framework.setSeedValue`` reaches the codes that draw random numbers while tracking.

A run split into chunks (one run per stretch of turns) needs a different seed per
chunk: with the same one, every chunk repeats the same quantum-excitation kicks.
"""

from types import SimpleNamespace

from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegant_run_setup_command


def test_elegant_writes_a_seed_it_is_given():
    command = elegant_run_setup_command(lattice="RING.lte", random_number_seed=7).write_Elegant()
    assert "random_number_seed = 7" in command


def test_elegant_unseeded_keeps_its_own_default():
    assert "random_number_seed" not in elegant_run_setup_command(lattice="RING.lte").write_Elegant()


def _ltt_seed(tmp_path, seed):
    (tmp_path / "RING.bmad").write_text("")
    line = SimpleNamespace(lattice_file=str(tmp_path / "RING.bmad"), _ramper_lines="",
                           runSettings=SimpleNamespace(seed=seed))
    init = bmadLattice._write_ltt_init(line, tmp_path / "RING.ltt.h5", "beam.dat", 10, 10)
    return next(row.split("=")[1].strip() for row in init.read_text().splitlines() if "ltt%random_seed" in row)


def test_bmad_writes_a_seed_it_is_given(tmp_path):
    assert _ltt_seed(tmp_path, 7) == "7"


def test_bmad_unseeded_keeps_its_fixed_default(tmp_path):
    assert _ltt_seed(tmp_path, 0) == "1"
