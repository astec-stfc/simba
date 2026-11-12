"""ASTRA's clock column: absolute for the reference particle (row 0), relative to it for the others
(ASTRA manual V3.2, Table 1). The reference clock was doubled at every hand-off between lines."""

import numpy as np
import pytest

from simba.Modules import Beams as rbf

T_REF = 33.84  # ns


@pytest.fixture
def astra_file(tmp_path):
    """Five particles at 10 m, 1 MeV/c spread about 100 MeV/c; the reference clock at T_REF."""
    rows = [[0, 0, 10.0, 0, 0, 100e6, T_REF, -1e-3, 1, 5]]
    for i, dz in enumerate([-2e-4, -1e-4, 1e-4, 2e-4]):
        rows.append([1e-4 * i, -1e-4 * i, dz, 1e3, -1e3, 1e6 * (i - 1.5), 0.0, -1e-3, 1, 5])
    filename = tmp_path / "in.astra"
    np.savetxt(filename, rows)
    return filename


def _written(beam, tmp_path):
    beam.write_astra_beam_file(str(tmp_path / "out.astra"))
    return np.loadtxt(tmp_path / "out.astra")


def _assert_relative_clocks(out, t_ns, ref):
    """Row 0 is particle ``ref`` with its absolute clock; the others are relative to it."""
    assert out[0, 6] == pytest.approx(t_ns[ref])
    others = np.delete(t_ns, ref)
    assert np.sort(out[1:, 6]) == pytest.approx(np.sort(others - t_ns[ref]), abs=1e-9)
    assert np.all(np.abs(out[1:, 6]) < 1e-2)


def test_a_beam_read_from_astra_keeps_its_reference_clock(astra_file, tmp_path):
    """The final ASTRA beam goes to the next line in memory, with t relative and the clock in toffset."""
    beam = rbf.beam()
    beam.read_astra_beam_file(str(astra_file))
    t_ns = 1e9 * np.asarray(beam._beam.t) + T_REF
    _assert_relative_clocks(_written(beam, tmp_path), t_ns, 0)


@pytest.mark.parametrize("named_reference", [True, False])
def test_a_beam_read_from_openpmd_is_not_doubled(astra_file, tmp_path, named_reference):
    """A line reading the previous line's openPMD end beam: absolute t, with the reference particle
    the file names (t in s), or none (the particle nearest the mean is taken)."""
    beam = rbf.beam()
    beam.read_astra_beam_file(str(astra_file))
    beam.write_openpmd_beam_file(str(tmp_path / "end.openpmd.hdf5"), toffset=float(beam._beam.toffset.val))
    beam = rbf.beam()
    beam.read_beam_file(str(tmp_path / "end.openpmd.hdf5"))
    assert beam.reference_particle is not None
    if not named_reference:
        beam.reference_particle = beam.reference_particle_index = None
    t_ns = 1e9 * np.asarray(beam._beam.t)
    out = _written(beam, tmp_path)
    _assert_relative_clocks(out, t_ns, beam.reference_particle_index)
    assert out[0, 6] == pytest.approx(T_REF, abs=1e-3)


def test_a_given_reference_particle_goes_in_row_0(astra_file, tmp_path):
    """Bmad's openPMD beams name a reference particle, in openPMD units (t in s), anywhere in the bunch."""
    beam = rbf.beam()
    beam.read_astra_beam_file(str(astra_file))
    beam.write_openpmd_beam_file(str(tmp_path / "end.openpmd.hdf5"), toffset=float(beam._beam.toffset.val))
    beam = rbf.beam()
    beam.read_beam_file(str(tmp_path / "end.openpmd.hdf5"))
    beam.reference_particle_index = 3
    beam.reference_particle = [float(np.asarray(getattr(beam._beam, c))[3]) for c in ("x", "y", "z", "px", "py", "pz", "t")]
    t_ns = 1e9 * np.asarray(beam._beam.t)
    out = _written(beam, tmp_path)
    assert out[0, 0] == pytest.approx(np.asarray(beam._beam.x)[3])
    _assert_relative_clocks(out, t_ns, 3)
