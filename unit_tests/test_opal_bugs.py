from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from laura.translator.converters.codes.opal import OpalOption
from simba.Codes.OPAL.OPAL import (
    BRHO_BETAGAMMA_1,
    _rms_emittance,
    canonical_emittances,
    dispersions,
    opalLattice,
    update_globals,
)
from simba.Modules.constants import m_e, speed_of_light, elementary_charge


def test_global_settings_override_defaults():
    g = update_globals({"NBIN": 3, "AUTOPHASE": 0})
    assert g["distribution"]["NBIN"] == 3
    assert g["option"]["AUTOPHASE"] == 0


def _write_opal_h5(filename, times):
    """An OPAL h5 file with one 3-particle step per ``time``, each particle at that time."""
    with h5py.File(filename, "w") as f:
        for i, t in enumerate(times):
            step = f.create_group(f"Step#{i}")
            step.attrs["MASS"] = m_e * speed_of_light**2 / elementary_charge * 1e-9
            step.attrs["TotalCharge"] = [1e-12]
            for key in ["x", "y", "z", "px", "py"]:
                step[key] = np.zeros(3)
            step["pz"] = np.full(3, 10.0)
            step["time"] = np.full(3, t)


@pytest.mark.parametrize(
    "monitor, spositions, default_step, expected_t",
    [
        (True, {"SCR": 1}, None, 9.0),  # the monitor file beats the nearest dump
        (False, {"SCR": 1}, None, 2.0),  # no monitor file: the dump nearest in s
        (False, {}, -1, 3.0),  # neither: the default step (the end object's last dump)
        (False, {}, None, None),  # nothing at all
        ("empty", {"SCR": 1}, None, 2.0),  # a monitor the bunch never crossed: as if there were none
        ("empty", {}, None, None),
    ],
)
def test_screens_read_their_monitor_file_first(tmp_path, monitor, spositions, default_step, expected_t):
    """Screens were read from the nearest whole-bunch dump, up to 3 mm (10 ps) off the
    plane, and looked up by z, which lags OPAL's s by ~6 mm after the injector dogleg.
    OPAL writes a monitor at the point where the line starts with no Step#0."""
    _write_opal_h5(tmp_path / "line.h5", [1.0, 2.0, 3.0])
    if monitor:
        _write_opal_h5(tmp_path / "SCR_opal.h5", [] if monitor == "empty" else [9.0])
    lattice = SimpleNamespace(global_parameters={"master_subdir": str(tmp_path)})
    beam = opalLattice.read_element_beam(
        lattice, "SCR", str(tmp_path / "line.h5"), spositions, default_step
    )
    if expected_t is None:
        assert beam is None
    else:
        assert np.allclose(beam._beam.t.val, expected_t)


def _settings(**globals_):
    return opalLattice.option_settings(SimpleNamespace(globalSettings=globals_))


def test_no_phase_space_dump_by_default():
    """A dump of the whole bunch every 10 steps was 8 GB for the JFEL linac; OPAL's largest
    PSDUMPFREQ means never, and still writes the last step."""
    option = OpalOption(**_settings())
    assert option.PSDUMPFREQ == 2147483647
    assert "PSDUMPFREQ = 2147483647" in option.write_opal()


def test_a_global_setting_turns_the_dump_back_on():
    assert OpalOption(**_settings(PSDUMPFREQ=10)).PSDUMPFREQ == 10


def _write_particles(filename, steps):
    """An OPAL h5 file with a step per ``(s, bz, x, y, px, py, pz)``; no B-ref when ``bz`` is None."""
    with h5py.File(filename, "w") as f:
        for i, (s, bz, x, y, px, py, pz) in enumerate(steps):
            step = f.create_group(f"Step#{i}")
            step.attrs["SPOS"] = [s]
            if bz is not None:
                step.attrs["B-ref"] = [0.0, 0.0, bz]
            for key, value in zip(["x", "y", "px", "py", "pz"], [x, y, px, py, pz]):
                step[key] = value


def _solenoid_beam(bz, n=2000):
    """Mechanical momenta of a beam whose canonical momenta are the cold-ish ``pcx``, ``pcy``."""
    rng = np.random.default_rng(1)
    x, y = rng.normal(0, 1e-3, (2, n))
    pcx, pcy = rng.normal(0, 1e-3, (2, n))
    k = bz / (2 * BRHO_BETAGAMMA_1)
    return x, y, pcx - k * y, pcy + k * x, np.full(n, 100.0), (x, pcx, y, pcy)


def test_canonical_emittance_from_a_monitor_file_takes_its_field_from_the_stat_file(tmp_path):
    """A monitor file has no B-ref; the .stat file's Bz_ref gives the field there."""
    x, y, px, py, pz, (_, pcx, _, pcy) = _solenoid_beam(0.1)
    _write_particles(tmp_path / "SCR_opal.h5", [(1.5, None, x, y, px, py, pz)])
    mechanical = _rms_emittance(x, px)
    emit = canonical_emittances([str(tmp_path / "SCR_opal.h5")], bz_ref=lambda s: 0.1)
    assert emit[1.5][0] == pytest.approx(_rms_emittance(x, pcx))
    assert emit[1.5][1] == pytest.approx(_rms_emittance(y, pcy))
    assert emit[1.5][0] < mechanical / 2  # the solenoid inflated the mechanical one
    assert canonical_emittances([str(tmp_path / "SCR_opal.h5")]) == {}  # no field, nothing to correct


def test_canonical_emittance_from_a_dump_uses_its_own_field(tmp_path):
    x, y, px, py, pz, (_, pcx, _, _) = _solenoid_beam(0.1)
    _write_particles(tmp_path / "line.h5", [(1.0, 0.0, x, y, px, py, pz), (2.0, 0.1, x, y, px, py, pz)])
    emit = canonical_emittances(str(tmp_path / "line.h5"))
    assert list(emit) == [2.0]  # the step with no field needs no correction
    assert emit[2.0][0] == pytest.approx(_rms_emittance(x, pcx))


def test_dispersion_from_monitor_files(tmp_path):
    rng = np.random.default_rng(2)
    n = 2000
    delta = rng.normal(0, 1e-3, n)
    x = 0.3 * delta + rng.normal(0, 1e-6, n)
    zeros = np.zeros(n)
    pz = 100.0 * (1 + delta)
    _write_particles(tmp_path / "A_opal.h5", [(4.0, None, x, zeros, zeros, zeros, pz)])
    disp = dispersions([str(tmp_path / "A_opal.h5")])
    assert disp[4.0][0] == pytest.approx(0.3, rel=1e-2)
    assert disp[4.0][2] == pytest.approx(0.0, abs=1e-6)
