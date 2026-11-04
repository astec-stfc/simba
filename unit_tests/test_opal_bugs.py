from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from simba.Codes.OPAL.OPAL import opalLattice, update_globals
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
    ],
)
def test_screens_read_their_monitor_file_first(tmp_path, monitor, spositions, default_step, expected_t):
    """Screens were read from the nearest whole-bunch dump, up to 3 mm (10 ps) off the
    plane, and looked up by z, which lags OPAL's s by ~6 mm after the injector dogleg."""
    _write_opal_h5(tmp_path / "line.h5", [1.0, 2.0, 3.0])
    if monitor:
        _write_opal_h5(tmp_path / "SCR_opal.h5", [9.0])
    lattice = SimpleNamespace(global_parameters={"master_subdir": str(tmp_path)})
    beam = opalLattice.read_element_beam(
        lattice, "SCR", str(tmp_path / "line.h5"), spositions, default_step
    )
    if expected_t is None:
        assert beam is None
    else:
        assert np.allclose(beam._beam.t.val, expected_t)
