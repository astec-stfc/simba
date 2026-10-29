"""Forward-mode gradients from Cheetah, checked against finite differences of the tracking."""

import os
import warnings

import h5py
import numpy as np
import pytest

pytest.importorskip("cheetah")

import simba.Framework as fw
from simba.Codes.Generators.Generators import frameworkGenerator
from simba.Modules.gradients import read_gradients
from laura.models.element import Screen
from helpers import fodo_machine, fodo_quads, read_beam


def _framework(tmp_path):
    """The FODO cell with a 10 cm screen between the quads, in Cheetah, gradients on for QUAD1F:k1l."""
    screen = Screen(
        name="SCR", machine_area="FODO",
        physical={"length": 0.1, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )
    quads = fodo_quads()
    machine, names, section = fodo_machine(tmp_path, middle=[quads[0], screen, quads[1]])
    settings = fw.FrameworkSettings()
    settings["global"] = {"gradients": {"parameters": ["QUAD1F:k1l"]}}
    settings.files = {
        "FODO": {
            "code": "cheetah",
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    return framework


def _track(framework, k1l):
    framework.modifyElement("QUAD1F", "k1l", k1l)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    subdir = framework.subdirectory
    beam = read_beam(subdir, "M3")
    particles = np.stack([getattr(beam, c).val for c in ("x", "cpx", "y", "cpy", "t", "cpz")], -1)
    with h5py.File(os.path.join(subdir, "FODO_twiss.cheetah.hdf5")) as f:
        beta_x = f["Twiss"]["beta_x"][()]
    return particles, beta_x


def test_cheetah_gradients_match_finite_differences(tmp_path):
    framework = _framework(tmp_path)
    # one beam for every run: the generator is unseeded
    frameworkGenerator(
        global_parameters={"master_subdir": framework.subdirectory},
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3, sigma_z=1e-3, sigma_pz=1e3,
        charge=100e-12,
    ).write()
    step = 1e-4
    plus, beta_plus = _track(framework, -1.0 + step)
    minus, beta_minus = _track(framework, -1.0 - step)
    _track(framework, -1.0)

    grads = read_gradients(os.path.join(framework.subdirectory, "FODO_gradients.cheetah.hdf5"))
    assert grads["parameters"] == ["QUAD1F:k1l"]
    np.testing.assert_allclose(grads["parameter_values"], [-1.0])
    assert grads["attrs"]["coordinates"].tolist() == ["x", "cpx", "y", "cpy", "t", "cpz"]
    assert any("SCR" in name for name in grads["Beams"])
    assert grads["Twiss"]["beta_x"].shape == (len(grads["Twiss"]["s"]), 1)

    fd = (plus - minus) / (2 * step)
    end = grads["Beams"]["M3"][..., 0]
    assert end.shape == fd.shape
    for column in range(6):
        np.testing.assert_allclose(
            end[:, column], fd[:, column], rtol=0, atol=1e-6 * np.abs(fd[:, column]).max(),
        )

    fd_beta = (beta_plus - beta_minus) / (2 * step)
    np.testing.assert_allclose(grads["Twiss"]["beta_x"][:, 0], fd_beta, rtol=0, atol=1e-6 * np.abs(fd_beta).max())
