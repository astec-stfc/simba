"""A standalone LAURA Wakefield (a single-mode dielectric-lined waveguide) loses the
analytic energy, and kicks an offset beam by the analytic amount, in elegant and
Ocelot; Ocelot used to drop the wake silently, then to ignore its transverse part."""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
from helpers import fodo_machine, read_beam, skip_missing
from laura.models.element import Wakefield
from laura.translator.utils.fields import FieldMap
from laura.translator.utils.fields.field_parameter import FieldParameter
from laura.translator.utils.units import UnitValue
from simba.Codes.Generators import frameworkGenerator

CHARGE, SIGMA_Z, P0C = 100e-12, 1e-4, 135e6
LENGTH, RADIUS, WAVELENGTH = 0.1, 0.5e-3, 1e-3
# single-mode DLW: W(s) = 2 kappa cos(k s), kappa = Z0 c / (pi a^2) per metre
KAPPA = 376.73 * 2.998e8 / (np.pi * RADIUS**2) * LENGTH
K = 2 * np.pi / WAVELENGTH
# dipole wake W_perp(s) = W_T sin(k s) per metre [V/C/m^2], on a beam offset by X0
W_T, X0 = 1e19, 5e-4


def wake_file(directory, transverse=False):
    s = np.linspace(0, 5 * WAVELENGTH, 2001)
    wake = FieldMap()
    wake.length = LENGTH
    wake.z = FieldParameter(name="z", value=UnitValue(s, "m"))
    if transverse:
        for plane in ("Wx", "Wy"):
            setattr(wake, plane, FieldParameter(name=plane, value=UnitValue(W_T * LENGTH * np.sin(K * s), "V/C/m")))
        wake.Wz = FieldParameter(name="Wz", value=UnitValue(0 * s, "V/C"))
        wake.field_type = "3DWake"
    else:
        wake.Wz = FieldParameter(name="Wz", value=UnitValue(2 * KAPPA * np.cos(K * s), "V/C"))
        wake.field_type = "LongitudinalWake"
    wake.filename = os.path.join(directory, "dlw.hdf5")
    wake.read = True
    wake.write_field_file(code="hdf5")
    return wake.filename


def seed(tmp_path_factory, **offset):
    directory = tmp_path_factory.mktemp("seed")
    np.random.seed(1)
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=P0C,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=SIGMA_Z, sigma_pz=1e3, charge=CHARGE, number_of_particles=20000, **offset,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    return seed(tmp_path_factory)


@pytest.fixture(scope="module")
def offset_beam(tmp_path_factory):
    return seed(tmp_path_factory, offset_x=X0)


def track(tmp_path, tmp_path_factory, beam, code, transverse=False, enabled=True):
    """The beam before (M1) and after (M3) a FODO with the wake in the middle."""
    skip_missing(code)
    elem = Wakefield(
        name="DLW1", machine_area="FODO",
        physical={"length": LENGTH, "middle": [0, 0, 0.75]},
        cavity={"n_cells": 1, "cell_length": LENGTH},
        # outside tmp_path, which the framework cleans
        simulation={"wakefield_definition": wake_file(str(tmp_path_factory.mktemp("wake")), transverse),
                    "wakefield_enable": enabled},
    )
    machine, _, section = fodo_machine(tmp_path, middle=[elem])
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
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
    shutil.copy(beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    return read_beam(framework.subdirectory, "M1"), read_beam(framework.subdirectory, "M3")


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("code", ["elegant", "ocelot"])
def test_wakefield_element_loses_the_analytic_energy(tmp_path, tmp_path_factory, seed_beam, code, enabled):
    """...and none with its wakefield_enable off (Ocelot used to ignore that)."""
    before, after = track(tmp_path, tmp_path_factory, seed_beam, code, enabled=enabled)
    loss = np.mean(before.cp.val) - np.mean(after.cp.val)
    # loss factor of a Gaussian bunch: kappa exp(-(k sigma_z)^2)
    expected = CHARGE * KAPPA * np.exp(-((K * SIGMA_Z) ** 2)) if enabled else 0.0
    assert loss == pytest.approx(expected, rel=0.05, abs=1.0)


@pytest.mark.parametrize("code", ["elegant", "ocelot"])
def test_wakefield_element_kicks_an_offset_beam(tmp_path, tmp_path_factory, offset_beam, code):
    """Particle by particle: dx' = L x0 sum_{ahead} q W_perp(dz) / p (Ocelot got none)."""
    before, after = track(tmp_path, tmp_path_factory, offset_beam, code, transverse=True)
    q = np.abs(before.Particles.charge.val)
    z = -(before.t.val - before.t.val.mean()) * 2.998e8  # head at +z
    dz = 2e-6
    edges = np.arange(z.min() - dz, z.max() + 2 * dz, dz)
    charge = np.histogram(z, edges, weights=q)[0]
    kernel = W_T * np.sin(K * np.arange(len(charge)) * dz)
    kick = np.array([charge[i:] @ kernel[: len(charge) - i] for i in range(len(charge))])
    expected = np.interp(z, 0.5 * (edges[1:] + edges[:-1]), kick) * LENGTH * before.x.val.mean() / P0C
    tracked = after.xp.val - before.xp.val
    assert np.polyfit(expected, tracked, 1)[0] == pytest.approx(1.0, abs=0.02)
    assert np.mean(tracked) == pytest.approx(np.mean(expected), rel=0.02)
    assert abs(np.mean(after.yp.val - before.yp.val)) < 0.01 * abs(np.mean(tracked))
