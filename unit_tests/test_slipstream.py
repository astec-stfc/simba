"""Slipstream through simba: a FODO cell against its transfer matrix, and a cavity against its voltage."""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
from helpers import fodo_machine, fodo_quads, read_beam, skip_missing
from laura.models.element import Marker
from simba.Codes.Generators import frameworkGenerator


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """One bunch for every run: ``frameworkGenerator`` is unseeded."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3, sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=64,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def track(tmp_path, seed_beam, **machine):
    """Track ``seed_beam`` from M1 to M3 in slipstream; the framework."""
    skip_missing("slipstream")
    machine, _, section = fodo_machine(tmp_path, **machine)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": "slipstream",
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
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    return framework


def transfer(k, length):
    """The 2 x 2 matrix of a thick quadrupole of strength ``k`` (focusing if positive)."""
    w = np.sqrt(abs(k))
    if k > 0:
        return np.array([[np.cos(w * length), np.sin(w * length) / w], [-w * np.sin(w * length), np.cos(w * length)]])
    return np.array([[np.cosh(w * length), np.sinh(w * length) / w], [w * np.sinh(w * length), np.cosh(w * length)]])


def drift(length):
    return np.array([[1.0, length], [0.0, 1.0]])


def test_a_fodo_cell_is_tracked_as_its_transfer_matrix_says(tmp_path, seed_beam):
    """The quadrupoles are 1 m long, so one kick each would be 16% off in x."""
    framework = track(tmp_path, seed_beam)
    start = read_beam(framework.subdirectory, "M1")
    end = read_beam(framework.subdirectory, "M3")
    # QUAD1F has k1l = -1 and QUAD1D has k1l = +1, both 1 m; the drifts are 0.25, 1.5 and 0.5 m
    for coord, k in (("x", (-1.0, 1.0)), ("y", (1.0, -1.0))):
        m = drift(0.5) @ transfer(k[1], 1.0) @ drift(1.5) @ transfer(k[0], 1.0) @ drift(0.25)
        slope = getattr(start, f"p{coord}").val / start.pz.val
        expected = m @ np.array([getattr(start, coord).val, slope])
        got = getattr(end, coord).val
        assert np.max(np.abs(got - expected[0])) < 2e-3 * got.std()
    assert end.energy.val.mean() == pytest.approx(start.energy.val.mean(), rel=1e-9)
    assert end.z.val.std() == pytest.approx(start.z.val.std(), rel=1e-2)  # R56 is 4e-3 of it


def test_the_beam_at_a_marker_inside_the_line_is_written(tmp_path, seed_beam):
    quads = fodo_quads()
    middle = Marker(name="M2", machine_area="FODO", hardware_class="Marker",
                    physical={"middle": {"x": 0.0, "y": 0.0, "z": 2.0}})
    framework = track(tmp_path, seed_beam, middle=[quads[0], middle, quads[1]])
    start = read_beam(framework.subdirectory, "M1")
    half = read_beam(framework.subdirectory, "M2")
    expected = (drift(0.75) @ transfer(1.0, 1.0) @ drift(0.25)) @ np.array([
        start.y.val, start.py.val / start.pz.val])  # QUAD1F focuses in y
    assert np.max(np.abs(half.y.val - expected[0])) < 2e-3 * half.y.val.std()


def test_a_cavity_on_crest_adds_its_voltage(tmp_path, seed_beam):
    cavity = {
        "cavity": {"phase": 0.0, "frequency": 3e9, "n_cells": 1, "cell_length": 0.5, "structure_Type": "StandingWave"},
        "simulation": {"field_amplitude": 5e6},
    }
    framework = track(tmp_path, seed_beam, cavity=cavity, cavity_length=0.5)
    start = read_beam(framework.subdirectory, "M1")
    end = read_beam(framework.subdirectory, "M3")
    gain = end.energy.val.mean() - start.energy.val.mean()
    # the bunch is 1 mm long, which is 0.06 rad of the 3 GHz wave: cos(0.06) is 0.998
    assert gain == pytest.approx(5e6, rel=5e-3)
