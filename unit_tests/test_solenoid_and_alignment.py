"""Solenoids and alignment errors through simba, against Xsuite: MAD-X's
solenoid ``ks`` needs the segment's rigidity, and its MAKETHIN drops EALIGN."""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
from helpers import fodo_machine, read_beam, skip_missing
from laura.models.element import Dipole, Quadrupole, Solenoid
from simba.Codes.Generators import frameworkGenerator


def solenoid():
    return Solenoid(
        name="SOL1", machine_area="FODO",
        magnetic={"length": 0.5, "fields": {"S0L": 0.02}},
        physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
    )


def misaligned_quad():
    return Quadrupole(
        name="QUAD1", machine_area="FODO",
        magnetic={"length": 1.0, "k1l": 1.0, "tilt": 0.1},
        physical={
            "length": 1.0,
            "middle": {"x": 0.0, "y": 0.0, "z": 2.5},
            "error": {
                "position": {"x": 1e-3, "y": -2e-3, "z": 3e-3},
                "rotation": {"theta": 4e-3, "phi": 5e-3, "psi": 6e-3},
            },
        },
    )


def misaligned_bend(error):
    """Untilted: MAD-X's MAKETHIN models a tilted strong bend 1e-4 off even
    without errors."""
    return Dipole(
        name="BEND1", machine_area="FODO",
        magnetic={"length": 1.0, "angle": 0.5},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 2.5}, "error": error},
    )


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=32,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def track(tmp_path, code, middle, seed_beam):
    skip_missing(code)
    machine, _, section = fodo_machine(tmp_path, middle=middle)
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
    framework = fw.Framework(
        machine=machine, directory=str(tmp_path), clean=True, verbose=False
    )
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    beam = read_beam(framework.subdirectory, "M3")
    return np.array([beam.x.val, beam.y.val])


@pytest.mark.parametrize("code", ["madx", "ocelot", "cheetah"])
def test_a_solenoid_turns_the_beam_as_xsuite_does(tmp_path, code, seed_beam):
    """One MAD-X slice is 2.4% off at this kL of 0.6; four are 5e-4."""
    expected = track(tmp_path / "x", "xsuite", [solenoid()], seed_beam)
    final = track(tmp_path / code, code, [solenoid()], seed_beam)
    assert close(final, expected, 1e-3)


def test_madx_keeps_alignment_errors_through_makethin(tmp_path, seed_beam):
    """The errors move x by 60% of its spread; four quadrupole slices are 3%
    off that, and 64 are 3e-4."""
    expected = track(tmp_path / "x", "xsuite", [misaligned_quad()], seed_beam)
    final = track(tmp_path / "m", "madx", [misaligned_quad()], seed_beam)
    assert close(final, expected, 5e-2)


@pytest.mark.parametrize(
    "error",
    [
        {"position": {"x": 1e-3}},
        {"position": {"z": 1e-3}},
        {"rotation": {"theta": 1e-3}},
        {"rotation": {"phi": 1e-3}},
        {"rotation": {"psi": 1e-2}},
    ],
    ids=["dx", "dz", "dx_rot", "dy_rot", "dz_rot"],
)
def test_madx_slices_follow_a_bends_arc(tmp_path, seed_beam, error):
    """Each slice is misaligned where it sits on the arc. Straight-line levers
    were 30% off on dx_rot and 5x on dy_rot; a thin slice cannot take the
    shift along the arc, which leaves dx 5% off."""
    def effect(code):
        bare = track(tmp_path / f"{code}0", code, [misaligned_bend({})], seed_beam)
        moved = track(tmp_path / code, code, [misaligned_bend(error)], seed_beam)
        return moved - bare

    got, want = effect("madx"), effect("xsuite")
    assert np.abs(got - want).max() < 0.1 * np.abs(want).max()


def close(final, expected, rtol):
    return all(
        np.abs(got - want).max() < rtol * np.abs(want).max()
        for got, want in zip(final, expected)
    )
