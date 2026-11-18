"""RF-Track through simba: the Bunch6d round trip, and a FODO cell against Xsuite."""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf
from helpers import fodo_machine, fodo_quads, read_beam, skip_missing
from laura.models.element import Screen
from simba.Codes.Generators import frameworkGenerator
from simba.Modules.Beams import rftrack as rbf_rftrack

pytest.importorskip("RF_Track")


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
        charge=100e-12, number_of_particles=256,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def test_a_beam_survives_the_bunch6d_round_trip(seed_beam):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, seed_beam)
    back = rbf.beam()
    rbf_rftrack.interpret_rftrack_bunch(back, rbf_rftrack.write_rftrack_bunch(beam))
    for key in ("x", "y", "t", "cpx", "cpy", "cpz"):
        assert np.allclose(getattr(back, key).val, getattr(beam, key).val, rtol=1e-12, atol=0), key
    assert float(back.Q.val) == pytest.approx(float(beam.Q.val), rel=1e-12)
    assert np.allclose(back.E0_eV, beam.E0_eV, rtol=1e-12)


def track(tmp_path, code, seed_beam, cathode=False):
    skip_missing(code)
    screen = Screen(
        name="SCR1", machine_area="FODO",
        physical={"length": 0.0, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )
    quads = fodo_quads()
    machine, _, section = fodo_machine(tmp_path, middle=[quads[0], screen, quads[1]])
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {"particle_definition": "laser"} if cathode else {},
            "output": {"start_element": "M1", "end_element": "M3"},
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    name = "laser" if cathode else "M1"
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, f"{name}.openpmd.hdf5"))
    framework.track()
    return framework.subdirectory


def test_a_fodo_cell_matches_xsuite_with_a_screen_on_the_way(tmp_path, seed_beam):
    """Measured 9e-7 of the spread at M3."""
    expected = read_beam(track(tmp_path / "x", "xsuite", seed_beam), "M3")
    subdir = track(tmp_path / "r", "rftrack", seed_beam)
    final, screen = read_beam(subdir, "M3"), read_beam(subdir, "SCR1")
    for key in ("x", "y", "cpx", "cpy"):
        want = getattr(expected, key).val
        assert np.abs(getattr(final, key).val - want).max() < 1e-5 * want.std(), key
    assert float(np.atleast_1d(screen.s)[0]) == pytest.approx(2.0)


def test_a_beam_from_the_cathode_is_tracked_through_a_volume(tmp_path, seed_beam):
    """A cathode beam is a plane crossing spread in time: through a Volume it ends as the
    same beam does through the Lattice from that plane. Measured 3e-3 of the spread."""
    expected = read_beam(track(tmp_path / "l", "rftrack", seed_beam), "M3")
    final = read_beam(track(tmp_path / "v", "rftrack", seed_beam, cathode=True), "M3")
    for key in ("x", "y", "cpx", "cpy", "cpz", "t"):
        want, got = getattr(expected, key).val, getattr(final, key).val
        if key == "t":  # the Volume's clock starts at the reference particle's emission
            want, got = want - want.mean(), got - got.mean()
        assert np.abs(got - want).max() < 5e-3 * want.std(), key
