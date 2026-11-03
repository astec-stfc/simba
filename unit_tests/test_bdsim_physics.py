"""BDSIM's physics list and options, and material that does nothing without them."""

import os
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from simba import exceptions

pytest.importorskip("pybdsim")
from simba.Codes.BDSIM.BDSIM import bdsimLattice  # noqa: E402

needs_bdsim = pytest.mark.skipif(shutil.which("bdsim") is None, reason="BDSIM not installed")


@pytest.mark.parametrize(
    "tracking, physics_list, options",
    [
        (None, "", {}),
        ({"physics_list": "em"}, "em", {"physicsList": '"em"'}),
        (
            {"physics_list": "em", "options": {"defaultRangeCut": 1e-3, "physicsVerbose": "1"}},
            "em",
            {"defaultRangeCut": 1e-3, "physicsVerbose": '"1"', "physicsList": '"em"'},
        ),
    ],
)
def test_the_physics_list_and_options_come_from_the_tracking_block(tracking, physics_list, options):
    line = SimpleNamespace(file_block={"tracking": tracking})
    line.physics_list = bdsimLattice.physics_list.fget(line)
    assert line.physics_list == physics_list
    assert bdsimLattice.bdsim_options.fget(line) == options


def _collimator_line(tmp_path, tracking, npart=400):
    """A 0.5 mm-radius copper collimator in a round 1 mm beam, through Framework.track."""
    from laura import LAURA
    from laura.exporters.yaml_exporter import export_machine
    from laura.models.element import Collimator, Marker

    import simba.Framework as fw
    import simba.Modules.Beams as rbf
    from simba.Codes.Generators import frameworkGenerator

    def marker(name, z):
        return Marker(name=name, machine_area="LINE", hardware_class="Marker",
                      physical={"middle": {"x": 0.0, "y": 0.0, "z": z}})

    col = Collimator(name="COL", machine_area="LINE", material="copper",
                     aperture={"shape": "circular", "radius": 5e-4},
                     physical={"length": 0.1, "middle": {"x": 0.0, "y": 0.0, "z": 0.5}})
    elements = [marker("M1", 0.0), col, marker("M3", 1.0)]
    section = {"sections": {"LINE": {"elements": [e.name for e in elements]}}}
    machine = LAURA(element_list=elements, section=section,
                    layout={"default_layout": "l", "layouts": {"l": ["LINE"]}})
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)}, filename="seed.openpmd.hdf5",
        initial_momentum=100e6, sigma_x=1e-3, sigma_px=1e3, sigma_y=1e-3, sigma_py=1e3,
        sigma_z=1e-4, sigma_pz=1e4, charge=1e-12, number_of_particles=npart,
    ).write()
    settings = fw.FrameworkSettings()
    settings.files = {"LINE": {
        "code": "bdsim", "charge": {"space_charge_mode": "False"}, "tracking": tracking,
        "output": {"start_element": "M1", "end_element": "M3"},
    }}
    settings.layout, settings.section = machine.layout, section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(tmp_path / "seed.openpmd.hdf5", os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        framework.track()
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    return beam, [w.message for w in caught]


@needs_bdsim
def test_without_a_physics_list_material_does_nothing_and_says_so(tmp_path):
    beam, caught = _collimator_line(tmp_path, {}, npart=100)
    assert len(beam.x) == 100
    assert any(isinstance(m, exceptions.MaterialWithoutPhysicsWarning) for m in caught)


@needs_bdsim
def test_a_round_collimator_passes_the_core_of_the_beam(tmp_path):
    # inside r = 0.5 mm of a round 1 mm Gaussian: 1 - exp(-r^2 / 2 sigma^2) = 11.75%
    npart = 400
    beam, caught = _collimator_line(tmp_path, {"physics_list": "em"}, npart=npart)
    expected = 1 - np.exp(-0.5**2 / 2)
    spread = np.sqrt(expected * (1 - expected) / npart)
    assert abs(len(beam.x) / npart - expected) < 4 * spread
    assert float(beam.Particles.total_charge) == pytest.approx(-1e-12 * len(beam.x) / npart)
    assert not any(isinstance(m, exceptions.MaterialWithoutPhysicsWarning) for m in caught)
