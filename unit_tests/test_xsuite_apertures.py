"""Collimators and apertures in Xsuite lines are black absorbers."""

import os
import shutil
import warnings

import numpy as np
import pytest
from scipy.special import erf

pytest.importorskip("xtrack")


def _through(tmp_path, aperture, npart=4000):
    """A round 1 mm, 100 MeV electron beam through a 0.1 m collimator at z = 0.5 m, in Xsuite."""
    from laura import LAURA
    from laura.exporters.yaml_exporter import export_machine
    from laura.models.element import Collimator, Marker

    import simba.Framework as fw
    import simba.Modules.Beams as rbf
    from simba.Codes.Generators import frameworkGenerator

    def marker(name, z):
        return Marker(name=name, machine_area="LINE", hardware_class="Marker",
                      physical={"middle": {"x": 0.0, "y": 0.0, "z": z}})

    col = Collimator(name="COL", machine_area="LINE", aperture=aperture,
                     physical={"length": 0.1, "middle": {"x": 0.0, "y": 0.0, "z": 0.5}})
    elements = [marker("M1", 0.0), col, marker("M3", 1.0)]
    section = {"sections": {"LINE": {"elements": [e.name for e in elements]}}}
    machine = LAURA(element_list=elements, section=section,
                    layout={"default_layout": "l", "layouts": {"l": ["LINE"]}})
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    # sigma 1 mm, divergence 1 urad: the beam hardly grows over the collimator
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)}, filename="seed.openpmd.hdf5",
        initial_momentum=100e6, sigma_x=1e-3, sigma_px=1e2, sigma_y=1e-3, sigma_py=1e2,
        sigma_z=1e-4, sigma_pz=1e4, charge=1e-12, number_of_particles=npart,
    ).write()
    settings = fw.FrameworkSettings()
    settings.files = {"LINE": {
        "code": "xsuite", "charge": {"space_charge_mode": "False"},
        "output": {"start_element": "M1", "end_element": "M3"},
    }}
    settings.layout, settings.section = machine.layout, section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(tmp_path / "seed.openpmd.hdf5", os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    return beam


@pytest.mark.parametrize(
    "aperture, expected",
    [
        # inside r = 0.5 mm of a round 1 mm Gaussian: 1 - exp(-r^2 / 2 sigma^2)
        ({"shape": "circular", "radius": 5e-4}, 1 - np.exp(-0.5**2 / 2)),
        # a 1 mm full-width horizontal slit: |x| < 0.5 mm, y open
        ({"shape": "planar", "horizontal_size": 1e-3}, erf(0.5 / np.sqrt(2))),
        # 1 mm x 2 mm rectangle: both planes
        ({"shape": "rectangular", "horizontal_size": 1e-3, "vertical_size": 2e-3},
         erf(0.5 / np.sqrt(2)) * erf(1 / np.sqrt(2))),
    ],
)
def test_a_collimator_passes_only_what_is_inside_it(tmp_path, aperture, expected):
    npart = 4000
    beam = _through(tmp_path, aperture, npart)
    passed = len(beam.x) / npart
    assert abs(passed - expected) < 4 * np.sqrt(expected * (1 - expected) / npart), (passed, expected)
    assert float(beam.Particles.total_charge) == pytest.approx(-1e-12 * passed)
    half_x = aperture.get("radius") or aperture["horizontal_size"] / 2
    assert np.max(np.abs(beam.x.val)) <= half_x * 1.001


def test_a_collimator_without_a_size_stays_a_drift(tmp_path):
    beam = _through(tmp_path, {"shape": "rectangular"}, npart=200)
    assert len(beam.x) == 200
