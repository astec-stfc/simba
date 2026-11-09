"""An aperture or collimator removes the particles outside it, in every code that has one.

MAD-X ignored the limits (``track`` without ``aperture``), Ocelot never activated its
``Aperture`` elements, and Cheetah only set a lost particle's survival probability to 0,
which the output beam did not read. The beam is cold, so a particle's x does not change
between the markers, and the aperture is placed in a gap of the sorted |x| so that the
number to survive does not depend on the code's integrator.
"""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
from helpers import fodo_machine, read_beam, skip_missing
from laura.models.element import Aperture, Collimator, Drift
from simba.Codes.Generators import frameworkGenerator

HALF_Y = 0.02  # far outside the beam

try:
    from laura.translator.converters.aperture import ApertureTranslator

    HALF_WIDTHS = hasattr(ApertureTranslator, "_half_sizes")
except ImportError:
    HALF_WIDTHS = False
needs_half_widths = pytest.mark.skipif(
    not HALF_WIDTHS,
    reason="needs LAURA with half-width apertures (laura#43), which laura-accelerator 1.1.0 lacks",
)


@pytest.fixture(scope="module")
def seed(tmp_path_factory):
    """A cold Gaussian bunch, written once, and the half width that splits it."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-3, sigma_px=1.0, sigma_y=1e-3, sigma_py=1.0,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=256,
    ).write()
    path = os.path.join(str(directory), "M1.openpmd.hdf5")
    x = np.sort(np.abs(read_beam(str(directory), "M1").x.val))
    lo, hi = len(x) // 2, len(x) * 9 // 10
    k = lo + int(np.argmax(np.diff(x[lo:hi])))  # the widest gap, away from the middle and the tail
    return path, (x[k] + x[k + 1]) / 2, k + 1


def element(thin, half_x):
    aperture = {"shape": "rectangular", "horizontal_size": half_x, "vertical_size": HALF_Y}
    if thin:
        return [
            Aperture(
                name="AP1", machine_area="FODO", aperture=aperture,
                physical={"length": 0.0, "middle": {"x": 0.0, "y": 0.0, "z": 1.0}},
            ),
            Drift(
                name="D1", hardware_class="Drift", hardware_type="Drift", machine_area="FODO",
                physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 1.5}},
            ),
        ]
    return [
        Collimator(
            name="COL1", machine_area="FODO", aperture=aperture,
            physical={"length": 0.1, "middle": {"x": 0.0, "y": 0.0, "z": 1.0}},
        )
    ]


@needs_half_widths
@pytest.mark.parametrize("thin", [True, False], ids=["aperture", "collimator"])
@pytest.mark.parametrize("code", ["madx", "ocelot", "cheetah"])
def test_the_beam_after_an_aperture_is_the_particles_inside_it(tmp_path, seed, code, thin):
    skip_missing(code)
    path, half_x, inside = seed
    machine, names, section = fodo_machine(tmp_path, middle=element(thin, half_x))
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
    settings.section = {"sections": {"FODO": section["sections"]["FODO"]}}
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(path, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()

    out = read_beam(framework.subdirectory, "M3")
    assert len(out.x.val) == inside
    assert np.abs(out.x.val).max() < half_x
    assert out.Q.val != 0
