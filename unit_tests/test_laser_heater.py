"""A laser heater (a LAURA wiggler with a laser) in Ocelot and elegant: the laser
adds the energy modulation LAURA gives, and nothing without pulse energy; elegant
follows the laser's time profile; codes that ignore the laser, or its profile,
say so."""

import os
import shutil
import warnings

import numpy as np
import pytest

import simba.Framework as fw
from simba import exceptions
from helpers import fodo_machine, read_beam, skip_missing
from laura.models.element import Wiggler
from laura.translator.utils.fields import FieldMap
from laura.translator.utils.fields.field_parameter import set_field
from scipy.constants import speed_of_light
from simba.Codes.Generators import frameworkGenerator

P0C = 135e6
GAMMA = np.hypot(P0C, 0.51099895e6) / 0.51099895e6


def wiggler(pulse_energy):
    """Ten 5 cm periods, the laser resonant, much wider (2 mm rms) and longer
    (1 ns) than the 0.1 mm, 1 mm bunch, so every electron sees its peak."""
    elem = Wiggler(
        name="LH1", machine_area="FODO",
        magnetic={"length": 0.5, "period": 0.05, "num_periods": 10, "strength": 1.53},
        physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
    )
    wavelength = elem.resonant_wavelength(GAMMA)
    elem.laser = {"wavelength": wavelength, "waist": 4e-3,
                  "pulse_energy": pulse_energy, "pulse_duration_fwhm": 1e-9}
    return elem


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    directory = tmp_path_factory.mktemp("seed")
    np.random.seed(1)
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=P0C,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        charge=100e-12, number_of_particles=5000,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def energy_spread(tmp_path, elem, seed_beam, code="ocelot"):
    """rms of cp [eV] at the end of the line, tracked in ``code``."""
    return np.std(track(tmp_path, elem, seed_beam, code).cp.val)


def track(tmp_path, elem, seed_beam, code):
    """The beam at the end of the line, tracked in ``code``."""
    skip_missing(code)
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
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    return read_beam(framework.subdirectory, "M3")


@pytest.mark.parametrize("code, scale, rel", [
    ("ocelot", 1.0, 0.03),
    # elegant's end poles cost about one period of interaction: (N-1)/N
    ("elegant", 0.9, 0.05),
])
def test_the_heater_adds_lauras_modulation(tmp_path, seed_beam, code, scale, rel):
    """Over many laser periods a peak modulation dE adds dE/sqrt(2) rms."""
    heater = wiggler(pulse_energy=5e-3)
    dE = scale * heater.laser_energy_modulation(GAMMA)
    assert 5e3 < dE < 5e4  # a heater-like 5-50 keV, not a vanishing effect
    off = energy_spread(tmp_path / "off", wiggler(pulse_energy=0.0), seed_beam, code)
    on = energy_spread(tmp_path / "on", heater, seed_beam, code)
    assert on == pytest.approx(np.sqrt(off**2 + dE**2 / 2), rel=rel)


SIGMA_T = 1e-3 / speed_of_light  # the bunch's rms length in time


def profiled_wiggler(directory, initial_position=0.0):
    """The heater with a Gaussian time profile of field rms SIGMA_T, at the peak
    power of ``wiggler(5e-3)``: pulse energy = P * integral(f**2 dt) = P * SIGMA_T * sqrt(pi)."""
    t = np.linspace(-8 * SIGMA_T, 8 * SIGMA_T, 401)
    profile = FieldMap()
    profile.reset_dicts()
    profile.field_type, profile.read = "LaserTimeProfile", True
    set_field(profile, "t", t, "s")
    set_field(profile, "Ex", np.exp(-t**2 / (2 * SIGMA_T**2)), "")
    profile.filename = str(directory / "src" / "profile.sdds")
    filename = profile.write_field_file("sdds", location=str(directory / "profile.sdds"))
    elem = wiggler(pulse_energy=wiggler(5e-3).laser.peak_power * SIGMA_T * np.sqrt(np.pi))
    elem.laser.pulse_duration_fwhm = None
    elem.laser.field_definition = os.path.abspath(filename)
    elem.laser.initial_position = initial_position
    return elem


@pytest.mark.parametrize("code, scale", [("elegant", 0.9), ("ocelot", 1.0)])
def test_a_pulse_as_long_as_the_bunch(tmp_path, seed_beam, code, scale):
    """With a Gaussian bunch and a Gaussian field envelope of the same rms (elegant: a
    time profile; Ocelot: pulse_duration_fwhm), the mean of f**2 over the bunch is
    1/sqrt(1 + 2): the added variance drops by sqrt(3)."""
    dE = scale * wiggler(5e-3).laser_energy_modulation(GAMMA)
    elem = profiled_wiggler(tmp_path) if code == "elegant" else gaussian_wiggler(0.0)
    off = energy_spread(tmp_path / "off", wiggler(pulse_energy=0.0), seed_beam, code)
    on = energy_spread(tmp_path / "on", elem, seed_beam, code)
    assert on == pytest.approx(np.sqrt(off**2 + dE**2 / (2 * np.sqrt(3))), rel=0.05)


def gaussian_wiggler(initial_position):
    """The heater with a Gaussian pulse of field rms SIGMA_T, as Ocelot takes it, at
    the peak power of ``wiggler(5e-3)`` (energy scaled with the duration)."""
    fwhm = 2 * np.sqrt(np.log(2)) * SIGMA_T
    elem = wiggler(pulse_energy=5e-3 * fwhm / 1e-9)
    elem.laser.pulse_duration_fwhm = fwhm
    elem.laser.initial_position = initial_position
    return elem


@pytest.mark.parametrize("code", ["elegant", "ocelot"])
def test_a_laser_ahead_of_the_bunch_heats_its_head(tmp_path, seed_beam, code):
    elem = profiled_wiggler(tmp_path, 2e-3) if code == "elegant" else gaussian_wiggler(2e-3)
    beam = track(tmp_path / "run", elem, seed_beam, code)
    t, cp = beam.t.val, beam.cp.val
    head, tail = t < np.median(t), t > np.median(t)
    assert np.std(cp[head]) > 3 * np.std(cp[tail])


def test_elegant_warns_when_its_cw_laser_over_heats(tmp_path, seed_beam):
    """A pulse as long as the bunch, with no time profile: sqrt(3) too much heating."""
    with pytest.warns(exceptions.LaserPulseIgnoredWarning, match=r"1\.7\dx"):
        energy_spread(tmp_path / "short", gaussian_wiggler(0.0), seed_beam, "elegant")
    with warnings.catch_warnings():
        warnings.simplefilter("error", exceptions.LaserPulseIgnoredWarning)
        energy_spread(tmp_path / "long", wiggler(pulse_energy=5e-3), seed_beam, "elegant")


def test_ocelot_warns_that_it_ignores_the_profile(tmp_path, seed_beam):
    with pytest.warns(exceptions.LaserTimeProfileUnsupportedWarning, match="LH1"):
        energy_spread(tmp_path / "run", profiled_wiggler(tmp_path), seed_beam, "ocelot")


def test_a_code_without_the_laser_warns(tmp_path, seed_beam):
    with pytest.warns(exceptions.LaserModulationUnsupportedWarning, match="LH1"):
        energy_spread(tmp_path, wiggler(pulse_energy=5e-3), seed_beam, "impactx")


def test_no_pulse_energy_is_a_plain_undulator(tmp_path, seed_beam):
    plain = wiggler(pulse_energy=0.0)
    plain.laser = None
    assert energy_spread(tmp_path / "a", wiggler(pulse_energy=0.0), seed_beam) == pytest.approx(
        energy_spread(tmp_path / "b", plain, seed_beam), rel=1e-9
    )
