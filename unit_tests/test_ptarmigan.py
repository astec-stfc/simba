"""Ptarmigan: a laser colliding with the beam (Compton scattering, radiation reaction, pairs)."""

import os
import shutil
import warnings
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c, e, h

import simba.Modules.Beams as rbf
from simba.Codes.Ptarmigan.Ptarmigan import gaussian_a0, ptarmiganLattice
from simba.Modules.units import UnitValue

PTARMIGAN = os.environ.get("PTARMIGAN", os.path.expanduser("~/ptarmigan/target/release/ptarmigan"))
needs_ptarmigan = pytest.mark.skipif(
    not (shutil.which("ptarmigan") or os.path.exists(PTARMIGAN)), reason="Ptarmigan not installed"
)

LASER = {"wavelength": 532e-9, "pulse_energy": 0.4, "waist": 20e-6, "pulse_duration_fwhm": 1e-12}
"""A laser whose head-on Compton probability, for a pencil beam at its focus, is
P = 2 sigma N_L / (pi w0^2) = 0.1085 at 1.3 GeV (Klein-Nishina sigma, N_L photons)."""


def _analytic_probability(gamma):
    photon = h * c / LASER["wavelength"]
    x = 4 * gamma * photon / (0.51099895e6 * e)
    sigma = 6.6524587e-29 * (1 - x + 1.3 * x**2)  # leading Klein-Nishina terms
    return 2 * sigma * (LASER["pulse_energy"] / photon) / (np.pi * LASER["waist"] ** 2)


def test_a0_of_a_gaussian_pulse():
    # the a0 that reproduced the analytic yield in Ptarmigan (0.1055 +- 0.0015 vs 0.1085)
    a0 = gaussian_a0(LASER["wavelength"], LASER["pulse_energy"], LASER["waist"], LASER["pulse_duration_fwhm"])
    assert a0 == pytest.approx(0.11123, rel=1e-4)


def test_tracking_keys_are_ptarmigans():
    line = SimpleNamespace(file_block={"tracking": {"pair_creation": False, "a0": 2.0}})
    assert ptarmiganLattice.control.fget(line) == {"pair_creation": False}
    line.file_block["tracking"]["radiation_reactoin"] = True
    with pytest.raises(ValueError, match="radiation_reactoin"):
        ptarmiganLattice.control.fget(line)


def test_elements_that_act_on_the_beam_are_warned_about():
    from simba import exceptions

    line = SimpleNamespace(objectname="IP", elements={
        "M1": SimpleNamespace(hardware_class="Marker"),
        "LASER": SimpleNamespace(hardware_class="Laser"),
        "Q1": SimpleNamespace(hardware_class="Magnet"),
        "COL": SimpleNamespace(hardware_class="Aperture"),
    })
    with pytest.warns(exceptions.UnmodelledElementsWarning, match="Q1, COL"):
        ptarmiganLattice.check_elements(line)


@pytest.mark.parametrize(
    "tracking, laser_a0, a0",
    [({"a0": 2.0}, 5.0, 2.0), ({}, 5.0, 5.0), ({}, None, 0.11123)],
)
def test_a0_from_tracking_then_the_laser_then_its_pulse(tracking, laser_a0, a0):
    laser = SimpleNamespace(name="LASER", laser=SimpleNamespace(a0=laser_a0, **LASER))
    line = SimpleNamespace(file_block={"tracking": tracking}, laser=laser, polarization="linear")
    assert ptarmiganLattice.a0.fget(line) == pytest.approx(a0, rel=1e-4)


def test_circular_polarization_has_a_lower_a0_at_the_same_intensity():
    # LUXE (Ptarmigan's examples/luxe.yml): 1.2 J, 0.8 um, 30 fs, circular, a0 = 5 sets the waist
    waist = 147.839 * np.sqrt(1.2) * 0.8e-6 / (5.0 * np.sqrt(30.0))
    laser = SimpleNamespace(name="LASER", laser=SimpleNamespace(
        a0=None, wavelength=0.8e-6, pulse_energy=1.2, waist=waist, pulse_duration_fwhm=30e-15,
    ))
    line = SimpleNamespace(file_block={}, laser=laser, polarization="circular")
    assert ptarmiganLattice.a0.fget(line) == pytest.approx(5.0, rel=1e-4)


def test_a_line_holds_one_laser():
    line = SimpleNamespace(objectname="IP", elements={"M1": SimpleNamespace(hardware_type="Marker")})
    with pytest.raises(ValueError, match="has 0"):
        ptarmiganLattice.laser.fget(line)


def _write_output(filename, species):
    """A Ptarmigan output file in the beam's frame, from ``{species: (position, momentum, weight)}``."""
    with h5py.File(filename, "w") as f:
        f["config/unit/momentum"] = "MeV/c"
        f["config/unit/position"] = "m"
        f["config/output/beam_defines_positive_z"] = True
        for name in ("electron", "positron", "photon"):
            position, momentum, weight = species.get(name, (np.zeros((0, 4)),) * 2 + (np.zeros(0),))
            f[f"final-state/{name}/position"] = position
            f[f"final-state/{name}/momentum"] = momentum
            f[f"final-state/{name}/weight"] = weight


def test_the_particles_are_drifted_to_the_end_of_the_line(tmp_path):
    # one electron and one photon, at the focus at t = 0, each 1 mrad off axis in x
    E = 1000.0  # MeV
    electron = (
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        np.array([[np.hypot(E, 0.51099895), 1e-3 * E, 0.0, E]]),
        np.array([3.0]),
    )
    photon = (np.array([[0.0, 0.0, 0.0, 0.0]]), np.array([[50.0, 0.05, 0.0, 50.0]]), np.array([7.0]))
    _write_output(tmp_path / "out.h5", {"electron": electron, "photon": photon})
    beam = rbf.beam()
    particles = rbf.ptarmigan.read_ptarmigan_output(str(tmp_path / "out.h5"))
    rbf.ptarmigan.ptarmigan_particles_to_beam(beam, particles, "electron", z_out=0.5, t_offset=1e-9)
    assert list(beam.particle_species) == ["electron", "photon"]
    # 0.5 m downstream at 1 mrad: x = 0.5 mm, for both
    np.testing.assert_allclose(beam.x.val, [5e-4, 5e-4], rtol=1e-6)
    # the photon takes 0.5 m / (c cos 1 mrad); the electron, a touch longer
    assert beam.t.val[1] == pytest.approx(1e-9 + 0.5 / c / np.cos(1e-3), rel=1e-9)
    assert beam.t.val[0] > beam.t.val[1]
    np.testing.assert_allclose(beam.Particles.nmacro.val, [3.0, 7.0])
    assert float(beam.Particles.total_charge) == pytest.approx(-3 * e)
    groups = beam.split_species()
    assert np.isclose(float(groups["photon"].Particles.total_charge), 0.0)


def test_the_beam_goes_in_relative_to_its_centre(tmp_path):
    beam = rbf.beam()
    rbf.ptarmigan.ptarmigan_particles_to_beam(
        beam, {"electron": {
            "position": np.array([[0.0, 0.0, 0.0, -1e-6], [0.0, 1e-6, 0.0, 1e-6]]),
            "momentum": np.array([[1e9, 0.0, 0.0, 1e9], [1e9, 0.0, 0.0, 1e9]]),
            "weight": np.array([2.0, 2.0]),
        }}, "electron", z_out=0.0, t_offset=0.0,
    )
    beam.set_z_from_t(10.0)  # somewhere along the machine
    rbf.ptarmigan.write_ptarmigan_beam_file(beam, str(tmp_path / "in.h5"))
    with h5py.File(tmp_path / "in.h5") as f:
        position = f["final-state/electron/position"][()]
        assert f["beam_axis"][()] == b"+z"
        np.testing.assert_allclose(f["final-state/electron/weight"][()], [2.0, 2.0])
    np.testing.assert_allclose(position[:, 0], 0.0)
    assert abs(np.mean(position[:, 3])) < 1e-12  # about the bunch's centre, not z = 10 m
    np.testing.assert_allclose(np.diff(position[:, 3]), [2e-6], rtol=1e-6)


PENCIL = {"initial_momentum": 1.3e9, "sigma_x": 1e-7, "sigma_px": 130, "sigma_y": 1e-7,
          "sigma_py": 130, "sigma_z": 1e-7, "sigma_pz": 1.3e3}
"""A 1.3 GeV pencil beam."""


def _laser_line(
    tmp_path, nparticles, collision_angle=0.0, tracking=None, laser=None, beam=None, charge=None,
    laser_error=None, beam_x=0.0,
):
    """
    A line M1, LASER (at z = 0.5 m), SCR, M3, run by Ptarmigan; the beam at M3.
    ``laser_error`` is the laser's ``physical.error``; ``beam_x`` moves the incoming beam.
    """
    from laura import LAURA
    from laura.exporters.yaml_exporter import export_machine
    from laura.models.element import Laser, Marker

    import simba.Framework as fw
    from simba.Codes.Generators import frameworkGenerator

    def marker(name, z):
        return Marker(name=name, machine_area="IP", hardware_class="Marker",
                      physical={"middle": {"x": 0.0, "y": 0.0, "z": z}})

    physical = {"length": 1e-3, "middle": {"x": 0.0, "y": 0.0, "z": 0.5}}
    if laser_error:
        physical["error"] = laser_error
    laser = Laser(name="LASER", machine_area="IP", physical=physical,
                  laser=laser or {**LASER, "polarization": "linear", "collision_angle": collision_angle})
    elements = [marker("M1", 0.0), laser, marker("SCR", 0.75), marker("M3", 1.0)]
    section = {"sections": {"IP": {"elements": [x.name for x in elements]}}}
    machine = LAURA(element_list=elements, section=section,
                    layout={"default_layout": "l", "layouts": {"l": ["IP"]}})
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    # by default, a 1.3 GeV pencil beam, one electron per macroparticle
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)}, filename="seed.openpmd.hdf5",
        charge=charge or nparticles * e, number_of_particles=nparticles, **(beam or PENCIL),
    ).write()
    if beam_x:
        seed = rbf.beam()
        rbf.openpmd.read_openpmd_beam_file(seed, str(tmp_path / "seed.openpmd.hdf5"))
        seed.Particles.x = UnitValue(seed.x.val + beam_x, "m")
        seed.write_openpmd_beam_file(str(tmp_path / "seed.openpmd.hdf5"))
    settings = fw.FrameworkSettings()
    settings.files = {"IP": {
        "code": "ptarmigan", "charge": {"space_charge_mode": "False"},
        "output": {"start_element": "M1", "end_element": "M3"},
        "tracking": tracking or {"pair_creation": False, "rng_seed": 1},
    }}
    settings.layout, settings.section = machine.layout, section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False)
    framework.executables.ptarmigan = [shutil.which("ptarmigan") or PTARMIGAN]
    framework.loadSettings(settings=settings)
    shutil.copy(tmp_path / "seed.openpmd.hdf5", os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    beam.subdirectory = framework.subdirectory
    return beam


@needs_ptarmigan
def test_head_on_compton_scattering_gives_the_analytic_yield_and_edge(tmp_path):
    n = 2000
    beam = _laser_line(tmp_path, n)
    assert beam.species == "electron" and len(beam.x) == n
    assert float(beam.Particles.total_charge) == pytest.approx(-n * e)
    assert float(beam.s) == pytest.approx(1.0)
    photons = beam.others["photon"]
    yield_ = photons.Particles.nmacro.val.sum() / n
    expected = _analytic_probability(1.3e9 / 0.51099895e6)
    assert abs(yield_ - expected) < 4 * np.sqrt(expected / n), (yield_, expected)
    # Compton edge, head-on: 4 gamma^2 E_L / (1 + 4 gamma E_L / m c^2) = 57.6 MeV
    edge = photons.Particles.energy.val.max() / 1e6
    assert 54.0 < edge < 57.7
    # the screen 0.25 m past the laser has the same particles, a quarter metre earlier
    screen = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(screen, os.path.join(beam.subdirectory, "SCR.openpmd.hdf5"))
    assert float(screen.s) == pytest.approx(0.75)
    assert len(screen.others["photon"].x) == len(photons.x)
    np.testing.assert_allclose(
        photons.t.val - screen.others["photon"].t.val, 0.25 / c, rtol=1e-4
    )


@needs_ptarmigan
def test_a_crossing_at_right_angles_halves_the_edge(tmp_path):
    beam = _laser_line(tmp_path, 2000, collision_angle=np.pi / 2)
    edge = beam.others["photon"].Particles.energy.val.max() / 1e6
    assert 24.0 < edge < 29.7  # 29.6 MeV at 90 degrees


def _photons_per_electron(beam, n):
    return beam.others["photon"].Particles.nmacro.val.sum() / n if "photon" in beam.others else 0.0


@needs_ptarmigan
def test_a_laser_one_waist_off_the_axis_scatters_exp_minus_two_as_much(tmp_path):
    # a pencil beam in a Gaussian focus sees exp(-2 d^2 / w0^2) of the peak intensity
    n = 4000
    error = {"position": {"x": LASER["waist"], "y": 0.0, "z": 0.0}}
    aligned = _photons_per_electron(_laser_line(tmp_path / "aligned", n), n)
    off = _laser_line(tmp_path / "off", n, laser_error=error)
    ratio = _photons_per_electron(off, n) / aligned
    spread = ratio * np.sqrt(1 / (ratio * aligned * n) + 1 / (aligned * n))
    assert abs(ratio - np.exp(-2)) < 4 * spread, (ratio, spread)
    # the electrons come out where they were: on the beam axis, not the laser's
    assert abs(np.mean(off.x.val)) < 1e-6


@needs_ptarmigan
def test_a_beam_steered_onto_the_misaligned_laser_scatters_as_if_aligned(tmp_path):
    # the beam and the laser both 20 um out in x: they meet; with the sign wrong, 40 um apart
    n = 4000
    error = {"position": {"x": 20e-6, "y": 0.0, "z": 0.0}}
    aligned = _photons_per_electron(_laser_line(tmp_path / "aligned", n), n)
    both = _laser_line(tmp_path / "both", n, laser_error=error, beam_x=20e-6)
    ratio = _photons_per_electron(both, n) / aligned
    assert abs(ratio - 1) < 4 * np.sqrt(2 / (aligned * n)), ratio
    assert np.mean(both.x.val) == pytest.approx(20e-6, abs=1e-6)


@needs_ptarmigan
def test_a_late_laser_meets_the_beam_away_from_its_focus(tmp_path):
    # late by 2 z_R / c, head-on: they meet z_R from the focus, where the intensity has halved
    n = 4000
    rayleigh = np.pi * LASER["waist"] ** 2 / LASER["wavelength"]
    on_time = _laser_line(tmp_path / "on", n)
    late = _laser_line(tmp_path / "late", n, tracking={
        "pair_creation": False, "rng_seed": 1, "timing_offset": 2 * rayleigh / c,
    })
    ratio = _photons_per_electron(late, n) / _photons_per_electron(on_time, n)
    aligned = _photons_per_electron(on_time, n) * n
    assert abs(ratio - 0.5) < 4 * ratio * np.sqrt(1 / (ratio * aligned) + 1 / aligned), ratio
    # the beam itself is not delayed: only where it meets the laser moves
    assert np.mean(late.t.val) == pytest.approx(np.mean(on_time.t.val), abs=1e-15)


def test_the_timing_offset_is_simbas_not_ptarmigans():
    line = SimpleNamespace(file_block={"tracking": {"timing_offset": 1e-12, "rng_seed": 3}})
    assert ptarmiganLattice.control.fget(line) == {"rng_seed": 3}
    assert ptarmiganLattice.timing_offset.fget(line) == 1e-12


def test_only_transverse_laser_errors_are_modelled():
    def line(error):
        return SimpleNamespace(objectname="IP", laser=SimpleNamespace(physical=SimpleNamespace(error=error)))

    def error(x=0.0, y=0.0, z=0.0, theta=0.0):
        return SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z),
                               rotation=SimpleNamespace(theta=theta, phi=0.0, psi=0.0))

    assert ptarmiganLattice.laser_offset.fget(line(error(x=1e-5, y=-2e-5))) == (1e-5, -2e-5)
    for unmodelled in (error(z=1e-3), error(theta=1e-3)):
        with pytest.warns(UserWarning, match="not modelled"):
            ptarmiganLattice.laser_offset.fget(line(unmodelled))


@needs_ptarmigan
def test_the_luxe_bunch_crossing(tmp_path):
    """
    Ptarmigan's ``examples/luxe.yml`` (LUXE, Abramowicz et al., EPJST 230, 2445 (2021),
    arXiv:2102.02032) through simba: a0 = 5, circular, 16.5 GeV, 17.2 degrees, with radiation
    reaction and pair creation. Measured against Ptarmigan run on that file: photons per
    electron 0.5584-0.5686 (four runs), electron energy loss 1.111 GeV in both at 100k
    electrons (simba 1.124 +- 0.007); positrons ~8e-6 per electron, but +-30% run to run.
    """
    import yaml

    p, n = 16.5e9, 20000
    beam = _laser_line(
        tmp_path, n,
        laser={"wavelength": 0.8e-6, "pulse_energy": 1.2, "pulse_duration_fwhm": 30e-15,
               "waist": 147.839 * np.sqrt(1.2) * 0.8e-6 / (5.0 * np.sqrt(30.0)),
               "polarization": "circular", "collision_angle": float(np.radians(-17.2))},
        beam={"initial_momentum": p, "sigma_x": 5e-6, "sigma_px": 8.672e-6 * p, "sigma_y": 5e-6,
              "sigma_py": 8.672e-6 * p, "sigma_z": 24e-6, "sigma_pz": 1e-3 * p},
        charge=1.5e9 * e,
        tracking={"dt_multiplier": 0.5, "radiation_reaction": True, "pair_creation": True,
                  "rng_seed": 0, "increase_pair_rate_by": 1.0e4},
    )
    with open(os.path.join(beam.subdirectory, "IP.yml")) as f:
        assert yaml.safe_load(f)["laser"]["a0"] == pytest.approx(5.0, rel=1e-5)
    electrons, photons = beam, beam.others["photon"]
    assert photons.Particles.nmacro.val.sum() / 1.5e9 == pytest.approx(0.563, rel=0.04)
    loss = 16.5e9 - np.average(electrons.Particles.energy.val, weights=electrons.Particles.nmacro.val)
    assert loss == pytest.approx(1.117e9, rel=0.04)
    # what the electrons lost, the photons carry (the pairs take a millionth of it)
    radiated = np.sum(photons.Particles.energy.val * photons.Particles.nmacro.val) / 1.5e9
    assert radiated == pytest.approx(loss, rel=0.005)
    assert "positron" in beam.others
