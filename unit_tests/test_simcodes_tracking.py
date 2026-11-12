import importlib.util
import os
import warnings
import numpy as np
import pytest
import simba.Framework as fw
from laura.translator.converters.codes import cheetah as laura_cheetah
from simba.Codes.Generators import frameworkGenerator
from laura.models.element import Marker, Plasma, Wiggler
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from helpers import skip_missing

needs_container = pytest.mark.needs_container

CODES = [
    pytest.param("astra", "docker", id="astra", marks=needs_container),
    pytest.param("elegant", "docker", id="elegant", marks=needs_container),
    pytest.param("csrtrack", "docker", id="csrtrack", marks=needs_container),
    pytest.param("ocelot", None, id="ocelot"),
    pytest.param("xsuite", None, id="xsuite"),
    pytest.param("cheetah", None, id="cheetah"),
    pytest.param("slipstream", None, id="slipstream"),
    pytest.param(
        "rftrack", None, id="rftrack",
        marks=pytest.mark.skipif(importlib.util.find_spec("RF_Track") is None, reason="RF-Track is not installed"),
    ),
    pytest.param("impactx", None, id="impactx"),
    pytest.param("opal", "docker", id="opal", marks=needs_container),
    pytest.param(
        "gpt", None, id="gpt",
        marks=pytest.mark.skipif("GPTLICENSE" not in os.environ, reason="requires a local GPT install and GPTLICENSE env var"),
    ),
]


def _track_lattice(tmp_path, code, container_runtime, middle_elements, before_track=None):
    framework = _run_lattice(tmp_path, code, container_runtime, middle_elements, before_track=before_track)
    return os.path.join(framework.subdirectory, "M3.openpmd.hdf5")


def _run_lattice(tmp_path, code, container_runtime, middle_elements, sigma_pz=1e3, cheetah_settings=None, before_track=None):
    m1 = Marker(name="M1", machine_area="FODO", hardware_class="Marker", physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}})
    end_z = middle_elements[-1].physical.middle.z + middle_elements[-1].physical.length
    m3 = Marker(name="M3", machine_area="FODO", hardware_class="Marker", physical={"middle": {"x": 0.0, "y": 0.0, "z": end_z}})
    element_names = ["M1"] + [e.name for e in middle_elements] + ["M3"]
    sections = {"sections": {"FODO": element_names}}
    layouts = {"default_layout": "line1", "layouts": {"line1": ["FODO"]}}
    machine = LAURA(element_list=[m1, *middle_elements, m3], layout=layouts, section=sections)
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)

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
    settings.section = {"sections": {"FODO": element_names}}
    settings.element_list = f"{tmp_path}/lattice"
    if cheetah_settings:
        settings["global"] = {**(settings.get("global") or {}), "CHEETAHsettings": cheetah_settings}

    framework = fw.Framework(
        machine=machine,
        directory=str(tmp_path),
        clean=True,
        verbose=False,
        container_runtime=container_runtime,
    )
    framework.loadSettings(settings=settings)

    gen = frameworkGenerator(
        global_parameters={"master_subdir": framework.subdirectory},
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3, sigma_z=1e-3, sigma_pz=sigma_pz,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12,
    )
    gen.write()
    if before_track:
        before_track(framework)

    framework.track()
    return framework


@pytest.mark.parametrize("code,container_runtime", CODES)
def test_code_tracks(tmp_path, code, container_runtime, monkeypatch, fodo_elements):
    skip_missing(code)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    output_file = _track_lattice(tmp_path, code, container_runtime, fodo_elements)
    assert os.path.isfile(output_file)
    assert not any(cwd.iterdir())


@pytest.mark.skipif(
    not hasattr(laura_cheetah, "choose_cheetah_tracking_method"), reason="needs a LAURA with cheetah_tracking_method"
)
@pytest.mark.parametrize("sigma_pz, settings, expected", [
    (1e3, None, "linear"),
    (2.5e5, None, "second_order"),
    (2.5e5, {"cheetah_tracking_method": "linear"}, "linear"),
])
def test_cheetah_auto_switches_to_second_order_for_a_large_energy_spread(tmp_path, fodo_elements, sigma_pz, settings, expected):
    """With a 5% energy spread, linear tracking through the FODO misses its chromatic
    aberration, so "auto" switches to second order and warns; a setting overrides it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        framework = _run_lattice(
            tmp_path, "cheetah", None, fodo_elements, sigma_pz=sigma_pz, cheetah_settings=settings
        )
    switched = [w for w in caught if "set cheetah_tracking_method" in str(w.message)]
    assert len(switched) == (expected == "second_order")
    segment = framework.latticeObjects["FODO"].segment
    import cheetah
    methods = {
        type(e).__name__: e.tracking_method for e in segment.flattened().elements
        if isinstance(e, (cheetah.Drift, cheetah.Quadrupole))
    }
    assert methods == {"Drift": expected, "Quadrupole": expected}


@pytest.mark.parametrize("code,container_runtime", [
    ("ocelot", None), ("xsuite", None), ("cheetah", None),
    pytest.param("astra", "docker", id="astra", marks=needs_container),
    pytest.param("opal", "docker", id="opal", marks=needs_container),
])
def test_output_timestamps(tmp_path, code, container_runtime, fodo_elements):
    """The bunch arrives L/(beta c) after it set off: Ocelot added the start z on top,
    Xsuite and Cheetah divided by c alone and dropped the input time, and Cheetah put
    every output at the start. Ocelot and Cheetah also read total energy as momentum.
    ASTRA's output dropped its reference particle's clock, and OPAL started again at 0 (the bunch sets off
    at 1 ns here)."""
    from scipy.constants import c
    from pmd_beamphysics import ParticleGroup
    import simba.Modules.Beams as rbf
    from simba.Modules.Beams import load_file
    from simba.Modules.Beams.openpmd import read_particle_group
    from simba.Modules.units import UnitValue

    def later(framework):
        path = os.path.join(framework.subdirectory, "M1.openpmd.hdf5")
        beam = rbf.beam()
        read_particle_group(beam, ParticleGroup(h5=path))
        beam.Particles.t = UnitValue(np.array(beam.Particles.t.val) + 1e-9, "s")
        beam.write_openpmd_beam_file(path)

    output_file = _track_lattice(tmp_path, code, container_runtime, fodo_elements, before_track=later)
    beam_in = load_file(os.path.join(os.path.dirname(output_file), "M1.openpmd.hdf5"))
    beam = load_file(output_file)
    flight = np.mean(beam.t.val) - np.mean(beam_in.t.val)
    # SIMBA read ASTRA's t as when each particle reaches the reference particle, not the screen: ~0.3 ps here
    assert flight == pytest.approx(4.25 / (np.mean(beam_in.Bz.val) * c), abs=1e-13)
    assert np.mean(beam.z.val) == pytest.approx(4.25, abs=1e-3)
    assert np.mean(beam.energy.val) == pytest.approx(np.mean(beam_in.energy.val), rel=1e-6)


def test_bmad_times_the_bunch_not_its_reference_particle(tmp_path, fodo_elements):
    """Bmad's lattice timed the mean but its beam file put z relative to the reference particle, so a bunch
    whose reference was off its centroid lost the difference: 0.25 ps a segment in the LCLS linac."""
    from scipy.constants import c
    from pmd_beamphysics import ParticleGroup
    import simba.Modules.Beams as rbf
    from simba.Modules.Beams import load_file
    from simba.Modules.Beams.openpmd import read_particle_group
    skip_missing("bmad")
    # the library the other tests start Tao from: pytao can start only one per process
    libtao = os.path.expanduser("~/Documents/bmad-ecosystem/production/lib/libtao.so")
    if not os.path.exists(libtao):
        pytest.skip("Bmad libtao not installed")

    def lead(framework):  # the reference is the particle at the front, mm ahead of the centroid
        path = os.path.join(framework.subdirectory, "M1.openpmd.hdf5")
        group, beam = ParticleGroup(h5=path), rbf.beam()
        read_particle_group(beam, group, reference_particle_index=int(np.argmax(group.z)))
        beam.write_openpmd_beam_file(path)
        framework["FODO"].libtao = libtao

    framework = _run_lattice(tmp_path, "bmad", None, fodo_elements, before_track=lead)
    beam_in, beam = (load_file(os.path.join(framework.subdirectory, f"{n}.openpmd.hdf5")) for n in ("M1", "M3"))
    assert beam_in.reference_particle_index is not None
    flight = np.mean(beam.t.val) - np.mean(beam_in.t.val)
    assert flight == pytest.approx(4.25 / (np.mean(beam_in.Bz.val) * c), abs=1e-13)


def test_reference_time_through_acceleration():
    """5 to 50 MeV, linearly in s, against the integral of ds/(beta c)."""
    from scipy.constants import c
    from simba.FrameworkHelperFunctions import reference_time
    mass, length = 0.511e6, 2.0
    energy = np.linspace(5e6, 50e6, 100001)
    p = np.sqrt(energy**2 - mass**2)
    expected = 1e-9 + np.trapezoid(energy / p, dx=length / 100000) / c
    assert reference_time(3.0 + length, p[-1], mass, (1e-9, 3.0, p[0])) == pytest.approx(expected, rel=1e-9)
    assert reference_time(length, p[0], mass) == pytest.approx(length * energy[0] / (p[0] * c))


def _cavity_line(tmp_path, code):
    from laura.models.element import RFCavity, Quadrupole
    skip_missing(code)
    cavity = RFCavity(
        name="CAV1", machine_area="FODO", physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 1.25}},
        cavity={"frequency": 3e9, "phase": 0.0, "n_cells": 10, "cell_length": 0.05, "structure_Type": "StandingWave"},
        simulation={"field_amplitude": 25e6},
    )
    m2 = Marker(name="M2", machine_area="FODO", hardware_class="Marker", physical={"middle": {"x": 0.0, "y": 0.0, "z": 1.5}})
    drift = Quadrupole(  # no strength: a drift that ends the line, at z = 5
        name="Q0", machine_area="FODO", magnetic={"length": 1.0, "k1l": 0.0},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
    )
    return _run_lattice(tmp_path, code, None, [cavity, m2, drift])


@pytest.mark.parametrize("code", ["ocelot", "xsuite", "cheetah", "impactx", "madx"])
def test_output_timestamps_across_a_cavity(tmp_path, code):
    """1 m of drift, then 0.5 m of cavity taking 5 to 30 MeV/c, linearly in s: int E/p ds = L dp/dE. MAD-X
    crossed it at the entrance's speed, 7 ps late here."""
    from scipy.constants import c
    from simba.Modules.Beams import load_file
    framework = _cavity_line(tmp_path, code)
    start, after = (load_file(os.path.join(framework.subdirectory, f"{n}.openpmd.hdf5")) for n in ("M1", "M2"))
    e0, e1, p0, p1 = (np.mean(b.val) for b in (start.energy, after.energy, start.cpz, after.cpz))
    flight = np.mean(after.t.val) - np.mean(start.t.val)
    # Xsuite's and ImpactX's kicks, 3 a cell, are 10 fs short of the ramp here; one kick was 3 ps
    assert flight == pytest.approx((1.0 * np.mean(start.energy.val / start.cpz.val) + 0.5 * (p1 - p0) / (e1 - e0)) / c, abs=2e-14)


@pytest.mark.parametrize("code", ["ocelot", "xsuite", "cheetah", "impactx", "madx"])
def test_output_timestamps_after_a_cavity(tmp_path, code):
    """In the drift after a cavity the bunch takes L/(beta c) at its new momentum,
    whatever the code's cavity model. The reference's time was the energy gain spread
    over the whole run from its start, 2.4 ps/m wrong here at 5 to 30 MeV/c."""
    from scipy.constants import c
    from simba.Modules.Beams import load_file
    framework = _cavity_line(tmp_path, code)
    after, end = (load_file(os.path.join(framework.subdirectory, f"{n}.openpmd.hdf5")) for n in ("M2", "M3"))
    assert np.mean(end.cp.val) > 25e6  # it did accelerate
    inv_beta = np.mean(after.energy.val / after.cpz.val)  # per particle, and pz: the angles are worth 1 fs here
    flight = np.mean(end.t.val) - np.mean(after.t.val)
    assert flight == pytest.approx(3.5 * inv_beta / c, abs=1e-15)


def test_waket_tracks(tmp_path):
    plasma = Plasma(
        name="TEST-PLASMA-01",
        machine_area="FODO",
        physical={"length": 0.01, "middle": {"x": 0.0, "y": 0.0, "z": 0.005}},
        laser={
            "profile_type": "gaussian",
            "initial_position": 0.00006,
            "waist": 0.00007,
            "pulse_energy": 3.0,
            "pulse_duration_fwhm": 50e-15,
            "wavelength": 800e-9,
        },
        plasma={
            "density": 1e23,
            "ramp_up": 0.001,
            "ramp_down": 0.001,
            "plateau": 0.01,
            "ramp_decay_length": 0.002,
            "density_profile": True,
        },
        simulation={
            "wakefield_model": "quasistatic_2d",
            "r_max": 0.001,
            "n_longitudinal": 100,
            "n_radial": 100,
            "n_out": 100,
            "min_longitudinal_position": -0.0001,
            "max_longitudinal_position": 0.0001,
        },
    )
    output_file = _track_lattice(tmp_path, "waket", None, [plasma])
    assert os.path.isfile(output_file)


@needs_container
def test_genesis_tracks(tmp_path):
    wiggler = Wiggler(
        name="TEST-WIGGLER-01",
        machine_area="FODO",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 1.0}, "length": 1.968},
        magnetic={
            "length": 1.968,
            "strength": 8.524,
            "period": 0.082,
            "num_periods": 24,
            "helical": True,
            "quadratic_roll_off_x": 0.5,
            "quadratic_roll_off_y": 0.5,
        },
    )
    output_file = _track_lattice(tmp_path, "genesis", "docker", [wiggler])
    assert os.path.isfile(output_file)
