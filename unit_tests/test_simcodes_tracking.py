import os
import numpy as np
import warnings
import pytest
import simba.Framework as fw
from simba.Codes.Generators import frameworkGenerator
from laura.models.element import Marker, Plasma, Wiggler
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine

needs_container = pytest.mark.needs_container

CODES = [
    pytest.param("astra", "docker", id="astra", marks=needs_container),
    pytest.param("elegant", "docker", id="elegant", marks=needs_container),
    pytest.param("csrtrack", "docker", id="csrtrack", marks=needs_container),
    pytest.param("ocelot", None, id="ocelot"),
    pytest.param("xsuite", None, id="xsuite"),
    pytest.param("cheetah", None, id="cheetah"),
    pytest.param("impactx", None, id="impactx"),
    pytest.param("opal", "docker", id="opal", marks=needs_container),
    pytest.param(
        "gpt", None, id="gpt",
        marks=pytest.mark.skipif("GPTLICENSE" not in os.environ, reason="requires a local GPT install and GPTLICENSE env var"),
    ),
]


def _track_lattice(tmp_path, code, container_runtime, middle_elements):
    framework = _run_lattice(tmp_path, code, container_runtime, middle_elements)
    return os.path.join(framework.subdirectory, "M3.openpmd.hdf5")


def _run_lattice(tmp_path, code, container_runtime, middle_elements, sigma_pz=1e3, cheetah_settings=None):
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

    framework.track()
    return framework


@pytest.mark.parametrize("code,container_runtime", CODES)
def test_code_tracks(tmp_path, code, container_runtime, monkeypatch, fodo_elements):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    output_file = _track_lattice(tmp_path, code, container_runtime, fodo_elements)
    assert os.path.isfile(output_file)
    assert not any(cwd.iterdir())


@pytest.mark.parametrize("sigma_pz, settings, expected", [
    (1e3, None, "linear"),
    (2.5e5, None, "second_order"),
    (2.5e5, {"cheetah_tracking_method": "linear"}, "linear"),
])
def test_cheetah_auto_switches_to_second_order_for_a_large_energy_spread(tmp_path, sigma_pz, settings, expected):
    """With a 5% energy spread, linear tracking through the FODO misses its chromatic
    aberration, so "auto" switches to second order and warns; a setting overrides it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        framework = _run_lattice(
            tmp_path, "cheetah", None, _fodo_elements(), sigma_pz=sigma_pz, cheetah_settings=settings
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


@pytest.mark.parametrize("code", ["ocelot", "xsuite", "cheetah"])
def test_output_timestamps(tmp_path, code):
    """The bunch arrives L/(beta c) after it set off: Ocelot added the start z on top,
    Xsuite and Cheetah divided by c alone and dropped the input time, and Cheetah put
    every output at the start. Ocelot and Cheetah also read total energy as momentum."""
    from scipy.constants import c
    from simba.Modules.Beams import load_file
    output_file = _track_lattice(tmp_path, code, None, _fodo_elements())
    beam_in = load_file(os.path.join(os.path.dirname(output_file), "M1.openpmd.hdf5"))
    beam = load_file(output_file)
    flight = np.mean(beam.t.val) - np.mean(beam_in.t.val)
    assert flight == pytest.approx(4.25 / (np.mean(beam_in.Bz.val) * c), abs=1e-13)
    assert np.mean(beam.z.val) == pytest.approx(4.25, abs=1e-3)
    assert np.mean(beam.energy.val) == pytest.approx(np.mean(beam_in.energy.val), rel=1e-6)


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
