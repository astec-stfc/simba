import os
import pytest
import simba.Framework as fw
from simba.Codes.Generators import frameworkGenerator
from laura.models.element import Quadrupole, Marker, Plasma, Wiggler
from laura import LAURA
from laura.Exporters.YAML import export_machine

RUN_SIMCODES_TESTS = os.environ.get("SIMBA_TEST_SIMCODES") == "1"
SKIP_REASON = (
    "set SIMBA_TEST_SIMCODES=1 to run real tracking codes "
    "(needs docker and the simcodes image; see docs/source/SimCodes.rst)"
)

CODES = [
    pytest.param("astra", "docker", id="astra"),
    pytest.param("elegant", "docker", id="elegant"),
    pytest.param("csrtrack", "docker", id="csrtrack"),
    pytest.param("ocelot", None, id="ocelot"),
    pytest.param("xsuite", None, id="xsuite"),
    pytest.param("cheetah", None, id="cheetah"),
    pytest.param("opal", "docker", id="opal"),
    pytest.param(
        "gpt", None, id="gpt",
        marks=pytest.mark.skipif("GPTLICENSE" not in os.environ, reason="requires a local GPT install and GPTLICENSE env var"),
    ),
]


def _track_lattice(tmp_path, code, container_runtime, middle_elements):
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
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3, sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12,
    )
    gen.write()

    framework.track()
    return os.path.join(framework.subdirectory, "M3.openpmd.hdf5")


def _fodo_elements():
    return [
        Quadrupole(name="QUAD1F", machine_area="FODO", magnetic={"length": 1.0, "k1l": -1}, physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}}),
        Quadrupole(name="QUAD1D", machine_area="FODO", magnetic={"length": 1.0, "k1l": 1.0}, physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}}),
    ]


@pytest.mark.skipif(not RUN_SIMCODES_TESTS, reason=SKIP_REASON)
@pytest.mark.parametrize("code,container_runtime", CODES)
def test_code_tracks(tmp_path, code, container_runtime):
    output_file = _track_lattice(tmp_path, code, container_runtime, _fodo_elements())
    assert os.path.isfile(output_file)


@pytest.mark.skipif(not RUN_SIMCODES_TESTS, reason=SKIP_REASON)
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


@pytest.mark.skipif(not RUN_SIMCODES_TESTS, reason=SKIP_REASON)
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
