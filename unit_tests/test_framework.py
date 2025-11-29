import os
import pytest
from unittest.mock import MagicMock
import shutil
from pydantic import ValidationError
import simba.Framework as fw
from simba.Codes.Generators import (
    frameworkGenerator,
    ASTRAGenerator,
    GPTGenerator,
)
from simba.Framework_lattices import (
    elegantLattice,
    astraLattice,
    cheetahLattice,
)
from nala.models.element import Quadrupole, Marker, Element
from nala import NALA
from nala.Exporters.YAML import export_machine

@pytest.fixture
def simple_machine():
    outdir = f"{os.path.dirname(os.path.abspath(__file__))}/framework"
    m1 = Marker(
        name="M1",
        machine_area="FODO",
        hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    q1f = Quadrupole(
        name="QUAD1F",
        machine_area="FODO",
        magnetic={"length": 1.0, "k1l": -1},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}}
    )
    q1d = Quadrupole(
        name="QUAD1D",
        machine_area="FODO",
        magnetic={"length": 1.0, "k1l": 1.0},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}}
    )
    m3 = Marker(
        name="M3",
        machine_area="FODO",
        hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 4.0}}
    )
    sections = {"sections": {"FODO": ["M1", "QUAD1F", "QUAD1D", "M3"]}}
    layouts = {"default_layout": "line1", "layouts": {"line1": ["FODO"]}}
    machine = NALA(element_list=[m1, q1f, q1d, m3], layout=layouts, section=sections)
    export_machine(path=f"{outdir}/lattice", machine=machine, overwrite=True)
    return machine, outdir

@pytest.fixture
def simple_generator():
    gen = frameworkGenerator(
        global_parameters={"master_subdir": f"{os.path.dirname(os.path.abspath(__file__))}"},
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4,
        sigma_px=1e3,
        sigma_y=1e-4,
        sigma_py=1e3,
        sigma_z=1e-3,
        sigma_pz=1e3,
        gaussian_cutoff_x=3,
        gaussian_cutoff_y=3,
        gaussian_cutoff_z=3,
        gaussian_cutoff_px=3,
        gaussian_cutoff_py=3,
        gaussian_cutoff_pz=3,
        charge=100e-12,
    )
    return gen.write()

def test_framework_initialization(simple_machine):
    machine, outdir = simple_machine
    framework = fw.Framework(
        machine=machine,
        directory=os.path.join(outdir, "ocelot"),
        clean=True,
        verbose=True
    )
    assert framework.machine == machine
    assert framework.directory == os.path.join(outdir, "ocelot")

def test_framework_settings_and_tracking(simple_machine, simple_generator):
    machine, outdir = simple_machine
    gen = simple_generator
    settings = fw.FrameworkSettings()
    files = {}
    for sec, elems in machine.sections.items():
        files[sec] = {
            "code": "ocelot",
            "charge": {"space_charge_mode": "False"},
            "input": {
                "twiss": {
                    "beta_x": 3.2844606,
                    "alpha_x": 2.48956886,
                    "nemit_x": 1e-6,
                    "beta_y": 3.2846606,
                    "alpha_y": -2.48956886,
                    "nemit_y": 1e-6,
                }
            },
            "output": {
                "start_element": elems[0].name,
                "end_element": elems[-1].name,
            },
        }
    settings.files = files
    settings.layout = machine.layout
    settings.section = {"sections": {name: e.names for name, e in machine.sections.items()}}
    settings.element_list = os.path.join(outdir, "lattice")
    framework = fw.Framework(
        machine=machine,
        directory=os.path.join(outdir, "ocelot"),
        clean=True,
        verbose=True
    )
    framework.loadSettings(settings=settings)
    framework.save_settings("test.def")
    framework.loadSettings(filename="test.def")
    framework.global_parameters["beam"] = MagicMock()
    framework["FODO"].lsc_enable = False
    framework["FODO"].csr_enable = False
    framework.set_lattice_prefix("FODO", f"{os.path.dirname(os.path.abspath(__file__))}/")
    framework.track = MagicMock()
    framework.track()
    shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__))}/framework")
    os.remove(f"{os.path.dirname(os.path.abspath(__file__))}/M1.openpmd.hdf5")
    with pytest.raises(FileNotFoundError):
        framework.loadSettings(filename="non_existent.def")
    with pytest.raises(ValueError):
        framework.loadSettings()

@pytest.fixture
def sample_framework(tmp_path):
    fw_obj = fw.Framework(directory=str(tmp_path))
    e1 = Element(name="E1", hardware_class="Magnet", hardware_type="Dipole", machine_area="A1")
    e2 = Element(name="E2", hardware_class="Magnet", hardware_type="Quadrupole", machine_area="A1")
    fw_obj.elementObjects = {"E1": e1, "E2": e2}
    fw_obj.original_elementObjects = {"E1": e1.model_copy(deep=True), "E2": e2.model_copy(deep=True)}
    return fw_obj

@pytest.fixture
def framework_with_machine(simple_machine):
    machine, outdir = simple_machine
    settings = fw.FrameworkSettings()
    files = {}
    for sec, elems in machine.sections.items():
        files[sec] = {
            "code": "ocelot",
            "charge": {"space_charge_mode": "False"},
            "input": {
                "twiss": {
                    "beta_x": 3.2844606,
                    "alpha_x": 2.48956886,
                    "nemit_x": 1e-6,
                    "beta_y": 3.2846606,
                    "alpha_y": -2.48956886,
                    "nemit_y": 1e-6,
                }
            },
            "output": {
                "start_element": elems[0].name,
                "end_element": elems[-1].name,
            },
        }
    settings.files = files
    settings.layout = machine.layout
    settings.section = {"sections": {name: e.names for name, e in machine.sections.items()}}
    settings.element_list = os.path.join(outdir, "lattice")
    framework = fw.Framework(
        machine=machine,
        directory=os.path.join(outdir, "ocelot"),
        clean=True,
        verbose=True
    )
    framework.loadSettings(settings=settings)
    return framework

def test_getElement(sample_framework):
    fw_obj = sample_framework
    assert fw_obj.getElement("E1").name == "E1"
    assert fw_obj.getElement("E1", "hardware_type") == "Dipole"
    with pytest.warns(UserWarning):
        assert fw_obj.getElement("NonExistent") == {}

def test_getElementType(framework_with_machine):
    fw_obj = framework_with_machine
    quads = fw_obj.getElementType("Quadrupole")
    assert any(e["name"] == "QUAD1F" for e in quads)
    elements = fw_obj.getElementType(["Quadrupole", "Marker"])
    assert any(e["name"] == "QUAD1F" for e in elements[0])
    assert any(e["name"] == "M1" for e in elements[1])

def test_modifyElement(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElement("E1", "name", "NewE1")
    assert fw_obj.elementObjects["E1"].name == "NewE1"

def test_modifyElements(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElements(["E1", "E2"], "alias", "mag")
    assert all(e.alias == "mag" for e in fw_obj.elementObjects.values())

def test_modifyElementType(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElementType("Dipole", "machine_area", "new_area")
    assert fw_obj.elementObjects["E1"].machine_area == "new_area"

def test_detect_changes(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElement("E1", "machine_area", "new_area")
    changes = fw_obj.detect_changes()
    assert "E1" in changes
    assert "machine_area" in str(changes["E1"])

def test_detect_changes_single(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElement("E1", "machine_area", "new_area")
    changes = fw_obj.detect_changes(elements=["E1"])
    assert "E1" in changes
    assert "machine_area" in str(changes["E1"])

def test_detect_changes_by_type(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElement("E1", "machine_area", "new_area")
    changes = fw_obj.detect_changes(elementtype="Dipole")
    assert "E1" in changes
    assert "machine_area" in str(changes["E1"])

def test_save_and_load_changes_file(sample_framework, tmp_path):
    fw_obj = sample_framework
    fw_obj.modifyElement("E2", "virtual_name", "VE2")
    changes_file = tmp_path / "changes.yaml"
    fw_obj.save_changes_file(filename=str(changes_file))
    assert changes_file.exists()
    loaded_changes = fw_obj.load_changes_file(filename=str(changes_file), apply=False)
    assert "E2" in loaded_changes
    with pytest.raises(ValueError):
        fw_obj.save_changes_file()
    assert isinstance(fw_obj.save_changes_file(dictionary=True), dict)
    fw_obj_copy = sample_framework
    fw_obj_copy.apply_changes(fw_obj.save_changes_file(dictionary=True))

def test_clear(sample_framework):
    fw_obj = sample_framework
    fw_obj.clear()
    assert fw_obj.elementObjects == {}
    assert fw_obj.latticeObjects == {}
    assert fw_obj.commandObjects == {}
    assert fw_obj.groupObjects == {}

def test_change_subdirectory(sample_framework):
    fw_obj = sample_framework
    fw_obj.change_subdirectory(direc="./new_subdir")
    assert os.path.isdir(fw_obj.global_parameters["master_subdir"])
    assert fw_obj.subdirectory == os.path.abspath("./new_subdir")
    shutil.rmtree("./new_subdir")

def test_change_lattice_code(framework_with_machine):
    framework_with_machine.change_Lattice_Code("FODO", "elegant")
    assert isinstance(framework_with_machine.latticeObjects["FODO"], elegantLattice)
    framework_with_machine.change_Lattice_Code("All", "cheetah")
    assert isinstance(framework_with_machine.latticeObjects["FODO"], cheetahLattice)
    framework_with_machine.change_Lattice_Code(["FODO"], "astra")
    assert isinstance(framework_with_machine.latticeObjects["FODO"], astraLattice)
    shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__))}/framework")

def test_modify_lattices(framework_with_machine):
    framework_with_machine.modifyLattices("FODO", "lsc_enable", False)
    assert not framework_with_machine.latticeObjects["FODO"].lsc_enable
    framework_with_machine.modifyLattices(["FODO"], "csr_enable", False)
    assert not framework_with_machine.latticeObjects["FODO"].csr_enable
    shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__))}/framework")

def test_change_generator(framework_with_machine):
    framework_with_machine.add_Generator(code="astra")
    assert isinstance(framework_with_machine.latticeObjects["generator"], ASTRAGenerator)
    framework_with_machine.add_Generator(code="gpt")
    assert isinstance(framework_with_machine.latticeObjects["generator"], GPTGenerator)
    framework_with_machine.add_Generator(code="simba")
    assert isinstance(framework_with_machine.latticeObjects["generator"], frameworkGenerator)
    framework_with_machine.change_generator("gpt")
    assert isinstance(framework_with_machine.latticeObjects["generator"], GPTGenerator)
    framework_with_machine.change_generator("simba")
    assert isinstance(framework_with_machine.latticeObjects["generator"], frameworkGenerator)
    with pytest.raises(ValidationError):
        with pytest.warns(UserWarning):
            framework_with_machine.change_generator("none")
    framework_with_machine.change_generator("ASTRA")
    assert isinstance(framework_with_machine.latticeObjects["generator"], ASTRAGenerator)
    shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__))}/framework")
