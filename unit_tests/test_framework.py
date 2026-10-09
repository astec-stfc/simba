import os
import pytest
import shutil
import types
from pydantic import ValidationError
import simba.Framework as fw
from simba.Framework_objects import chicane
from simba.Codes.Generators import (
    frameworkGenerator,
    ASTRAGenerator,
    GPTGenerator,
)
from simba.Framework_lattices import (
    elegantLattice,
    astraLattice,
    cheetahLattice,
    bmadLattice,
)
from laura.models.element import Dipole, Quadrupole, Marker, PhysicalBaseElement
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine

@pytest.fixture
def simple_machine(tmp_path):
    # not beside this file: under xdist, one test's rmtree raced another's
    # writes, and left unit_tests/framework behind
    outdir = str(tmp_path / "framework")
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
    machine = LAURA(element_list=[m1, q1f, q1d, m3], layout=layouts, section=sections)
    export_machine(path=f"{outdir}/lattice", machine=machine, overwrite=True)
    return machine, outdir

@pytest.fixture
def simple_generator(tmp_path):
    gen = frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
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
    gen.write()
    return tmp_path

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

@pytest.fixture
def sample_framework(tmp_path):
    fw_obj = fw.Framework(directory=str(tmp_path))
    e1 = PhysicalBaseElement(name="E1", hardware_class="Magnet", hardware_type="Dipole", machine_area="A1")
    e2 = PhysicalBaseElement(name="E2", hardware_class="Magnet", hardware_type="Quadrupole", machine_area="A1")
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

def test_framework_settings_and_tracking(framework_with_machine, simple_generator):
    framework = framework_with_machine
    framework.save_settings("test.def", directory=str(simple_generator))
    framework.loadSettings(filename=str(simple_generator / "test.def"))
    framework["FODO"].lsc_enable = False
    framework["FODO"].csr_enable = False
    framework.set_lattice_prefix("FODO", f"{simple_generator}/")
    framework.track()
    assert os.path.isfile(os.path.join(framework.directory, "M3.openpmd.hdf5"))
    with pytest.raises(FileNotFoundError):
        framework.loadSettings(filename="non_existent.def")
    with pytest.raises(ValueError):
        framework.loadSettings()

def test_getElement(sample_framework):
    fw_obj = sample_framework
    assert fw_obj.getElement("E1").name == "E1"
    assert fw_obj.getElement("E1", "hardware_type") == "Dipole"
    with pytest.warns(UserWarning):
        assert fw_obj.getElement("NonExistent") == {}

def test_original_elements_are_copies_sharing_one_trajectory(framework_with_machine):
    """Copied per element, each took the section trajectory: CLIC DR needed 8.8 GB."""
    fw_obj = framework_with_machine
    names = [n for n in fw_obj.elementObjects if fw_obj.elementObjects[n].physical._trajectory]
    assert len(names) > 1
    originals = [fw_obj.original_elementObjects[n] for n in names]
    assert all(o is not fw_obj.elementObjects[n] for o, n in zip(originals, names))
    assert len({id(o.physical._trajectory) for o in originals}) == 1
    assert originals[0].physical._trajectory is not fw_obj.elementObjects[names[0]].physical._trajectory


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


def test_check_lattice_after_chicane_angle_change(sample_framework):
    dipoles = {
        f"D{index + 1}": Dipole(
            name=f"D{index + 1}",
            machine_area="A1",
            magnetic={"length": 0.2, "angle": 0.0},
            physical={"length": 0.2, "middle": {"x": 0, "y": 0, "z": index}},
        )
        for index in range(4)
    }
    sample_framework.elementObjects = dipoles

    chicane(
        "bunch_compressor",
        sample_framework,
        "chicane",
        list(dipoles),
    ).set_angle(0.1)

    assert sample_framework.check_lattice()


def test_check_lattice_catches_a_magnet_the_layout_was_not_told_about(sample_framework):
    """A field change without a move leaves tracking and layout disagreeing."""
    def dipole(layout_angle):
        return Dipole(
            name="D1",
            machine_area="A1",
            magnetic={"length": 0.2, "angle": 0.1},
            physical={
                "length": 0.2,
                "middle": {"x": 0, "y": 0, "z": 1.0},
                "physical_angle": layout_angle,
            },
        )

    sample_framework.elementObjects = {"D1": dipole(0.1)}
    assert sample_framework.check_lattice()

    sample_framework.elementObjects = {"D1": dipole(0.13)}
    assert not sample_framework.check_lattice()


def test_modifyElements(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElements(["E1", "E2"], "alias", "mag")
    assert all(e.alias == ["mag"] for e in fw_obj.elementObjects.values())

def test_modifyElementType(sample_framework):
    fw_obj = sample_framework
    fw_obj.modifyElementType("Dipole", "machine_area", "new_area")
    assert fw_obj.elementObjects["E1"].machine_area == "new_area"

@pytest.mark.parametrize(
    "kwargs", [{}, {"elements": ["E1"]}, {"elementtype": "Dipole"}], ids=["all", "single", "by_type"]
)
def test_detect_changes(sample_framework, kwargs):
    fw_obj = sample_framework
    fw_obj.modifyElement("E1", "machine_area", "new_area")
    changes = fw_obj.detect_changes(**kwargs)
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
    framework_with_machine.change_Lattice_Code("FODO", "bmad")
    assert isinstance(framework_with_machine.latticeObjects["FODO"], bmadLattice)
    framework_with_machine.change_Lattice_Code("All", "cheetah")
    assert isinstance(framework_with_machine.latticeObjects["FODO"], cheetahLattice)
    framework_with_machine.change_Lattice_Code(["FODO"], "astra")
    assert isinstance(framework_with_machine.latticeObjects["FODO"], astraLattice)


def test_modify_lattices(framework_with_machine):
    framework_with_machine.modifyLattices("FODO", "lsc_enable", False)
    assert not framework_with_machine.latticeObjects["FODO"].lsc_enable
    framework_with_machine.modifyLattices(["FODO"], "csr_enable", False)
    assert not framework_with_machine.latticeObjects["FODO"].csr_enable

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
    with pytest.raises(ValidationError), pytest.warns(UserWarning):
        framework_with_machine.change_generator("none")
    framework_with_machine.change_generator("ASTRA")
    assert isinstance(framework_with_machine.latticeObjects["generator"], ASTRAGenerator)


def test_modify_lattice_with_lists(framework_with_machine):
    framework_with_machine.modifyLattice("FODO", ["lsc_enable", "csr_enable"], [False, False])
    assert not framework_with_machine.latticeObjects["FODO"].lsc_enable
    assert not framework_with_machine.latticeObjects["FODO"].csr_enable


def test_detect_changes_generator_reports_only_changed_fields(sample_framework, tmp_path):
    gen = frameworkGenerator(global_parameters={"master_subdir": str(tmp_path)}, charge=1e-12)
    sample_framework.generator = gen
    sample_framework.original_elementObjects["generator"] = gen.model_copy(deep=True)
    gen.charge = 2e-12
    assert sample_framework.detect_changes(elements=["generator"]) == {"generator": {"charge": 2e-12}}


def test_framework_directory_element_prints_laura_element(sample_framework):
    directory = types.SimpleNamespace(framework=sample_framework)
    assert fw.frameworkDirectory.element(directory, "E1").name == "E1"


def test_yaml_checker_builds_a_framework(tmp_path, monkeypatch):
    from simba import yaml_checker
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        yaml_checker.main([str(tmp_path / "missing.def")])
