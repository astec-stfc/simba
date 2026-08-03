import pytest

import simba.Framework as fw
from laura.models.element import Quadrupole, Marker
from laura import LAURA
from laura.Exporters.YAML import export_machine


@pytest.fixture
def fodo_lattice(tmp_path):
    """A real `ocelotLattice` (frameworkLattice subclass) for a 4-element FODO line."""
    m1 = Marker(
        name="M1",
        machine_area="FODO",
        hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    q1f = Quadrupole(
        name="QUAD1F",
        machine_area="FODO",
        magnetic={"length": 1.0, "k1l": -1},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
    )
    q1d = Quadrupole(
        name="QUAD1D",
        machine_area="FODO",
        magnetic={"length": 1.0, "k1l": 1.0},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
    )
    m3 = Marker(
        name="M3",
        machine_area="FODO",
        hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
    )
    sections = {"sections": {"FODO": ["M1", "QUAD1F", "QUAD1D", "M3"]}}
    layouts = {"default_layout": "line1", "layouts": {"line1": ["FODO"]}}
    machine = LAURA(element_list=[m1, q1f, q1d, m3], layout=layouts, section=sections)
    export_machine(path=str(tmp_path / "lattice"), machine=machine, overwrite=True)

    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": "ocelot",
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
        }
    }
    settings.layout = machine.layout
    settings.section = sections
    settings.element_list = str(tmp_path / "lattice")

    framework = fw.Framework(machine=machine, directory=str(tmp_path / "ocelot"), clean=True)
    framework.loadSettings(settings=settings)
    return framework["FODO"]


def test_quadrupoles_property(fodo_lattice):
    names = {q.name for q in fodo_lattice.quadrupoles}
    assert names == {"QUAD1F", "QUAD1D"}


@pytest.mark.parametrize(
    "prop", ["cavities", "solenoids", "dipoles", "kickers", "wakefields"]
)
def test_empty_type_properties(fodo_lattice, prop):
    assert getattr(fodo_lattice, prop) == []


def test_get_element_type_returns_names(fodo_lattice):
    names = fodo_lattice.getElementType("quadrupole", param="name")
    assert set(names) == {"QUAD1F", "QUAD1D"}


def test_get_element_type_with_list_of_types(fodo_lattice):
    quads, markers = fodo_lattice.getElementType(["quadrupole", "marker"])
    assert {q.name for q in quads} == {"QUAD1F", "QUAD1D"}
    assert {m.name for m in markers} == {"M1", "M3"}


def test_get_element_type_with_list_of_params_zips(fodo_lattice):
    zipped = list(fodo_lattice.getElementType("quadrupole", param=["name", "hardware_type"]))
    assert set(zipped) == {("QUAD1F", "Quadrupole"), ("QUAD1D", "Quadrupole")}


def test_set_element_type_updates_values(fodo_lattice):
    fodo_lattice.setElementType("quadrupole", "virtual_name", ["NQ1", "NQ2"])
    assert {q.virtual_name for q in fodo_lattice.quadrupoles} == {"NQ1", "NQ2"}


def test_set_element_type_raises_on_length_mismatch(fodo_lattice):
    with pytest.raises(ValueError):
        fodo_lattice.setElementType("quadrupole", "virtual_name", ["ONLY_ONE"])
