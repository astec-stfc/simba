import pytest

import simba.Framework as fw
from simba.Framework_objects import global_error
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


def test_insert_element_inserts_into_element_objects_in_order(fodo_lattice):
    lattice = fodo_lattice
    original_order = list(lattice.elementObjects.keys())
    new_marker = Marker(
        name="M2",
        machine_area="FODO",
        hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )

    lattice.insert_element(2, new_marker)

    expected_order = original_order[:2] + ["M2"] + original_order[2:]
    assert list(lattice.elementObjects.keys()) == expected_order
    assert lattice.elementObjects["M2"] is new_marker
    assert lattice.allElements == expected_order


def test_end_property_with_zstop_finds_matching_element(fodo_lattice):
    lattice = fodo_lattice
    # QUAD1D ends at z=3.75 (middle 3.25 + length/2); M3 is a zero-length marker at z=4.0.
    lattice.file_block["output"] = {"zstop": 3.75}
    assert lattice.end == "QUAD1D"


def test_end_property_with_zstop_falls_back_to_first_element_past_stop(fodo_lattice):
    lattice = fodo_lattice
    # Only M3 (end.z=4.0) ends past 3.9, so the "first element past zstop" fallback
    # branch must pick it regardless of elementObjects' iteration order.
    lattice.file_block["output"] = {"zstop": 3.9}
    assert lattice.end == "M3"


def test_global_error_still_constructs_after_dead_code_removal():
    err = global_error(
        objectname="TEST_global_error",
        objecttype="global_error",
        global_parameters={"master_subdir": ".", "master_lattice": "."},
    )
    assert err.objectname == "TEST_global_error"
    assert not hasattr(err, "_write_ASTRA")
    assert not hasattr(err, "_write_GPT")
    assert not hasattr(err, "add_Error")
