"""Tests for Framework.replace_element and Framework/frameworkLattice.disable_screens,
ported from SimFrame (SimulationFramework.Framework)."""
import os

import pytest

import simba.Framework as fw
from laura.models.element import Quadrupole, Drift, Marker, Screen, Beam_Position_Monitor
from laura import LAURA


# ---------------------------------------------------------------------------
# replace_element
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_framework_with_quad(tmp_path):
    fw_obj = fw.Framework(directory=str(tmp_path))
    quad = Quadrupole(
        name="Q1",
        machine_area="A1",
        magnetic={"length": 0.3, "k1l": -1.5},
        physical={"length": 0.3, "middle": {"x": 0.0, "y": 0.0, "z": 1.0}},
    )
    fw_obj.elementObjects = {"Q1": quad}
    return fw_obj


def test_replace_element_changes_type(sample_framework_with_quad):
    fw_obj = sample_framework_with_quad
    new_elem = fw_obj.replace_element(name="Q1", type="Drift")
    assert isinstance(new_elem, Drift)
    assert new_elem.name == "Q1"
    assert fw_obj.elementObjects["Q1"] is new_elem


def test_replace_element_carries_over_shared_properties(sample_framework_with_quad):
    fw_obj = sample_framework_with_quad
    new_elem = fw_obj.replace_element(name="Q1", type="Drift")
    # `physical` is a shared field name/shape between Quadrupole and Drift.
    assert new_elem.physical.length == pytest.approx(0.3)
    assert new_elem.machine_area == "A1"


def test_replace_element_kwargs_override_original(sample_framework_with_quad):
    fw_obj = sample_framework_with_quad
    new_elem = fw_obj.replace_element(
        name="Q1", type="Drift", machine_area="A2"
    )
    assert new_elem.machine_area == "A2"


def test_replace_element_uses_name_from_kwargs(sample_framework_with_quad):
    fw_obj = sample_framework_with_quad
    new_elem = fw_obj.replace_element(type="Drift", name="Q1")
    assert new_elem.name == "Q1"


def test_replace_element_raises_without_name(sample_framework_with_quad):
    fw_obj = sample_framework_with_quad
    with pytest.raises(NameError):
        fw_obj.replace_element(type="Drift")


def test_replace_element_propagates_into_lattice_objects(sample_framework_with_quad):
    fw_obj = sample_framework_with_quad

    class MockLattice:
        def __init__(self, elementObjects):
            self.elementObjects = elementObjects

    fw_obj.latticeObjects["L1"] = MockLattice({"Q1": fw_obj.elementObjects["Q1"]})
    new_elem = fw_obj.replace_element(name="Q1", type="Drift")
    assert fw_obj.latticeObjects["L1"].elementObjects["Q1"] is new_elem


# ---------------------------------------------------------------------------
# Framework.disable_screens (delegation)
# ---------------------------------------------------------------------------

def test_framework_disable_screens_delegates_to_lattices(tmp_path):
    fw_obj = fw.Framework(directory=str(tmp_path))

    class MockLattice:
        def __init__(self):
            self.calls = []

        def disable_screens(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    latt = MockLattice()
    fw_obj.latticeObjects["L1"] = latt
    fw_obj.disable_screens("all", disable=True)
    assert latt.calls == [(("all",), {"disable": True})]


def test_framework_disable_screens_swallows_attribute_error(tmp_path):
    fw_obj = fw.Framework(directory=str(tmp_path))

    class MockLatticeNoDisable:
        pass

    fw_obj.latticeObjects["L1"] = MockLatticeNoDisable()
    # Should not raise, even though MockLatticeNoDisable has no disable_screens.
    fw_obj.disable_screens("all")


# ---------------------------------------------------------------------------
# frameworkLattice.disable_screens (real lattice, filtering logic)
# ---------------------------------------------------------------------------

@pytest.fixture
def framework_with_screens(tmp_path):
    outdir = str(tmp_path)
    m1 = Marker(
        name="M1", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    q1f = Quadrupole(
        name="QUAD1F", machine_area="FODO",
        magnetic={"length": 1.0, "k1l": -1},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
    )
    scr1 = Screen(
        name="SCR1", machine_area="FODO",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 1.5}},
    )
    bpm1 = Beam_Position_Monitor(
        name="BPM1", machine_area="FODO",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )
    q1d = Quadrupole(
        name="QUAD1D", machine_area="FODO",
        magnetic={"length": 1.0, "k1l": 1.0},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
    )
    m3 = Marker(
        name="M3", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
    )
    sections = {"sections": {"FODO": ["M1", "QUAD1F", "SCR1", "BPM1", "QUAD1D", "M3"]}}
    layouts = {"default_layout": "line1", "layouts": {"line1": ["FODO"]}}
    machine = LAURA(
        element_list=[m1, q1f, scr1, bpm1, q1d, m3], layout=layouts, section=sections
    )
    from laura.Exporters.YAML import export_machine

    export_machine(path=f"{outdir}/lattice", machine=machine, overwrite=True)

    settings = fw.FrameworkSettings()
    files = {}
    for sec, elems in machine.sections.items():
        files[sec] = {
            "code": "ocelot",
            "charge": {"space_charge_mode": "False"},
            "input": {
                "twiss": {
                    "beta_x": 3.2844606, "alpha_x": 2.48956886, "nemit_x": 1e-6,
                    "beta_y": 3.2846606, "alpha_y": -2.48956886, "nemit_y": 1e-6,
                }
            },
            "output": {"start_element": elems[0].name, "end_element": elems[-1].name},
        }
    settings.files = files
    settings.layout = machine.layout
    settings.section = {"sections": {name: e.names for name, e in machine.sections.items()}}
    settings.element_list = os.path.join(outdir, "lattice")

    framework = fw.Framework(machine=machine, directory=os.path.join(outdir, "ocelot"), clean=True)
    framework.loadSettings(settings=settings)
    return framework


def test_disable_screens_all(framework_with_screens):
    lattice = framework_with_screens["FODO"]
    lattice.disable_screens("all")
    assert lattice.disabled_elements == {"M1", "SCR1", "BPM1", "M3"}


def test_disable_screens_specific_names(framework_with_screens):
    lattice = framework_with_screens["FODO"]
    lattice.disable_screens(["SCR1"])
    assert lattice.disabled_elements == {"SCR1"}


def test_disable_screens_excludes(framework_with_screens):
    lattice = framework_with_screens["FODO"]
    lattice.disable_screens("all", excludes=["BPM1"])
    assert "BPM1" not in lattice.disabled_elements
    assert "SCR1" in lattice.disabled_elements


def test_disable_screens_re_enable(framework_with_screens):
    lattice = framework_with_screens["FODO"]
    lattice.disable_screens("all")
    lattice.disable_screens(["SCR1"], disable=False)
    assert "SCR1" not in lattice.disabled_elements
    assert "BPM1" in lattice.disabled_elements


def test_framework_disable_screens_reaches_real_lattice(framework_with_screens):
    framework_with_screens.disable_screens("all")
    assert framework_with_screens["FODO"].disabled_elements == {"M1", "SCR1", "BPM1", "M3"}


def test_disable_screens_excludes_element_from_section(framework_with_screens):
    """The real point of disabling: the element must vanish from `section`
    (and therefore from every per-code writer's output), not just be flagged."""
    lattice = framework_with_screens["FODO"]
    assert "SCR1" in lattice.section.elements.elements
    lattice.disable_screens(["SCR1"])
    assert "SCR1" not in lattice.section.elements.elements
    assert "SCR1" not in lattice.elements
    # Everything else is untouched.
    assert "BPM1" in lattice.section.elements.elements
    assert "QUAD1F" in lattice.section.elements.elements


def test_disable_screens_re_enable_restores_section(framework_with_screens):
    lattice = framework_with_screens["FODO"]
    lattice.disable_screens(["SCR1"])
    assert "SCR1" not in lattice.section.elements.elements
    lattice.disable_screens(["SCR1"], disable=False)
    assert "SCR1" in lattice.section.elements.elements
