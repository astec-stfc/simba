import types

import pytest
from laura.models.element import (
    Combined_Corrector, Dipole, Horizontal_Corrector, PhysicalBaseElement, Quadrupole, RFCavity,
)
from laura.models.physical import PhysicalElement
from laura.translator.utils.fields import FieldMap

from simba.Framework_objects import frameworkGroup, frameworkLattice, r56_group
from simba.FrameworkHelperFunctions import convert_numpy_types, convert_outputs


def _element(z, length):
    return PhysicalBaseElement(
        name="E", hardware_class="Magnet", hardware_type="Quadrupole", machine_area="A",
        physical=PhysicalElement(middle=[0, 0, z + length / 2.0], length=length),
    )


def test_end_from_zstop():
    latt = types.SimpleNamespace(
        file_block={"output": {"zstop": 2.0}},
        elementObjects={"A": _element(0.0, 1.0), "B": _element(1.0, 1.0), "C": _element(2.0, 1.0)},
    )
    assert frameworkLattice.end.fget(latt) == "B"


def test_lattice_repr_is_a_string():
    latt = types.SimpleNamespace(name="L", elements={"A": None, "B": None})
    latt.__str__ = lambda: frameworkLattice.__str__(latt)
    assert frameworkLattice.__repr__(latt) == "L = (A, B, )"


def test_lattice_set_element_type():
    q1 = Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A")
    d1 = Dipole(name="D1", hardware_class="Magnet", machine_area="A")
    frameworkLattice.setElementType(types.SimpleNamespace(getElementType=lambda typ: [q1]), "quadrupole", "virtual_name", ["V"])
    assert q1.virtual_name == "V"
    frameworkLattice.setElementType(types.SimpleNamespace(getElementType=lambda typ: [d1]), "dipole", "angle", [0.1])
    assert d1.magnetic.multipoles.K0L.normal == pytest.approx(0.1)


def test_wakefields_and_cavity_wakefields_skips_cavities_without_wakes():
    with_wake = RFCavity(name="C1", hardware_class="RF", machine_area="A")
    with_wake.simulation.wakefield_definition = "wake.sdds"
    without = RFCavity(name="C2", hardware_class="RF", machine_area="A")
    latt = types.SimpleNamespace(cavities=[with_wake, without], getElementType=lambda typ: [])
    assert frameworkLattice.wakefields_and_cavity_wakefields.fget(latt) == [with_wake]


def test_type_properties_match_laura_hardware_types():
    elements = {
        "C1": RFCavity(name="C1", hardware_class="RF", machine_area="A"),
        "H1": Horizontal_Corrector(name="H1", machine_area="A"),
        "K1": Combined_Corrector(name="K1", machine_area="A"),
        "Q1": Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A"),
    }
    latt = types.SimpleNamespace(elements=elements)
    latt.getElementType = types.MethodType(frameworkLattice.getElementType, latt)
    assert [e.name for e in frameworkLattice.cavities.fget(latt)] == ["C1"]
    assert sorted(e.name for e in frameworkLattice.kickers.fget(latt)) == ["H1", "K1"]


def test_convert_numpy_types_dumps_laura_fieldmap():
    assert isinstance(convert_numpy_types(FieldMap()), dict)


def _framework(*names):
    elements = {n: Quadrupole(name=n, hardware_class="Magnet", machine_area="A") for n in names}
    return types.SimpleNamespace(elementObjects=elements, groupObjects={})


def test_group_get_parameter_prefers_group_attribute():
    group = frameworkGroup("G", _framework("Q1"), "element_group", ["Q1"])
    group.custom = 5
    assert group.get_Parameter("custom") == 5
    assert group.get_Parameter("machine_area") == "A"


def test_r56_group_update_elements_with_a_list():
    fw = _framework("Q1", "Q2", "Q3")
    group = r56_group("R", fw, "r56_group", ["Q1", ["Q2", "Q3"]], ratios=[], keys=[])
    group.updateElements(["Q2", "Q3"], "virtual_name", "V")
    assert [fw.elementObjects[n].virtual_name for n in ("Q1", "Q2", "Q3")] == ["", "V", "V"]


@pytest.mark.parametrize("workers", [1, 3])
def test_convert_outputs_converts_every_item(tmp_path, workers):
    names = [f"S{i}" for i in range(7)]
    convert_outputs(lambda n: (tmp_path / n).write_text(n), names, workers)
    assert sorted(p.name for p in tmp_path.iterdir()) == names


def test_convert_outputs_raises_a_workers_error(tmp_path):
    def convert(n):
        if n == "BAD":
            raise ValueError("no beam file for BAD")
    with pytest.raises(ValueError, match="BAD"):
        convert_outputs(convert, ["S1", "BAD", "S2"], workers=2)
