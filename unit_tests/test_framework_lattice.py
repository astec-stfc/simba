import pytest
import simba.Framework as sfw
from nala.models.element import Quadrupole, Marker, Element

@pytest.fixture
def framework_with_elements(tmp_path):
    fw_obj = sfw.Framework(directory=str(tmp_path))
    quad1 = Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A1")
    quad2 = Quadrupole(name="Q2", hardware_class="Magnet", machine_area="A1")
    marker = Element(name="M1", hardware_class="Marker", hardware_type="Marker", machine_area="A1")
    fw_obj.elementObjects = {"Q1": quad1, "Q2": quad2, "M1": marker}
    return fw_obj

def test_get_element_returns_full_object(framework_with_elements):
    fw = framework_with_elements
    elem = fw.getElement("Q1")
    assert isinstance(elem, Quadrupole)
    assert elem.name == "Q1"

def test_get_element_returns_specific_param(framework_with_elements):
    fw = framework_with_elements
    param = fw.getElement("Q1", "hardware_class")
    assert param == "Magnet"

def test_get_element_nonexistent_returns_empty_dict(framework_with_elements):
    fw = framework_with_elements
    result = fw.getElement("NON_EXISTENT")
    assert result == {}

def test_get_element_type_returns_all(framework_with_elements):
    fw = framework_with_elements
    quads = fw.getElementType("Quadrupole")
    names = [e["name"] for e in quads]
    assert set(names) == {"Q1", "Q2"}

def test_get_element_type_with_param(framework_with_elements):
    fw = framework_with_elements
    classes = fw.getElementType("Quadrupole", param="hardware_class")
    assert all(c == "Magnet" for c in classes)

def test_set_element_type_updates_values(framework_with_elements):
    fw = framework_with_elements
    fw.setElementType("Quadrupole", "virtual_name", ["NewQ1", "NewQ2"])
    assert fw.elementObjects["Q1"].virtual_name == "NewQ1"
    assert fw.elementObjects["Q2"].virtual_name == "NewQ2"

def test_set_element_type_raises_on_length_mismatch(framework_with_elements):
    fw = framework_with_elements
    with pytest.raises(ValueError):
        fw.setElementType("Quadrupole", "virtual_name", ["OnlyOne"])

def test_set_lattice_prefix_sets_prefix(framework_with_elements):
    class MockLattice:
        def __init__(self):
            self.prefix = None
        def set_prefix(self, p):
            self.prefix = p
    fw = framework_with_elements
    fw.latticeObjects["L1"] = MockLattice()
    fw.set_lattice_prefix("L1", "prefix_value")
    assert fw.latticeObjects["L1"].prefix == "prefix_value"