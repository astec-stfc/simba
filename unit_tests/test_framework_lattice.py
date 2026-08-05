import importlib
import inspect
import pkgutil
import re

import pytest
import simba.Framework as sfw
import simba.Codes
from simba.Codes.ASTRA.ASTRA import astra_newrun
from simba.Framework_objects import frameworkLattice
from laura.models.element import Quadrupole, Marker, Element

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
    with pytest.warns(UserWarning):
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

def _all_lattice_classes():
    for m in pkgutil.walk_packages(simba.Codes.__path__, "simba.Codes."):
        try:
            importlib.import_module(m.name)
        except Exception:
            pass
    todo, seen = list(frameworkLattice.__subclasses__()), []
    while todo:
        cls = todo.pop()
        if cls not in seen:
            seen.append(cls)
            todo.extend(cls.__subclasses__())
    return seen

def test_no_property_setter_shadowed_by_inherited_field():
    """A property whose name is already a pydantic field on the parent is silently dropped
    by the metaclass, so assignment writes the field and the setter never runs. This broke
    astraLattice.sample_interval (n_red stayed 1 however it was set)."""
    broken = {}
    for cls in _all_lattice_classes():
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        setters = set(re.findall(r"@(\w+)\.setter", src))
        lost = sorted(p for p in setters if p not in cls.__dict__ and p in cls.model_fields)
        if lost:
            broken[cls.__name__] = lost
    assert not broken, f"property setters dropped by pydantic: {broken}"

@pytest.mark.parametrize("interval", [1, 8, 64])
def test_astra_sample_interval_written_as_n_red(interval):
    """astraLattice.preProcess copies sample_interval into this header; guard the mapping."""
    header = astra_newrun(
        global_parameters={},
        input_particle_definition="test.astra",
        sample_interval=interval,
    )
    assert f"n_red = {interval}," in header.write_ASTRA()
