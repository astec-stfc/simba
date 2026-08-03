import numpy as np
import pytest
from pydantic import ValidationError

from simba.Framework_objects import (
    runSetup,
    frameworkObject,
    frameworkCommand,
    frameworkCounter,
    getGrids,
)


# --- runSetup -----------------------------------------------------------

def test_run_setup_defaults():
    rs = runSetup()
    assert rs.nruns == 1
    assert rs.seed == 0
    assert rs.elementErrors is None
    assert rs.elementScan is None


def test_set_n_runs_accepts_int_and_float():
    rs = runSetup()
    rs.setNRuns(5)
    assert rs.nruns == 5
    rs.setNRuns(3.7)
    assert rs.nruns == 3


def test_set_n_runs_rejects_bad_type():
    rs = runSetup()
    with pytest.raises(TypeError):
        rs.setNRuns("bad")


def test_set_seed_value():
    rs = runSetup()
    rs.setSeedValue(42.9)
    assert rs.seed == 42


def test_set_seed_value_rejects_bad_type():
    rs = runSetup()
    with pytest.raises(TypeError):
        rs.setSeedValue("bad")


def test_load_element_errors_from_dict():
    rs = runSetup()
    rs.loadElementErrors({"elements": {"Q1": {"sigma": 0.1}}, "nruns": 5, "seed": 42})
    assert rs.elementErrors == {"Q1": {"sigma": 0.1}}
    assert rs.elementScan is None
    assert rs.nruns == 5
    assert rs.seed == 42


def test_load_element_errors_from_yaml_file(tmp_path):
    p = tmp_path / "errors.yaml"
    p.write_text("elements:\n  Q1:\n    sigma: 0.1\n")
    rs = runSetup()
    rs.loadElementErrors(str(p))
    assert rs.elementErrors == {"Q1": {"sigma": 0.1}}


def test_load_element_errors_bad_type_warns():
    rs = runSetup()
    with pytest.warns(UserWarning):
        rs.loadElementErrors(12345)
    assert rs.elementErrors is None


def test_set_element_scan():
    rs = runSetup()
    rs.setElementScan("Q1", "k1l", [0.0, 1.0], multiplicative=True)
    assert rs.elementScan == {
        "name": "Q1",
        "item": "k1l",
        "min": 0.0,
        "max": 1.0,
        "multiplicative": True,
    }
    assert rs.elementErrors is None


def test_set_element_scan_accepts_ndarray_range():
    rs = runSetup()
    rs.setElementScan("Q1", "k1l", np.array([0.0, 2.0]))
    assert rs.elementScan["min"] == 0.0
    assert rs.elementScan["max"] == 2.0


def test_set_element_scan_rejects_non_string_name_or_item():
    rs = runSetup()
    with pytest.raises(TypeError):
        rs.setElementScan(1, "k1l", [0.0, 1.0])


def test_set_element_scan_rejects_bad_range():
    rs = runSetup()
    with pytest.raises(TypeError):
        rs.setElementScan("Q1", "k1l", [0.0, 1.0, 2.0])


def test_set_element_scan_rejects_non_bool_multiplicative():
    rs = runSetup()
    with pytest.raises(ValueError):
        rs.setElementScan("Q1", "k1l", [0.0, 1.0], multiplicative="notabool")


# --- frameworkObject ------------------------------------------------------

def test_framework_object_construction_sets_allowed_keywords():
    obj = frameworkObject(name="M1", type="marker")
    assert obj.objectname == "M1"
    assert obj.objecttype == "marker"
    assert "subelement" in obj.allowedkeywords


def test_framework_object_unknown_type_raises_name_error():
    with pytest.raises(NameError):
        frameworkObject(name="M1", type="not_a_real_type")


def test_framework_object_non_string_name_raises_validation_error():
    with pytest.raises(ValidationError):
        frameworkObject(name=123, type="marker")


def test_framework_object_non_string_type_raises_validation_error():
    with pytest.raises(ValidationError):
        frameworkObject(name="M1", type=123)


def test_add_property_sets_allowed_keyword():
    obj = frameworkObject(name="M1", type="marker")
    obj.add_property("subelement", True)
    assert obj.subelement is True


def test_add_property_ignores_disallowed_keyword():
    obj = frameworkObject(name="M1", type="marker")
    obj.add_property("totally_not_a_keyword", 5)
    assert not hasattr(obj, "totally_not_a_keyword")


def test_add_properties_sets_multiple_allowed_keywords():
    obj = frameworkObject(name="M1", type="marker")
    obj.add_properties(subelement=True, totally_not_a_keyword=5)
    assert obj.subelement is True
    assert not hasattr(obj, "totally_not_a_keyword")


def test_add_default():
    obj = frameworkObject(name="M1", type="marker")
    obj.add_default("subelement", False)
    assert obj.objectdefaults == {"subelement": False}


def test_parameters_lists_object_properties():
    obj = frameworkObject(name="M1", type="marker")
    assert set(obj.parameters) == set(obj.objectproperties.keys())
    assert "objectname" in obj.parameters


def test_object_properties_contains_declared_fields():
    obj = frameworkObject(name="M1", type="marker")
    props = obj.objectproperties
    assert props["objectname"] == "M1"
    assert props["objecttype"] == "marker"


def test_repr_lists_set_allowed_keywords():
    obj = frameworkObject(name="M1", type="marker")
    obj.add_property("subelement", True)
    assert "subelement = True" in repr(obj)


# --- frameworkCounter -----------------------------------------------------

def test_framework_counter_peeks_next_value_without_mutating():
    # counter() doesn't store anything in the dict itself -- it just reports what
    # the *next* count would be (1 while unset, else current value + 1).
    fc = frameworkCounter()
    assert fc.counter("quadrupole") == 1
    assert fc.counter("quadrupole") == 1
    fc.add("quadrupole")
    assert fc.counter("quadrupole") == 2


def test_framework_counter_uses_sub_mapping():
    fc = frameworkCounter(sub={"quad": "quadrupole"})
    fc.add("quad")
    assert fc.counter("quadrupole") == 2
    assert fc.value("quad") == 1


def test_framework_counter_value_defaults_to_one():
    fc = frameworkCounter()
    assert fc.value("quadrupole") == 1
    fc.counter("quadrupole")
    assert fc.value("quadrupole") == 1


def test_framework_counter_add():
    fc = frameworkCounter()
    assert fc.add("quadrupole", 3) == 3
    assert fc.add("quadrupole", 2) == 5


def test_framework_counter_subtract():
    fc = frameworkCounter()
    assert fc.subtract("quadrupole") == 0
    fc.add("quadrupole", 3)
    assert fc.subtract("quadrupole") == 2


# --- getGrids ---------------------------------------------------------

def test_get_grid_sizes_rounds_to_nearest_power_of_8():
    g = getGrids()
    assert g.getGridSizes(1000) == 8  # cube root of 1000 = 10, nearest power-of-8 is 8
    assert g.getGridSizes(1) == 4  # floor is enforced at 4


def test_find_nearest():
    g = getGrids()
    assert g.find_nearest(np.array([2, 4, 8, 16]), 10) == 8


# --- frameworkCommand ------------------------------------------------------

def test_framework_command_unknown_type_raises_name_error():
    with pytest.raises(NameError):
        frameworkCommand(name="c1", type="not_a_real_command")


def test_write_elegant():
    cmd = frameworkCommand(name="cp1", type="change_particle", mass_ratio=1.0, charge_ratio=2.0)
    output = cmd.write_Elegant()
    assert output.startswith("&change_particle\n")
    assert "mass_ratio = 1.0" in output
    assert "charge_ratio = 2.0" in output
    assert output.endswith("&end\n")


def test_write_mad8():
    cmd = frameworkCommand(name="cp1", type="change_particle", mass_ratio=1.0)
    # write_MAD8 only ever emits declared frameworkObject fields (not the extra
    # "allowed keyword" attributes), so command-specific params never appear here;
    # it's marked "# TODO deprecated?" in-source and untouched by this test.
    assert cmd.write_MAD8() == "change_particle;\n"


def test_write_genesis():
    cmd = frameworkCommand(name="ab1", type="alter_beam", dgamma=0.5)
    output = cmd.write_Genesis()
    assert output == "&alter_beam\n\tdgamma = 0.5\n&end\n"
