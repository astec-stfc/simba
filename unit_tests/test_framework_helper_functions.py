import os
import types
import numpy as np
import pytest
from pydantic import BaseModel

import simba.FrameworkHelperFunctions as fh


def test_read_and_save_file(tmp_path):
    p = tmp_path / "f.txt"
    fh.saveFile(str(p), lines=["a\n", "b\n"])
    assert fh.readFile(str(p)) == ["a\n", "b\n"]


def test_find_setting_and_value():
    dictionary = {
        "e1": {"type": "quad", "value": 1},
        "e2": {"type": "quad", "value": 2},
        "e3": {"type": "dipole", "value": 1},
    }
    found = fh.findSetting("type", "quad", dictionary)
    assert [f[0] for f in found] == ["e1", "e2"]
    assert fh.findSettingValue("value", dictionary) == []


def test_line_replace_and_replace_string_scalar():
    line = "before $FOO$ after"
    assert fh.lineReplaceFunction(line, "FOO", "bar") == "before bar after"
    assert fh.lineReplaceFunction(line, "MISSING", "bar") == line

    lines = ["$FOO$", "no match", "$FOO$"]
    assert fh.replaceString(lines, "FOO", "bar") == ["bar", "no match", "bar"]


def test_replace_string_list():
    lines = ["$FOO$", "$FOO$"]
    assert fh.replaceString(lines, "FOO", ["one", "two"]) == ["one", "two"]


def test_chop():
    assert fh.chop(1e-10) == 0
    assert fh.chop(1.0) == 1.0
    assert fh.chop([1e-10, 1.0, 1e-9]) == [0, 1.0, 0]


def test_chunks():
    assert list(fh.chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_dot():
    assert fh.dot([1, 2, 3], [4, 5, 6]) == 1 * 4 + 2 * 5 + 3 * 6


def test_sort_by_position_function():
    element = ("name", {"position_start": [0, 0, 3.5]})
    assert fh.sortByPositionFunction(element) == 3.5


def test_rotation_matrix_zero_angle():
    m = fh.rotationMatrix(0)
    assert np.allclose(m, np.identity(3))


def test_get_parameter_dict():
    d = {"Length": 2.0}
    assert fh.getParameter(d, "length") == 2.0
    assert fh.getParameter(d, "missing", default=42) == 42


def test_get_parameter_list_of_dicts_last_wins():
    dicts = [{"length": 1.0}, {"length": 2.0}]
    assert fh.getParameter(dicts, "length") == 2.0
    assert fh.getParameter([], "length", default=7) == 7


def test_get_parameter_invalid_type():
    assert fh.getParameter(5, "length", default=99) == 99


def test_format_optional_string():
    assert fh.formatOptionalString("1.0", "length") == " length=1.0\n"
    assert fh.formatOptionalString("None", "length") == ""
    assert fh.formatOptionalString("1.0", "length", n=2) == " length(2)=1.0\n"


def test_create_optional_string():
    d = {"length": 3.0}
    assert fh.createOptionalString(d, "length") == " length=3.0\n"
    assert fh.createOptionalString(d, "missing") == ""


def test_isevaluable():
    assert fh.isevaluable(None, "1 + 1") is True
    assert fh.isevaluable(None, "not valid python (") is False


def test_path_function():
    assert fh.path_function(".", None) == os.path.abspath(".")


def test_expand_substitution_no_dollar():
    class Fake:
        global_parameters = {"master_lattice": ".", "master_subdir": "."}

    assert fh.expand_substitution(Fake(), "plain_string") == "plain_string"


def test_expand_substitution_non_string_passthrough():
    class Fake:
        global_parameters = {"master_lattice": ".", "master_subdir": "."}

    assert fh.expand_substitution(Fake(), 5) == 5


def test_expand_substitution_evaluable_expression():
    class Fake:
        global_parameters = {"master_lattice": ".", "master_subdir": "."}

    result = fh.expand_substitution(Fake(), "$1+1$")
    assert result == "2"


def test_check_value_dict_with_default():
    result = fh.checkValue(None, {"value": None, "default": 5})
    assert result == 5


def test_check_value_dict_list_with_defaults():
    result = fh.checkValue(
        None, {"type": "list", "value": [None, 2], "default": [1, 1]}
    )
    assert result == [1, 2]


def test_check_value_dict_list_no_default():
    result = fh.checkValue(None, {"type": "list", "value": [1, None]})
    assert result == [1, None]


def test_check_value_string_attr_lookup():
    class Fake:
        foo = "bar"

    assert fh.checkValue(Fake(), "foo") == "bar"
    assert fh.checkValue(Fake(), "missing", default="dflt") == "dflt"


def test_clean_directory(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    fh.clean_directory(str(tmp_path))
    assert not f.exists()


def test_list_add():
    assert fh.list_add([1, 2, 3], [10, 20, 30]) == [11, 22, 33]


def test_symlink_calls_os_symlink(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "symlink", lambda source, link_name: calls.append((source, link_name)))
    fh.symlink("src", "dst")
    assert calls == [("src", "dst")]


def test_symlink_swallows_file_exists_error(monkeypatch):
    def raiser(source, link_name):
        raise FileExistsError()

    monkeypatch.setattr(os, "symlink", raiser)
    fh.symlink("src", "dst")  # should not raise


def test_copylink(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dst = tmp_path / "dst.txt"
    fh.copylink(str(src), str(dst))
    assert dst.read_text() == "hello"


def test_copylink_missing_source_does_not_raise(tmp_path):
    fh.copylink(str(tmp_path / "missing.txt"), str(tmp_path / "dst.txt"))


def test_convert_numpy_types():
    assert fh.convert_numpy_types(np.float64(1.5)) == 1.5
    assert isinstance(fh.convert_numpy_types(np.float64(1.5)), float)
    assert isinstance(fh.convert_numpy_types(np.int32(3)), int)
    assert fh.convert_numpy_types([np.float64(1.0), np.float64(2.0)]) == [1.0, 2.0]
    assert fh.convert_numpy_types({"a": np.int64(2)}) == {"a": 2}
    assert fh.convert_numpy_types("plain") == "plain"


def test_normalize():
    assert fh.normalize({"a": 1}) == {"a": 1.0}
    assert fh.normalize([1, 2]) == [1.0, 2.0]
    assert fh.normalize(np.array([1, 2])) == [1, 2]
    assert fh.normalize(np.float64(2.0)) == 2.0
    assert fh.normalize("plain") == "plain"


def test_deepdiff_to_nested_empty():
    assert fh.deepdiff_to_nested({}) == {}


def test_deepdiff_to_nested_values_changed():
    diff = {
        "values_changed": {
            "root['a']['b']": {"old_value": 1, "new_value": 2},
            "root['a'][0]": {"old_value": "x", "new_value": "y"},
        }
    }
    nested = fh.deepdiff_to_nested(diff)
    assert nested["a"]["b"] == {"old": 1, "new": 2}
    assert nested["a"][0] == {"old": "x", "new": "y"}


def test_compare_multiple_models():
    class Foo(BaseModel):
        name: str
        value: float

    old = Foo(name="A", value=1.0)
    new = Foo(name="A", value=2.0)
    result = fh.compare_multiple_models([(old, new)])
    assert result["A"]["value"] == {"old": 1.0, "new": 2.0}


def test_set_deep_attr():
    obj = types.SimpleNamespace(a=types.SimpleNamespace(b=types.SimpleNamespace(c=1)))
    fh.set_deep_attr(obj, "a.b.c", 42)
    assert obj.a.b.c == 42


def test_flatten_changes_dict():
    nested = {"a": {"b": {"old": 1, "new": 2}}, "c": {"old": 3, "new": 4}}
    flattened = dict(fh.flatten_changes_dict(nested))
    assert flattened == {"a.b": 2, "c": 4}
