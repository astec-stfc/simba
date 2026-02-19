"""Tests for FrameworkHelperFunctions utility functions."""
import os
import tempfile
import pytest
import numpy as np

from simba.FrameworkHelperFunctions import (
    readFile,
    saveFile,
    findSetting,
    findSettingValue,
    lineReplaceFunction,
    replaceString,
    chop,
    chunks,
    dot,
    rotationMatrix,
    getParameter,
    formatOptionalString,
    createOptionalString,
    isevaluable,
    list_add,
    convert_numpy_types,
    normalize,
    deepdiff_to_nested,
    flatten_changes_dict,
    set_deep_attr,
    expand_substitution_recursive,
)


# ── File I/O ────────────────────────────────────────────────────────────────

class TestFileIO:

    def test_save_and_read_file(self, tmp_path):
        filepath = str(tmp_path / "test.txt")
        saveFile(filepath, ["line1\n", "line2\n"])
        content = readFile(filepath)
        assert content == ["line1\n", "line2\n"]

    def test_save_file_append_mode(self, tmp_path):
        filepath = str(tmp_path / "test.txt")
        saveFile(filepath, ["line1\n"])
        saveFile(filepath, ["line2\n"], mode="a")
        content = readFile(filepath)
        assert content == ["line1\n", "line2\n"]


# ── findSetting / findSettingValue ──────────────────────────────────────────

class TestFindSetting:

    def test_find_existing_setting(self):
        d = {
            "elem1": {"code": "astra", "length": 1.0},
            "elem2": {"code": "elegant", "length": 2.0},
        }
        result = findSetting("code", "astra", d)
        assert len(result) == 1
        assert result[0][0] == "elem1"

    def test_find_no_match(self):
        d = {"elem1": {"code": "astra"}}
        result = findSetting("code", "gpt", d)
        assert result == []

    def test_find_multiple_matches(self):
        d = {
            "a": {"type": "quad"},
            "b": {"type": "quad"},
            "c": {"type": "dipole"},
        }
        result = findSetting("type", "quad", d)
        assert len(result) == 2


# ── lineReplaceFunction / replaceString ─────────────────────────────────────

class TestStringReplacement:

    def test_line_replace(self):
        result = lineReplaceFunction("value=$param$", "param", "42")
        assert result == "value=42"

    def test_line_replace_no_match(self):
        result = lineReplaceFunction("value=10", "param", "42")
        assert result == "value=10"

    def test_line_replace_with_list(self):
        # replaceString handles the list case by initialising lineIterator
        lines = ["x=$param$\n", "y=$param$\n"]
        result = replaceString(lines, "param", [10, 20])
        assert result[0] == "x=10\n"
        assert result[1] == "y=20\n"

    def test_replace_string_scalar(self):
        lines = ["a=$x$\n", "b=5\n", "c=$x$\n"]
        result = replaceString(lines, "x", "10")
        assert result[0] == "a=10\n"
        assert result[1] == "b=5\n"
        assert result[2] == "c=10\n"


# ── chop ────────────────────────────────────────────────────────────────────

class TestChop:

    def test_small_number_chopped(self):
        assert chop(1e-10) == 0

    def test_larger_number_preserved(self):
        assert chop(0.5) == 0.5

    def test_negative_small(self):
        assert chop(-1e-10) == 0

    def test_list_input(self):
        result = chop([1e-10, 0.5, -1e-10])
        assert result == [0, 0.5, 0]

    def test_custom_delta(self):
        assert chop(0.05, delta=0.1) == 0
        assert chop(0.2, delta=0.1) == 0.2


# ── chunks ──────────────────────────────────────────────────────────────────

class TestChunks:

    def test_even_split(self):
        result = list(chunks([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = list(chunks([1, 2, 3, 4, 5], 2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        result = list(chunks([1, 2, 3], 10))
        assert result == [[1, 2, 3]]


# ── dot ─────────────────────────────────────────────────────────────────────

class TestDot:

    def test_simple_dot_product(self):
        assert dot([1, 0, 0], [0, 1, 0]) == 0

    def test_parallel_vectors(self):
        assert dot([1, 2, 3], [1, 2, 3]) == 14

    def test_negative(self):
        assert dot([1, 0, 0], [-1, 0, 0]) == -1


# ── rotationMatrix ──────────────────────────────────────────────────────────

class TestRotationMatrix:

    def test_identity_at_zero(self):
        R = rotationMatrix(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_ninety_degrees(self):
        R = rotationMatrix(np.pi / 2)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R[1, 1], 1.0, atol=1e-10)

    def test_orthogonal(self):
        R = rotationMatrix(0.3)
        RtR = np.array(R) @ np.array(R.T)
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-10)


# ── getParameter ────────────────────────────────────────────────────────────

class TestGetParameter:

    def test_from_dict(self):
        assert getParameter({"Length": 1.5}, "length") == 1.5

    def test_default_when_missing(self):
        assert getParameter({"x": 1}, "y", default=99) == 99

    def test_from_list_of_dicts(self):
        # Later dict takes precedence
        result = getParameter([{"a": 1}, {"a": 2}], "a")
        assert result == 2

    def test_non_dict_returns_default(self):
        assert getParameter("not_a_dict", "param", default=42) == 42

    def test_case_insensitive(self):
        assert getParameter({"MyParam": "val"}, "myparam") == "val"


# ── formatOptionalString ────────────────────────────────────────────────────

class TestFormatOptionalString:

    def test_with_value(self):
        result = formatOptionalString("10", "sigma_x")
        assert "sigma_x=10" in result

    def test_none_returns_empty(self):
        result = formatOptionalString("None", "sigma_x")
        assert result == ""

    def test_with_index(self):
        result = formatOptionalString("5", "sigma", n=1)
        assert "sigma(1)=5" in result


# ── isevaluable ─────────────────────────────────────────────────────────────

class TestIsevaluable:

    def test_valid_expression(self):
        # isevaluable takes self as first arg (unused mostly)
        assert isevaluable(None, "1 + 2") is True

    def test_invalid_expression(self):
        assert isevaluable(None, "undefined_variable") is False

    def test_math_expression(self):
        assert isevaluable(None, "3.14 * 2") is True


# ── list_add ────────────────────────────────────────────────────────────────

class TestListAdd:

    def test_simple_add(self):
        assert list_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

    def test_float_add(self):
        result = list_add([0.1, 0.2], [0.3, 0.4])
        assert all(abs(a - b) < 1e-10 for a, b in zip(result, [0.4, 0.6]))


# ── convert_numpy_types ─────────────────────────────────────────────────────

class TestConvertNumpyTypes:

    def test_float64(self):
        result = convert_numpy_types(np.float64(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_int64(self):
        result = convert_numpy_types(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_array(self):
        result = convert_numpy_types(np.array([1.0, 2.0]))
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_nested_dict(self):
        result = convert_numpy_types({"a": np.float64(1.0), "b": np.int32(2)})
        assert result == {"a": 1.0, "b": 2}
        assert isinstance(result["a"], float)
        assert isinstance(result["b"], int)

    def test_passthrough(self):
        assert convert_numpy_types("hello") == "hello"
        assert convert_numpy_types(42) == 42


# ── normalize ───────────────────────────────────────────────────────────────

class TestNormalize:

    def test_none_becomes_empty_dict(self):
        assert normalize(None) == {}

    def test_empty_dict(self):
        assert normalize({}) == {}

    def test_nested_dict(self):
        result = normalize({"a": {"b": 1}})
        assert result == {"a": {"b": 1.0}}

    def test_numpy_array(self):
        result = normalize(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_numpy_scalar(self):
        result = normalize(np.float64(3.14))
        assert isinstance(result, float)

    def test_int_to_float(self):
        result = normalize(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_list_recursion(self):
        result = normalize([None, {"a": 1}, [2]])
        assert result == [{}, {"a": 1.0}, [2.0]]

    def test_string_passthrough(self):
        assert normalize("hello") == "hello"


# ── deepdiff_to_nested ─────────────────────────────────────────────────────

class TestDeepdiffToNested:

    def test_empty_diff(self):
        assert deepdiff_to_nested({}) == {}

    def test_simple_change(self):
        diff = {
            "values_changed": {
                "root['a']": {"old_value": 1, "new_value": 2},
            }
        }
        result = deepdiff_to_nested(diff)
        assert result == {"a": {"old": 1.0, "new": 2.0}}

    def test_nested_change(self):
        diff = {
            "values_changed": {
                "root['a']['b']": {"old_value": "x", "new_value": "y"},
            }
        }
        result = deepdiff_to_nested(diff)
        assert result == {"a": {"b": {"old": "x", "new": "y"}}}


# ── flatten_changes_dict ───────────────────────────────────────────────────

class TestFlattenChangesDict:

    def test_simple(self):
        d = {"a": {"old": 1, "new": 2}}
        result = flatten_changes_dict(d)
        assert ("a", 2) in result

    def test_nested(self):
        d = {"a": {"b": {"old": 1, "new": 2}}}
        result = flatten_changes_dict(d)
        assert ("a.b", 2) in result

    def test_leaf_value(self):
        d = {"x": 42}
        result = flatten_changes_dict(d)
        assert ("x", 42) in result


# ── set_deep_attr ───────────────────────────────────────────────────────────

class TestSetDeepAttr:

    def test_simple(self):
        class Obj:
            value = 0
        obj = Obj()
        set_deep_attr(obj, "value", 42)
        assert obj.value == 42

    def test_nested(self):
        class Inner:
            x = 0
        class Outer:
            def __init__(self):
                self.inner = Inner()
        obj = Outer()
        set_deep_attr(obj, "inner.x", 99)
        assert obj.inner.x == 99


# ── expand_substitution_recursive ───────────────────────────────────────────

class TestExpandSubstitutionRecursive:

    def test_no_substitution_passthrough(self):
        """Non-string/non-container values pass through unchanged."""
        assert expand_substitution_recursive(None, 42) == 42
        assert expand_substitution_recursive(None, 3.14) == 3.14

    def test_string_without_dollar_passthrough(self):
        result = expand_substitution_recursive(None, "hello")
        assert result == "hello"

    def test_dict_recursion(self):
        result = expand_substitution_recursive(None, {"a": "hello", "b": 42})
        assert result == {"a": "hello", "b": 42}

    def test_list_recursion(self):
        result = expand_substitution_recursive(None, ["hello", 42])
        assert result == ["hello", 42]

    def test_tuple_preserved(self):
        result = expand_substitution_recursive(None, ("hello", 42))
        assert isinstance(result, tuple)
        assert result == ("hello", 42)

    def test_nested_dict_list(self):
        data = {"elements": [{"name": "elem1"}, {"name": "elem2"}]}
        result = expand_substitution_recursive(None, data)
        assert result == data
