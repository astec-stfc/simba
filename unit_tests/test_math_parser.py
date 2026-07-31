import math
import pytest

from simba.Modules.MathParser import MathParser


def test_docstring_example():
    data = {"r": 3.4, "theta": 3.141592653589793}
    parser = MathParser(data)
    assert parser.parse("r*cos(theta)") == pytest.approx(-3.4)
    data["theta"] = 0.0
    assert parser.parse("r*cos(theta)") == pytest.approx(3.4)


def test_binary_operators():
    parser = MathParser({})
    assert parser.parse("2+3") == 5
    assert parser.parse("2-3") == -1
    assert parser.parse("2*3") == 6
    assert parser.parse("7/2") == 3.5
    assert parser.parse("2**3") == 8
    assert parser.parse("7%3") == 1
    assert parser.parse("7//2") == 3
    assert parser.parse("5^1") == 4  # BitXor


def test_unary_operators():
    parser = MathParser({"r": 3.4})
    assert parser.parse("-r") == pytest.approx(-3.4)
    assert parser.parse("+r") == pytest.approx(3.4)


def test_name_lookup_from_vars_and_math_module():
    parser = MathParser({"r": 2})
    assert parser.parse("r") == 2
    assert parser.parse("pi") == pytest.approx(math.pi)


def test_unknown_name_raises_name_error():
    parser = MathParser({})
    with pytest.raises(NameError):
        parser.parse("doesnotexist")


def test_underscore_name_raises_name_error():
    parser = MathParser({})
    with pytest.raises(NameError):
        parser.parse("_secret")


def test_math_disabled_raises_name_error_for_math_functions():
    parser = MathParser({}, math=False)
    with pytest.raises(NameError):
        parser.parse("cos(0)")


def test_attribute_access():
    class Obj:
        x = 5

    parser = MathParser({"a": Obj()})
    assert parser.parse("a.x") == 5


def test_call_with_args_and_kwargs():
    parser = MathParser({"f": lambda x, y=1: x + y})
    assert parser.parse("f(2)") == 3
    assert parser.parse("f(2, y=3)") == 5


def test_unsupported_node_raises_type_error():
    parser = MathParser({"a": 1, "b": 2})
    with pytest.raises(TypeError):
        parser.parse("a and b")
