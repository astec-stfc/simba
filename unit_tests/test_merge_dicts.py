from simba.Modules.merge_two_dicts import merge_two_dicts, merge_dicts


def test_merge_two_dicts_normal():
    x = {"a": 1, "b": 2}
    y = {"b": 3, "c": 4}
    assert merge_two_dicts(y, x) == {"a": 1, "b": 3, "c": 4}


def test_merge_two_dicts_y_not_dict():
    assert merge_two_dicts(None, {"a": 1}) == {"a": 1}


def test_merge_two_dicts_x_not_dict():
    assert merge_two_dicts({"a": 1}, None) == {"a": 1}


def test_merge_two_dicts_neither_dict():
    assert merge_two_dicts(None, None) == {}


def test_merge_dicts_priority_order():
    a = {"x": 1}
    b = {"x": 2, "y": 2}
    c = {"x": 3, "y": 3, "z": 3}
    # first dict has highest priority
    assert merge_dicts(a, b, c) == {"x": 1, "y": 2, "z": 3}


def test_merge_dicts_single_dict():
    assert merge_dicts({"a": 1}) == {"a": 1}
