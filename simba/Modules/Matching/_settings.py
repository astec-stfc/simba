"""What the matchers share: naming elements, and returning settings that keep functional definitions."""

import contextlib
from functools import reduce

import numpy as np

FUNCTIONAL = "functional_definitions"
"""Key of the matched functional definitions in a matcher's result."""


def expand(lattice, names) -> list:
    """Element names, with any of the settings' ``groups`` replaced by their members."""
    groups = getattr(lattice, "groupObjects", None) or {}
    out = []
    for name in [names] if isinstance(names, str) else names:
        out += expand(lattice, groups[name].elements) if name in groups else [name]
    return list(dict.fromkeys(out))


def get_path(obj, path: str):
    """``obj.a.b.c`` for ``path`` ``"a.b.c"``."""
    return reduce(getattr, path.split("."), obj)


@contextlib.contextmanager
def temporarily(lattice, knobs, values):
    """
    With each knob's field set to its value, on its elements or in the functional
    definition it names, then put back: for building a model with LAURA's exporters.
    """
    from laura.models.base_models import IgnoreExtra

    from ...FrameworkHelperFunctions import set_deep_attr

    registry = IgnoreExtra.functional_definitions
    section = getattr(lattice.section, "functional_definitions", None)
    section = section if isinstance(section, dict) else {}
    saved_fields, saved_definitions = [], []
    try:
        for knob, value in zip(knobs, values):
            if knob.definition is not None:
                saved_definitions.append((knob.definition, registry.get(knob.definition), section.get(knob.definition)))
                registry[knob.definition] = section[knob.definition] = float(value)
            else:
                for element in knob.elements:
                    saved_fields.append((element, knob.field, get_path(element, knob.field)))
                    set_deep_attr(element, knob.field, float(value))
        yield
    finally:
        for element, path, value in reversed(saved_fields):
            set_deep_attr(element, path, value)
        for name, old, old_section in reversed(saved_definitions):
            registry[name] = old
            if old_section is not None:
                section[name] = old_section


def set_value(result: dict, definitions: dict, element, path: str, current, value: float, matcher: str) -> None:
    """Set ``path`` on ``element``, or the functional definition ``current`` names."""
    if not isinstance(current, str):
        result.setdefault(element.name, {})[path] = value
    elif current in definitions and not np.isclose(definitions[current], value, rtol=1e-12, atol=0):
        raise ValueError(
            f"{matcher} needs functional definition {current!r} to be both "
            f"{definitions[current]} and {value} (from {element.name}.{path})"
        )
    else:
        definitions[current] = value


def check_definitions_stay_inside(lattice, definitions: dict, inside: set, matcher: str) -> None:
    """Refuse to change a definition that elements other than those named ``inside`` use too."""
    from laura.models.base_models import functional_references

    machine = getattr(lattice, "machine", None)
    elements = machine.elements if machine is not None else lattice.elements
    outside = {}
    for name in list(elements.keys()):
        if name in inside:
            continue
        for ref in functional_references(elements[name]) & definitions.keys():
            outside.setdefault(ref, []).append(name)
    if outside:
        raise ValueError(
            f"{matcher} would change functional definitions that other elements use "
            f"too: {outside}; match those as well, or give the matched ones their own "
            "definitions"
        )
