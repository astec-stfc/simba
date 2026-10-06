"""
simba's warnings and errors live in :mod:`simba.exceptions`, one class each,
so they can be filtered by kind rather than by matching message text.
"""

import inspect
import re
import warnings

import pytest

from simba import exceptions
from simba.Framework_objects import frameworkLattice

GROUPS = (
    exceptions.UnsupportedWarning,
    exceptions.SettingWarning,
    exceptions.GeometryWarning,
    exceptions.PhysicsWarning,
)


def specific_warnings():
    return [
        klass
        for _, klass in inspect.getmembers(exceptions, inspect.isclass)
        if issubclass(klass, exceptions.SimbaWarning)
        and klass is not exceptions.SimbaWarning
        and klass not in GROUPS
    ]


@pytest.mark.parametrize("klass", specific_warnings(), ids=lambda k: k.__name__)
def test_every_warning_is_in_exactly_one_group(klass):
    assert sum(issubclass(klass, group) for group in GROUPS) == 1
    assert issubclass(klass, UserWarning)


def test_frameworklattice_words_none_of_its_own_warnings():
    """Every warning frameworkLattice gives is a class from simba.exceptions."""
    source = inspect.getsource(frameworkLattice)
    bare = [call for call in re.findall(r"warn\(\s*(\S+)", source)
            if not call.startswith("exceptions.")]
    assert bare == []


def test_a_group_can_be_silenced_as_a_whole():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", exceptions.UnsupportedWarning)
        warnings.warn(exceptions.TurnsUnsupportedWarning("RING", "ocelot", 10, ""))
        warnings.warn(exceptions.RampOneTurnWarning("RING"))
    assert [type(w.message) for w in caught] == [exceptions.RampOneTurnWarning]


def test_the_message_is_the_one_it_always_was():
    message = str(exceptions.RampOverrunWarning("RING", 20, 10))
    assert message == (
        "Line 'RING' ramps out to turn 20, but only 10 turns are tracked, so "
        "the run ends part-way up the ramp."
    )


def test_a_wrong_species_is_still_a_value_error():
    with pytest.raises(ValueError, match="tracks only electrons"):
        raise exceptions.WrongSpeciesError("RING", "elegant", 938.272e6, 1)


@pytest.mark.parametrize("value, shown", [("two", "'two'"), (0, "0"), (-3, "-3")])
def test_a_bad_superperiod_count_says_why(value, shown):
    message = str(exceptions.BadSuperperiodsWarning("RING", value))
    assert f"nsuperperiods={shown}" in message
    assert ("whole number" in message) == isinstance(value, str)
