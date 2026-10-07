"""A line, not an element, decides whether CSR and LSC are modelled."""

import types

import pytest
from laura.models.element import Drift, Quadrupole
from simba.Framework_objects import frameworkLattice


class FakeLattice:
    """Enough of a lattice for the settings pass and the two flags it sets."""

    def __init__(self, **file_block):
        self.file_block = file_block
        # the real class defaults, so a change to them is seen here; they are
        # pydantic private attributes, so not plain class attributes
        defaults = frameworkLattice.__private_attributes__
        self._csr_enable = defaults["_csr_enable"].default
        self._lsc_enable = defaults["_lsc_enable"].default
        self.section = types.SimpleNamespace(
            csr_enable=self._csr_enable, lsc_enable=self._lsc_enable
        )
        self.elementObjects = {
            "D1": Drift(name="D1", hardware_class="Drift", machine_area="A1"),
            "Q1": Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A1"),
        }

    csr_enable = frameworkLattice.csr_enable
    lsc_enable = frameworkLattice.lsc_enable
    lsc_in_use = frameworkLattice.lsc_in_use
    _apply_collective_settings = frameworkLattice._apply_collective_settings


def lattice(**file_block):
    latt = FakeLattice(**file_block)
    latt._apply_collective_settings()
    return latt


def test_csr_is_on_and_lsc_off_when_the_settings_file_says_nothing():
    latt = lattice(input={})
    assert latt.csr_enable is True
    assert latt.lsc_enable is False
    assert latt.section.lsc_enable is False


def test_an_element_is_not_doing_lsc_unless_it_asks():
    """LAURA's own default is off too, so an untouched element does no LSC."""
    latt = lattice(input={})
    assert latt.elementObjects["D1"].simulation.lsc_enable is False
    assert latt.lsc_in_use is False


def test_a_stage_can_turn_lsc_on():
    latt = lattice(input={}, lsc_enable=True)
    assert latt.lsc_enable is True
    assert latt.csr_enable is True
    assert latt.elementObjects["D1"].simulation.lsc_enable is True
    assert latt.lsc_in_use is True


def test_a_stage_can_turn_lsc_off_explicitly():
    latt = lattice(input={}, lsc_enable=False)
    assert latt.lsc_enable is False
    assert latt.elementObjects["D1"].simulation.lsc_enable is False


def test_a_stage_can_turn_csr_off():
    latt = lattice(input={}, csr_enable=False)
    assert latt.csr_enable is False
    assert latt.lsc_enable is False
    assert latt.elementObjects["D1"].simulation.csr_enable is False


def test_the_flags_reach_the_section_the_code_writes_from():
    """`section` is what the translators export, so the flag has to land there."""
    latt = lattice(input={}, csr_enable=False, lsc_enable=True)
    assert latt.section.csr_enable is False
    assert latt.section.lsc_enable is True


@pytest.mark.parametrize("value", [True, 1, "yes"])
def test_a_truthy_setting_turns_lsc_on(value):
    assert lattice(input={}, lsc_enable=value).lsc_enable is True


def test_an_explicit_null_is_not_a_setting():
    """`lsc_enable:` with nothing after it is YAML for None, not for on."""
    latt = lattice(input={}, lsc_enable=None)
    assert latt.lsc_enable is False
    assert latt.section.lsc_enable is False
