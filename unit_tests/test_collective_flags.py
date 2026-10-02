"""A line, not an element, decides whether CSR and LSC are modelled.

The element files carry `csr_enable`/`lsc_enable`, but which of them is right
depends on where the line sits: the same drift earns a wake between the
compressors and does not in a transport line at 13 GeV, where elegant's LSC
kicks fall over on a bunch two hundredths of a millimetre long. A settings
file can now say so per stage, and one that says nothing keeps both on as
before.
"""

import types

import pytest
from laura.models.element import Drift, Quadrupole
from simba.Framework_objects import frameworkLattice


class FakeLattice:
    """Enough of a lattice for the settings pass and the two flags it sets."""

    def __init__(self, **file_block):
        self.file_block = file_block
        self._csr_enable = True
        self._lsc_enable = True
        self.section = types.SimpleNamespace(csr_enable=True, lsc_enable=True)
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


def test_both_are_on_when_the_settings_file_says_nothing():
    """The backwards-compatible case, and the one every existing .def is in."""
    latt = lattice(input={})
    assert latt.csr_enable is True
    assert latt.lsc_enable is True
    assert latt.elementObjects["D1"].simulation.lsc_enable is not False


def test_a_stage_can_turn_lsc_off():
    latt = lattice(input={}, lsc_enable=False)
    assert latt.lsc_enable is False
    assert latt.csr_enable is True
    assert latt.elementObjects["D1"].simulation.lsc_enable is False
    assert latt.lsc_in_use is False


def test_a_stage_can_turn_csr_off():
    latt = lattice(input={}, csr_enable=False)
    assert latt.csr_enable is False
    assert latt.lsc_enable is True
    assert latt.elementObjects["D1"].simulation.csr_enable is False


def test_turning_them_off_reaches_the_section_the_code_writes_from():
    """`section` is what the translators export, so the flag has to land there."""
    latt = lattice(input={}, csr_enable=False, lsc_enable=False)
    assert latt.section.csr_enable is False
    assert latt.section.lsc_enable is False


@pytest.mark.parametrize("value", [True, 1, "yes"])
def test_a_truthy_setting_leaves_them_on(value):
    assert lattice(input={}, lsc_enable=value).lsc_enable is True


def test_an_explicit_null_is_not_a_setting():
    """`lsc_enable:` with nothing after it is YAML for None, not for off."""
    latt = lattice(input={}, lsc_enable=None)
    assert latt.lsc_enable is True
    assert latt.section.lsc_enable is True
