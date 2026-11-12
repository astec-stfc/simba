"""Each element decides whether CSR and LSC are modelled; a line's settings block overrides them all."""

import types

import pytest
from laura.models.element import Dipole, Drift, Quadrupole
from simba.Framework_objects import frameworkLattice


class FakeLattice:
    """Enough of a lattice for the settings pass and the flags it sets."""

    def __init__(self, number_of_bins=None, **file_block):
        self.file_block = file_block
        self.globalSettings = {"charge": None}
        # the real class defaults, so a change to them is seen here; they are
        # pydantic private attributes, so not plain class attributes
        defaults = frameworkLattice.__private_attributes__
        self._csr_enable = defaults["_csr_enable"].default
        self._lsc_enable = defaults["_lsc_enable"].default
        self._csr_bins = defaults["_csr_bins"].default
        self._number_of_bins = number_of_bins
        self.section = types.SimpleNamespace(
            csr_enable=self._csr_enable, lsc_enable=self._lsc_enable
        )
        self.elementObjects = {
            "D1": Drift(name="D1", hardware_class="Drift", machine_area="A1"),
            "Q1": Quadrupole(name="Q1", hardware_class="Magnet", machine_area="A1"),
            "B1": Dipole(name="B1", hardware_class="Magnet", machine_area="A1"),
        }

    def _machine_space_charge(self):
        return types.SimpleNamespace(number_of_bins=self._number_of_bins)

    @property
    def dipoles(self):
        return [self.elementObjects["B1"]]

    closed_geometry = False
    periodic = frameworkLattice.periodic
    csr_enable = frameworkLattice.csr_enable
    csr_bins = frameworkLattice.csr_bins
    lsc_enable = frameworkLattice.lsc_enable
    lsc_in_use = frameworkLattice.lsc_in_use
    space_charge_mode = frameworkLattice.space_charge_mode
    space_charge_modes_off = frameworkLattice.space_charge_modes_off
    _apply_collective_settings = frameworkLattice._apply_collective_settings


def lattice(closed=False, csr_on_bend=None, number_of_bins=None, **file_block):
    latt = FakeLattice(number_of_bins=number_of_bins, **file_block)
    if csr_on_bend is not None:
        latt.elementObjects["B1"].simulation.csr_enable = csr_on_bend
    latt.closed_geometry = closed
    latt._apply_collective_settings()
    return latt


def test_unstated_the_line_follows_its_elements():
    """LAURA's defaults: CSR on, LSC off; nothing is pushed onto the section."""
    latt = lattice(input={})
    assert latt.csr_enable is True
    assert latt.lsc_enable is False
    assert latt.section.csr_enable is None
    assert latt.section.lsc_enable is None


def test_issue_50_csr_off_on_the_bends_is_off_for_the_line():
    """Ocelot, Bmad, ImpactX and GPT read the line's flag, so it must say what ELEGANT's elements say."""
    latt = lattice(input={}, csr_on_bend=False)
    assert latt.csr_enable is False


def test_a_stated_flag_overrides_the_elements():
    latt = lattice(input={}, csr_on_bend=False, csr_enable=True)
    assert latt.csr_enable is True
    assert latt.elementObjects["B1"].simulation.csr_enable is True


def test_csr_bins_come_from_the_bends_unless_stated():
    assert lattice(input={}).csr_bins == 100  # LAURA's dipole default
    latt = lattice(input={}, number_of_bins=30)
    assert latt.csr_bins == 30
    assert latt.elementObjects["B1"].simulation.csr_bins == 30
    latt = lattice(input={}, number_of_bins=30, csr_bins=40)
    assert latt.csr_bins == 40
    assert latt.elementObjects["B1"].simulation.csr_bins == 40


@pytest.mark.parametrize(
    "line, global_, mode",
    [
        (None, None, ""),
        (None, "3D", "3d"),
        ("2d", "3d", "2d"),
        ("off", "3d", ""),
        (False, "3d", ""),
        ("None", None, ""),
    ],
)
def test_space_charge_mode_line_then_global(line, global_, mode):
    latt = lattice(input={})
    if line is not None:
        latt.file_block["charge"] = {"space_charge_mode": line}
    if global_ is not None:
        latt.globalSettings = {"charge": {"space_charge_mode": global_}}
    assert latt.space_charge_mode == mode


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
    assert latt.section.lsc_enable is None


@pytest.mark.parametrize(
    "closed, file_block",
    [(True, {}), (False, {"tracking": {"periodic": True}})],
)
def test_a_ring_defaults_csr_off(closed, file_block):
    """Closed or periodic means a ring: no CSR, so ELEGANT writes CSBEND."""
    latt = lattice(closed=closed, input={}, **file_block)
    assert latt.csr_enable is False
    assert latt.section.csr_enable is False
    assert latt.elementObjects["Q1"].simulation.csr_enable is False


def test_a_ring_can_still_ask_for_csr():
    latt = lattice(closed=True, input={}, csr_enable=True)
    assert latt.csr_enable is True


def test_an_injection_study_on_a_ring_is_still_a_ring():
    latt = lattice(closed=True, input={}, tracking={"periodic": False})
    assert latt.csr_enable is False


@pytest.mark.parametrize("supports, warns", [(False, True), (True, False)])
def test_a_code_without_csr_says_so(supports, warns):
    from simba import exceptions
    fake = types.SimpleNamespace(
        csr_enable=True, dipoles=["B1"], lsc_enable=False, supports_csr=supports,
        supports_lsc=False, objectname="L1", code="astra",
        codes_that_can=frameworkLattice.codes_that_can,
    )
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frameworkLattice.check_collective_supported(fake)
    found = [w for w in caught if issubclass(w.category, exceptions.CollectiveUnsupportedWarning)]
    assert bool(found) == warns
    if warns:
        assert "bmad, csrtrack, elegant, gpt, impactx and ocelot are the codes that can" in str(found[0].message)
