"""Synchrotron radiation in a ring, and the silent wrongness without it."""

import warnings

import pytest

from simba.Codes.ASTRA.ASTRA import astraLattice
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice


class FakeBeam:
    def __init__(self, species):
        self.species = species


class FakeLine:
    """A lattice stub carrying only what the radiation code reads."""

    def __init__(self, tracking=None, species="electron"):
        self.file_block = {"tracking": tracking or {}}
        self.objectname = "RING"
        self.code = "xsuite"
        self.global_parameters = {"beam": FakeBeam(species)}
        self.radiates_by_default = False
        self.supports_radiation = True

    def _machine_geometry(self):
        return None

    radiation = frameworkLattice.radiation
    periodic = frameworkLattice.periodic
    turns = frameworkLattice.turns
    check_radiation = frameworkLattice.check_radiation
    check_radiation_supported = frameworkLattice.check_radiation_supported
    codes_that_can = frameworkLattice.codes_that_can


RING = {"turns": 100000, "periodic": True}


# --- reading the setting ------------------------------------------------


def test_no_setting_means_not_stated():
    """`None` is 'leave each code alone', not 'off' -- elegant and Bmad
    radiate by default and must not be silently switched off."""
    assert FakeLine().radiation is None


def test_false_is_an_explicit_refusal():
    """Distinct from absent, because it has to actually turn elegant and
    Bmad off rather than leaving them."""
    assert FakeLine({"radiation": False}).radiation == "off"


def test_true_is_taken_as_the_mean_model():
    """Damping and energy loss, but no excitation -- the conservative
    reading of a bare `radiation: true`."""
    assert FakeLine({"radiation": True}).radiation == "mean"


def test_a_named_model_is_passed_through():
    assert FakeLine({"radiation": "quantum"}).radiation == "quantum"


# --- which codes simba can switch it on for -----------------------------


@pytest.mark.parametrize(
    "cls",
    [xsuiteLattice, ocelotLattice, bmadLattice, elegantLattice],
    ids=lambda c: c.__name__,
)
def test_the_four_that_can(cls):
    """Xsuite `configure_radiation`, Ocelot `SpontanRadEffects` per dipole,
    elegant `SYNCH_RAD`/`ISR` on the bends, Bmad `bmad_com[...]` -- the last
    two straight from LAURA's element `sr_enable`/`isr_enable`."""
    assert cls.supports_radiation is True


def test_a_code_with_no_radiation_at_all():
    assert astraLattice.supports_radiation is False


@pytest.mark.parametrize("cls", [elegantLattice, bmadLattice], ids=lambda c: c.__name__)
def test_elegant_and_bmad_radiate_without_being_asked(cls):
    """LAURA's element `sr_enable` and `isr_enable` both default True and
    both codes read them, so these two radiate out of the box."""
    assert cls.radiates_by_default is True


@pytest.mark.parametrize("cls", [xsuiteLattice, ocelotLattice], ids=lambda c: c.__name__)
def test_xsuite_and_ocelot_do_not(cls):
    assert cls.radiates_by_default is False


def test_the_base_class_assumes_it_cannot():
    assert frameworkLattice.supports_radiation is False


def test_asking_a_code_with_no_switch_warns():
    """MAD-X writes SR off whatever is asked, so `radiation: quantum` there
    was dropped without a word."""
    line = FakeLine({"radiation": "quantum"})
    line.code, line.supports_radiation = "madx", False
    with pytest.warns(
        UserWarning,
        match="not applied.*bmad, elegant, ocelot and xsuite are the codes that can",
    ):
        line.check_radiation_supported()


@pytest.mark.parametrize("tracking", [{}, {"radiation": "quantum"}], ids=["unstated", "switchable"])
def test_nothing_to_say_otherwise(tracking):
    line = FakeLine(tracking)
    if not tracking:
        line.supports_radiation = False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_radiation_supported()


# --- the warning --------------------------------------------------------


def test_a_long_electron_ring_without_radiation_warns():
    with pytest.warns(UserWarning, match="synchrotron radiation"):
        FakeLine(RING).check_radiation()


def test_a_code_that_already_radiates_is_silent():
    """elegant and Bmad need no warning -- they are already doing it, and
    saying otherwise would be the misinformation this check exists to stop."""
    line = FakeLine(RING)
    line.radiates_by_default = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_radiation()


def test_the_warning_says_what_is_wrong_with_the_answer():
    """Not that something failed -- that the numbers mean something else."""
    with pytest.warns(UserWarning, match="never reach equilibrium"):
        FakeLine(RING).check_radiation()


def test_asking_for_radiation_silences_it():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FakeLine({**RING, "radiation": "quantum"}).check_radiation()


def test_a_short_run_is_silent():
    """Well inside a damping time, so radiation would change little anyway."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FakeLine({"turns": 100, "periodic": True}).check_radiation()


def test_a_transfer_line_is_silent():
    """No turns to damp over; this is the ordinary non-ring case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FakeLine({"turns": 100000, "periodic": False}).check_radiation()


@pytest.mark.parametrize("species", ["electron", "positron", "proton", ""])
def test_the_warning_does_not_depend_on_species(species):
    """The check is species-blind. A proton ring at these energies radiates
    negligibly, so this is noise there -- but species came from matching on
    a free-text string, and defaulting to silence is the wrong way round for
    a check whose whole purpose is to catch a plausible wrong answer."""
    with pytest.warns(UserWarning):
        FakeLine(RING, species=species).check_radiation()


# --- the measurement the warning is based on ----------------------------


def test_radiation_off_means_no_energy_loss_at_all():
    """Not 'a small amount' -- exactly zero, with infinite partition numbers
    and no equilibrium emittance in the table."""
    xt = pytest.importorskip("xtrack")
    import math

    import numpy as np

    ncell = 16
    angle = 2 * math.pi / ncell
    els, nms = [], []
    for i in range(ncell):
        for nm, el in (
            (f"qf{i}", xt.Quadrupole(length=0.3, k1=1.2)),
            (f"d{i}a", xt.Drift(length=0.5)),
            (f"b{i}", xt.Bend(length=1.0, angle=angle)),
            (f"d{i}b", xt.Drift(length=0.5)),
            (f"qd{i}", xt.Quadrupole(length=0.3, k1=-1.2)),
            (f"d{i}c", xt.Drift(length=0.5)),
        ):
            nms.append(nm)
            els.append(el)
    nms.append("rf")
    els.append(xt.Cavity(voltage=5e6, frequency=100e6, lag=180))
    line = xt.Line(elements=els, element_names=nms)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()

    line.configure_radiation(model=None)
    off = line.twiss(radiation_analysis=True)
    assert float(off.energy_loss) == 0.0
    assert "eq_gemitt_x" not in off.keys()
    assert np.isinf(np.asarray(off.partition_numbers)).all()

    line.configure_radiation(model="mean")
    on = line.twiss(radiation_analysis=True)
    assert float(on.energy_loss) > 1e3
    assert float(on.eq_gemitt_x) > 0
    assert np.isfinite(np.asarray(on.partition_numbers)).all()
    # damping in a number of turns a real study would reach
    assert 1 / abs(on.damping_constants_turns[0]) < 1e6


# --- the flag reaches LAURA's section flags (Bmad) ----------------------


class FakeSection:
    sr_enable = None
    isr_enable = None


class FakeRadiationLine(FakeLine):
    """Adds the bits `_apply_radiation_to_section` touches."""

    def __init__(self, tracking=None):
        super().__init__(tracking)
        self.section = FakeSection()
        self.elementObjects = {}

    _apply_radiation_to_section = frameworkLattice._apply_radiation_to_section


def test_an_unstated_setting_leaves_lauras_flags_untouched():
    """The regression this nearly shipped: forcing them off would have
    switched off the radiation elegant and Bmad already do."""
    line = FakeRadiationLine()
    line._apply_radiation_to_section()
    assert line.section.sr_enable is None
    assert line.section.isr_enable is None


def test_an_explicit_off_does_turn_them_off():
    line = FakeRadiationLine({"radiation": False})
    line._apply_radiation_to_section()
    assert line.section.sr_enable is False
    assert line.section.isr_enable is False


def test_mean_sets_damping_but_not_fluctuations():
    """`sr_enable` -> bmad_com[radiation_damping_on]."""
    line = FakeRadiationLine({"radiation": "mean"})
    line._apply_radiation_to_section()
    assert line.section.sr_enable is True
    assert line.section.isr_enable is False


def test_quantum_sets_both():
    """Only with the fluctuations does an equilibrium emittance exist."""
    line = FakeRadiationLine({"radiation": "quantum"})
    line._apply_radiation_to_section()
    assert line.section.sr_enable is True
    assert line.section.isr_enable is True
