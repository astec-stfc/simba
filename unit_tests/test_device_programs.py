"""An element's strength as a program over turn number."""

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import ELEGANT, needs_elegant
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice
from simba.Modules.DeviceProgram import DeviceProgram

PULSE = {"turns": [1, 4, 5], "values": [0.0, 1.0e-3, 0.0]}
"""One turn of kick on turn 4, the shape both R19 devices have."""

LENGTH = 10.0
"""Ring circumference for the backend tests, in metres."""


class FakeLine:
    """A lattice stub carrying only what the program code reads."""

    def __init__(self, tracking=None, code="astra", supports=False):
        self.file_block = {"tracking": tracking or {}}
        self.code = code
        self.objectname = "RING"
        self.supports_programs = supports
        self.supports_ramp = supports
        # no superperiods: a turn is one pass, as every test below assumes
        self.supports_nsuperperiods = True
        self._elements = {}

    @property
    def elements(self):
        return self._elements

    turns = frameworkLattice.turns
    nsuperperiods = frameworkLattice.nsuperperiods
    passes_per_turn = frameworkLattice.passes_per_turn
    programs = frameworkLattice.programs
    ramp = frameworkLattice.ramp
    ramped = frameworkLattice.ramped
    ramp_clock = frameworkLattice.ramp_clock
    check_programs_supported = frameworkLattice.check_programs_supported
    codes_that_can = frameworkLattice.codes_that_can
    check_programs_fit = frameworkLattice.check_programs_fit
    program_is_vertical = frameworkLattice.program_is_vertical
    program_attribute = frameworkLattice.program_attribute
    apply_programs = frameworkLattice.apply_programs


def program(**overrides):
    """One :class:`DeviceProgram` over :data:`PULSE`."""
    return DeviceProgram.from_dict({"element": "KICK1", **PULSE, **overrides})


def test_no_setting_means_no_programs():
    assert FakeLine().programs == []
    assert FakeLine({"turns": 10}).programs == []


def test_a_program_is_read_from_the_tracking_block():
    line = FakeLine({"turns": 10, "programs": [{"element": "KICK1", **PULSE}]})
    (read,) = line.programs
    assert read.element == "KICK1"
    assert read.turns == [1, 4, 5]
    assert read.values == [0.0, 1.0e-3, 0.0]


def test_a_lone_mapping_is_taken_as_one_program():
    """YAML lets a one-item list be written without the dash."""
    line = FakeLine({"programs": {"element": "KICK1", **PULSE}})
    assert len(line.programs) == 1


def test_an_entry_naming_no_element_warns_and_is_dropped():
    line = FakeLine({"programs": [PULSE, {"element": "KICK1", **PULSE}]})
    with pytest.warns(UserWarning, match="names no element"):
        assert len(line.programs) == 1


def test_mismatched_knots_warn_and_are_dropped():
    line = FakeLine({"programs": [{"element": "K", "turns": [1, 2], "values": [0.0]}]})
    with pytest.warns(UserWarning, match="pair up one to one"):
        assert line.programs == []


def test_turns_out_of_order_are_refused():
    with pytest.raises(ValueError, match="not strictly ascending"):
        DeviceProgram(element="K", turns=[1, 5, 3], values=[0.0, 1.0, 0.0])


def test_turn_zero_is_refused():
    """Silently shifting a 0-based turn is how a kicker fires on the wrong turn."""
    with pytest.raises(ValueError, match="numbered from 1"):
        DeviceProgram(element="K", turns=[0, 3], values=[0.0, 1.0])


def test_an_unknown_interpolation_warns_and_falls_back():
    with pytest.warns(UserWarning, match="not one of"):
        assert program(interpolation="cubic").interpolation == "hold"


def test_a_setting_simba_does_not_read_warns():
    with pytest.warns(UserWarning, match="does not read"):
        program(t_offset=1e-6)


def test_hold_is_the_default():
    assert program().interpolation == "hold"


def test_hold_gives_a_clean_single_turn_pulse():
    kicks = [program().value_at(turn) for turn in range(1, 8)]
    assert kicks == [0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0]


def test_linear_leaks_the_kick_before_the_kicker_fires():
    """Why ``hold`` exists: two thirds of the kick arrives a turn early."""
    leaked = [program(interpolation="linear").value_at(t) for t in (2, 3)]
    assert leaked == pytest.approx([1.0e-3 / 3, 2.0e-3 / 3])


def test_spline_runs_through_the_knots():
    splined = program(interpolation="spline")
    for turn, value in zip(PULSE["turns"], PULSE["values"]):
        assert splined.value_at(turn) == pytest.approx(value)


def test_a_spline_through_two_knots_warns_and_goes_linear():
    with pytest.warns(UserWarning, match="do not define one"):
        value = DeviceProgram(
            element="K", turns=[1, 3], values=[0.0, 1.0], interpolation="spline"
        ).value_at(2)
    assert value == pytest.approx(0.5)


def test_the_value_is_held_outside_the_programmed_turns():
    """As every code does, so a pulse must come back down with a final knot."""
    bumper = DeviceProgram(element="K", turns=[3, 6], values=[0.0, 1.0e-3])
    assert bumper.value_at(1) == 0.0
    assert bumper.value_at(1000) == 1.0e-3


@pytest.mark.parametrize("rule", ("hold", "linear", "spline"))
def test_linear_knots_reproduce_the_rule_at_every_turn(rule):
    """Every code joins samples with straight lines, so the backends rest on this."""
    programmed = program(interpolation=rule)
    turns, values = programmed.linear_knots()
    for turn in range(1, 8):
        assert np.interp(turn, turns, values) == pytest.approx(
            programmed.value_at(turn), abs=1e-15
        )


def test_a_hold_step_is_written_between_two_tracked_turns():
    """Half a turn either side is also the tolerance on a slightly-off ``T_rev``."""
    turns, _ = program().linear_knots()
    risers = [t for t in turns if t != int(t)]
    assert risers == [3.25, 3.75, 4.25, 4.75]


def test_peak_keeps_its_sign():
    assert DeviceProgram(element="K", turns=[1, 2], values=[0.0, -2.0]).peak == -2.0


def test_factor_knots_are_the_shape_without_the_amplitude():
    _, factors = program().factor_knots(1.0)
    assert max(factors) == pytest.approx(1.0)


def test_time_knots_put_the_origin_turn_at_zero():
    """elegant measures from the firing pass, Xsuite from the start of the run."""
    period = 1e-6
    from_start, _ = program().time_knots(period)
    from_firing, _ = program().time_knots(period, origin_turn=4)
    assert from_start[0] == pytest.approx(0.0)
    assert from_firing[0] == pytest.approx(-3 * period)


def test_a_code_that_cannot_says_so():
    line = FakeLine({"programs": [{"element": "KICK1", **PULSE}]}, code="astra")
    with pytest.warns(UserWarning, match="no way to change an element"):
        line.check_programs_supported()


def test_a_code_that_can_says_nothing(recwarn):
    line = FakeLine(
        {"programs": [{"element": "KICK1", **PULSE}]}, code="elegant", supports=True
    )
    line.check_programs_supported()
    assert not [w for w in recwarn if "no way to change" in str(w.message)]


def test_nothing_is_said_when_nothing_is_programmed(recwarn):
    FakeLine({"turns": 10}, code="astra").check_programs_supported()
    assert len(recwarn) == 0


def test_a_run_that_ends_before_the_program_does_warns():
    """Otherwise a clean run of a machine where the kicker never fired."""
    line = FakeLine({"turns": 3, "programs": [{"element": "KICK1", **PULSE}]})
    with pytest.warns(UserWarning, match="ends part-way"):
        line.check_programs_fit()


def test_a_program_left_switched_on_warns():
    """A pulse with no closing knot is a bumper (elegant's ``ramp_elements`` trap)."""
    line = FakeLine(
        {
            "turns": 1000,
            "programs": [{"element": "K", "turns": [1, 4], "values": [0.0, 1.0e-3]}],
        }
    )
    with pytest.warns(UserWarning, match="holds for the remaining"):
        line.check_programs_fit()


def test_a_pulse_that_comes_back_down_is_quiet(recwarn):
    line = FakeLine({"turns": 1000, "programs": [{"element": "K", **PULSE}]})
    line.check_programs_fit()
    assert len(recwarn) == 0


class FakeElement:
    def __init__(self, hardware_type):
        self.hardware_type = hardware_type


@pytest.mark.parametrize(
    "hardware, vertical",
    (
        ("Horizontal_AC_Dipole", False),
        ("Vertical_AC_Dipole", True),
        ("Horizontal_Corrector", False),
        ("Vertical_Corrector", True),
        ("Quadrupole", False),
    ),
)
def test_the_plane_comes_from_the_lattice_not_the_built_element(hardware, vertical):
    """A kicker's program starts at zero, so the built element has no plane to read."""
    line = FakeLine()
    line._elements["K"] = FakeElement(hardware)
    assert line.program_is_vertical("K") is vertical


def test_an_unknown_element_is_treated_as_horizontal():
    assert FakeLine().program_is_vertical("nope") is False


# The backends borrow the real lattice methods and run them: inspecting what
# simba would write is not enough, as each convention has a wrong version that
# exports, parses and tracks without complaint.

CLIGHT = 299792458.0


class FakeXsuite(FakeLine):
    """``xsuiteLattice``'s program binding, over a real three-element ring."""

    bind_programs = xsuiteLattice.bind_programs
    revolution_period = xsuiteLattice.revolution_period
    program_attributes = xsuiteLattice.program_attributes
    program_signs = xsuiteLattice.program_signs

    def __init__(self, tracking, element=None):
        super().__init__(tracking, code="xsuite", supports=True)
        xt = pytest.importorskip("xtrack")
        self.line = xt.Line(
            elements=[
                xt.Drift(length=LENGTH / 2),
                element if element is not None else xt.Multipole(knl=[0.0], length=0.0),
                xt.Drift(length=LENGTH / 2),
            ],
            element_names=["d1", "KICK1", "d2"],
        )
        self.line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV)
        self.line.build_tracker()

    def kicks(self, turns=8):
        """``px`` at the start of each turn, so index ``i`` is after ``i`` turns."""
        xt = pytest.importorskip("xtrack")
        self.bind_programs()
        particles = self.line.build_particles(x=0, px=0, y=0, py=0)
        monitor = xt.ParticlesMonitor(
            start_at_turn=0, stop_at_turn=turns, num_particles=1
        )
        self.line.track(particles, num_turns=turns, turn_by_turn_monitor=monitor)
        return np.asarray(monitor.px).ravel()


class TestXsuite:
    """``t_turn_s`` bound to a ``FunctionPieceWiseLinear``: no Python loop."""

    @staticmethod
    def line(**overrides):
        return FakeXsuite({"programs": [{"element": "KICK1", **PULSE, **overrides}]})

    def test_the_kick_lands_on_the_programmed_turn(self):
        px = self.line().kicks()
        assert list(px[:4]) == [0.0, 0.0, 0.0, 0.0]
        assert px[4] == pytest.approx(1.0e-3)

    def test_it_is_one_turn_wide(self):
        """``px`` is cumulative, so a second firing would show as a second step."""
        px = self.line().kicks()
        assert px[5] == pytest.approx(px[4])
        assert px[7] == pytest.approx(px[4])

    def test_linear_interpolation_would_have_kicked_early(self):
        """Why ``hold`` is the default, measured through the tracker."""
        px = self.line(interpolation="linear").kicks()
        assert px[2] == pytest.approx(1.0e-3 / 3, rel=1e-6)

    def test_a_positive_value_deflects_toward_positive_x(self):
        """``knl[0]`` negated; nothing downstream of a sign error looks wrong."""
        assert self.line().kicks()[4] > 0

    def test_an_element_simba_has_no_default_for_says_so(self):
        xt = pytest.importorskip("xtrack")
        line = FakeXsuite(
            {"programs": [{"element": "KICK1", **PULSE}]},
            element=xt.Cavity(voltage=0.0, frequency=1e6),
        )
        with pytest.warns(UserWarning, match="no default attribute"):
            line.bind_programs()

    def test_a_named_parameter_is_set_verbatim(self):
        """The escape hatch: no plane lookup and no sign flip."""
        xt = pytest.importorskip("xtrack")
        line = FakeXsuite(
            {
                "programs": [
                    {"element": "KICK1", "parameter": "voltage",
                     "turns": [1, 2], "values": [0.0, 5.0e6]}
                ]
            },
            element=xt.Cavity(voltage=0.0, frequency=1e6),
        )
        line.bind_programs()
        line.line.vars["t_turn_s"] = 10 * line.revolution_period
        assert float(line.line["KICK1"].voltage) == pytest.approx(5.0e6)

    def test_an_element_that_is_not_there_says_so(self):
        line = FakeXsuite({"programs": [{"element": "NOPE", **PULSE}]})
        with pytest.warns(UserWarning, match="not in the xsuite lattice"):
            line.bind_programs()

    def test_a_vertical_element_with_no_default_says_so(self):
        """Rather than ``TypeError: 'NoneType' object is not subscriptable``."""
        xt = pytest.importorskip("xtrack")
        line = FakeXsuite(
            {"programs": [{"element": "KICK1", **PULSE}]},
            element=xt.Cavity(voltage=0.0, frequency=1e6),
        )
        line._elements["KICK1"] = SimpleNamespace(hardware_type="Vertical_Corrector")
        with pytest.warns(UserWarning, match="no default attribute"):
            line.bind_programs()

    def test_the_revolution_period_uses_beta0(self):
        """Not c: one part in ``2*gamma**2`` per turn adds up to a whole turn."""
        line = self.line()
        beta0 = float(np.atleast_1d(line.line.particle_ref.beta0)[0])
        assert line.revolution_period == pytest.approx(
            LENGTH / (beta0 * CLIGHT), rel=1e-15
        )
        assert line.revolution_period > LENGTH / CLIGHT


class FakeOcelot(FakeLine):
    """``ocelotLattice.apply_programs`` over a real three-element ring."""

    apply_programs = ocelotLattice.apply_programs

    def __init__(self, tracking, element=None):
        super().__init__(tracking, code="ocelot", supports=True)
        pytest.importorskip("ocelot")
        from ocelot.cpbd.elements import Drift, Hcor
        from ocelot.cpbd.magnetic_lattice import MagneticLattice

        self.kicker = element if element is not None else Hcor(
            l=0.0, angle=0.0, eid="KICK1"
        )
        self.lat_obj = MagneticLattice(
            (
                Drift(l=LENGTH / 2, eid="d1"),
                self.kicker,
                Drift(l=LENGTH / 2, eid="d2"),
            )
        )

    def kicks(self, turns=7):
        from ocelot.cpbd.beam import ParticleArray
        from ocelot.cpbd.navi import Navigator
        from ocelot.cpbd.track import track

        particles = ParticleArray(n=1)
        particles.rparticles[:] = 0.0
        particles.E = 1.0
        history = []
        for turn in range(1, turns + 1):
            self.apply_programs(turn)
            navigator = Navigator(self.lat_obj)
            navigator.go_to_start()
            _, particles = track(
                self.lat_obj, particles, navi=navigator, calc_tws=False,
                print_progress=False,
            )
            history.append(float(particles.rparticles[1, 0]))
        return history


class TestOcelot:
    """The attribute set between turns of simba's own loop."""

    @staticmethod
    def line(**overrides):
        return FakeOcelot({"programs": [{"element": "KICK1", **PULSE, **overrides}]})

    def test_the_kick_lands_on_the_programmed_turn(self):
        px = self.line().kicks()
        assert px[:3] == [0.0, 0.0, 0.0]
        assert px[3] == pytest.approx(1.0e-3)

    def test_the_corrector_reads_its_angle_at_apply_time(self):
        """No ``update_transfer_maps()``, yet the kick stops (``px`` is cumulative)."""
        px = self.line().kicks()
        assert px[4] == pytest.approx(px[3])
        assert px[6] == pytest.approx(px[3])

    def test_a_positive_value_deflects_toward_positive_x(self):
        assert self.line().kicks()[3] > 0

    def test_an_element_that_is_not_there_says_so(self):
        line = FakeOcelot({"programs": [{"element": "NOPE", **PULSE}]})
        with pytest.warns(UserWarning, match="not in the ocelot lattice"):
            line.apply_programs(4)

    def test_an_element_without_the_attribute_says_so(self):
        from ocelot.cpbd.elements import Marker

        line = FakeOcelot(
            {"programs": [{"element": "KICK1", **PULSE}]},
            element=Marker(eid="KICK1"),
        )
        with pytest.warns(UserWarning, match="no 'angle' to set"):
            line.apply_programs(4)


class FakeMadx(FakeLine):
    """``madxLattice.apply_programs`` against a real, thin-sliced sequence."""

    apply_programs = madxLattice.apply_programs
    bind_programs = madxLattice.bind_programs
    check_programs_present = madxLattice.check_programs_present
    program_variable = staticmethod(madxLattice.program_variable)
    program_attribute = frameworkLattice.program_attribute
    program_attributes = madxLattice.program_attributes

    def __init__(self, tracking, etype="HKICKER", length=0.0, attributes=""):
        super().__init__(tracking, code="madx", supports=True)
        cpymad = pytest.importorskip("cpymad.madx")
        self.segments = [["D1", "KICK1", "D2"]]
        self._madx = cpymad.Madx(stdout=False)
        # in `track_one_turn`'s order: turn 1 set before the sequence exists,
        # then the sequence, the binding, and the slicing
        self.apply_programs(1)
        self._madx.input(
            f"""
            D1: DRIFT, L={(LENGTH - length) / 2};
            KICK1: {etype}, L={length}{attributes};
            D2: DRIFT, L={(LENGTH - length) / 2};
            RING: SEQUENCE, L={LENGTH};
              D1, AT={(LENGTH - length) / 4};
              KICK1, AT={LENGTH / 2};
              D2, AT={LENGTH - (LENGTH - length) / 4};
            ENDSEQUENCE;
            BEAM, PARTICLE=ELECTRON, PC=1.0;
            """
        )
        self.bind_programs(self._madx, self.segments[0])
        self._madx.input(
            """
            USE, SEQUENCE=RING;
            SELECT, FLAG=MAKETHIN, CLEAR;
            SELECT, FLAG=MAKETHIN, CLASS=QUADRUPOLE, SLICE=4;
            MAKETHIN, SEQUENCE=RING, STYLE=TEAPOT;
            USE, SEQUENCE=RING;
            """
        )

    def kicks(self, turns=7, coordinate="px"):
        history = []
        for turn in range(1, turns + 1):
            self.apply_programs(turn)
            self._madx.input("USE, SEQUENCE=RING;")
            self._madx.input(
                """
                TRACK, ONEPASS, ONETABLE;
                  START, X=0, PX=0, Y=0, PY=0, T=0, PT=0;
                  OBSERVE, PLACE=#E;
                  RUN, TURNS=1;
                ENDTRACK;
                """
            )
            history.append(float(list(getattr(self._madx.table.trackone, coordinate))[-1]))
        return history

    def close(self):
        self._madx.exit()


class TestMadx:
    """``name->attribute`` re-stated between turns of :func:`run_segments`."""

    @pytest.fixture(autouse=True)
    def _run_somewhere_disposable(self, tmp_path, monkeypatch):
        """MAD-X ``TRACK`` drops ``checkpoint_restart.dat`` in the working directory."""
        monkeypatch.chdir(tmp_path)

    @staticmethod
    def line(**overrides):
        return FakeMadx({"programs": [{"element": "KICK1", **PULSE, **overrides}]})

    def test_the_kick_reaches_a_thin_sliced_sequence(self):
        """The sliced sequence is only re-``USE``d on later turns, not rebuilt."""
        line = self.line()
        try:
            px = line.kicks()
        finally:
            line.close()
        assert px[:3] == [0.0, 0.0, 0.0]
        assert px[3] == pytest.approx(1.0e-3)

    def test_it_comes_back_down(self):
        """Each turn is tracked from zero, so this reads the value back directly."""
        line = self.line()
        try:
            assert line.kicks()[4] == pytest.approx(0.0)
        finally:
            line.close()

    def test_a_positive_value_deflects_toward_positive_x(self):
        line = self.line()
        try:
            assert line.kicks()[3] > 0
        finally:
            line.close()

    def test_the_attribute_is_chosen_from_the_madx_base_type(self):
        """``kick`` for an ``hkicker``, ``volt`` for an ``hacdipole``."""
        line = FakeMadx(
            {"programs": [{"element": "KICK1", "turns": [1, 2], "values": [0.0, 7.0]}]},
            etype="HACDIPOLE",
        )
        try:
            line.apply_programs(2)
            assert float(line._madx.elements["kick1"].volt) == pytest.approx(7.0)
        finally:
            line.close()

    def test_an_element_that_is_not_there_says_so(self):
        line = FakeMadx({"programs": [{"element": "NOPE", **PULSE}]})
        try:
            with pytest.warns(UserWarning, match="not in the madx lattice"):
                line.check_programs_present()
        finally:
            line.close()

    def test_a_type_with_no_default_attribute_says_so(self):
        """When its segment is defined, which is when MAD-X can say."""
        with pytest.warns(UserWarning, match="no default attribute"):
            FakeMadx(
                {"programs": [{"element": "KICK1", **PULSE}]}, etype="MARKER"
            ).close()

    def test_turn_one_is_set_before_the_element_exists(self):
        """Turn 1 used to be tracked at the lattice's own value, with a warning."""
        line = FakeMadx(
            {"programs": [{"element": "KICK1", "turns": [1, 2], "values": [5.0e-4, 0.0]}]}
        )
        try:
            assert float(line._madx.elements["kick1"].kick) == pytest.approx(5.0e-4)
        finally:
            line.close()

    def test_a_sliced_element_follows_its_program(self):
        """``MAKETHIN`` slices keep their cut strength; ``QF->K1`` moved nothing."""
        line = FakeMadx(
            {"programs": [{"element": "KICK1", "parameter": "k1",
                           "turns": [1, 2], "values": [1.0, 0.25]}]},
            etype="QUADRUPOLE", length=1.0,
        )
        try:
            slices = [name for name in line._madx.elements if name.startswith("kick1..")]
            assert len(slices) == 4
            line.apply_programs(2)
            for name in slices:
                assert float(line._madx.elements[name].knl[1]) == pytest.approx(0.25 / 4)
        finally:
            line.close()

    def test_a_named_knl_follows_its_program_and_keeps_its_other_orders(self):
        """``M->KNL = ...`` does nothing at all to a multipole."""
        line = FakeMadx(
            {"programs": [{"element": "KICK1", "parameter": "knl",
                           "turns": [1, 2], "values": [0.0, 3.0e-4]}]},
            etype="MULTIPOLE", attributes=", KNL={0.0, 0.5}",
        )
        try:
            line.apply_programs(2)
            knl = [float(value) for value in line._madx.elements["kick1"].knl]
            assert knl[:2] == pytest.approx([3.0e-4, 0.5])
        finally:
            line.close()

    @pytest.mark.parametrize("hardware, attribute", [
        ("Horizontal_Kicker", "hkick"), ("Vertical_Kicker", "vkick"),
    ])
    def test_a_kicker_is_kicked_in_its_own_plane(self, hardware, attribute):
        """A ``KICKER`` has both planes, and was always given ``hkick``."""
        line = FakeMadx(
            {"programs": [{"element": "KICK1", "turns": [1, 2], "values": [0.0, 2.0e-4]}]},
            etype="KICKER",
        )
        try:
            line._elements["KICK1"] = SimpleNamespace(hardware_type=hardware)
            line.bind_programs(line._madx, line.segments[0])
            line.apply_programs(2)
            assert float(getattr(line._madx.elements["kick1"], attribute)) == pytest.approx(2.0e-4)
        finally:
            line.close()

    @pytest.mark.parametrize("hardware, coordinate", [
        ("Horizontal_Kicker", "px"), ("Vertical_Kicker", "py"),
    ])
    def test_a_multipole_cannot_be_kicked_and_says_so(self, hardware, coordinate):
        """MAD-X kicks by ``-(KNL[0] - ANGLE)`` and ``ANGLE`` defaults to
        ``KNL[0]``, so a program bound to ``KNL`` silently moved nothing."""
        with pytest.warns(UserWarning, match="kicks nothing"):
            line = FakeMadx(
                {"programs": [{"element": "KICK1", **PULSE}]},
                etype="MULTIPOLE", attributes=", KNL={0.0, 0.0}, KSL={0.0, 0.0}",
            )
        try:
            line._elements["KICK1"] = SimpleNamespace(hardware_type=hardware)
            with pytest.warns(UserWarning, match="kicks nothing"):
                line.bind_programs(line._madx, line.segments[0])
            assert line.kicks(coordinate=coordinate) == [0.0] * 7
            for attribute in ("knl", "ksl"):
                line._madx.input(f"kick1, {attribute}:={{1.0e-3, 0.0}};")
                assert line.kicks(coordinate=coordinate) == [0.0] * 7, attribute
        finally:
            line.close()


class FakeElegant(FakeLine):
    """``elegantLattice``'s ``&alter_elements`` commands and SDDS sidecar."""

    program_commands = elegantLattice.program_commands
    write_program_waveform = elegantLattice.write_program_waveform
    program_elements = elegantLattice.program_elements
    _elegant_type = elegantLattice._elegant_type

    def __init__(self, tracking, directory, hardware="ac_dipole"):
        super().__init__(tracking, code="elegant", supports=True)
        from laura.models.element import HorizontalACDipole, HorizontalCorrector

        self.files = []
        self.global_parameters = {"master_subdir": str(directory)}
        self.revolution_period = LENGTH / CLIGHT
        if hardware == "ac_dipole":
            element = HorizontalACDipole(
                name="KICK1", machine_area="INJ",
                physical={"length": 0.0},
                simulation={"field_amplitude": 1.0},
            )
        else:
            element = HorizontalCorrector(
                name="KICK1", machine_area="INJ",
                physical={"length": 0.0},
                magnetic={"horizontal_kick": 0.0},
            )
        self._elements["KICK1"] = element


@pytest.fixture
def elegant_line(tmp_path):
    return FakeElegant(
        {"turns": 8, "programs": [{"element": "KICK1", **PULSE}]}, tmp_path
    )


class TestElegantCommands:
    """``FIRE_ON_PASS`` plus a ``WAVEFORM``; ``&ramp_elements`` cannot express a pulse."""

    def test_an_ac_dipole_gets_three_alter_elements_commands(self, elegant_line):
        commands = elegant_line.program_commands()
        items = sorted(c.item for c in commands.values())
        assert items == ["ANGLE", "FIRE_ON_PASS", "WAVEFORM"]

    def test_each_command_names_the_element_it_alters(self, elegant_line):
        """``objectname`` aliases ``name``, which once named the command and left
        no target, an error elegant only raises at run time."""
        commands = elegant_line.program_commands()
        for command in commands.values():
            assert "name = KICK1" in command.write_Elegant()

    def test_fire_on_pass_is_zero_based(self, elegant_line):
        """The program starts at turn 1 (pass 0); the waveform carries the wait."""
        commands = elegant_line.program_commands()
        (fire,) = [c for c in commands.values() if c.item == "FIRE_ON_PASS"]
        assert fire.value == 0

    def test_the_strength_goes_out_as_the_peak(self, elegant_line):
        """``BUMPER`` is a strength times a waveform in ``[-1, 1]``."""
        commands = elegant_line.program_commands()
        (angle,) = [c for c in commands.values() if c.item == "ANGLE"]
        assert angle.value == pytest.approx(1.0e-3)

    def test_the_waveform_is_written_beside_the_lattice(self, elegant_line, tmp_path):
        elegant_line.program_commands()
        written = tmp_path / "KICK1_program.sdds"
        assert written.is_file()
        assert "t" in written.read_text() and "factor" in written.read_text()

    def test_the_waveform_time_axis_is_seconds_from_the_firing_pass(
        self, elegant_line
    ):
        """It keeps running across passes, so a whole program is one table."""
        programmed = elegant_line.programs[0]
        times, _ = programmed.factor_knots(
            elegant_line.revolution_period, origin_turn=programmed.first_turn
        )
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(4 * elegant_line.revolution_period)

    def test_an_element_elegant_cannot_program_says_so(self, tmp_path):
        """A corrector is an ``HKICK``, which has no ``WAVEFORM``."""
        line = FakeElegant(
            {"turns": 8, "programs": [{"element": "KICK1", **PULSE}]},
            tmp_path,
            hardware="corrector",
        )
        with pytest.warns(UserWarning, match="Only a BUMPER or MBUMPER"):
            assert line.program_commands() == {}


@needs_elegant
class TestElegantRun:
    """The commands and the sidecar, through a real elegant run."""

    @staticmethod
    def kicks(line, tmp_path, passes=8):
        """``xp`` after each pass, with simba's own commands and waveform."""
        commands = line.program_commands()
        altered = "".join(command.write_Elegant() for command in commands.values())
        (tmp_path / "ring.lte").write_text(
            f"D1: DRIF, L={LENGTH / 2}\n"
            "KICK1: BUMPER, L=0.0\n"
            f"D2: DRIF, L={LENGTH / 2}\n"
            'W: WATCH, FILENAME="w-%03ld.sdds", MODE=coordinate, INTERVAL=1\n'
            "RING: LINE=(D1,KICK1,D2,W)\n"
        )
        (tmp_path / "run.ele").write_text(
            "&run_setup\n"
            '  lattice = "ring.lte", use_beamline = RING, p_central_mev = 1000,\n'
            '  default_order = 1, output = "out.sdds"\n'
            "&end\n" + altered
            + f"&run_control n_passes = {passes} &end\n"
            "&bunched_beam n_particles_per_bunch = 1, emit_x = 0, emit_y = 0,\n"
            "  sigma_dp = 0, sigma_s = 0 &end\n"
            "&track &end\n"
        )
        subprocess.run(
            [ELEGANT, "run.ele"], cwd=tmp_path, check=True, capture_output=True
        )
        read = subprocess.run(
            ["sdds2stream", "-col=xp", "w-001.sdds"],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )
        return [float(value) for value in read.stdout.split()]

    def test_the_kick_lands_on_the_programmed_turn(self, elegant_line, tmp_path):
        xp = self.kicks(elegant_line, tmp_path)
        assert xp[:3] == [0.0, 0.0, 0.0]
        assert xp[3] == pytest.approx(1.0e-3, rel=1e-3)

    def test_it_is_one_turn_wide(self, elegant_line, tmp_path):
        """The time axis keeps running, so only ``hold`` closing the window stops it."""
        xp = self.kicks(elegant_line, tmp_path)
        assert xp[7] == pytest.approx(xp[3], rel=1e-3)

    def test_a_positive_value_deflects_toward_positive_x(
        self, elegant_line, tmp_path
    ):
        assert self.kicks(elegant_line, tmp_path)[3] > 0


class RecordingTao:
    """A Tao stand-in that keeps the commands it was given."""

    def __init__(self):
        self.commands = []

    def cmd(self, command, raises=True):
        self.commands.append(command)
        return []


class FakeBmad(FakeLine):
    """``bmadLattice.apply_programs``, which drives Tao by ``set element``."""

    apply_programs = bmadLattice.apply_programs
    program_attributes = bmadLattice.program_attributes
    _bmad_type = bmadLattice._bmad_type

    def __init__(self, tracking, hardware="Horizontal_AC_Dipole"):
        super().__init__(tracking, code="bmad", supports=True)
        from laura.models.element import (
            HorizontalACDipole,
            HorizontalCorrector,
            VerticalACDipole,
        )

        self.tao = RecordingTao()
        kinds = {
            "Horizontal_AC_Dipole": HorizontalACDipole,
            "Vertical_AC_Dipole": VerticalACDipole,
        }
        if hardware in kinds:
            element = kinds[hardware](
                name="KICK1", machine_area="INJ",
                physical={"length": 0.0},
                simulation={"field_amplitude": 1.0},
            )
        else:
            element = HorizontalCorrector(
                name="KICK1", machine_area="INJ",
                physical={"length": 0.0},
                magnetic={"horizontal_kick": 0.0},
            )
        self._elements["KICK1"] = element


class TestBmad:
    """Programs only inside the Tao turn loops, hence no ``supports_turns``."""

    @staticmethod
    def line(**kwargs):
        return FakeBmad({"programs": [{"element": "KICK1", **PULSE}]}, **kwargs)

    def test_an_ac_kicker_is_programmed_in_integrated_field(self):
        """``bl_hkick``, not ``hkick``, which is in radians; invisible in the output."""
        line = self.line()
        line.apply_programs(4)
        assert line.tao.commands == ["set element KICK1 bl_hkick = 0.001"]

    def test_the_vertical_plane_gets_the_vertical_attribute(self):
        line = self.line(hardware="Vertical_AC_Dipole")
        line.apply_programs(4)
        assert line.tao.commands == ["set element KICK1 bl_vkick = 0.001"]

    def test_a_corrector_is_programmed_in_radians(self):
        """An ``hkicker``'s ``kick`` *is* the angle, so no prefix."""
        line = self.line(hardware="corrector")
        line.apply_programs(4)
        assert line.tao.commands == ["set element KICK1 kick = 0.001"]

    def test_the_quiet_turns_are_set_too(self):
        """Tao holds whatever it was last given."""
        line = self.line()
        line.apply_programs(5)
        assert line.tao.commands == ["set element KICK1 bl_hkick = 0.0"]

    def test_nothing_is_driven_before_tao_exists(self, recwarn):
        """``apply_programs`` is reachable before a session is open."""
        line = self.line()
        line.tao = None
        line.apply_programs(4)
        assert len(recwarn) == 0

    def test_an_element_that_is_not_there_says_so(self):
        """Even with the attribute named: Tao's ``raises=False`` swallowed it."""
        line = FakeBmad(
            {"programs": [{"element": "NOPE", "parameter": "kick", **PULSE}]}
        )
        with pytest.warns(UserWarning, match="not in the bmad lattice"):
            line.apply_programs(4)
        assert line.tao.commands == []
