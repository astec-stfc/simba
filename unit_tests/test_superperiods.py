"""One sector of an N-fold-symmetric ring, tracked N times per turn.

A real ring is usually built as N identical sectors, and writing the lattice
out N times is both tedious and a lie about what the machine is. ``tracking:
{nsuperperiods: N}`` says the line is one sector, and the backends traverse it
N times before calling it a turn.

**A turn is N passes, and stays one turn.** That is the whole convention, and
it was measured rather than assumed, in both codes that have a native notion
of it:

* Ocelot's ``track_nturns(..., nsuperperiods=N)`` on a sector is bit-identical
  to ``nsuperperiods=1`` on N copies of that sector, and both record one point
  per turn rather than one per pass.
* Xtrack has no notion of a sector at all, so ``num_turns`` is multiplied and
  everything counted per turn is converted back. The final state is
  bit-identical to N copies tracked once per turn; ``at_turn`` counts passes
  and is floor-divided; and a ``ParticlesMonitor`` records the state at the
  *start* of each pass -- the first sample is the launch condition -- so the
  sector boundaries fall at 0, N, 2N and the stride starts from zero. Taking
  ``[N-1::N]`` instead is off by one sector and silently measures a different
  trajectory, which is why it is pinned below.

The failure this guards against is not a crash. A sector tracked once per turn
runs, converges, and describes one Nth of a ring -- a machine that does not
exist. Hence ``check_nsuperperiods_supported`` being loud, and
``check_superperiods_close`` checking the declared count against the geometry.
"""

import math
import warnings

import pytest

from simba.Codes.ASTRA.ASTRA import astraLattice
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

CAN = [ocelotLattice, madxLattice, xsuiteLattice]
CANNOT = [elegantLattice, bmadLattice, astraLattice]


class FakeLine:
    """A lattice stub carrying only what the superperiod code reads."""

    def __init__(self, file_block=None, code="ocelot", supports=True):
        self.file_block = file_block or {}
        self.code = code
        self.objectname = "RING"
        self.supports_nsuperperiods = supports

    nsuperperiods = frameworkLattice.nsuperperiods
    passes_per_turn = frameworkLattice.passes_per_turn
    check_nsuperperiods_supported = frameworkLattice.check_nsuperperiods_supported


# --- reading the count --------------------------------------------------


def test_no_setting_means_one_pass_per_turn():
    assert FakeLine().nsuperperiods == 1


def test_an_empty_tracking_block_means_one():
    assert FakeLine({"tracking": {}}).nsuperperiods == 1


def test_a_null_tracking_block_means_one():
    """A key present but empty is how YAML hands over ``tracking:``."""
    assert FakeLine({"tracking": None}).nsuperperiods == 1


def test_the_count_is_read_from_the_files_block():
    assert FakeLine({"tracking": {"nsuperperiods": 4}}).nsuperperiods == 4


def test_a_string_count_is_coerced():
    """Every other numeric setting arrives coerced rather than type-checked."""
    assert FakeLine({"tracking": {"nsuperperiods": "6"}}).nsuperperiods == 6


def test_a_nonsense_count_warns_and_falls_back_to_one():
    line = FakeLine({"tracking": {"nsuperperiods": "many"}})
    with pytest.warns(UserWarning, match="not a whole number"):
        assert line.nsuperperiods == 1


def test_zero_superperiods_warns_and_falls_back_to_one():
    """Zero passes per turn is not a smaller ring, it is no tracking."""
    line = FakeLine({"tracking": {"nsuperperiods": 0}})
    with pytest.warns(UserWarning, match="not a count"):
        assert line.nsuperperiods == 1


def test_a_negative_count_warns_and_falls_back_to_one():
    line = FakeLine({"tracking": {"nsuperperiods": -4}})
    with pytest.warns(UserWarning, match="not a count"):
        assert line.nsuperperiods == 1


# --- asked for against actually happening -------------------------------


def test_passes_per_turn_is_the_count_on_a_capable_code():
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, supports=True)
    assert line.passes_per_turn == 4


def test_passes_per_turn_is_one_on_a_code_that_cannot():
    """`nsuperperiods` is what was asked for; this is what will happen."""
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, supports=False)
    assert line.nsuperperiods == 4
    assert line.passes_per_turn == 1


def test_passes_per_turn_is_one_by_default_everywhere():
    for supports in (True, False):
        assert FakeLine(supports=supports).passes_per_turn == 1


# --- which codes can honour it ------------------------------------------


@pytest.mark.parametrize("cls", CAN, ids=lambda c: c.__name__)
def test_the_ones_that_can(cls):
    assert cls.supports_nsuperperiods is True


@pytest.mark.parametrize("cls", CANNOT, ids=lambda c: c.__name__)
def test_the_ones_that_cannot(cls):
    assert cls.supports_nsuperperiods is False


def test_the_base_class_assumes_it_cannot():
    """So a backend gains superperiods by declaring it, never by omission."""
    assert frameworkLattice.supports_nsuperperiods is False


def test_the_warning_names_every_code_that_can():
    """The named list and the declared flags have to agree, or the advice
    sends the user to a backend that will warn at them again."""
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, code="elegant", supports=False)
    with pytest.warns(UserWarning) as caught:
        line.check_nsuperperiods_supported()
    message = str(caught[0].message).lower()
    for cls in CAN:
        name = cls.model_fields["code"].default
        # the warning spells it MAD-X; the class calls itself madx
        assert name.lower() in message or name.lower() == "madx" and "mad-x" in message


# --- the warning --------------------------------------------------------


def test_asking_a_code_that_cannot_warns():
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, code="elegant", supports=False)
    with pytest.warns(UserWarning, match="superperiods"):
        line.check_nsuperperiods_supported()


def test_the_warning_says_it_is_a_different_machine():
    """Not a coarser one. Every other unsupported setting degrades the run;
    this one changes what is being tracked."""
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, code="elegant", supports=False)
    with pytest.warns(UserWarning, match="one 4th of the intended ring"):
        line.check_nsuperperiods_supported()


def test_a_capable_code_is_silent():
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, code="ocelot", supports=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_nsuperperiods_supported()


def test_one_superperiod_is_silent_everywhere():
    """The default must never warn, on any code."""
    for code, supports in (("elegant", False), ("ocelot", True)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FakeLine({}, code=code, supports=supports).check_nsuperperiods_supported()


def test_it_is_checked_during_preprocessing():
    """One settings file driving several codes is ordinary, so the warning
    has to reach the user on the run rather than on an explicit call."""
    import inspect

    source = inspect.getsource(frameworkLattice.preProcess)
    assert "check_nsuperperiods_supported()" in source


# --- the declared count against the geometry ----------------------------
#
# Getting the count wrong is otherwise invisible: six sectors of a four-fold
# ring tracks perfectly and models nothing. The net bend answers it --
# N * angle should be a whole number of turns.

from laura.models.element import Dipole, Drift
from laura.models.element_list import MachineModel


def sector(nbend, angle, nsuperperiods=1, turns=1000):
    """A line of `nbend` bends of `angle`, each followed by a 1 m drift."""
    elements, order = {}, []
    for i in range(nbend):
        bend, drift = f"B{i}", f"D{i}"
        elements[bend] = Dipole(
            name=bend,
            hardware_class="Magnet",
            machine_area="A",
            magnetic={"magnetic_length": 1.0, "k0l": angle},
            physical={"length": 1.0},
        )
        elements[drift] = Drift(
            name=drift,
            hardware_class="Drift",
            hardware_type="Drift",
            machine_area="A",
            physical={"length": 1.0},
        )
        order += [bend, drift]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MachineModel(
            elements=elements,
            section={"sections": {"RING": order}},
            layout={"layouts": {"M": ["RING"]}, "default_layout": "M"},
        )
    return SectorLine(model, order, nsuperperiods, turns)


class SectorLine:
    """A stub exposing what the closure checks read off real geometry."""

    def __init__(self, model, order, nsuperperiods, turns):
        self.startObject = model[order[0]]
        self.endObject = model[order[-1]]
        self.elements = {name: model[name] for name in order}
        self.file_block = {
            "tracking": {"turns": turns, "nsuperperiods": nsuperperiods}
        }
        self.objectname = "RING"
        self.code = "ocelot"

    def _machine_geometry(self):
        """No layout behind this stub; `periodic` is never set here."""
        return None

    turns = frameworkLattice.turns
    periodic = frameworkLattice.periodic
    closed_geometry = frameworkLattice.closed_geometry
    nsuperperiods = frameworkLattice.nsuperperiods
    net_bend_angle = frameworkLattice.net_bend_angle
    check_turns_closed = frameworkLattice.check_turns_closed
    check_superperiods_close = frameworkLattice.check_superperiods_close


def test_a_correctly_counted_sector_is_silent():
    """One quarter of a ring, declared as one of four."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sector(1, math.pi / 2, nsuperperiods=4).check_turns_closed()


def test_declaring_superperiods_suppresses_the_closure_complaint():
    """A sector is *meant* to be open, so the plain closure test would only
    ever fire spuriously here."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sector(1, math.pi / 2, nsuperperiods=4).check_turns_closed()
    assert not [c for c in caught if "does not close" in str(c.message)]


def test_an_open_sector_still_warns_without_the_declaration():
    """Which is the behaviour this setting exists to give a way out of."""
    with pytest.warns(UserWarning, match="does not close"):
        sector(1, math.pi / 2, nsuperperiods=1).check_turns_closed()


def test_the_closure_warning_suggests_the_setting():
    """The message that used to say only 'track the whole ring instead'."""
    with pytest.warns(UserWarning, match="nsuperperiods: 4"):
        sector(1, math.pi / 2, nsuperperiods=1).check_turns_closed()


def test_a_wrong_count_warns():
    """Six sectors of a four-fold ring bend through one and a half turns."""
    with pytest.warns(UserWarning, match="do not make a closed ring"):
        sector(1, math.pi / 2, nsuperperiods=6).check_turns_closed()


def test_the_wrong_count_warning_suggests_the_right_one():
    with pytest.warns(UserWarning, match="suggests 4 instead"):
        sector(1, math.pi / 2, nsuperperiods=6).check_turns_closed()


def test_a_count_that_overshoots_by_a_whole_turn_is_accepted():
    """Eight quarter-sectors bend through two turns. That is a figure of
    eight or a double pass, not an error, and the check only asks for a whole
    number of turns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sector(1, math.pi / 2, nsuperperiods=8).check_turns_closed()


def test_a_straight_sector_is_not_judged():
    """No net bend says nothing either way -- a chicane-like sector can be
    perfectly periodic."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sector(4, 0.0, nsuperperiods=4).check_turns_closed()


def test_one_turn_never_checks_anything():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sector(1, math.pi / 2, nsuperperiods=6, turns=1).check_turns_closed()


# --- the revolution period is the ring's, not the sector's --------------
#
# This is what device programs convert against, so getting it wrong puts a
# kicker in the wrong turn rather than merely the wrong place.


class PeriodLine:
    """A stub exposing just the revolution-period path."""

    def __init__(self, nsuperperiods=1, supports=True, length=10.0):
        self.file_block = {"tracking": {"nsuperperiods": nsuperperiods}}
        self.objectname = "RING"
        self.code = "ocelot"
        self.supports_nsuperperiods = supports
        self.end = "END"
        self.entrance_s = 0.0
        self.machine = _Machine(length)
        self.global_parameters = {"beam": _Beam()}

    nsuperperiods = frameworkLattice.nsuperperiods
    passes_per_turn = frameworkLattice.passes_per_turn
    revolution_period = frameworkLattice.revolution_period


class _Machine:
    def __init__(self, length):
        self.length = length

    def get_elements_s_pos(self, end=None):
        return {"END": self.length}


class _Beam:
    """Ultra-relativistic enough that beta0 is 1 to the test's tolerance."""

    gamma = 1000.0
    BetaGamma = math.sqrt(1000.0**2 - 1)


def test_the_period_is_the_sector_crossing_time_by_default():
    from scipy.constants import speed_of_light

    period = PeriodLine(nsuperperiods=1).revolution_period
    assert period == pytest.approx(10.0 / speed_of_light, rel=1e-5)


def test_the_period_counts_every_pass():
    four = PeriodLine(nsuperperiods=4).revolution_period
    one = PeriodLine(nsuperperiods=1).revolution_period
    assert four == pytest.approx(4 * one)


def test_a_code_that_cannot_repeat_gets_the_sector_period():
    """It is going to track the sector once, so a program stated in turns has
    to land on the sector crossings -- the warning is what says the machine is
    wrong, not a silently mismatched clock on top of it."""
    asked = PeriodLine(nsuperperiods=4, supports=False).revolution_period
    one = PeriodLine(nsuperperiods=1).revolution_period
    assert asked == pytest.approx(one)


# --- the backends -------------------------------------------------------


def test_ocelot_hands_the_count_to_track_nturns():
    """The single-particle paths take it natively, so there is no loop to
    write: `track_nturns` has had the argument all along.

    Asserted as the invariant -- *every* call site passes it -- rather than
    as a count of them, which is the mistake three earlier tests here made
    and which breaks the moment a fourth legitimate call appears.
    """
    import inspect

    source = inspect.getsource(ocelotLattice)
    calls = source.count("track_nturns(")
    assert calls
    assert source.count("nsuperperiods=self.nsuperperiods") == calls


def test_ocelot_loops_the_sector_for_bunch_tracking():
    """`track` does not take the argument, so the bunch path loops."""
    import inspect

    source = inspect.getsource(ocelotLattice.run)
    assert "for sector in range(1, self.nsuperperiods + 1)" in source


def test_madx_loops_the_sector_inside_the_turn():
    """simba owns the MAD-X turn loop, so a superperiod is an inner loop over
    the same already-thin sequences."""
    import inspect

    source = inspect.getsource(madxLattice.run_segments)
    assert "for sector in range(1, self.nsuperperiods + 1)" in source


@pytest.mark.parametrize(
    "method", [ocelotLattice.run, madxLattice.run_segments],
    ids=["ocelot", "madx"],
)
def test_only_the_last_pass_of_a_turn_records(method):
    """Output is counted per turn, so a screen gives one beam file per turn
    rather than N. Both loops gate their writing on being the last sector --
    without it a four-fold ring quadruples every file it writes."""
    import inspect

    source = inspect.getsource(method)
    assert "sector == self.nsuperperiods" in source


def test_xsuite_multiplies_num_turns_rather_than_looping():
    """Xtrack has no notion of a sector, so every tracking call asks for
    `turns * passes_per_turn` and converts back."""
    import inspect

    source = inspect.getsource(xsuiteLattice)
    calls = [
        line.strip()
        for line in source.splitlines()
        if "num_turns=" in line and ".track(" in line
    ]
    assert calls
    for call in calls:
        assert "num_turns=self.turns)" not in call, call
        assert "num_turns=self.turns," not in call, call


def test_xsuite_converts_at_turn_back_into_turns():
    """`at_turn` counts passes, and a dynamic aperture is quoted in turns."""
    import inspect

    source = inspect.getsource(xsuiteLattice.run_dynamic_aperture)
    assert "// self.passes_per_turn" in source


# --- measured against the codes themselves ------------------------------


# Ocelot's `track_nturns` calls `aperture_limit`, which traces the lattice on
# a 1000-point grid and walks off the end of the sequence when the total
# length is not a round number -- an IndexError from deep inside `trace_z`,
# nothing to do with superperiods. Every length below is therefore 1 m.


def _xtrack_sector(copies=1):
    """A FODO cell with sextupoles, repeated `copies` times."""
    import xtrack as xt

    elements, names = [], []
    for copy in range(copies):
        for i in range(4):
            elements += [
                xt.Drift(length=0.5),
                xt.Multipole(knl=[0.0, 0.3], length=0.0),
                xt.Drift(length=0.5),
                xt.Multipole(knl=[0.0, -0.3], length=0.0),
                xt.Multipole(knl=[0.0, 0.0, 2.0], length=0.0),
            ]
            names += [f"d{i}a_{copy}", f"qf{i}_{copy}", f"d{i}b_{copy}",
                      f"qd{i}_{copy}", f"sx{i}_{copy}"]
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
    line.build_tracker()
    return line


def test_xsuite_a_sector_n_times_is_n_copies_once():
    """The measurement the backend is built on. Sextupoles are present so
    that agreement is not a linear coincidence."""
    pytest.importorskip("xtrack")
    import numpy as np

    n, turns = 3, 20
    one, many = _xtrack_sector(1), _xtrack_sector(n)
    a = one.build_particles(x=[1e-3], y=[0.5e-3])
    b = many.build_particles(x=[1e-3], y=[0.5e-3])
    one.track(a, num_turns=turns * n)
    many.track(b, num_turns=turns)
    assert float(a.x[0]) == float(b.x[0])
    assert float(a.px[0]) == float(b.px[0])


def test_xsuite_at_turn_counts_passes():
    """Which is why the dynamic-aperture scan floor-divides it."""
    pytest.importorskip("xtrack")

    n, turns = 3, 20
    line = _xtrack_sector(1)
    particles = line.build_particles(x=[1e-3], y=[0.5e-3])
    line.track(particles, num_turns=turns * n)
    assert int(particles.at_turn[0]) == turns * n
    assert int(particles.at_turn[0]) // n == turns


def test_xsuite_a_monitor_samples_the_start_of_each_pass():
    """So the sector boundaries are at 0, N, 2N and the stride starts from
    zero. `[N-1::N]` is off by one sector, runs, and is wrong."""
    pytest.importorskip("xtrack")
    import numpy as np
    import xtrack as xt

    n, turns = 3, 20
    one, many = _xtrack_sector(1), _xtrack_sector(n)
    a = one.build_particles(x=[1e-3], y=[0.5e-3])
    b = many.build_particles(x=[1e-3], y=[0.5e-3])
    monitor_a = xt.ParticlesMonitor(
        start_at_turn=0, stop_at_turn=turns * n, num_particles=1
    )
    monitor_b = xt.ParticlesMonitor(
        start_at_turn=0, stop_at_turn=turns, num_particles=1
    )
    one.track(a, num_turns=turns * n, turn_by_turn_monitor=monitor_a)
    many.track(b, num_turns=turns, turn_by_turn_monitor=monitor_b)

    strided = np.asarray(monitor_a.x)[0][::n]
    per_turn = np.asarray(monitor_b.x)[0]
    assert strided.shape == per_turn.shape == (turns,)
    assert np.array_equal(strided, per_turn)
    # and the off-by-one-sector slice is not merely noisier, it is a
    # different trajectory
    assert not np.allclose(np.asarray(monitor_a.x)[0][n - 1 :: n], per_turn)


def test_ocelot_a_sector_n_times_is_n_copies_once():
    """The same measurement against Ocelot, whose `track_nturns` takes the
    count natively."""
    pytest.importorskip("ocelot")
    import numpy as np
    from ocelot.cpbd.elements import Drift, Quadrupole, Sextupole
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.track import Track_info, track_nturns
    from ocelot.cpbd.beam import Particle

    def cell(copies):
        sequence = []
        for _ in range(copies):
            for sign in (1, -1):
                sequence += [
                    Drift(l=1.0),
                    Quadrupole(l=1.0, k1=sign * 0.3),
                    Sextupole(l=1.0, k2=2.0),
                    Drift(l=1.0),
                ]
        return MagneticLattice(sequence)

    n, turns = 3, 20
    results = []
    for copies, nsuper in ((1, n), (n, 1)):
        track_list = [Track_info(Particle(x=1e-4, y=0.5e-4), 1e-4, 0.5e-4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tracked = track_nturns(
                cell(copies),
                turns,
                track_list,
                nsuperperiods=nsuper,
                save_track=True,
                print_progress=False,
            )
        results.append(np.array(tracked[0].get_x()))

    assert results[0].shape == results[1].shape
    assert np.max(np.abs(results[0] - results[1])) == 0.0


def test_ocelot_records_one_point_per_turn_not_per_pass():
    """`nsuperperiods` does not multiply the record, which is what lets the
    turn suffixes and the beam files stay per turn."""
    pytest.importorskip("ocelot")
    from ocelot.cpbd.elements import Drift, Quadrupole
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.track import Track_info, track_nturns
    from ocelot.cpbd.beam import Particle

    turns = 20
    lattice = MagneticLattice(
        [Drift(l=1.0), Quadrupole(l=1.0, k1=0.3), Drift(l=1.0),
         Quadrupole(l=1.0, k1=-0.3)]
    )
    track_list = [Track_info(Particle(x=1e-5), 1e-5, 0.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracked = track_nturns(
            lattice, turns, track_list, nsuperperiods=4,
            save_track=True, print_progress=False,
        )
    # the launch condition and then one point per turn -- not per pass, which
    # would be 4x this. `track_reference_particle` drops the leading sample.
    assert len(tracked[0].get_x()) == turns + 1


# --- one sample per completed turn, in every code -----------------------
#
# Found while striding the Xsuite monitor: the codes did not agree on what
# `track_reference_particle` returns. elegant reads one WATCH page per pass
# and Bmad reads the bunch at END once per turn, so both give `turns` samples
# at the ends of turns. Ocelot's `p_list` opens with the launch condition and
# is `turns + 1` long, and the Xsuite monitor samples the start of each pass,
# so it gave the launch and then turns 1..turns-1 -- the same length as the
# others and shifted a turn earlier. They are all end-of-turn now.


class FakeXsuiteRing:
    """Enough of `xsuiteLattice` to call `track_reference_particle` for real.

    The method reads a line, a closed orbit, an aperture scan and a turn
    count, and nothing else -- so the trajectory it returns can be measured
    rather than the source inspected for how it was written.
    """

    track_reference_particle = xsuiteLattice.track_reference_particle

    def __init__(self, copies=1, nsuperperiods=1, turns=20):
        self.line = _xtrack_sector(copies)
        self.context = None
        self.objectname = "RING"
        self.code = "xsuite"
        self.supports_nsuperperiods = True
        self.file_block = {
            "tracking": {"turns": turns, "nsuperperiods": nsuperperiods}
        }

    def read_closed_orbit(self):
        """A straight FODO sector's closed orbit is the axis."""
        return None

    turns = frameworkLattice.turns
    nsuperperiods = frameworkLattice.nsuperperiods
    passes_per_turn = frameworkLattice.passes_per_turn
    da_settings = frameworkLattice.da_settings


def test_the_xsuite_trajectory_is_one_sample_per_turn():
    pytest.importorskip("xtrack")

    trajectory = FakeXsuiteRing(turns=20).track_reference_particle()
    assert set(trajectory) == {"x", "px", "y", "py"}
    for values in trajectory.values():
        assert values.shape == (20,)


def test_the_xsuite_trajectory_does_not_open_with_the_launch_point():
    """Which is what a raw monitor gives, and is a turn out."""
    pytest.importorskip("xtrack")

    line = FakeXsuiteRing(turns=20)
    nudge = 1e-3 / 100.0
    assert line.track_reference_particle()["x"][0] != pytest.approx(nudge)


def test_a_sector_trajectory_matches_the_whole_ring_turn_for_turn():
    """The measurement R17 rests on, through the real method: three sectors
    with `nsuperperiods: 3` against the same ring written out three times."""
    pytest.importorskip("xtrack")
    import numpy as np

    sector = FakeXsuiteRing(copies=1, nsuperperiods=3, turns=20)
    whole = FakeXsuiteRing(copies=3, nsuperperiods=1, turns=20)
    for name in ("x", "px", "y", "py"):
        a = sector.track_reference_particle()[name]
        b = whole.track_reference_particle()[name]
        assert a.shape == b.shape == (20,)
        assert np.max(np.abs(a - b)) == 0.0


def test_the_ocelot_trajectory_drops_its_launch_sample():
    """Ocelot's is pinned on the source: the method needs a lattice object,
    a beam and a closed orbit read back from a run, which is a whole
    integration test for one slice. The behaviour it encodes --
    ``p_list`` being ``turns + 1`` long -- is measured just above."""
    import inspect

    assert "p_list[1:]" in inspect.getsource(
        ocelotLattice.track_reference_particle
    )


def test_xsuite_reference_particle_is_as_long_as_the_turn_count():
    """The monitor is one sample short of a turn-per-turn record, so the
    final particle state has to be appended. Measured end to end."""
    pytest.importorskip("xtrack")
    import numpy as np
    import xtrack as xt

    n, turns = 3, 20
    line = _xtrack_sector(1)
    particles = line.build_particles(x=[1e-3], y=[0.5e-3])
    monitor = xt.ParticlesMonitor(
        start_at_turn=0, stop_at_turn=turns * n, num_particles=1
    )
    line.track(particles, num_turns=turns * n, turn_by_turn_monitor=monitor)
    trajectory = np.append(
        np.asarray(monitor.x)[0][n::n], float(np.atleast_1d(particles.x)[0])
    )
    assert trajectory.shape == (turns,)
    # the last entry is where the particle actually ended up, not a monitor
    # sample -- that is the whole point of appending it
    assert trajectory[-1] == float(np.atleast_1d(particles.x)[0])
    assert trajectory[0] != float(np.atleast_1d(monitor.x)[0][0])
