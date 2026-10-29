"""``tracking: {nsuperperiods: N}``: one sector of an N-fold ring, tracked N
times per turn."""

import math
import warnings

import pytest

from helpers import BentLine, xtrack_sector
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

CAN = [ocelotLattice, madxLattice, xsuiteLattice]


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
    codes_that_can = frameworkLattice.codes_that_can


def test_no_setting_means_one_pass_per_turn():
    assert FakeLine().nsuperperiods == 1


def test_an_empty_tracking_block_means_one():
    assert FakeLine({"tracking": {}}).nsuperperiods == 1


def test_a_null_tracking_block_means_one():
    """How YAML hands over a bare ``tracking:``."""
    assert FakeLine({"tracking": None}).nsuperperiods == 1


def test_the_count_is_read_from_the_files_block():
    assert FakeLine({"tracking": {"nsuperperiods": 4}}).nsuperperiods == 4


def test_a_string_count_is_coerced():
    assert FakeLine({"tracking": {"nsuperperiods": "6"}}).nsuperperiods == 6


def test_a_nonsense_count_warns_and_falls_back_to_one():
    line = FakeLine({"tracking": {"nsuperperiods": "many"}})
    with pytest.warns(UserWarning, match="not a whole number"):
        assert line.nsuperperiods == 1


def test_zero_superperiods_warns_and_falls_back_to_one():
    line = FakeLine({"tracking": {"nsuperperiods": 0}})
    with pytest.warns(UserWarning, match="not a count"):
        assert line.nsuperperiods == 1


def test_a_negative_count_warns_and_falls_back_to_one():
    line = FakeLine({"tracking": {"nsuperperiods": -4}})
    with pytest.warns(UserWarning, match="not a count"):
        assert line.nsuperperiods == 1


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


def test_asking_a_code_that_cannot_warns():
    """It is a different machine, not a coarser one; and the warning names
    every code that can."""
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, code="elegant", supports=False)
    with pytest.warns(UserWarning, match="superperiods.*one 4th of the intended ring") as caught:
        line.check_nsuperperiods_supported()
    message = str(caught[0].message).lower()
    for cls in CAN:
        name = cls.model_fields["code"].default
        # the warning spells it MAD-X; the class calls itself madx
        assert name.lower() in message or name.lower() == "madx" and "mad-x" in message


@pytest.mark.filterwarnings("error")
def test_a_capable_code_is_silent():
    line = FakeLine({"tracking": {"nsuperperiods": 4}}, code="ocelot", supports=True)
    line.check_nsuperperiods_supported()


@pytest.mark.filterwarnings("error")
def test_one_superperiod_is_silent_everywhere():
    for code, supports in (("elegant", False), ("ocelot", True)):
        FakeLine({}, code=code, supports=supports).check_nsuperperiods_supported()


def test_it_is_checked_during_preprocessing():
    import inspect

    source = inspect.getsource(frameworkLattice.preProcess)
    assert "check_nsuperperiods_supported()" in source


@pytest.mark.filterwarnings("error")
def test_a_correctly_counted_sector_is_silent():
    """A quarter ring declared as one of four: no closure complaint either."""
    BentLine(1, math.pi / 2, nsuperperiods=4).check_turns_closed()


def test_an_open_sector_warns_and_suggests_the_setting():
    with pytest.warns(UserWarning, match="does not close.*nsuperperiods: 4"):
        BentLine(1, math.pi / 2, nsuperperiods=1).check_turns_closed()


def test_a_wrong_count_warns_and_suggests_the_right_one():
    """Six sectors of a four-fold ring bend through one and a half turns."""
    with pytest.warns(UserWarning, match="do not make a closed ring.*suggests 4 instead"):
        BentLine(1, math.pi / 2, nsuperperiods=6).check_turns_closed()


@pytest.mark.filterwarnings("error")
def test_a_count_that_overshoots_by_a_whole_turn_is_accepted():
    """Two turns of bend is a double pass, not an error."""
    BentLine(1, math.pi / 2, nsuperperiods=8).check_turns_closed()


@pytest.mark.filterwarnings("error")
def test_a_straight_sector_is_not_judged():
    BentLine(4, 0.0, nsuperperiods=4).check_turns_closed()


@pytest.mark.filterwarnings("error")
def test_one_turn_never_checks_anything():
    BentLine(1, math.pi / 2, nsuperperiods=6, turns=1).check_turns_closed()


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
    pass_length = frameworkLattice.pass_length
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
    """It tracks the sector once, so programs land on the sector crossings."""
    asked = PeriodLine(nsuperperiods=4, supports=False).revolution_period
    one = PeriodLine(nsuperperiods=1).revolution_period
    assert asked == pytest.approx(one)


def test_ocelot_hands_the_count_to_track_nturns():
    """Driven through ``_track_nturns`` and recorded; counting the source
    text broke when the calls were gathered there."""
    import ocelot.cpbd.track as octrack
    from ocelot import Drift as OcelotDrift, MagneticLattice

    seen = {}

    def recorder(lat, nturns, track_list, nsuperperiods=1, **kwargs):
        seen.update(nturns=nturns, nsuperperiods=nsuperperiods)
        return track_list

    class FakeOcelot:
        _track_nturns = ocelotLattice._track_nturns
        # a real lattice: the call is made inside `lattice_pass`, which reads it
        lat_obj = MagneticLattice([OcelotDrift(l=1.0)])
        turns, nsuperperiods = 7, 4

    original = octrack.track_nturns
    octrack.track_nturns = recorder
    try:
        FakeOcelot()._track_nturns([], save_track=False)
    finally:
        octrack.track_nturns = original
    assert seen == {"nturns": 7, "nsuperperiods": 4}


class TurnLine:
    """Enough of a line for the shared turn loop, recording its calls."""

    def __init__(self, turns, nsuperperiods, write_turns=False):
        self.turns = turns
        self.passes_per_turn = nsuperperiods
        self.write_turns = write_turns
        self.calls = []
        self._rf_corrections = None
        self._rf_phase0 = None

    def begin_rf_phases(self):
        self.calls.append(("begin",))

    def apply_rf_phases(self, pass_index):
        self.calls.append(("rf", pass_index))

    def apply_programs(self, turn):
        self.calls.append(("programs", turn))

    run_turns = frameworkLattice.run_turns
    end_turns = frameworkLattice.end_turns
    pass_index = frameworkLattice.pass_index
    output_turns = frameworkLattice.output_turns


def _passes(line):
    passes = []
    line.run_turns(lambda *args: passes.append(args))
    return passes


@pytest.mark.parametrize("cls", [ocelotLattice, madxLattice], ids=["ocelot", "madx"])
def test_the_bunch_paths_loop_through_the_shared_turn_loop(cls):
    import inspect

    source = inspect.getsource(cls.run if cls is ocelotLattice else cls.run_segments)
    assert "self.run_turns(" in source


def test_the_turn_loop_tracks_every_pass_of_every_turn():
    passes = _passes(TurnLine(turns=3, nsuperperiods=4))
    assert [(turn, index) for turn, index, _, _ in passes] == [
        (turn, (turn - 1) * 4 + sector)
        for turn in range(1, 4)
        for sector in range(4)
    ]


@pytest.mark.parametrize("write_turns", [False, True])
def test_only_the_last_pass_of_a_turn_records(write_turns):
    passes = _passes(TurnLine(turns=3, nsuperperiods=4, write_turns=write_turns))
    recorded = [(turn, index, name) for turn, index, name, record in passes if record]
    if write_turns:
        assert recorded == [(1, 3, 1), (2, 7, 2), (3, 11, 3)]
    else:
        assert recorded == [(3, 11, None)]


def test_programs_are_set_per_turn_and_rf_per_pass():
    line = TurnLine(turns=2, nsuperperiods=2)
    _passes(line)
    assert line.calls == [
        ("begin",),
        ("programs", 1), ("rf", 0), ("rf", 1),
        ("programs", 2), ("rf", 2), ("rf", 3),
        # put back as turn 1 had it, for the optics
        ("rf", None), ("programs", 1),
    ]


def test_the_line_is_put_back_even_if_a_pass_fails():
    line = TurnLine(turns=2, nsuperperiods=1)

    def fail(*args):
        raise RuntimeError("tracking failed")

    with pytest.raises(RuntimeError):
        line.run_turns(fail)
    assert line.calls[-2:] == [("rf", None), ("programs", 1)]


def test_xsuite_multiplies_num_turns_rather_than_looping():
    """Xtrack has no sector, so it tracks `turns * passes_per_turn`."""
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
    import inspect

    source = inspect.getsource(xsuiteLattice.run_dynamic_aperture)
    assert "// self.passes_per_turn" in source


def test_xsuite_a_sector_n_times_is_n_copies_once():
    """The sextupoles keep the agreement from being a linear coincidence."""
    pytest.importorskip("xtrack")

    n, turns = 3, 20
    one, many = xtrack_sector(1), xtrack_sector(n)
    a = one.build_particles(x=[1e-3], y=[0.5e-3])
    b = many.build_particles(x=[1e-3], y=[0.5e-3])
    one.track(a, num_turns=turns * n)
    many.track(b, num_turns=turns)
    assert float(a.x[0]) == float(b.x[0])
    assert float(a.px[0]) == float(b.px[0])


def test_xsuite_at_turn_counts_passes():
    pytest.importorskip("xtrack")

    n, turns = 3, 20
    line = xtrack_sector(1)
    particles = line.build_particles(x=[1e-3], y=[0.5e-3])
    line.track(particles, num_turns=turns * n)
    assert int(particles.at_turn[0]) == turns * n
    assert int(particles.at_turn[0]) // n == turns


def test_xsuite_a_monitor_samples_the_start_of_each_pass():
    """So the stride starts from zero; `[N-1::N]` runs and is wrong."""
    pytest.importorskip("xtrack")
    import numpy as np
    import xtrack as xt

    n, turns = 3, 20
    one, many = xtrack_sector(1), xtrack_sector(n)
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


class FakeXsuiteRing:
    """Enough of `xsuiteLattice` to call `track_reference_particle` for real."""

    track_reference_particle = xsuiteLattice.track_reference_particle
    single_particle_line = xsuiteLattice.single_particle_line

    def __init__(self, copies=1, nsuperperiods=1, turns=20):
        self.line = xtrack_sector(copies)
        self.context = None
        self.objectname = "RING"
        self.code = "xsuite"
        self.supports_nsuperperiods = True
        self.file_block = {
            "tracking": {"turns": turns, "nsuperperiods": nsuperperiods}
        }

    def read_closed_orbit(self):
        """A straight FODO sector's closed orbit is the axis."""

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
    """A raw monitor does, and is a turn out."""
    pytest.importorskip("xtrack")

    line = FakeXsuiteRing(turns=20)
    nudge = 1e-3 / 100.0
    assert line.track_reference_particle()["x"][0] != pytest.approx(nudge)


def test_a_sector_trajectory_matches_the_whole_ring_turn_for_turn():
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
    """Pinned on the source; ``p_list`` being ``turns + 1`` long is measured above."""
    import inspect

    assert "p_list[1:]" in inspect.getsource(
        ocelotLattice.track_reference_particle
    )


def test_xsuite_reference_particle_is_as_long_as_the_turn_count():
    """The monitor is a sample short, so the final state is appended."""
    pytest.importorskip("xtrack")
    import numpy as np
    import xtrack as xt

    n, turns = 3, 20
    line = xtrack_sector(1)
    particles = line.build_particles(x=[1e-3], y=[0.5e-3])
    monitor = xt.ParticlesMonitor(
        start_at_turn=0, stop_at_turn=turns * n, num_particles=1
    )
    line.track(particles, num_turns=turns * n, turn_by_turn_monitor=monitor)
    trajectory = np.append(
        np.asarray(monitor.x)[0][n::n], float(np.atleast_1d(particles.x)[0])
    )
    assert trajectory.shape == (turns,)
    assert trajectory[-1] == float(np.atleast_1d(particles.x)[0])
    assert trajectory[0] != float(np.atleast_1d(monitor.x)[0][0])
