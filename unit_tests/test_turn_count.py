"""A turn count is a tracking setting, and only three codes can honour it.

A ring's turns are strict repeats -- nothing in the lattice differs between
them -- so the count lives in the ``files:`` block rather than in the layout,
which is the only shape that survives a million of them. The unit is the line,
and a ring spanning several sections is written as one section naming them
(section orders nest), so the line is the whole group.

elegant (``n_passes``), Xsuite (``num_turns``) and Ocelot (``track_nturns``)
can track turns. Everything else tracks a line once, and says so rather than
quietly tracking one turn when a thousand were asked for.
"""

import warnings

import pytest

from simba.Codes.ASTRA.ASTRA import astraLattice
from simba.Codes.Cheetah.Cheetah import cheetahLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.GPT.GPT import gptLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.OPAL.OPAL import opalLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

CAN_TURN = [elegantLattice, xsuiteLattice, ocelotLattice]
CANNOT = [astraLattice, gptLattice, cheetahLattice, opalLattice]


class FakeLine:
    """A lattice stub carrying only what the turn-count code reads."""

    def __init__(self, file_block=None, code="astra", supports=False):
        self.file_block = file_block or {}
        self.code = code
        self.objectname = "RING"
        self.supports_turns = supports

    turns = frameworkLattice.turns
    check_turns_supported = frameworkLattice.check_turns_supported


# --- reading the count --------------------------------------------------


def test_no_setting_means_one_turn():
    assert FakeLine().turns == 1


def test_an_empty_tracking_block_means_one_turn():
    assert FakeLine({"tracking": {}}).turns == 1


def test_a_null_tracking_block_means_one_turn():
    """A key present but empty is how YAML hands over ``tracking:``."""
    assert FakeLine({"tracking": None}).turns == 1


def test_the_count_is_read_from_the_files_block():
    assert FakeLine({"tracking": {"turns": 1000}}).turns == 1000


def test_a_string_count_is_coerced():
    """Every other numeric setting arrives coerced rather than type-checked."""
    assert FakeLine({"tracking": {"turns": "512"}}).turns == 512


# --- which codes can honour it ------------------------------------------


@pytest.mark.parametrize("cls", CAN_TURN, ids=lambda c: c.__name__)
def test_the_three_that_can(cls):
    assert cls.supports_turns is True


@pytest.mark.parametrize("cls", CANNOT, ids=lambda c: c.__name__)
def test_the_ones_that_cannot(cls):
    assert cls.supports_turns is False


def test_the_base_class_assumes_it_cannot():
    """So a backend gains turns by declaring it, never by omission."""
    assert frameworkLattice.supports_turns is False


# --- the warning --------------------------------------------------------


def test_asking_a_single_pass_code_for_turns_warns():
    line = FakeLine({"tracking": {"turns": 1000}}, code="astra", supports=False)
    with pytest.warns(UserWarning, match="tracks a line once"):
        line.check_turns_supported()


def test_the_warning_names_the_code_and_the_count():
    line = FakeLine({"tracking": {"turns": 1000}}, code="astra", supports=False)
    with pytest.warns(UserWarning, match=r"1000 turns.*astra"):
        line.check_turns_supported()


def test_a_capable_code_is_silent():
    line = FakeLine({"tracking": {"turns": 1000}}, code="elegant", supports=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_turns_supported()


def test_one_turn_is_silent_everywhere():
    """The default must never warn, on any code."""
    for code, supports in (("astra", False), ("elegant", True)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FakeLine({}, code=code, supports=supports).check_turns_supported()


def test_it_warns_rather_than_refusing():
    """One settings file driving several codes is ordinary."""
    line = FakeLine({"tracking": {"turns": 5}}, code="gpt", supports=False)
    with pytest.warns(UserWarning):
        line.check_turns_supported()
    assert line.turns == 5


# --- turns only mean something on a closed path -------------------------
#
# A turn count wraps the line onto its own start. Asking a transfer line for a
# thousand turns is not a smaller ring, it is incoherent -- and until this
# check nothing said so. A superperiod is the legitimate exception: one sector
# of an N-fold-symmetric ring is open on its own.

import math

from laura.models.element import Dipole, Drift
from laura.models.elementList import MachineModel


def ring(nbend, angle, turns=1000):
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
    return ClosureLine(model, order, turns)


class ClosureLine:
    """A stub exposing what `check_turns_closed` reads off real geometry."""

    def __init__(self, model, order, turns):
        self.startObject = model[order[0]]
        self.endObject = model[order[-1]]
        self.elements = {name: model[name] for name in order}
        self.file_block = {"tracking": {"turns": turns}}
        self.objectname = "RING"
        self.code = "elegant"

    turns = frameworkLattice.turns
    net_bend_angle = frameworkLattice.net_bend_angle
    check_turns_closed = frameworkLattice.check_turns_closed


def test_a_closed_ring_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ring(4, math.pi / 2).check_turns_closed()


def test_an_open_line_warns():
    with pytest.warns(UserWarning, match="does not close"):
        ring(4, 0.0).check_turns_closed()


def test_the_warning_reports_the_gap():
    """Four 1 m bends and four 1 m drifts, dead straight: 8 m from home."""
    with pytest.warns(UserWarning, match=r"ends 8 m from where it starts"):
        ring(4, 0.0).check_turns_closed()


def test_one_turn_never_checks_closure():
    """A single pass down an open line is the ordinary case."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ring(4, 0.0, turns=1).check_turns_closed()


def test_a_superperiod_is_named_as_such():
    """Half a ring closes after two, and the message should say so."""
    with pytest.warns(UserWarning, match=r"1/2 fraction of a turn"):
        ring(2, math.pi / 2).check_turns_closed()


def test_a_quarter_ring_is_named_as_such():
    with pytest.warns(UserWarning, match=r"1/4 fraction of a turn"):
        ring(1, math.pi / 2).check_turns_closed()


def test_a_straight_line_gets_no_superperiod_hint():
    """No bending at all is not a sector of anything."""
    with pytest.warns(UserWarning) as caught:
        ring(4, 0.0).check_turns_closed()
    assert "superperiod" not in str(caught[0].message)


def test_the_net_bend_of_a_closed_ring_is_a_full_turn():
    assert ring(4, math.pi / 2).net_bend_angle == pytest.approx(2 * math.pi)


def test_the_net_bend_of_a_straight_line_is_zero():
    assert ring(4, 0.0).net_bend_angle == pytest.approx(0.0)


# --- one file per turn --------------------------------------------------
#
# N turns through one screen is N beams wanting one filename: the collision S4
# fixed across lines, now within one. The turn suffix only appears when turns
# were asked for, so a single-pass run keeps the names it always had.


class NamingLine:
    """A stub exposing just the naming path."""

    def __init__(self, turns=1, colliding=(), name="LINAC"):
        self.file_block = {"tracking": {"turns": turns}}
        self.colliding_outputs = set(colliding)
        self.objectname = name

    turns = frameworkLattice.turns
    output_basename = frameworkLattice.output_basename


def test_a_single_turn_run_is_unchanged():
    assert NamingLine(turns=1).output_basename("SCR", turn=1) == "SCR"


def test_no_turn_given_is_unchanged():
    assert NamingLine(turns=100).output_basename("SCR") == "SCR"


def test_turns_qualify_the_name():
    assert NamingLine(turns=100).output_basename("SCR", turn=7) == "SCR-t007"


def test_the_index_is_padded_to_the_count():
    """So the files sort in turn order rather than lexically."""
    assert NamingLine(turns=1000).output_basename("SCR", turn=7) == "SCR-t0007"
    assert NamingLine(turns=9).output_basename("SCR", turn=7) == "SCR-t7"


def test_every_turn_gets_a_distinct_name():
    line = NamingLine(turns=5)
    names = {line.output_basename("SCR", turn=t) for t in range(1, 6)}
    assert len(names) == 5


def test_a_line_collision_and_a_turn_compose():
    line = NamingLine(turns=20, colliding=["SCR"], name="PASS2")
    assert line.output_basename("SCR", turn=3) == "PASS2-SCR-t03"


def test_a_pass_selector_and_a_turn_compose():
    line = NamingLine(turns=20)
    assert line.output_basename("SCR#2", turn=3) == "SCR.2-t03"


# --- the monitor has to be sized for the turns --------------------------


def test_a_screen_monitor_is_sized_for_one_turn_by_default():
    """`stop_at_turn` is how many slots a ParticlesMonitor has."""
    pytest.importorskip("xtrack")
    from laura.models.element import Screen
    from laura.translator.converters.converter import translate_elements

    screen = Screen(name="SCR", machine_area="A", physical={"length": 0.0})
    _, _, properties = translate_elements([screen])["SCR"].to_xsuite(beam_length=10)
    assert properties["stop_at_turn"] == 1


def test_a_multi_turn_line_resizes_its_monitors():
    """Otherwise every turn after the first is silently dropped."""
    pytest.importorskip("xtrack")
    from laura.models.element import Screen, Drift
    from laura.models.elementList import SectionLattice, ElementList
    from laura.translator.converters.section import SectionLatticeTranslator

    screen = Screen(name="SCR", machine_area="A", physical={"length": 0.0})
    drift = Drift(
        name="D1",
        hardware_class="Drift",
        hardware_type="Drift",
        machine_area="A",
        physical={"length": 1.0},
    )
    section = SectionLattice(
        name="S",
        order=["D1", "SCR"],
        elements=ElementList(elements={"D1": drift, "SCR": screen}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        line = SectionLatticeTranslator.from_section(section).to_xsuite(
            beam_length=10, turns=250
        )
    assert line["SCR"].stop_at_turn == 250


# --- splitting a monitor's record into turns ----------------------------
#
# `monitor.x` is (particle, turn), but `data.to_dict()` flattens it and
# carries an `at_turn` column saying which turn each row came from. Masking on
# that is exact; slicing by stride would assume an ordering.


def tracked_monitor(num_particles=4, num_turns=3):
    """A real xtrack monitor, tracked, so the layout is measured not assumed."""
    pytest.importorskip("xtrack")
    import xtrack as xt
    import xpart as xp

    env = xt.Environment()
    line = env.new_line()
    line.append("d1", xt.Drift(length=1.0))
    line.append(
        "mon",
        xt.ParticlesMonitor(
            num_particles=num_particles, start_at_turn=0, stop_at_turn=num_turns
        ),
    )
    line.particle_ref = xt.Particles(
        p0c=[1e9], mass0=[xp.ELECTRON_MASS_EV], q0=-1
    )
    particles = xt.Particles(
        p0c=1e9,
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        x=[1e-3 * (i + 1) for i in range(num_particles)],
    )
    line.build_tracker()
    line.track(particles, num_turns=num_turns)
    return line["mon"]


def test_a_monitor_records_one_row_per_particle_per_turn():
    data = tracked_monitor(4, 3).data.to_dict()
    import numpy as np

    assert np.asarray(data["x"]).size == 12
    assert list(np.asarray(data["at_turn"])) == [0, 1, 2] * 4


def test_selecting_a_turn_keeps_one_row_per_particle():
    from simba.Codes.Xsuite.Xsuite import _select_turn
    import numpy as np

    data = tracked_monitor(4, 3).data.to_dict()
    for turn in range(3):
        selected = _select_turn(data, turn)
        assert np.asarray(selected["x"]).size == 4
        assert set(np.asarray(selected["at_turn"])) == {turn}


def test_every_particle_appears_once_in_a_turn():
    from simba.Codes.Xsuite.Xsuite import _select_turn
    import numpy as np

    data = tracked_monitor(4, 3).data.to_dict()
    ids = np.asarray(_select_turn(data, 1)["particle_id"])
    assert sorted(ids) == [0, 1, 2, 3]


def test_the_turns_partition_the_record():
    """No row is dropped and none is counted twice."""
    from simba.Codes.Xsuite.Xsuite import _select_turn
    import numpy as np

    data = tracked_monitor(4, 3).data.to_dict()
    total = sum(np.asarray(_select_turn(data, t)["x"]).size for t in range(3))
    assert total == np.asarray(data["x"]).size


def test_a_dump_without_at_turn_is_returned_untouched():
    from simba.Codes.Xsuite.Xsuite import _select_turn

    data = {"x": [1, 2, 3]}
    assert _select_turn(data, 0) is data


# --- ocelot: the named mechanism was the wrong one ----------------------
#
# `track_nturns` sounds like the answer and is not: it takes single Particles
# wrapped in Track_info for dynamic-aperture studies, never a ParticleArray or
# a Navigator, so it carries none of this backend's physics or output. Turns
# come from looping `track` and feeding the bunch back in.


def test_ocelot_track_nturns_takes_a_track_list_not_a_bunch():
    """Pins why it is not used: the signature is the evidence."""
    pytest.importorskip("ocelot")
    import inspect
    from ocelot.cpbd.track import track_nturns

    parameters = list(inspect.signature(track_nturns).parameters)
    assert parameters[:3] == ["lat", "nturns", "track_list"]
    assert "nsuperperiods" in parameters


def test_ocelot_track_takes_a_particle_array_and_a_navigator():
    """Which is what this backend uses, and why the loop goes around it."""
    pytest.importorskip("ocelot")
    import inspect
    from ocelot.cpbd.track import track

    parameters = list(inspect.signature(track).parameters)
    assert parameters[:3] == ["lattice", "p_array", "navi"]


def test_a_navigator_can_be_rewound_for_the_next_turn():
    pytest.importorskip("ocelot")
    from ocelot.cpbd.navi import Navigator

    assert hasattr(Navigator, "go_to_start")


def test_the_ocelot_loop_rebuilds_the_navigator_each_turn():
    """The monitor filenames are fixed when the processes are built, so a
    shared navigator would write every turn to the same file."""
    import inspect
    from simba.Codes.Ocelot.Ocelot import ocelotLattice

    source = inspect.getsource(ocelotLattice.run)
    assert "self.navi_setup(turn=" in source
    # Called, not merely mentioned -- the comment above the loop names
    # `track_nturns` precisely to say why it is the wrong one.
    assert "track_nturns(" not in source


# --- elegant, the one wired so far --------------------------------------


def test_elegant_no_longer_hardcodes_one_pass():
    """All three ``run_control`` sites used to pass the literal 1."""
    import inspect

    source = inspect.getsource(elegantLattice)
    assert "n_passes=1" not in source
    assert source.count("n_passes=self.turns") == 3
