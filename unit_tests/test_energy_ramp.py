"""Energy ramps: ``tracking: {ramp: ...}``, one model, every code that can.

The model is :mod:`simba.Modules.EnergyRamp`'s. The ramp sets the
*reference* momentum at the start of each turn. Every particle keeps its
absolute momentum and arrival time. Normalised strengths stay put, so the
fields follow the reference, and the RF does the accelerating.

Three layers:

* the ramp itself -- reading the block, the momentum per pass, the clock;
* the line -- when a run counts as ramped, and the warnings;
* the codes -- a closed FODO cell with no RF, ramped 2% over 10 turns,
  through Xsuite, elegant, Ocelot and MAD-X. With no RF nothing gains
  energy, so the beam falls behind the ramp. The quads, fixed in
  normalised strength, get stronger relative to the beam turn by turn, and
  the orbit visibly departs from the unramped run's. The codes must agree
  on that departure.
"""

import os
import shutil
import warnings

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.Generators import frameworkGenerator
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import OUTPUT_TURN_SEPARATOR as SEPARATOR
from simba.Framework_objects import frameworkLattice
from simba.Modules.DeviceProgram import DeviceProgram
from simba.Modules.EnergyRamp import (
    SPEED_OF_LIGHT,
    EnergyRamp,
    RampClock,
    beta_from_p0c,
)

ELECTRON = 510998.95
PROTON = 938.27208816e6


def ramp(**entry):
    return EnergyRamp.from_dict({"turns": [1, 11], "momentum": [5e6, 6e6], **entry})


# --- reading the block ---------------------------------------------------


def test_a_ramp_in_momentum():
    read = ramp()
    assert read.quantity == "momentum"
    assert read.p0c_at(1, ELECTRON) == 5e6
    assert read.p0c_at(11, ELECTRON) == 6e6


def test_the_default_is_linear():
    """A ramp is smooth, unlike a kicker, whose default is to hold."""
    read = ramp()
    assert read.interpolation == "linear"
    assert read.p0c_at(6, ELECTRON) == pytest.approx(5.5e6)


def test_hold_is_still_available():
    assert ramp(interpolation="hold").p0c_at(6, ELECTRON) == 5e6


def test_a_ramp_in_kinetic_energy_is_turned_into_momentum():
    """160 MeV kinetic, the PSB's injection energy, as a proton's p0c."""
    read = EnergyRamp.from_dict({"turns": [1, 2], "kinetic_energy": [160e6, 160e6]})
    energy = 160e6 + PROTON
    assert read.p0c_at(1, PROTON) == pytest.approx(np.sqrt(energy**2 - PROTON**2))


def test_outside_its_knots_a_ramp_holds_its_ends():
    read = EnergyRamp.from_dict({"turns": [3, 5], "momentum": [5e6, 6e6]})
    assert read.p0c_at(1, ELECTRON) == 5e6
    assert read.p0c_at(50, ELECTRON) == 6e6


@pytest.mark.parametrize(
    "entry",
    [
        {"turns": [1, 2]},
        {"turns": [1, 2], "momentum": [1e9, 2e9], "kinetic_energy": [1e9, 2e9]},
    ],
    ids=["neither", "both"],
)
def test_a_ramp_states_exactly_one_quantity(entry):
    with pytest.raises(ValueError, match="exactly one"):
        EnergyRamp.from_dict(entry)


def test_the_knots_must_pair_up():
    with pytest.raises(ValueError):
        EnergyRamp.from_dict({"turns": [1, 2, 3], "momentum": [1e9, 2e9]})


def test_a_ramp_cannot_go_through_zero():
    with pytest.raises(ValueError, match="positive"):
        EnergyRamp.from_dict({"turns": [1, 2], "momentum": [1e9, 0.0]})


def test_a_setting_simba_does_not_read_is_reported():
    with pytest.warns(UserWarning, match="frequency"):
        ramp(frequency=[1.0, 2.0])


# --- the momentum on each pass -------------------------------------------


def test_one_value_per_pass_and_one_beyond():
    """``turns * passes + 1``: the last is where the final pass ends."""
    assert len(ramp().p0c_per_pass(10, ELECTRON)) == 11
    assert len(ramp().p0c_per_pass(10, ELECTRON, passes_per_turn=4)) == 41


def test_a_superperiod_does_not_ramp_within_a_turn():
    """The reference changes at the start of a turn and is held across its
    passes: a superperiod is a piece of one turn, not a turn of its own."""
    per_pass = ramp().p0c_per_pass(10, ELECTRON, passes_per_turn=4)
    per_turn = ramp().p0c_per_pass(10, ELECTRON)
    assert np.array_equal(per_pass[:-1].reshape(10, 4), np.repeat(per_turn[:-1, None], 4, 1))


def test_the_energy_the_rf_has_to_find():
    read = ramp()
    energy = np.hypot(read.p0c_per_pass(10, ELECTRON), ELECTRON)
    assert np.allclose(read.energy_gain_per_turn(10, ELECTRON), np.diff(energy))


# --- the clock -----------------------------------------------------------


def test_a_flat_ramp_is_a_steady_revolution():
    flat = EnergyRamp.from_dict({"turns": [1, 2], "momentum": [5e6, 5e6]})
    clock = flat.clock(10, 4.25, ELECTRON)
    period = 4.25 / (beta_from_p0c(5e6, ELECTRON) * SPEED_OF_LIGHT)
    assert np.allclose(clock.times, np.arange(11) * period, rtol=1e-14)


def test_turn_one_starts_at_zero():
    assert ramp().clock(10, 4.25, ELECTRON)(1) == 0.0


def test_the_clock_is_xsuites():
    """Turn ``n`` at the time Xtrack's own ``EnergyProgram`` puts it, so
    one knot per pass lands Xsuite on integer turns and it never
    interpolates the momentum."""
    xt = pytest.importorskip("xtrack")
    length = 4.25
    read = ramp()
    clock = read.clock(10, length, ELECTRON)
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.particle_ref = xt.Particles(p0c=5e6, mass0=ELECTRON)
    line.build_tracker()
    line.energy_program = xt.EnergyProgram(
        t_s=clock.times, p0c=read.p0c_per_pass(10, ELECTRON)
    )
    for turn in range(10):
        assert line.energy_program.get_t_s_at_turn(turn) == pytest.approx(
            clock.times[turn], rel=1e-12, abs=1e-20
        )


def test_the_clock_runs_on_past_its_last_pass():
    """At the last pass's rate: a device program can outlast the ramp."""
    clock = RampClock(times=np.array([0.0, 1.0, 3.0]))
    assert clock(3) == 3.0
    assert clock(5) == 7.0
    assert clock(2.5) == 2.0


def test_a_device_program_reads_the_ramp_clock():
    """A kicker programmed on turn 5 fires at turn 5's time, which under a
    ramp is not 4 periods of turn 1."""
    clock = ramp().clock(10, 4.25, ELECTRON)
    kick = DeviceProgram.from_dict({"element": "K", "turns": [3, 5], "values": [0.0, 1.0]})
    times, _ = kick.time_knots(1.0, origin_turn=3, clock=clock)
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(clock(5) - clock(3), rel=1e-12)
    # and that is not two of turn 1's periods: the bunch is faster by then
    assert abs(times[-1] - 2 * clock(2)) > 1e-14


# --- the line ------------------------------------------------------------


class FakeBeam:
    def __init__(self, cp=5e6, rest_energy=ELECTRON):
        self.cp = type("V", (), {"val": np.full(4, cp)})()
        self.particle_rest_energy_eV = type("V", (), {"val": np.array([rest_energy])})()


class FakeCavity:
    hardware_type = "RFCavity"

    def __init__(self, voltage):
        self.simulation = type("S", (), {"field_amplitude": voltage})()


class FakeLine:
    """A lattice stub carrying only what the ramp code reads."""

    def __init__(self, tracking, supports=True, beam=None, voltage=None, code="xsuite"):
        self.file_block = {"tracking": tracking}
        self.objectname = "RING"
        self.code = code
        self.supports_ramp = supports
        self.supports_nsuperperiods = True
        self.global_parameters = {"beam": beam}
        self.elements = {} if voltage is None else {"CAV": FakeCavity(voltage)}

    turns = frameworkLattice.turns
    nsuperperiods = frameworkLattice.nsuperperiods
    passes_per_turn = frameworkLattice.passes_per_turn
    ramp = frameworkLattice.ramp
    ramped = frameworkLattice.ramped
    ramp_p0c = frameworkLattice.ramp_p0c
    rest_energy = frameworkLattice.rest_energy
    _input_reference = None
    _input_mean = frameworkLattice._input_mean
    reference_p0c = frameworkLattice.reference_p0c
    accelerating_cavities = frameworkLattice.accelerating_cavities
    cavity_voltage = staticmethod(frameworkLattice.cavity_voltage)
    rf_voltage = frameworkLattice.rf_voltage
    check_ramp = frameworkLattice.check_ramp
    codes_that_can = frameworkLattice.codes_that_can
    check_ramp_beam = frameworkLattice.check_ramp_beam


RAMP = {"turns": [1, 10], "momentum": [5e6, 5.1e6]}


@pytest.mark.parametrize(
    "cls,can",
    [
        (xsuiteLattice, True),
        (elegantLattice, True),
        (ocelotLattice, True),
        (madxLattice, True),
        (bmadLattice, False),
    ],
    ids=lambda c: getattr(c, "__name__", str(c)),
)
def test_which_codes_ramp(cls, can):
    """Bmad's tracking path here is single-pass, so it has no turn to
    change the reference between."""
    assert cls.supports_ramp is can


def test_no_ramp_means_not_ramped():
    line = FakeLine({"turns": 10})
    assert line.ramp is None and not line.ramped
    assert line.ramp_p0c(5) is None


def test_a_ramp_over_turns_is_ramped():
    line = FakeLine({"turns": 10, "ramp": RAMP}, beam=FakeBeam())
    assert line.ramped
    assert line.ramp_p0c(10) == 5.1e6


def test_one_turn_is_not_ramped():
    line = FakeLine({"turns": 1, "ramp": RAMP})
    assert not line.ramped and line.ramp_p0c(1) is None
    with pytest.warns(UserWarning, match="tracks one turn"):
        line.check_ramp()


def test_a_code_that_cannot_ramp_warns_and_is_not_ramped():
    line = FakeLine({"turns": 10, "ramp": RAMP}, supports=False, code="bmad")
    assert not line.ramped
    with pytest.warns(UserWarning, match="fixed energy"):
        line.check_ramp()


def test_a_run_shorter_than_its_ramp_warns():
    line = FakeLine({"turns": 5, "ramp": RAMP})
    with pytest.warns(UserWarning, match="part-way up"):
        line.check_ramp()


def test_an_unreadable_ramp_warns_and_is_ignored():
    line = FakeLine({"turns": 10, "ramp": {"turns": [1, 2]}})
    with pytest.warns(UserWarning, match="exactly one"):
        assert line.ramp is None


def test_a_beam_off_the_ramp_warns():
    line = FakeLine({"turns": 10, "ramp": RAMP}, beam=FakeBeam(cp=4.9e6), voltage=1e6)
    with pytest.warns(UserWarning, match="off the ramp"):
        line.check_ramp_beam()


def test_no_rf_warns():
    line = FakeLine({"turns": 10, "ramp": RAMP}, beam=FakeBeam())
    with pytest.warns(UserWarning, match="no RF cavity"):
        line.check_ramp_beam()


def test_too_little_voltage_warns():
    """2% of 5 MeV/c over 9 turns is about 11 keV a turn."""
    line = FakeLine({"turns": 10, "ramp": RAMP}, beam=FakeBeam(), voltage=1e3)
    with pytest.warns(UserWarning, match="falls off the ramp"):
        line.check_ramp_beam()


def test_enough_voltage_on_the_ramp_is_silent():
    line = FakeLine({"turns": 10, "ramp": RAMP}, beam=FakeBeam(), voltage=1e6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line.check_ramp_beam()


def test_the_rest_energy_survives_a_beam_not_yet_read():
    """``preProcess`` used to read it off a beam with no rest energy set,
    and every ramped run crashed there."""
    beam = FakeBeam()
    beam.particle_rest_energy_eV = None
    beam.particle_mass = type("V", (), {"val": np.array([1.67262192595e-27])})()
    line = FakeLine({"turns": 10, "ramp": RAMP}, beam=beam)
    assert line.rest_energy == pytest.approx(PROTON, rel=1e-8)
    assert FakeLine({"turns": 10}, beam=None).rest_energy == pytest.approx(ELECTRON, rel=1e-6)


# --- the codes -----------------------------------------------------------

TURNS = 10
CODES = ["xsuite", "elegant", "ocelot", "madx"]


def _machine(tmp_path):
    """The FODO cell of ``test_madx_native_turns.py``, called a ring."""
    quads = [
        Quadrupole(
            name="QUAD1F", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": -1},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
        ),
        Quadrupole(
            name="QUAD1D", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": 1.0},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
        ),
    ]
    markers = [
        Marker(
            name=name, machine_area="FODO", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": z}},
        )
        for name, z in (("M1", 0.0), ("M3", 4.25))
    ]
    names = ["M1", "QUAD1F", "QUAD1D", "M3"]
    section = {"sections": {"FODO": {"elements": names, "geometry": "closed"}}}
    machine = LAURA(
        element_list=[markers[0], *quads, markers[1]],
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section=section,
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, section


def _beam(subdir, name):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(subdir, f"{name}.openpmd.hdf5"))
    return {k: np.array(getattr(beam, k).val) for k in ("x", "px", "cp", "t")}


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """Generated once: ``frameworkGenerator`` is unseeded."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=16,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def _track(tmp_path, code, tracking, seed_beam):
    """Track and return the final beam at M3, and every turn's if written.

    LSC and CSR are off: Ocelot applies LSC by default, and its energy kick
    (1.5e5 eV in the first turn of this 100 pC bunch) would swamp the ramp.
    """
    if code == "elegant" and shutil.which("elegant") is None:
        pytest.skip("elegant is not installed")
    machine, section = _machine(tmp_path)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            "tracking": tracking,
            "lsc_enable": False,
            "csr_enable": False,
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    subdir = framework.subdirectory
    turns = {}
    for turn in range(1, TURNS + 1):
        name = f"M3{SEPARATOR}{turn:0{len(str(TURNS))}d}"
        if os.path.isfile(os.path.join(subdir, f"{name}.openpmd.hdf5")):
            turns[turn] = _beam(subdir, name)
    return {"final": _beam(subdir, "M3"), "turns": turns, "subdir": subdir}


@pytest.fixture(scope="module")
def runs(tmp_path_factory, seed_beam):
    """Every code, ramped and flat; cached, since each is a real run."""
    cache = {}

    def get(code, ramped):
        key = (code, ramped)
        if key not in cache:
            tracking = {"turns": TURNS, "write_turns": True}
            if ramped:
                tracking["ramp"] = RAMP
            directory = tmp_path_factory.mktemp(f"{code}_{'ramp' if ramped else 'flat'}")
            cache[key] = _track(directory, code, tracking, seed_beam)
        return cache[key]

    return get


@pytest.mark.parametrize("code", CODES)
def test_the_ramp_moves_the_orbit(code, runs):
    """Guards everything below: an ignored ramp would agree with itself.
    It was the first thing measured -- elegant re-centred ``p_central`` on
    the beam after every element and its ramp did nothing at all."""
    ramped, flat = runs(code, True)["final"], runs(code, False)["final"]
    assert np.abs(ramped["x"] - flat["x"]).max() > 5e-5


@pytest.mark.parametrize("code", CODES)
def test_no_rf_means_no_energy_change(code, runs, seed_beam):
    """The reference moves, the particles do not. Moving the reference
    without recomputing each particle's offset from it would change its
    energy by the 2% of the ramp, 1e5 eV here. (Xsuite drifts by a few eV
    with or without a ramp; that is its own.)"""
    start = _beam(os.path.dirname(seed_beam), "M1")["cp"]
    final = runs(code, True)["final"]["cp"]
    assert np.allclose(final, start, rtol=1e-5, atol=0)


@pytest.mark.parametrize("code", ["xsuite", "ocelot"])
def test_the_thick_codes_agree_on_the_ramped_orbit(code, runs):
    """Xsuite and Ocelot against elegant: the same thick quadrupoles, so
    the same answer (measured 1.0e-7 m and 2.7e-6 m, against a 1.6e-4 m
    effect)."""
    a, b = runs(code, True)["final"]["x"], runs("elegant", True)["final"]["x"]
    assert np.abs(a - b).max() < 1e-5


def test_madx_agrees_up_to_its_thin_lenses(runs):
    """MAD-X tracks sliced thin lenses, which differ from the thick codes
    by about 2e-5 m here with or without a ramp; the ramp moves it 1.2e-4 m."""
    a, b = runs("madx", True)["final"]["x"], runs("elegant", True)["final"]["x"]
    assert np.abs(a - b).max() < 5e-5


def test_madx_and_elegant_agree_on_time_every_turn(runs):
    """Both carry absolute time, and the ramp's reference clock is theirs:
    measured to 4e-16 s over 10 turns. Xsuite's ``t`` is relative to the
    reference and Ocelot's is a turn ahead, with or without a ramp."""
    madx, elegant = runs("madx", True)["turns"], runs("elegant", True)["turns"]
    assert set(madx) == set(elegant) == set(range(1, TURNS + 1))
    for turn in madx:
        assert np.allclose(madx[turn]["t"], elegant[turn]["t"], rtol=0, atol=1e-13), turn


def test_elegants_reference_follows_the_program(runs):
    """``pCentral`` in elegant's own watch file, pass by pass: pass ``k``
    (0-based) at turn ``k + 1``'s momentum."""
    pytest.importorskip("sdds")
    import sdds

    data = sdds.SDDS(0)
    data.load(os.path.join(runs("elegant", True)["subdir"], "M3.SDDS"))
    p_central = np.array(data.parameterData[data.parameterName.index("pCentral")])
    expected = EnergyRamp.from_dict(RAMP).p0c_per_pass(TURNS, ELECTRON)[:TURNS] / ELECTRON
    assert np.allclose(p_central, expected, rtol=1e-6)
