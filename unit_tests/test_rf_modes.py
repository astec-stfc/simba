"""``tracking: {rf: follow | fixed}``: how the RF keeps time, the same in every code."""

import os
import shutil
import warnings

import numpy as np
import pytest
from scipy.constants import c

import simba.Framework as fw
from helpers import fodo_machine, read_beam, skip_missing
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.Generators import frameworkGenerator
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice
from simba.Modules.EnergyRamp import rf_phase_slip, wrap_phase

ELECTRON = 510998.95
TURNS = 10
CAVITY_LENGTH = 0.2
RING = 4.0 + 1.5 * CAVITY_LENGTH  # where fodo_machine puts M3
BETA = 5e6 / np.hypot(5e6, ELECTRON)
HARMONIC = 10 * c * BETA / RING
RAMP = {"turns": [1, TURNS], "momentum": [5e6, 5.04e6]}


def test_without_a_ramp_fixed_and_follow_agree():
    beta = np.full(20, 0.99)
    for frequency in (HARMONIC, 1.003 * HARMONIC):
        assert np.allclose(
            rf_phase_slip("fixed", frequency, 1.3, RING, beta),
            rf_phase_slip("follow", frequency, 1.3, RING, beta),
            atol=1e-9,
        )


def test_on_the_harmonic_nothing_slips():
    beta = np.full(20, BETA)
    assert np.allclose(rf_phase_slip("fixed", HARMONIC, 2.0, RING, beta), 0, atol=1e-9)


def test_off_the_harmonic_follow_slips_by_the_fraction_each_pass():
    beta = np.linspace(BETA, 0.999, 5)
    frequency = 10.25 * c * BETA / RING
    slip = rf_phase_slip("follow", frequency, 2.0, RING, beta)
    assert np.allclose(slip, wrap_phase(2 * np.pi * 0.25 * np.arange(5)))


def test_fixed_times_the_reference_as_it_travels():
    """Each pass at its own speed: the ramp clock's mid-point put elegant
    4 eV out on the first ramped turn."""
    beta = np.array([0.990, 0.991, 0.993])
    s = 1.7
    arrival = np.array([
        s / (beta[0] * c),
        RING / (beta[0] * c) + s / (beta[1] * c),
        RING / (beta[0] * c) + RING / (beta[1] * c) + s / (beta[2] * c),
    ])
    expected = wrap_phase(2 * np.pi * HARMONIC * (arrival - arrival[0]))
    assert np.allclose(rf_phase_slip("fixed", HARMONIC, s, RING, beta), expected)


def test_synchronous_never_slips():
    assert not np.any(rf_phase_slip("synchronous", 1.003 * HARMONIC, 1, RING, [0.9, 0.95]))


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="follow, fixed"):
        rf_phase_slip("locked", HARMONIC, 0, RING, [0.9])


def test_a_phase_wraps_into_one_turn():
    assert np.allclose(wrap_phase([np.pi, -np.pi, 3 * np.pi / 2, 0.1]), [np.pi, np.pi, -np.pi / 2, 0.1])


@pytest.mark.parametrize(
    "cls,native",
    [
        (elegantLattice, "fixed"),
        (xsuiteLattice, "synchronous"),
        (madxLattice, "synchronous"),
        (ocelotLattice, "synchronous"),
        (bmadLattice, None),
    ],
    ids=lambda c: getattr(c, "__name__", str(c)),
)
def test_what_each_code_does_natively(cls, native):
    assert cls.native_rf == native


class FakeCavity:
    hardware_type = "RFCavity"

    def __init__(self, frequency, voltage=1e5):
        self.cavity = type("C", (), {"frequency": frequency})()
        self.simulation = type("S", (), {"field_amplitude": voltage})()


class FakeLine:
    """A lattice stub carrying only what the RF code reads."""

    def __init__(self, tracking, native, frequency=HARMONIC):
        self.file_block = {"tracking": tracking}
        self.objectname = "RING"
        self.code = "somecode"
        self.native_rf = native
        self.supports_ramp = True
        self.supports_nsuperperiods = True
        beam = type("B", (), {})()
        beam.cp = type("V", (), {"val": np.full(4, 5e6)})()
        beam.particle_rest_energy_eV = type("V", (), {"val": np.array([ELECTRON])})()
        self.global_parameters = {"beam": beam}
        self.elements = {"CAV": FakeCavity(frequency)}
        self.pass_length = RING

    def getSValues(self, as_dict=False, at_entrance=False):
        return {"CAV": 4.0 if at_entrance else 4.2}

    turns = frameworkLattice.turns
    nsuperperiods = frameworkLattice.nsuperperiods
    passes_per_turn = frameworkLattice.passes_per_turn
    ramp = frameworkLattice.ramp
    ramped = frameworkLattice.ramped
    rest_energy = frameworkLattice.rest_energy
    _input_reference = None
    _input_mean = frameworkLattice._input_mean
    design_p0c = None
    reference_p0c = frameworkLattice.reference_p0c
    rf_mode = frameworkLattice.rf_mode
    pass_p0c = frameworkLattice.pass_p0c
    pass_beta0 = frameworkLattice.pass_beta0
    accelerating_cavities = frameworkLattice.accelerating_cavities
    live_cavities = frameworkLattice.live_cavities
    cavity_voltage = staticmethod(frameworkLattice.cavity_voltage)
    rf_phase_corrections = frameworkLattice.rf_phase_corrections


def test_follow_is_the_default():
    assert FakeLine({"turns": 10}, "fixed").rf_mode == "follow"


def test_an_unknown_mode_warns_and_follows():
    with pytest.warns(UserWarning, match="Using follow"):
        assert FakeLine({"turns": 10, "rf": "locked"}, "fixed").rf_mode == "follow"


@pytest.mark.parametrize("native", ["fixed", "synchronous"])
def test_a_flat_ring_on_the_harmonic_needs_nothing(native):
    for mode in ("fixed", "follow"):
        assert FakeLine({"turns": 10, "rf": mode}, native).rf_phase_corrections() == {}


def test_one_pass_needs_nothing():
    line = FakeLine({"turns": 1, "ramp": RAMP}, "fixed", frequency=1.003 * HARMONIC)
    assert line.rf_phase_corrections() == {}


def test_a_ramp_moves_a_follower_only_for_fixed():
    assert FakeLine({"turns": 10, "ramp": RAMP, "rf": "follow"}, "synchronous").rf_phase_corrections() == {}
    moved = FakeLine({"turns": 10, "ramp": RAMP, "rf": "fixed"}, "synchronous").rf_phase_corrections()
    assert set(moved) == {"CAV"} and moved["CAV"][0] == 0 and moved["CAV"][-1] < 0


def test_a_ramp_moves_a_fixed_oscillator_only_for_follow():
    """Equal and opposite to the follower's correction."""
    assert FakeLine({"turns": 10, "ramp": RAMP, "rf": "fixed"}, "fixed").rf_phase_corrections() == {}
    follow = FakeLine({"turns": 10, "ramp": RAMP, "rf": "follow"}, "fixed").rf_phase_corrections()
    fixed = FakeLine({"turns": 10, "ramp": RAMP, "rf": "fixed"}, "synchronous").rf_phase_corrections()
    assert np.allclose(follow["CAV"], -fixed["CAV"])


def test_off_the_harmonic_a_follower_is_moved_even_without_a_ramp():
    line = FakeLine({"turns": 10}, "synchronous", frequency=1.002 * HARMONIC)
    assert set(line.rf_phase_corrections()) == {"CAV"}


def test_a_slip_too_small_to_matter_over_the_run_is_left_alone():
    """CLIC DR sat 1.5e-11 off the harmonic; moving its lag cost Xsuite its
    closed orbit and every ring parameter."""
    line = FakeLine({"turns": 20}, "synchronous", frequency=(1 + 1.5e-11) * HARMONIC)
    assert line.rf_phase_corrections() == {}


def test_a_cavity_with_no_voltage_is_never_moved():
    """LAURA's default cavity (3 GHz, 0 V) switched MAD-X's native turns off."""
    line = FakeLine({"turns": 10}, "synchronous", frequency=1.002 * HARMONIC)
    line.elements["CAV"].simulation.field_amplitude = 0.0
    assert line.rf_phase_corrections() == {}


def test_a_code_that_cannot_move_its_rf_warns():
    line = FakeLine({"turns": 10, "ramp": RAMP, "rf": "fixed"}, None)
    with pytest.warns(UserWarning, match="cannot move a cavity's phase"):
        assert line.rf_phase_corrections() == {}


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """Generated once: ``frameworkGenerator`` is unseeded."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-5, sigma_px=1e2, sigma_y=1e-5, sigma_py=1e2,
        sigma_z=1e-5, sigma_pz=1e2,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=1e-15, number_of_particles=16,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def _track(tmp_path, code, tracking, seed_beam, frequency=HARMONIC):
    """Track the ring; return the lattice and the bunch-mean energy gain per turn."""
    skip_missing(code)
    cavity = {
        "cavity": {"frequency": frequency, "phase": 60.0},
        "simulation": {"field_amplitude": 2.0e4},
    }
    machine, _, section = fodo_machine(
        tmp_path, cavity=cavity, closed=True, cavity_length=CAVITY_LENGTH
    )
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            "tracking": {"turns": TURNS, "write_turns": True, **tracking},
            "lsc_enable": False,
            "csr_enable": False,
        }
    }
    settings.layout = machine.layout
    settings.section = {"sections": {"FODO": section["sections"]["FODO"]}}
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    subdir = framework.subdirectory
    cp = [np.mean(read_beam(os.path.dirname(seed_beam), "M1").cp.val)]
    cp.extend(np.mean(read_beam(subdir, "M3", turn).cp.val) for turn in range(1, TURNS + 1))
    return framework["FODO"], np.diff(cp)


@pytest.fixture(scope="module")
def runs(tmp_path_factory, seed_beam):
    """Every code and case; cached, since each is a real run."""
    cache = {}
    cases = {
        "fixed": ({"ramp": RAMP, "rf": "fixed"}, HARMONIC),
        "follow": ({"ramp": RAMP, "rf": "follow"}, HARMONIC),
        "flat": ({}, HARMONIC),
        "off_harmonic": ({}, 10.02 / 10 * HARMONIC),
    }

    def get(code, case):
        if (code, case) not in cache:
            tracking, frequency = cases[case]
            directory = tmp_path_factory.mktemp(f"{code}_{case}")
            cache[code, case] = _track(directory, code, tracking, seed_beam, frequency)
        return cache[code, case]

    return get


def test_the_two_modes_differ(runs):
    """Guards the agreement below: turn 10 is 9050 eV fixed, 9500 eV follow."""
    fixed, follow = runs("elegant", "fixed")[1], runs("elegant", "follow")[1]
    assert follow[-1] - fixed[-1] > 300


@pytest.mark.parametrize("mode", ["fixed", "follow"])
@pytest.mark.parametrize("code", ["xsuite", "madx", "ocelot"])
def test_every_code_runs_the_rf_it_is_asked_for(code, mode, runs):
    """Against elegant; worst measured is MAD-X, 23 eV in 9500 on turn 10."""
    gains, reference = runs(code, mode)[1], runs("elegant", mode)[1]
    assert np.allclose(gains, reference, rtol=0, atol=40)


@pytest.mark.parametrize("code", ["xsuite", "madx", "ocelot"])
def test_off_the_harmonic_every_code_slips_as_elegant_does(code, runs):
    """7.2 deg a turn takes the gain from 10 to 20 keV; followers stayed at 10."""
    gains, reference = runs(code, "off_harmonic")[1], runs("elegant", "off_harmonic")[1]
    assert reference[-1] > 19000
    assert np.allclose(gains, reference, rtol=0, atol=40)


def test_ocelots_cavity_leaves_a_rings_reference_alone(runs):
    """Its cavity map moved the reference by ``V cos(phi)``, so the gain
    stayed at 10052 eV where elegant's falls to 9050."""
    gains, reference = runs("ocelot", "flat")[1], runs("elegant", "flat")[1]
    assert reference[0] - reference[-1] > 900
    assert np.allclose(gains, reference, rtol=0, atol=40)


def test_ocelot_finds_a_rings_optics_with_a_cavity_in_it(runs):
    """``twiss(tws0=None)`` refused a lattice with a cavity."""
    lattice = runs("ocelot", "flat")[0]
    periodic = lattice._ocelot_periodic()
    assert periodic and periodic[0].beta_x > 0


class PhaseLine:
    """A code with one cavity at 10 deg, moved by `apply_rf_phases`."""

    objectname = "RING"
    code = "fake"
    turns = 1
    rf_phase_sign = -1.0
    rf_phase_per_radian = 180 / np.pi

    def __init__(self, corrections, phases=None):
        self.corrections = corrections
        self.phases = {"CAV": 10.0} if phases is None else phases
        self._rf_corrections = None
        self._rf_phase0 = None

    def rf_phase_corrections(self):
        return self.corrections

    def cavity_phase(self, name):
        return self.phases.get(name)

    def set_cavity_phase(self, name, phase):
        self.phases[name] = phase

    begin_rf_phases = frameworkLattice.begin_rf_phases
    apply_rf_phases = frameworkLattice.apply_rf_phases
    rf_phase_shifts = frameworkLattice.rf_phase_shifts
    end_turns = frameworkLattice.end_turns


def test_a_phase_is_moved_in_the_codes_own_sign_and_units():
    line = PhaseLine({"CAV": np.array([0.0, 0.1, 0.2])})
    line.begin_rf_phases()
    line.apply_rf_phases(2)
    assert line.phases["CAV"] == pytest.approx(10.0 - np.degrees(0.2))
    line.apply_rf_phases(None)
    assert line.phases["CAV"] == 10.0


def test_a_phase_changed_between_runs_is_the_one_moved_from():
    """Ocelot kept its first run's phases for good."""
    line = PhaseLine({"CAV": np.array([0.0, 0.1])})
    line.begin_rf_phases()
    line.apply_rf_phases(1)
    line.end_turns()
    line.phases["CAV"] = 20.0
    line.begin_rf_phases()
    line.apply_rf_phases(1)
    assert line.phases["CAV"] == pytest.approx(20.0 - np.degrees(0.1))


def test_a_cavity_the_code_has_not_got_yet_is_read_when_it_has():
    """MAD-X has a segment's elements only once it has been tracked."""
    line = PhaseLine({"CAV": np.array([0.0, 0.1, 0.2])}, phases={})
    line.begin_rf_phases()
    line.apply_rf_phases(0)
    line.phases["CAV"] = 10.0
    line.apply_rf_phases(1)
    assert line.phases["CAV"] == pytest.approx(10.0 - np.degrees(0.1))


def test_a_cavity_the_code_never_has_is_named_once():
    line = PhaseLine({"CAV": np.array([0.0, 0.1])}, phases={})
    line.begin_rf_phases()
    line.apply_rf_phases(0)
    line.apply_rf_phases(1)
    with pytest.warns(UserWarning, match="CAV") as caught:
        line.end_turns()
    assert len(caught) == 1


@pytest.mark.parametrize(
    "cls, per_radian",
    [(madxLattice, 1 / (2 * np.pi)), (elegantLattice, 180 / np.pi)],
    ids=["madx", "elegant"],
)
def test_each_code_says_its_own_units(cls, per_radian):
    """MAD-X's LAG is in turns; the rest are in degrees."""
    assert cls.rf_phase_per_radian == pytest.approx(per_radian)
