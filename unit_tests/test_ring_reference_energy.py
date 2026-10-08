"""A ring's reference momentum is its section's design ``reference_energy``,
and its reference time ``tracking: reference_t0``, else the beam's reference
particle's.

A closed or periodic line used to take its reference from the incoming
bunch's mean momentum and time, which absorbs any injection energy or timing
offset: the beam then tracks as if the ring's magnets were set for it and
its RF phased to it. With a design reference, the offset is tracked.
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
from laura.models.element import Marker, Quadrupole, RFCavity
from simba import exceptions
from simba.Codes.Generators import frameworkGenerator

ELECTRON = 0.51099895e6
OFFSET = 5e-3
"""The beam's momentum offset from the design in the tracking tests: inside
the 1 % :class:`OffDesignEnergyWarning` threshold, and big enough that the
k1 it changes moves the orbit well clear of the codes' own noise."""

CODES = ["xsuite", "elegant", "ocelot", "madx"]


def _machine(tmp_path, closed=True, reference_energy=None, scale=1.0, cavity=None):
    """The FODO cell of ``test_madx_native_turns.py``, its quadrupoles' k1l
    multiplied by ``scale``; ``cavity``, RFCavity fields, adds a 0.2 m cavity
    after them (thick, for Ocelot, which divides by its length)."""
    quads = [
        Quadrupole(
            name="QUAD1F", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": -1.0 * scale},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
        ),
        Quadrupole(
            name="QUAD1D", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": 1.0 * scale},
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
    if cavity is not None:
        quads.append(
            RFCavity(
                name="CAV1", machine_area="FODO",
                physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": 3.95}},
                **cavity,
            )
        )
    names = ["M1", *(element.name for element in quads), "M3"]
    entry = {"elements": names}
    if closed:
        entry["geometry"] = "closed"
    if reference_energy is not None:
        entry["reference_energy"] = float(reference_energy)
    section = {"sections": {"FODO": entry}}
    machine = LAURA(
        element_list=[markers[0], *quads, markers[1]],
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section=section,
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, section


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


def _beam(subdir, name):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(subdir, f"{name}.openpmd.hdf5"))
    return {k: np.array(getattr(beam, k).val) for k in ("x", "px", "y", "py", "cp", "t")}


def _mean_cp(seed_beam):
    return float(np.mean(_beam(os.path.dirname(seed_beam), "M1")["cp"]))


def _energy(p0c):
    return float(np.hypot(p0c, ELECTRON))


def _framework(tmp_path, seed_beam, code="xsuite", tracking=None, **machine):
    machine, section = _machine(tmp_path, **machine)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            "tracking": tracking or {"turns": 1},
            "lsc_enable": False,
            "csr_enable": False,
        }
    }
    settings.layout = machine.layout
    # `settings.section` replaces the machine's sections wholesale, so the
    # geometry and energy have to be in it
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    return framework


def _loaded(tmp_path, seed_beam, **kw):
    """The lattice object with its input beam read, and the warnings that
    reading raised."""
    lattice = _framework(tmp_path, seed_beam, **kw).latticeObjects["FODO"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lattice.preProcess()
    return lattice, [w.message for w in caught]


# --- which reference -----------------------------------------------------


def test_a_ring_takes_its_reference_from_the_section(tmp_path, seed_beam):
    design = _mean_cp(seed_beam) / (1 + OFFSET)
    lattice, _ = _loaded(tmp_path, seed_beam, reference_energy=_energy(design))
    assert lattice.design_p0c == pytest.approx(design, rel=1e-12)
    assert lattice.reference_p0c == pytest.approx(design, rel=1e-12)


def test_the_beam_is_tracked_at_its_offset(tmp_path, seed_beam):
    """Xsuite's particles carry the offset as delta against the design."""
    design = _mean_cp(seed_beam) / (1 + OFFSET)
    lattice, _ = _loaded(tmp_path, seed_beam, reference_energy=_energy(design))
    assert lattice.pin.p0c[0] == pytest.approx(design, rel=1e-12)
    assert np.mean(lattice.pin.delta) == pytest.approx(OFFSET, rel=1e-6)


def test_a_ring_without_an_energy_uses_the_beam(tmp_path, seed_beam):
    lattice, _ = _loaded(tmp_path, seed_beam)
    assert lattice.design_p0c is None
    assert lattice.reference_p0c == pytest.approx(_mean_cp(seed_beam), rel=1e-12)


def test_an_open_line_ignores_the_energy(tmp_path, seed_beam):
    """A linac's reference follows its beam, whatever the section says."""
    design = _mean_cp(seed_beam) / (1 + OFFSET)
    lattice, _ = _loaded(
        tmp_path, seed_beam, closed=False, reference_energy=_energy(design)
    )
    assert lattice.design_p0c is None
    assert lattice.reference_p0c == pytest.approx(_mean_cp(seed_beam), rel=1e-12)


def test_a_ramp_owns_the_reference(tmp_path, seed_beam):
    framework = _framework(
        tmp_path, seed_beam,
        tracking={"turns": 10, "ramp": {"turns": [1, 10], "momentum": [5e6, 5.1e6]}},
        reference_energy=_energy(4e6),
    )
    assert framework.latticeObjects["FODO"].design_p0c is None


# --- the warning ---------------------------------------------------------


def test_a_beam_far_from_the_design_warns(tmp_path, seed_beam):
    design = _mean_cp(seed_beam) / 1.05
    _, caught = _loaded(tmp_path, seed_beam, reference_energy=_energy(design))
    assert any(isinstance(w, exceptions.OffDesignEnergyWarning) for w in caught)


def test_an_injection_offset_does_not_warn(tmp_path, seed_beam):
    design = _mean_cp(seed_beam) / (1 + OFFSET)
    _, caught = _loaded(tmp_path, seed_beam, reference_energy=_energy(design))
    assert not any(isinstance(w, exceptions.OffDesignEnergyWarning) for w in caught)


# --- Bmad ----------------------------------------------------------------


def test_bmad_prefers_the_design_to_its_reference_particle(tmp_path, seed_beam):
    """Bmad otherwise takes its reference from the beam's ``ref_idx``
    particle."""
    design = _mean_cp(seed_beam) / (1 + OFFSET)
    lattice, _ = _loaded(tmp_path, seed_beam, code="bmad", reference_energy=_energy(design))
    assert lattice._reference_p0c() == pytest.approx(design, rel=1e-12)
    assert lattice._reference_energy() == pytest.approx(_energy(design), rel=1e-12)


# --- tracking ------------------------------------------------------------


TURNS = 3
"""Enough for a ramp, which needs more than one turn."""


def _track(tmp_path, seed_beam, code, tracking=None, **machine):
    if code == "elegant" and shutil.which("elegant") is None:
        pytest.skip("elegant is not installed")
    framework = _framework(
        tmp_path, seed_beam, code=code, tracking={"turns": TURNS, **(tracking or {})}, **machine
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return _beam(framework.subdirectory, "M3")


@pytest.fixture(scope="module")
def runs(tmp_path_factory, seed_beam):
    """Per code: the ring at its design energy, the same ring with no energy
    and its quadrupoles scaled to what the design k1 is at the beam's
    momentum, the ring with no energy and unscaled quadrupoles, and the ring
    under a flat ramp at the design momentum."""
    cache = {}
    p_mean = _mean_cp(seed_beam)
    design = p_mean / (1 + OFFSET)

    def get(code):
        if code not in cache:
            cases = {
                "design": {"reference_energy": _energy(design)},
                "scaled": {"scale": design / p_mean},
                "beam": {},
                "ramp": {"tracking": {"ramp": {"turns": [1, TURNS], "momentum": [design, design]}}},
            }
            cache[code] = {
                name: _track(tmp_path_factory.mktemp(f"{code}_{name}"), seed_beam, code, **kw)
                for name, kw in cases.items()
            }
        return cache[code]

    return get


AGREEMENT = {"xsuite": 1e-6, "madx": 1e-6, "elegant": 2 * OFFSET, "ocelot": 0.05}
"""How closely each code's design and rescaled rings agree, as a fraction of
the design energy's effect. Xsuite and MAD-X track a quadrupole exactly in
delta, so the two are the same ring. elegant's QUAD and Ocelot's matrices
expand in delta to second order, and the expansion's error, second order in
an offset first order in it, is ~OFFSET of the effect. Worst over 8 beams:
elegant 5.3e-4, Ocelot 7.4e-3 -- and one Ocelot beam above 1e-2, hence its
looser bound, still far from the whole effect a missing design would give."""


@pytest.mark.parametrize("code", CODES)
def test_the_magnets_are_set_for_the_design(code, runs):
    """A k1 set at the design momentum is k1 * p_design / p at the beam's,
    so the design ring and the rescaled one track the same."""
    run = runs(code)
    for coord in ("x", "px", "y", "py"):
        effect = np.abs(run["beam"][coord] - run["design"][coord]).max()
        difference = np.abs(run["design"][coord] - run["scaled"][coord]).max()
        assert difference < AGREEMENT[code] * effect


@pytest.mark.parametrize("code", CODES)
def test_a_ramp_is_a_design_too(code, runs):
    """A flat ramp at the design momentum is the design ring: the beam's
    offset from the ramp is tracked from the first pass."""
    run = runs(code)
    for coord in ("x", "px", "y", "py"):
        effect = np.abs(run["beam"][coord] - run["design"][coord]).max()
        difference = np.abs(run["ramp"][coord] - run["design"][coord]).max()
        assert difference < 1e-6 * effect


@pytest.mark.parametrize("code", CODES)
def test_the_design_energy_changes_the_orbit(code, runs):
    """Guards the test above: a design energy that did nothing would agree
    with itself."""
    run = runs(code)
    assert np.abs(run["design"]["x"] - run["beam"]["x"]).max() > 1e-8


@pytest.mark.parametrize("code", CODES)
def test_the_particles_keep_their_momentum(code, runs, seed_beam):
    """Only the reference moves: re-expressing each particle against it
    must not change its energy."""
    start = _beam(os.path.dirname(seed_beam), "M1")["cp"]
    assert np.allclose(runs(code)["design"]["cp"], start, rtol=1e-6, atol=0)


# --- the reference time --------------------------------------------------

BETA = 5e6 / np.hypot(5e6, ELECTRON)
FREQUENCY = 10 * 299792458.0 * BETA / 4.0
"""h = 10 on a 4 m cell at the seed's 5 MeV/c."""
PHASE = 60.0
"""Off crest, so a timing error changes the energy the bunch gains."""
LATE = 1 / (20 * FREQUENCY)
"""The bunch's lateness on the reference, an 18 degree phase error."""


def _cavity(phase=PHASE):
    return {
        "cavity": {"frequency": FREQUENCY, "phase": phase},
        "simulation": {"field_amplitude": 1.0e5},
    }


def test_a_ring_takes_its_reference_time_from_the_setting(tmp_path, seed_beam):
    lattice, _ = _loaded(tmp_path, seed_beam, tracking={"turns": 1, "reference_t0": 1e-9})
    assert lattice.reference_t0 == 1e-9
    beta0 = lattice.reference_p0c / lattice.reference_energy
    t_mean, z_mean = lattice._input_mean("t"), lattice._input_mean("z")
    assert lattice.reference_z0 == pytest.approx(
        z_mean + beta0 * 299792458.0 * (t_mean - 1e-9), rel=1e-12
    )


def test_an_open_line_ignores_the_setting(tmp_path, seed_beam):
    lattice, _ = _loaded(
        tmp_path, seed_beam, closed=False, tracking={"turns": 1, "reference_t0": 1e-9}
    )
    assert lattice.reference_t0 == lattice._input_mean("t")
    assert lattice.reference_z0 == lattice._input_mean("z")


def test_else_a_ring_takes_the_reference_particles_time(tmp_path, seed_beam):
    lattice, _ = _loaded(tmp_path, seed_beam)
    lattice._input_particle = {"t": 2e-9, "z": 0.1}
    assert (lattice.reference_t0, lattice.reference_z0) == (2e-9, 0.1)
    lattice.file_block["tracking"]["reference_t0"] = 1e-9
    assert lattice.reference_t0 == 1e-9


def test_else_the_mean(tmp_path, seed_beam):
    lattice, _ = _loaded(tmp_path, seed_beam)
    assert lattice._input_particle is None
    assert lattice.reference_t0 == lattice._input_mean("t")


@pytest.fixture(scope="module")
def timed(tmp_path_factory, seed_beam):
    """Per code: the bunch ``LATE`` on a ring's reference time, the bunch
    on time with the cavity phased ``LATE`` later instead, and the bunch on
    time."""
    cache = {}
    t_mean = float(np.mean(_beam(os.path.dirname(seed_beam), "M1")["t"]))
    shift = 360.0 * FREQUENCY * LATE

    def get(code):
        if code not in cache:
            cases = {
                "late": {"cavity": _cavity(), "tracking": {"reference_t0": t_mean - LATE}},
                "phased": {"cavity": _cavity(PHASE - shift)},
                "on time": {"cavity": _cavity()},
            }
            cache[code] = {
                name: _track(tmp_path_factory.mktemp(f"{code}_{name}"), seed_beam, code, **kw)
                for name, kw in cases.items()
            }
        return cache[code]

    return get


TIMING = {
    "xsuite": {"cp": 1e-6, "t": 1e-6, "x": 1e-6, "px": 1e-6},
    "madx": {"cp": 1e-6, "t": 1e-6, "x": 1e-6, "px": 1e-6},
    "elegant": {"cp": 1e-6, "t": 1e-6, "x": 1e-6, "px": 1e-6},
    "ocelot": {"cp": 2e-2, "t": 2e-2},
}
"""How closely each code's late and rephased runs agree, by coordinate, as a
fraction of the lateness's effect. elegant only does once simba moves its
cavities' phases (:meth:`~simba.Codes.Elegant.Elegant.elegantLattice.rf_fiducial_corrections`):
an RFCA phases itself to the bunch. Ocelot's cavity map takes its transverse
part -- end focusing and adiabatic damping -- from the *reference*
particle's energy gain, so a bunch 18 degrees off the reference is focused
as if it gained the reference's energy: x and px are 30-40 % out whatever
the lateness, which no reference simba gives it can fix. Its cp and t,
measured up to 0.8 % and 1.2 % at a quarter of ``LATE``, are its own
expansion about the reference phase."""


@pytest.mark.parametrize("code", CODES)
def test_a_late_bunch_sees_the_rf_late(code, timed):
    """Arriving ``LATE`` on the reference is the same as the cavity's phase
    moving by ``LATE``: the same particles at the same times see the same
    voltage."""
    run = timed(code)
    for coord, tolerance in TIMING[code].items():
        effect = np.abs(run["on time"][coord] - run["late"][coord]).max()
        difference = np.abs(run["late"][coord] - run["phased"][coord]).max()
        assert difference < tolerance * effect, (coord, difference, effect)


@pytest.mark.parametrize("code", CODES)
def test_lateness_changes_the_energy(code, timed):
    """Guards the test above."""
    run = timed(code)
    assert np.abs(run["on time"]["cp"] - run["late"]["cp"]).max() > 100.0
