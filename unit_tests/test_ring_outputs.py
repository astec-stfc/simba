"""What a ring run writes, and where, the same in every code."""

import json
import os
import re
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole
from simba.Codes.Generators import frameworkGenerator
from simba.Framework_objects import OUTPUT_TURN_SEPARATOR as SEPARATOR
from simba.Framework_objects import frameworkLattice
from simba.Modules import constants

TURNS = 4
RING_CODES = ["elegant", "xsuite", "madx", "ocelot"]
END_Z = 3.25 + 1.0
BMAD_SO = os.path.expanduser("~/Documents/bmad-ecosystem/production/lib/libtao.so")


def _machine(tmp_path):
    """A FODO cell with a marker in the middle, called a ring."""
    elements = [
        Marker(
            name="M1", machine_area="FODO", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
        ),
        Quadrupole(
            name="QUAD1F", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": -1},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
        ),
        Marker(
            name="MID", machine_area="FODO", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
        ),
        Quadrupole(
            name="QUAD1D", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": 1.0},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
        ),
        Marker(
            name="M3", machine_area="FODO", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": END_Z}},
        ),
    ]
    names = [e.name for e in elements]
    section = {"sections": {"FODO": {"elements": names, "geometry": "closed"}}}
    machine = LAURA(
        element_list=elements,
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section=section,
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, section


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """Generated once: ``frameworkGenerator`` is unseeded. Placed at s = 100 m,
    nowhere near the lattice, so an s taken from it rather than from the
    lattice shows."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-4, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=1e-15, number_of_particles=64,
    ).write()
    path = os.path.join(str(directory), "M1.openpmd.hdf5")
    beam = _read(path)
    beam.s = 100.0
    rbf.openpmd.write_openpmd_beam_file(beam, path)
    return path


def _read(path):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, path)
    return beam


def _skip_missing(code):
    if code == "elegant" and shutil.which("elegant") is None:
        pytest.skip("elegant is not installed")
    module = {"xsuite": "xtrack", "madx": "cpymad", "ocelot": "ocelot", "bmad": "pytao"}
    if code in module:
        pytest.importorskip(module[code])


def _framework(tmp_path, code, tracking, seed_beam, inputs=None):
    _skip_missing(code)
    machine, section = _machine(tmp_path)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": inputs or {},
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
    return framework


def _track(tmp_path, code, tracking, seed_beam):
    framework = _framework(tmp_path, code, tracking, seed_beam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return framework.subdirectory


def _turn(subdir, element, turn):
    """``element``'s beam on ``turn``, from its one multi-turn file."""
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(
        beam, os.path.join(subdir, f"{element}.openpmd.hdf5"), turn=turn
    )
    return beam


def _turns(subdir, element):
    return rbf.openpmd.openpmd_turns(os.path.join(subdir, f"{element}.openpmd.hdf5"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory, seed_beam):
    """Each code's ring run, writing every turn; cached, since each is real."""
    cache = {}

    def get(code, **tracking):
        key = (code, tuple(sorted(tracking.items())))
        if key not in cache:
            directory = tmp_path_factory.mktemp(code)
            cache[key] = _track(
                directory, code, {"turns": TURNS, "write_turns": True, **tracking},
                seed_beam,
            )
        return cache[key]

    return get


# --- markers --------------------------------------------------------------


@pytest.mark.parametrize("code", RING_CODES)
@pytest.mark.parametrize("element", ["MID", "M3"])
def test_a_marker_is_recorded_on_every_turn(code, element, runs):
    """In the one file, a turn to an iteration, the end of the line too."""
    assert _turns(runs(code), element) == list(range(1, TURNS + 1)), code


@pytest.mark.parametrize("code", RING_CODES)
def test_no_file_is_written_per_turn(code, runs):
    subdir = runs(code)
    per_turn = re.compile(rf"{SEPARATOR}\d+\.")
    assert not [name for name in os.listdir(subdir) if per_turn.search(name)]


@pytest.mark.parametrize("code", RING_CODES)
def test_each_turn_says_its_turn(code, runs):
    subdir = runs(code)
    for turn in range(1, TURNS + 1):
        assert _turn(subdir, "MID", turn).turn == turn


@pytest.mark.parametrize("code", RING_CODES)
def test_the_end_of_the_line_reads_as_its_last_turn(code, runs):
    """What the next line takes."""
    assert _read(os.path.join(runs(code), "M3.openpmd.hdf5")).turn == TURNS


@pytest.mark.parametrize("code", RING_CODES)
@pytest.mark.parametrize(
    "tracking",
    [{}, {"write_turns": False}, {"turns": 1}],
    ids=["every_turn", "last_turn", "one_turn"],
)
def test_the_input_beam_is_not_overwritten(code, tracking, runs, seed_beam):
    """The start marker's file *is* the input beam: recording at markers
    must not write over it (Xsuite's first version did)."""
    written = _read(os.path.join(runs(code, **tracking), "M1.openpmd.hdf5"))
    assert written.turn is None
    np.testing.assert_array_equal(written.x.val, _read(seed_beam).x.val)


@pytest.mark.parametrize("code", [c for c in RING_CODES if c != "elegant"])
@pytest.mark.parametrize("element", ["MID", "M3"])
def test_every_code_records_the_same_turn_as_elegant(code, element, runs):
    """The beam is mismatched, so its size changes turn to turn, and each
    code's turn ``k`` must be nearest elegant's turn ``k``.

    Nearest rather than within a tolerance: measured, Xsuite and Ocelot agree
    with elegant to 2e-5, but MAD-X's thin lenses put it up to 4% out (at M3
    on turn 3, where the beam is smallest), as far as one turn is from the
    next there.
    """
    if shutil.which("elegant") is None:
        pytest.skip("elegant is not installed")

    def sizes(subdir):
        return np.array([
            [
                _turn(subdir, element, turn).sigmas.sigma_x.val,
                _turn(subdir, element, turn).sigmas.sigma_y.val,
            ]
            for turn in range(1, TURNS + 1)
        ])

    ours, theirs = sizes(runs(code)), sizes(runs("elegant"))
    for turn in range(TURNS):
        distance = np.linalg.norm((theirs - ours[turn]) / theirs, axis=1)
        assert np.argmin(distance) == turn, (code, element, turn + 1, distance)


def test_the_mismatch_really_does_change_the_size_turn_to_turn(runs):
    """Else the test above could not tell one turn from the next."""
    subdir = runs("xsuite")
    sizes = [
        _turn(subdir, "MID", turn).sigmas.sigma_x.val for turn in range(1, TURNS + 1)
    ]
    assert np.min(np.abs(np.diff(sizes)) / sizes[0]) > 1e-2


# --- superperiods ---------------------------------------------------------


@pytest.mark.parametrize("element", ["MID", "M3"])
def test_xsuite_superperiods_record_the_last_pass_of_each_turn(element, runs):
    """Two superperiods for ``TURNS`` turns is the cell for ``2 * TURNS``:
    turn ``k`` of the first is turn ``2k`` of the second, exactly."""
    two = runs("xsuite", nsuperperiods=2)
    cells = _track_cells(runs)
    for turn in range(1, TURNS + 1):
        ours = _turn(two, element, turn)
        theirs = _turn(cells, element, 2 * turn)
        np.testing.assert_allclose(ours.x.val, theirs.x.val, rtol=0, atol=1e-15)


def _track_cells(runs):
    """The single cell, ``2 * TURNS`` times."""
    return runs("xsuite", turns=2 * TURNS)


def test_xsuite_superperiods_record_every_turn(runs):
    """The monitors were sized to ``turns``, short of a run of passes."""
    two = runs("xsuite", nsuperperiods=2)
    for turn in range(1, TURNS + 1):
        assert _turn(two, "MID", turn).turn == turn


# --- cp -------------------------------------------------------------------


def test_xsuite_reads_back_the_total_momentum_as_cp(runs):
    """Xsuite's ``p0c * (1 + delta)`` is each particle's total momentum, and
    that is what simba's ``cp`` must come back as."""
    subdir = runs("xsuite")
    # the last turn's, which Xsuite also writes as its own
    with open(os.path.join(subdir, "MID.xsuite.json")) as handle:
        particles = json.load(handle)
    total = np.asarray(particles["p0c"]) * (1 + np.asarray(particles["delta"]))
    alive = np.asarray(particles["state"]) > 0
    cp = _turn(subdir, "MID", TURNS).cp.val
    np.testing.assert_allclose(cp, total[alive], rtol=1e-12)


# --- programs -------------------------------------------------------------


def test_madx_programs_a_sliced_quadrupole_as_xsuite_does(tmp_path, seed_beam):
    program = {"element": "QUAD1F", "parameter": "k1",
               "turns": [1, TURNS], "values": [-1.05, -0.9]}

    def sizes(code, **extra):
        subdir = _track(
            tmp_path / f"{code}{len(extra)}", code,
            {"turns": TURNS, "write_turns": True, **extra}, seed_beam,
        )
        return np.array([
            _turn(subdir, "MID", turn).sigmas.sigma_x.val for turn in range(1, TURNS + 1)
        ])

    effect = {
        code: sizes(code, programs=[program]) - sizes(code)
        for code in ("madx", "xsuite")
    }
    assert np.min(np.abs(effect["xsuite"])) > 5e-6
    np.testing.assert_allclose(effect["madx"], effect["xsuite"], rtol=0.1)


# --- the input beam's name ------------------------------------------------


@pytest.mark.parametrize("block, name", [
    ({}, "M1"),
    ({"input": {}}, "M1"),
    ({"input": {"particle_definition": "initial_distribution"}}, "laser"),
    ({"input": {"particle_definition": "BEAM"}}, "BEAM"),
])
def test_the_input_beam_is_named_once_for_every_code(block, name):
    """Eight codes each had this as their own copy."""
    line = SimpleNamespace(file_block=block, start="M1")
    assert frameworkLattice.input_particle_definition.fget(line) == name


class _Read(Exception):
    pass


@pytest.mark.parametrize("code", [
    "astra", "csrtrack", "elegant", "genesis", "gpt", "opal", "ocelot",
    "xsuite", "bmad", "madx", "cheetah", "waket",
])
@pytest.mark.parametrize("block, name", [
    ({}, "M1"),
    ({"particle_definition": "initial_distribution"}, "laser"),
    ({"particle_definition": "BEAM"}, "BEAM"),
])
def test_every_code_reads_the_input_beam_it_is_given(
    code, block, name, tmp_path, seed_beam, monkeypatch
):
    """CSRTrack, elegant and Genesis read the start's file whatever was
    stated, and CSRTrack's ``initial_distribution`` branch could not be
    reached."""
    def read(self, prefix, definition, *args, **kwargs):
        raise _Read(definition)

    monkeypatch.setattr(frameworkLattice, "load_input_beam", read)
    line = _framework(tmp_path, code, {}, seed_beam, inputs=block)["FODO"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_Read) as read_from:
            line.preProcess()
    assert str(read_from.value) == name


# --- species --------------------------------------------------------------


@pytest.mark.parametrize("species", ["proton", "positron"])
@pytest.mark.parametrize("code, refused", [
    ("elegant", True), ("ocelot", True), ("cheetah", True),
    ("xsuite", False), ("madx", False), ("bmad", False),
])
def test_an_electron_only_code_refuses_anything_else(
    code, refused, species, tmp_path, seed_beam
):
    beam = _read(seed_beam)
    beam.set_species(species)
    (tmp_path / "beam").mkdir()
    other = str(tmp_path / "beam" / "other.openpmd.hdf5")
    rbf.openpmd.write_openpmd_beam_file(beam, other)
    line = _framework(tmp_path / "run", code, {}, other)["FODO"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if refused:
            with pytest.raises(ValueError, match="tracks only electrons"):
                line.preProcess()
        else:
            line.preProcess()
            assert line.reference_charge == 1


# --- which codes can ------------------------------------------------------


@pytest.mark.parametrize("flag", [
    "supports_turns", "supports_periodic", "supports_frequency_map",
    "supports_single_particle", "supports_dynamic_aperture",
    "supports_nsuperperiods", "supports_radiation", "supports_programs",
    "supports_ramp",
])
def test_the_codes_that_can_are_read_off_the_codes(flag):
    """The warnings named them by hand, and had fallen behind: Bmad and
    MAD-X were missing from several."""
    import simba.Framework_lattices as lattices

    can = sorted(
        cls.model_fields["code"].default
        for cls in vars(lattices).values()
        if isinstance(cls, type) and issubclass(cls, frameworkLattice)
        and getattr(cls, flag)
    )
    sentence = frameworkLattice.codes_that_can(flag)
    assert can, flag
    for code in can:
        assert code in sentence, (flag, code)
    assert sentence.count(",") + sentence.count(" and ") == len(can) - 1


# --- charge ---------------------------------------------------------------


@pytest.mark.parametrize("code", RING_CODES)
@pytest.mark.parametrize("element", ["MID", "M3"])
def test_every_code_writes_the_input_charge(code, element, runs, seed_beam):
    """No particle is lost here, so the charge out is the charge in. Xsuite's
    was ``sum(q0)``, its charge *state*: -1 C for each macroparticle's -1."""
    written = _turn(runs(code), element, TURNS)
    seed = _read(seed_beam)
    assert len(written.x.val) == len(seed.x.val)
    np.testing.assert_allclose(
        float(written.total_charge.val), float(seed.total_charge.val), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(written.particle_charge.val), -constants.elementary_charge,
        rtol=1e-12,
    )


def test_xsuite_macroparticles_carry_the_bunchs_charge(seed_beam, tmp_path):
    pytest.importorskip("xtrack")
    seed = _read(seed_beam)
    particles = rbf.beam.write_xsuite_beam_file(seed, write=False)
    charge = abs(float(seed.total_charge.val))
    real = np.sum(np.asarray(particles.weight)) * constants.elementary_charge
    assert real == pytest.approx(charge, rel=1e-12)
    back = rbf.beam()
    rbf.xsuite.read_xsuite_beam_file(back, particles)
    assert float(back.total_charge.val) == pytest.approx(-charge, rel=1e-12)


def _xsuite_with_losses(lost):
    """Four macroparticles of 10 e each, those in ``lost`` lost, and put at
    the end of the arrays, as Xsuite does with the particles it loses."""
    import xtrack as xt

    particles = xt.Particles(
        p0c=5e6, mass0=xt.ELECTRON_MASS_EV, q0=-1,
        x=[0.0, 1e-4, 2e-4, 3e-4], weight=[10.0] * 4,
    )
    particles.state[list(lost)] = -1
    order = [i for i in range(4) if i not in lost] + list(lost)
    for name in ("x", "state", "particle_id", "weight", "zeta", "delta"):
        getattr(particles, name)[:] = np.asarray(getattr(particles, name))[order]
    return particles


def test_xsuite_drops_its_lost_particles():
    """A lost particle has state <= 0 and keeps the coordinates it was lost
    at. It was read back as part of the beam, charge and all."""
    pytest.importorskip("xtrack")
    beam = rbf.beam()
    rbf.xsuite.read_xsuite_beam_file(beam, _xsuite_with_losses([1]))
    np.testing.assert_array_equal(beam.x.val, [0.0, 2e-4, 3e-4])
    assert float(beam.total_charge.val) == pytest.approx(
        -30 * constants.elementary_charge, rel=1e-12
    )
    assert np.shape(beam.z.val) == (3,)


@pytest.mark.parametrize("lost, index, found", [([1], 2, 1), ([1], 1, None), ([], 2, 2)])
def test_xsuite_finds_the_reference_by_its_id(lost, index, found):
    """Xsuite moves lost particles to the end, so the n-th particle read was
    no longer the n-th written. The reference is found by ``particle_id``, and
    a lost reference is no reference."""
    pytest.importorskip("xtrack")
    beam = rbf.beam()
    rbf.xsuite.read_xsuite_beam_file(beam, _xsuite_with_losses(lost), ref_index=index)
    assert beam.reference_particle_index == found


def test_xsuite_without_a_reference_has_one_z_per_particle():
    """``t[None]`` made z a (1, N) array."""
    pytest.importorskip("xtrack")
    beam = rbf.beam()
    rbf.xsuite.read_xsuite_beam_file(beam, _xsuite_with_losses([]))
    assert np.shape(beam.z.val) == (4,)


# --- s --------------------------------------------------------------------


@pytest.mark.parametrize("code", RING_CODES)
def test_s_is_the_lattices_not_the_incoming_beams(code, runs):
    """The seed beam says s = 100 m; the lattice starts at 0."""
    subdir = runs(code)
    middle = _turn(subdir, "MID", TURNS)
    end = _turn(subdir, "M3", TURNS)
    assert float(np.mean(middle.s.val)) == pytest.approx(2.0, abs=1e-6), code
    assert float(np.mean(end.s.val)) == pytest.approx(END_Z, abs=1e-6), code


@pytest.mark.parametrize(
    "name, turns, writes",
    [("M1", 1, False), ("M1", 4, False), ("MID", 1, True), ("MID", 4, True)],
)
def test_the_start_is_never_written(name, turns, writes):
    """Its file is the incoming beam; in a ring its turns are the end's."""
    line = SimpleNamespace(start="M1", turns=turns)
    assert frameworkLattice.writes_output(line, name) is writes


@pytest.mark.parametrize(
    "index, interval, sampled",
    [(None, 4, None), (8, 4, 2), (6, 4, None), (6, 1, 6), (0, 8, 0)],
)
def test_the_reference_particle_follows_the_sampling(index, interval, sampled):
    """Ocelot tracked every n-th particle, and went on naming the full beam's
    reference index: on a sampled run every beam it wrote took the wrong
    particle as its reference. The beam is now sampled as it is read, and
    :meth:`sample_beam` carries the index across."""
    line = SimpleNamespace(sample_interval=interval)
    assert frameworkLattice.sampled_index(line, index) == sampled


@pytest.mark.parametrize("code", RING_CODES + ["bmad"])
def test_every_code_tracks_every_nth_particle(code, tmp_path, seed_beam):
    if code == "bmad" and not os.path.exists(BMAD_SO):
        pytest.skip("Bmad libtao not installed")
    seed = _read(seed_beam)

    def track(interval, beam):
        framework = _framework(
            tmp_path / str(interval), code, {}, beam,
            inputs={"sample_interval": interval},
        )
        if code == "bmad":
            framework["FODO"].libtao = BMAD_SO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            framework.track()
        return _read(os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))

    sampled, full = track(2, seed_beam), track(1, seed_beam)
    assert len(sampled.x) == len(seed.x) // 2
    np.testing.assert_allclose(sampled.x.val, full.x.val[::2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(sampled.cp.val, full.cp.val[::2], rtol=1e-14)
    assert float(sampled.total_charge) == pytest.approx(float(seed.total_charge))
    if code != "xsuite":
        # a single Xsuite pass recentres zeta on the beam's own mean before
        # every element (Xsuite.run), so its t is still the sample's
        np.testing.assert_allclose(sampled.t.val, full.t.val[::2], rtol=0, atol=1e-18)


def _referenced(beam, recorded=None):
    line = SimpleNamespace(
        global_parameters={"beam": beam}, _input_reference=recorded, design_p0c=None,
        fixed_reference=False,
    )
    line._input_mean = lambda coord: frameworkLattice._input_mean(line, coord)
    return line


def test_the_reference_is_the_incoming_beams_before_sampling(seed_beam):
    seed = _read(seed_beam)
    full = float(np.mean(seed.cp.val))
    every_other = SimpleNamespace(sample_interval=2)
    every_other.sampled_index = lambda i: frameworkLattice.sampled_index(every_other, i)
    sampled = frameworkLattice.sample_beam(every_other, seed)
    line = _referenced(sampled, {"cp": full, "t": 1e-9, "z": 0.25})
    assert frameworkLattice.reference_p0c.fget(line) == full
    assert frameworkLattice.reference_t0.fget(line) == 1e-9
    assert frameworkLattice.reference_z0.fget(line) == 0.25
    assert float(np.mean(sampled.cp.val)) != pytest.approx(full, rel=1e-6)


def test_without_one_recorded_the_reference_is_the_beams_own(seed_beam):
    seed = _read(seed_beam)
    line = _referenced(seed)
    assert frameworkLattice.reference_p0c.fget(line) == pytest.approx(
        float(np.mean(seed.cp.val)), rel=1e-15
    )


def test_xsuite_measures_delta_and_zeta_from_the_reference_its_given(seed_beam):
    """``delta`` was taken from the beam's own mean momentum and the particles
    given ``p0c``, so a p0c other than the mean put every particle off."""
    pytest.importorskip("xtrack")
    seed = _read(seed_beam)
    p0c, t0 = 1.001 * float(np.mean(seed.cp.val)), float(np.mean(seed.t.val)) + 1e-12
    particles = seed.write_xsuite_beam_file(write=False, p0c=p0c, t0=t0)
    np.testing.assert_allclose(
        np.asarray(particles.p0c) * (1 + np.asarray(particles.delta)),
        seed.cp.val, rtol=1e-14,
    )
    beta0 = float(np.asarray(particles.beta0)[0])
    np.testing.assert_allclose(
        -np.asarray(particles.zeta) / (beta0 * constants.speed_of_light) + t0,
        seed.t.val, rtol=0, atol=1e-21,
    )


def test_cheetah_gets_back_the_energies_it_was_given(seed_beam):
    """Cheetah's delta is dE / p0c; simba wrote dE / E, 1/beta0 off: 25 eV
    here, where dE is 5 keV. What is left, 2e-6 eV, is Cheetah's electron mass
    not quite being simba's."""
    pytest.importorskip("cheetah")
    seed = _read(seed_beam)
    energy = 1.001 * float(np.mean(seed.energy.val))
    particle_beam = seed.write_cheetah_beam_file(write=False, energy=energy)
    np.testing.assert_allclose(
        particle_beam.energies.numpy(), seed.energy.val, rtol=0, atol=1e-4
    )


def _spread_beam(seed_beam):
    """The seed with a 1% momentum spread, so p_x / p0 and the slope
    p_x / p_z differ by enough to see."""
    seed = _read(seed_beam)
    rng = np.random.default_rng(1)
    scale = 1 + 0.01 * rng.standard_normal(len(seed.x))
    for coord in ("px", "py", "pz"):
        setattr(seed._beam, coord, getattr(seed._beam, coord) * scale)
    return seed


def test_ocelot_is_given_px_over_p0_and_read_back_from_it(seed_beam):
    """Ocelot's x' is p_x / p0 (ParticleArray); simba wrote and read the slope
    p_x / p_z, off by 1 + delta. The round trip hid it: both ends agreed."""
    pytest.importorskip("ocelot")
    from simba.Modules.Beams import ocelot as rbf_ocelot

    beam = _spread_beam(seed_beam)
    energy = float(np.mean(beam.energy.val))
    p0c = np.sqrt(energy**2 - float(beam.E0_eV) ** 2)
    parray = rbf_ocelot.particle_group_to_parray(beam, energy=energy)
    np.testing.assert_allclose(parray.px(), beam.cpx.val / p0c, rtol=1e-12)
    np.testing.assert_allclose(parray.py(), beam.cpy.val / p0c, rtol=1e-12)

    back = rbf.beam()
    rbf_ocelot.particle_array_to_beam(back, parray)
    for coord in ("cpx", "cpy", "cpz"):
        np.testing.assert_allclose(
            getattr(back, coord).val, getattr(beam, coord).val, rtol=1e-10
        )


def test_cheetah_is_given_px_over_p0c_and_read_back_from_it(seed_beam):
    """Cheetah's px is p_x / p0c (``ParticleBeam.from_openpmd_file``)."""
    pytest.importorskip("cheetah")
    from simba.Modules.Beams import cheetah as rbf_cheetah

    beam = _spread_beam(seed_beam)
    energy = float(np.mean(beam.energy.val))
    p0c = np.sqrt(energy**2 - float(beam.E0_eV) ** 2)
    particle_beam = beam.write_cheetah_beam_file(write=False, energy=energy)
    np.testing.assert_allclose(particle_beam.px.numpy(), beam.cpx.val / p0c, rtol=1e-12)
    np.testing.assert_allclose(particle_beam.py.numpy(), beam.cpy.val / p0c, rtol=1e-12)

    back = rbf.beam()
    rbf_cheetah.interpret_cheetah_ParticleBeam(back, particle_beam)
    # Cheetah's electron is 0.0125 eV lighter than simba's: 1e-9 in cp here
    for coord in ("cpx", "cpy", "cpz"):
        np.testing.assert_allclose(
            getattr(back, coord).val, getattr(beam, coord).val, rtol=1e-8
        )


def _fake(start_s, first_length):
    line = SimpleNamespace(
        start_s=start_s,
        startObject=SimpleNamespace(physical=SimpleNamespace(length=first_length)),
    )
    return frameworkLattice.entrance_s.fget(line)


def test_the_entrance_is_the_start_on_a_marker():
    assert _fake(12.0, 0.0) == 12.0


def test_the_entrance_is_before_a_first_element_with_length():
    """`start_s` is at the exit of the first element."""
    assert _fake(12.0, 0.5) == 11.5


# --- rematching -----------------------------------------------------------


@pytest.mark.parametrize("code", RING_CODES + ["bmad"])
def test_every_code_rematches_to_the_input_twiss(code, tmp_path, seed_beam):
    framework = _framework(
        tmp_path, code, {"turns": 1}, seed_beam,
        inputs={"twiss": {"beta_x": 7.0, "alpha_x": -1.5}},
    )
    lattice = framework["FODO"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lattice.preProcess()
    twiss = lattice.global_parameters["beam"].twiss
    assert float(twiss.beta_x.val) == pytest.approx(7.0, rel=1e-6), code
    assert float(twiss.alpha_x.val) == pytest.approx(-1.5, rel=1e-6), code


@pytest.mark.parametrize("code", RING_CODES + ["bmad"])
def test_a_plane_without_twiss_is_left_alone(code, tmp_path, seed_beam):
    framework = _framework(
        tmp_path, code, {"turns": 1}, seed_beam,
        inputs={"twiss": {"beta_x": 7.0, "alpha_x": -1.5}},
    )
    lattice = framework["FODO"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lattice.preProcess()
    # not exactly: rematching x keeps each particle's cp and y', so moving x'
    # moves py a little (measured: beta_y by 3.6e-8, `rematchXPlane` alone)
    beta_y = float(lattice.global_parameters["beam"].twiss.beta_y.val)
    assert beta_y == pytest.approx(float(_read(seed_beam).twiss.beta_y.val), rel=1e-6)


# --- one-turn map ---------------------------------------------------------


def _one_turn_map(tmp_path, code, tracking, seed_beam):
    framework = _framework(tmp_path, code, tracking, seed_beam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return np.asarray(framework["FODO"].one_turn_map)


@pytest.mark.parametrize(
    "tracking",
    [
        {"turns": 3},
        {"turns": 3, "write_turns": True},
        {"turns": 1, "nsuperperiods": 2},
    ],
    ids=["turns", "write_turns", "superperiods"],
)
def test_madxs_one_turn_map_is_the_lines_however_it_is_tracked(
    tracking, tmp_path, seed_beam
):
    """MAD-X's turn loop (which programs, ramps, moving RF and several
    segments all need) took the model optics on every pass it did not record,
    and the sector maps accumulate: three turns gave M^3, two superperiods
    M^2, and recording every turn no map at all. The map is the line's, as
    every other code's is."""
    one = _one_turn_map(tmp_path / "one", "madx", {"turns": 1}, seed_beam)
    looped = _one_turn_map(
        tmp_path / "looped", "madx", {**tracking, "native_turns": False}, seed_beam
    )
    np.testing.assert_allclose(looped, one, rtol=0, atol=1e-12)


# --- t --------------------------------------------------------------------


def _clock(seed_beam, element_s, turn):
    """The reference's time at ``element_s`` on ``turn``: the incoming beam's
    mean ``t``, plus a period per turn before it."""
    beam = _read(seed_beam)
    cp = float(np.mean(beam.cp.val))
    beta0 = cp / np.hypot(cp, beam.E0_eV.val)
    c = 299792458.0
    return float(np.mean(beam.t.val)) + ((turn - 1) * END_Z + element_s) / (beta0 * c)


@pytest.mark.parametrize("code", RING_CODES)
@pytest.mark.parametrize("element, element_s", [("MID", 2.0), ("M3", END_Z)])
def test_t_is_on_the_reference_clock(code, element, element_s, runs, seed_beam):
    """A period is 14 ns; the centroid moves 8 fs off the reference in four
    turns, as it does in elegant."""
    subdir = runs(code)
    for turn in range(1, TURNS + 1):
        t = _turn(subdir, element, turn).t.val
        clock = _clock(seed_beam, element_s, turn)
        assert float(np.mean(t)) == pytest.approx(clock, abs=2e-14), (code, turn)


@pytest.mark.parametrize("code", [c for c in RING_CODES if c != "elegant"])
@pytest.mark.parametrize("element", ["MID", "M3"])
def test_every_code_gives_each_particle_the_t_elegant_does(code, element, runs):
    if shutil.which("elegant") is None:
        pytest.skip("elegant is not installed")
    for turn in range(1, TURNS + 1):
        ours = _turn(runs(code), element, turn).t.val
        theirs = _turn(runs("elegant"), element, turn).t.val
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-15)


def test_xsuite_native_time_is_its_own_zeta(runs, tmp_path, seed_beam):
    """`native_times` gives back what Xsuite itself wrote: its ``.xsuite.json``
    is the last turn's, so turn 1 is a one-turn run's."""
    framework = _framework(
        tmp_path, "xsuite", {"turns": TURNS, "write_turns": True}, seed_beam
    )
    lattice = framework["FODO"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lattice.preProcess()
    for element in ("MID", "M3"):
        for turn, subdir in ((1, runs("xsuite", turns=1)), (TURNS, runs("xsuite"))):
            with open(os.path.join(subdir, f"{element}.xsuite.json")) as handle:
                particles = json.load(handle)
            alive = np.asarray(particles["state"]) > 0
            beam = _read(os.path.join(subdir, f"{element}.openpmd.hdf5"))
            np.testing.assert_allclose(
                lattice.native_times(beam, element, turn),
                np.asarray(particles["zeta"])[alive],
                rtol=0, atol=1e-12,
            )


def _code_class(code):
    if code == "xsuite":
        from simba.Codes.Xsuite.Xsuite import xsuiteLattice
        return xsuiteLattice
    if code == "ocelot":
        from simba.Codes.Ocelot.Ocelot import ocelotLattice
        return ocelotLattice
    if code == "madx":
        from simba.Codes.MADX.MADX import madxLattice
        return madxLattice
    from simba.Codes.Elegant.Elegant import elegantLattice
    return elegantLattice


class _Clocked:
    """A line on a two-pass clock, borrowing a code's native time."""

    def __init__(self, code):
        self.reference_clock = (1e-9, np.array([0.0, 2e-8, 4e-8]), np.array([0.9, 0.95]), None)
        self.native_time_scale = _code_class(code).native_time_scale

    reference_time = frameworkLattice.reference_time
    time_to_native = frameworkLattice.time_to_native
    time_from_native = frameworkLattice.time_from_native


@pytest.mark.parametrize("code", RING_CODES)
def test_native_time_round_trips(code):
    line = _Clocked(code)
    t = line.reference_time(1.5, 1) + np.array([-1e-12, 0.0, 2e-12])
    native = line.time_to_native(t, 1.5, 1)
    np.testing.assert_allclose(line.time_from_native(native, 1.5, 1), t, rtol=0, atol=1e-24)


@pytest.mark.parametrize(
    "code, sign", [("xsuite", -1), ("ocelot", +1), ("madx", -1)]
)
def test_a_late_particle_has_each_codes_own_sign(code, sign):
    """Behind the reference: zeta < 0 in Xsuite, tau > 0 in Ocelot, T < 0 in
    MAD-X; elegant's own is absolute ``t``."""
    line = _Clocked(code)
    late = line.reference_time(1.5, 1) + 1e-12
    assert np.sign(line.time_to_native(late, 1.5, 1)) == sign


def test_elegants_native_time_is_t():
    line = _Clocked("elegant")
    assert line.time_to_native(3e-9, 1.5, 1) == 3e-9


def test_xsuite_zeta_scales_with_beta0():
    """``zeta = -beta0 c (t - t_ref)``: a picosecond late at beta0 = 0.95."""
    line = _Clocked("xsuite")
    late = line.reference_time(1.5, 1) + 1e-12
    assert line.time_to_native(late, 1.5, 1) == pytest.approx(-0.95 * 299792458.0 * 1e-12)
