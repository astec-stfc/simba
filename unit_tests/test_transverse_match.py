"""``match``: quadrupole strengths for target Twiss, through :mod:`simba.Modules.Matching`."""

import os
import shutil
import warnings

import numpy as np
import pytest

import simba.Framework as fw
from helpers import fodo_machine, fodo_quads, read_beam, skip_missing
from laura.models.element import Dipole, Sextupole
from simba.Codes.Generators import frameworkGenerator

INPUT_TWISS = {"beta_x": 10.0, "alpha_x": 1.0, "nemit_x": 1e-6,
               "beta_y": 10.0, "alpha_y": -1.0, "nemit_y": 1e-6}

ACC_ROOT_DIR = os.path.expanduser("~/Documents/bmad-ecosystem")
BMAD_SO = os.path.join(ACC_ROOT_DIR, "production", "lib", "libtao.so")
if os.path.exists(BMAD_SO):
    # as a Bmad installation sets it: how matching in Bmad finds libtao for any line
    os.environ.setdefault("ACC_ROOT_DIR", ACC_ROOT_DIR)


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp("seed"))
    frameworkGenerator(
        global_parameters={"master_subdir": directory},
        filename="M1.openpmd.hdf5", initial_momentum=100e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=5000,
    ).write()
    return os.path.join(directory, "M1.openpmd.hdf5")


def track(tmp_path, code, seed_beam, match, middle=None, definitions=None, groups=None, quiet=True,
          line=None, end_marker=None):
    """The FODO cell, its input beam rematched to INPUT_TWISS, which the match must start from.
    ``quiet`` silences warnings; ``line`` adds to the line's block; ``end_marker`` replaces M3."""
    skip_missing(code)
    if "bmad" in (code, (match or {}).get("code")) and not os.path.exists(BMAD_SO):
        pytest.skip("Bmad libtao not installed")
    machine, _, section = fodo_machine(tmp_path, middle=middle, functional_definitions=definitions,
                                       end_marker=end_marker)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {"twiss": INPUT_TWISS},
            "output": {"start_element": "M1", "end_element": "M3"},
            "match": match,
            **(line or {}),
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    settings.functional_definitions = dict(definitions or {})
    settings.groups = dict(groups or {})
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    framework["FODO"].lsc_enable = False
    framework["FODO"].csr_enable = False
    if code == "bmad":
        framework["FODO"].libtao = BMAD_SO
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        framework.track()
    return framework, read_beam(framework.subdirectory, "M3")


BETA_5 = {"variables": ["QUAD1F", "QUAD1D"], "targets": {"M3": {"beta_x": 5.0, "beta_y": 5.0}}}


MATCHERS = ["ocelot", "xsuite", "madx", "cheetah", "bmad", "elegant"]


@pytest.mark.parametrize("code", MATCHERS)
def test_each_code_matches_its_own_line(tmp_path, code, seed_beam):
    """By default the line's own code matches. The old match started from the beam
    file, not the rematched input beam: it gave beta (2.85, 4.28) here, not (5, 5)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, end = track(tmp_path, code, seed_beam, BETA_5, quiet=False)
    assert not [w for w in caught if "match" in str(w.message)]
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)
    assert end.twiss.beta_y.val == pytest.approx(5.0, rel=1e-2)


@pytest.mark.parametrize("matcher", MATCHERS)
def test_any_code_matches_a_line_tracked_in_another(tmp_path, matcher, seed_beam):
    """Each code matches for a line elegant tracks, and elegant agrees."""
    skip_missing(matcher)
    _, end = track(tmp_path, "elegant", seed_beam, {**BETA_5, "code": matcher})
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)
    assert end.twiss.beta_y.val == pytest.approx(5.0, rel=1e-2)


def test_a_line_whose_code_cannot_match_falls_back(tmp_path, seed_beam):
    """ImpactX has no backend, so the first of FALLBACK that can matches, without a word."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, end = track(tmp_path, "impactx", seed_beam, BETA_5, quiet=False)
    assert not [w for w in caught if "match" in str(w.message)]
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)


DISPERSION = {"variables": ["QUAD1F", "QUAD1D"],
              "targets": {"M3": {"beta_x": 5.0, "eta_x": ["a<", 1.0]}}}


def test_a_code_that_cannot_target_it_hands_over(tmp_path, seed_beam):
    """Cheetah's Twiss has no dispersion: its line is matched in the first of FALLBACK, with a warning."""
    from simba.Modules.Matching.transverse import FALLBACK

    with pytest.warns(UserWarning, match=f"cheetah cannot match this.*eta_x.*matching in {FALLBACK[0]}"):
        _, end = track(tmp_path, "cheetah", seed_beam, DISPERSION, quiet=False)
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)


def test_naming_a_code_that_cannot_target_it_is_refused(tmp_path, seed_beam):
    with pytest.raises(ValueError, match="cannot run in cheetah.*eta_x"):
        track(tmp_path, "ocelot", seed_beam, {**DISPERSION, "code": "cheetah"})


def test_ocelots_names_still_work(tmp_path, seed_beam):
    _, end = track(tmp_path, "ocelot", seed_beam,
                   {**BETA_5, "targets": {"M3": {"beta_x": 5.0, "Dx": ["a<", 1.0]}}})
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)


@pytest.mark.parametrize("matcher", ["ocelot", "xsuite", "madx"])
def test_the_phase_advance_is_in_units_of_two_pi(tmp_path, matcher, seed_beam):
    """Ocelot works in radians, Xsuite and MAD-X in 2 pi: each must reach the same
    target, as Xsuite's own twiss of the matched line reads it."""
    from simba.Modules.Matching.backends.xsuite import XsuiteBackend
    from simba.Modules.Matching.transverse import parse

    skip_missing(matcher)
    skip_missing("xsuite")
    block = {"variables": ["QUAD1F", "QUAD1D"], "targets": {"M3": {"beta_x": 5.0, "mux": 0.1}}}
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**block, "code": matcher})
    line = framework["FODO"]
    values = [line.elements[q].magnetic.k1l for q in ("QUAD1F", "QUAD1D")]
    spec = parse(line, {**block, "variables": ["QUAD1F", "QUAD1D"]})
    end = dict(XsuiteBackend(line, spec).twiss(values))["M3"]
    assert end["mux"] == pytest.approx(0.1, abs=1e-4)
    assert end["beta_x"] == pytest.approx(5.0, rel=1e-3)


def test_an_inequality_is_met(tmp_path, seed_beam):
    match = {"variables": ["QUAD1F", "QUAD1D"],
             "targets": {"M3": {"beta_x": 5.0, "beta_y": ["<", 4.0]}}}
    _, end = track(tmp_path, "ocelot", seed_beam, match)
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)
    assert end.twiss.beta_y.val < 4.0 * 1.01


def test_targets_out_of_reach_warn(tmp_path, seed_beam):
    """With beta_x at 5, these two quadrupoles cannot take beta_y below about 3.2:
    Ocelot gives its best try without a word."""
    match = {"variables": ["QUAD1F", "QUAD1D"],
             "targets": {"M3": {"beta_x": 5.0, "beta_y": ["<", 3.0]}}}
    with pytest.warns(UserWarning, match="M3.beta_y"):
        track(tmp_path, "ocelot", seed_beam, match, quiet=False)


def test_disabled_changes_nothing(tmp_path, seed_beam):
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**BETA_5, "enable": False})
    assert framework["FODO"].elements["QUAD1F"].magnetic.k1l == pytest.approx(-1.0)


def test_a_group_names_the_variables(tmp_path, seed_beam):
    groups = {"fodo_quads": {"type": "element_group", "elements": ["QUAD1F", "QUAD1D"]}}
    _, end = track(tmp_path, "ocelot", seed_beam, {**BETA_5, "variables": "fodo_quads"}, groups=groups)
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)


def test_definitions_are_matched_not_overwritten(tmp_path, seed_beam):
    quads = fodo_quads()
    quads[0].magnetic.k1l, quads[1].magnetic.k1l = "kf", "kd"
    framework, end = track(tmp_path, "ocelot", seed_beam, BETA_5, middle=quads,
                           definitions={"kf": -1.0, "kd": 1.0})
    elements = framework["FODO"].elements
    assert (elements["QUAD1F"].magnetic.k1l, elements["QUAD1D"].magnetic.k1l) == ("kf", "kd")
    assert framework.settings["functional_definitions"]["kf"] != pytest.approx(-1.0)
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)


def test_anything_but_a_quadrupole_names_its_setting(tmp_path, seed_beam):
    """A bare name means a quadrupole's k1l. The old match took a sextupole's name,
    which Ocelot never varied, and wrote back wherever the minimiser left it."""
    middle = fodo_quads() + [Sextupole(
        name="SEXT1", machine_area="FODO", magnetic={"length": 0.2, "k2l": 1.0},
        physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
    )]
    with pytest.raises(ValueError, match="SEXT1.angle"):
        track(tmp_path, "ocelot", seed_beam, {**BETA_5, "variables": ["QUAD1F", "SEXT1"]}, middle=middle)
    with pytest.raises(ValueError, match="no number nope"):
        track(tmp_path / "2", "ocelot", seed_beam, {**BETA_5, "variables": ["QUAD1F", "SEXT1.nope"]}, middle=middle)


IMAGING = {"variables": ["QUAD1F", "QUAD1D"], "targets": {"M3": {"R12": 0.0, "R34": 0.0}}}


@pytest.mark.parametrize(
    "matcher,checker",
    [("ocelot", "madx"), ("xsuite", "ocelot"), ("madx", "xsuite"), ("elegant", "xsuite"), ("xsuite", "elegant"),
     ("bmad", "xsuite"), ("xsuite", "bmad")],
)
def test_transfer_matrix_targets(tmp_path, matcher, checker, seed_beam):
    """Point-to-point imaging, solved in one code and read back in another: Ocelot's
    match has no matrix targets, so it takes the shared solver."""
    from simba.Modules.Matching.backends import backends
    from simba.Modules.Matching.transverse import parse

    skip_missing(matcher)
    skip_missing(checker)
    framework, _ = track(tmp_path, "elegant", seed_beam, {**IMAGING, "code": matcher})
    line = framework["FODO"]
    values = [line.elements[q].magnetic.k1l for q in ("QUAD1F", "QUAD1D")]
    spec = parse(line, IMAGING)
    if backends()[checker].cannot(line, spec):
        pytest.skip(f"{checker}: {backends()[checker].cannot(line, spec)}")
    end = dict(backends()[checker](line, spec).twiss(values))["M3"]
    assert end["R12"] == pytest.approx(0.0, abs=1e-6)
    assert end["R34"] == pytest.approx(0.0, abs=1e-6)


def bend(angle):
    """A 1 m sector bend entering on the axis at 1.5 m (its middle is on the arc, off
    the axis); fodo_machine ends the line about 0.5 m after it, see :func:`marker_after`."""
    dipole = Dipole(
        name="BEND1", machine_area="FODO", magnetic={"length": 1.0, "angle": angle},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )
    shift = np.array([0.0, 0.0, 1.5]) - np.asarray(dipole.physical.start.array, dtype=float)
    x, y, z = np.asarray(dipole.physical.middle.array, dtype=float) + shift
    dipole.physical.middle = {"x": float(x), "y": float(y), "z": float(z)}
    return dipole


def marker_after(dipole, distance=0.5):
    """M3, ``distance`` on from ``dipole`` along its exit direction: a layout Bmad can
    follow, which places elements where they are, with patches between."""
    from laura.models.element import Marker

    end = np.asarray(dipole.physical.end.array, dtype=float)
    direction = np.asarray(dipole.physical.end_rotation_matrix) @ [0.0, 0.0, 1.0]
    x, _, z = end + distance * direction
    return Marker(
        name="M3", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": float(x), "y": 0.0, "z": float(z)},
                  "global_rotation": {"theta": float(np.arctan2(-direction[0], direction[2]))}},
    )


def test_every_code_gives_the_same_canonical_matrix(tmp_path, seed_beam):
    """Ocelot's (tau, p), MAD-X's (t, pt) and elegant's (s, delta) read into Xsuite's
    (zeta, delta); elegant's path length is not a rescale, so it gets R5j and R56 its own way."""
    from simba.Modules.Matching.backends import MATRIX, backends
    from simba.Modules.Matching.transverse import parse

    block = {"variables": ["QUAD1F"], "targets": {"M3": {"R16": 0.0, "R56": 0.0, "R51": 0.0}}}
    # FINT said outright: left out, each code takes its own default (0.5 in elegant, 0 in MAD-X)
    dipole = bend(0.2)
    dipole.magnetic.edge_field_integral = 0.5
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**BETA_5, "enable": False},
                         middle=fodo_quads()[:1] + [dipole], end_marker=marker_after(dipole))
    line = framework["FODO"]
    spec = parse(line, block)
    skip_missing("xsuite")
    matrices = {}
    for code in ("ocelot", "xsuite", "madx", "elegant", "bmad"):
        if backends()[code].cannot(line, spec):
            continue
        end = dict(backends()[code](line, spec).twiss([-1.0]))["M3"]
        matrices[code] = [end[q] for q in sorted(MATRIX)]
    assert abs(matrices["xsuite"][sorted(MATRIX).index("R16")]) > 1e-2, "the bend must disperse"
    assert len(matrices) > 1
    for code, matrix in matrices.items():
        assert matrix == pytest.approx(matrices["xsuite"], abs=1e-5), code


def test_rebuilding_leaves_the_elements_as_they_were(tmp_path, seed_beam):
    from simba.Modules.Matching.backends import backends
    from simba.Modules.Matching.transverse import parse

    block = {"variables": ["BEND1.angle"], "targets": {"M3": {"R16": 0.1}}}
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**block, "enable": False}, middle=[bend(0.05)])
    line = framework["FODO"]
    backend = backends()["ocelot"](line, parse(line, block))
    assert dict(backend.twiss([0.3]))["M3"]["R16"] != pytest.approx(dict(backend.twiss([0.05]))["M3"]["R16"])
    assert line.elements["BEND1"].magnetic.angle == pytest.approx(0.05)


BEND_ANGLE = {"variables": ["BEND1.angle"], "targets": {"M3": {"R16": [">", 0.08]}}}


@pytest.mark.parametrize("matcher", ["ocelot", "xsuite", "madx", "elegant"])
def test_a_dipole_angle_is_matched_by_rebuilding(tmp_path, matcher, seed_beam):
    """Any field LAURA exports: here a sector bend's angle, for R16 = L (1 - cos a) / a
    plus the drift to M3 at the slope sin a."""
    from scipy.optimize import brentq

    skip_missing(matcher)
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**BEND_ANGLE, "code": matcher}, middle=[bend(0.05)])
    elements = framework["FODO"].elements
    drift = elements["M3"].physical.middle.z - elements["BEND1"].physical.end.z
    expected = brentq(lambda a: (1 - np.cos(a)) / a + drift * np.sin(a) - 0.08, 0.01, 1.0)
    assert elements["BEND1"].magnetic.angle == pytest.approx(expected, rel=1e-3)


def test_bmad_will_not_vary_a_lone_dipole_angle(tmp_path, seed_beam):
    """Its lattice keeps elements where they are placed: the patch after the bend would
    take the angle's change back out."""
    with pytest.raises(ValueError, match="cannot run in bmad.*BEND1.*chicane"):
        track(tmp_path, "ocelot", seed_beam, {**BEND_ANGLE, "code": "bmad"}, middle=[bend(0.05)])


@pytest.mark.parametrize(
    "match,error,text",
    [
        ({**BETA_5, "variables": ["QUAD1F", "NOPE"]}, KeyError, "NOPE"),
        ({**BETA_5, "targets": {"NOPE": {"beta_x": 5.0}}}, KeyError, "NOPE"),
        ({**BETA_5, "targets": {"M3": {"beta": 5.0}}}, KeyError, "beta"),
        # the form the old docstring gave, which Ocelot never read
        ({**BETA_5, "targets": {"M3": {"beta_x": {"mode": "greaterthan", "value": 8.0}}}}, ValueError, "operator"),
        ({**BETA_5, "targets": {"M3": {"beta_x": [">=", 8.0]}}}, ValueError, "operator"),
    ],
    ids=["unknown-variable", "unknown-target", "unknown-key", "old-inequality", "bad-operator"],
)
def test_bad_blocks_are_refused(tmp_path, seed_beam, match, error, text):
    with pytest.raises(error, match=text):
        track(tmp_path, "ocelot", seed_beam, match)


def chicane_dipoles():
    """Four 0.2 m dipoles, straight to start with: the group's dipoleangle bends them."""
    return [
        Dipole(name=f"D{i}", machine_area="FODO", magnetic={"length": 0.2, "angle": 0.0},
               physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": z}})
        for i, z in enumerate((0.2, 1.4, 3.4, 4.6), start=1)
    ]


CHICANE = {"bc": {"type": "chicane", "elements": ["D1", "D2", "D3", "D4"]}}
R56 = {"variables": ["bc.angle"], "targets": {"M3": {"R56": ["a>", 0.0]}}}


def chicane_r56(tmp_path, seed_beam, angle):
    """The chicane's canonical R56 at ``angle``, read in Xsuite."""
    from simba.Modules.Matching.backends import backends
    from simba.Modules.Matching.transverse import parse

    framework, _ = track(tmp_path, "ocelot", seed_beam, {**R56, "enable": False}, middle=chicane_dipoles(),
                         groups=CHICANE, line={"groups": {"bc": {"dipoleangle": 0.05}}})
    line = framework["FODO"]
    block = {"variables": ["bc.angle"], "targets": {"M3": {"R56": 0.0}}}
    return dict(backends()["xsuite"](line, parse(line, block)).twiss([angle]))["M3"]["R56"]


@pytest.mark.parametrize("matcher", ["ocelot", "xsuite", "madx", "elegant", "bmad"])
def test_a_chicane_angle_is_matched_through_its_group(tmp_path, matcher, seed_beam):
    """The group lays the chicane out at each angle tried; the angle that gives the
    target R56 is the one Xsuite reads it at, and the settings keep it."""
    skip_missing(matcher)
    skip_missing("xsuite")
    target = chicane_r56(tmp_path / "target", seed_beam, 0.1)
    block = {"variables": ["bc.angle"], "targets": {"M3": {"R56": target}}, "code": matcher}
    framework, _ = track(tmp_path / "match", "ocelot", seed_beam, block, middle=chicane_dipoles(),
                         groups=CHICANE, line={"groups": {"bc": {"dipoleangle": 0.05}}})
    group = framework.groupObjects["bc"]
    assert group.angle == pytest.approx(0.1, rel=1e-4)
    assert framework["FODO"].groupSettings["bc"]["dipoleangle"] == pytest.approx(0.1, rel=1e-4)
    d4 = framework["FODO"].elements["D4"].physical
    assert d4.end.x == pytest.approx(0.0, abs=1e-9), "the chicane is laid out closed"


def test_numeric_edges_of_a_matched_dipole_warn(tmp_path, seed_beam):
    edged = bend(0.05)
    edged.magnetic.exit_edge_angle = 0.05
    block = {"variables": ["BEND1.angle"], "targets": {"M3": {"R16": [">", 0.08]}}}
    with pytest.warns(UserWarning, match="BEND1.*edge"):
        track(tmp_path, "ocelot", seed_beam, block, middle=[edged], quiet=False)


def test_edges_written_with_the_angle_follow_it(tmp_path, seed_beam):
    """A bend with edges 'angle/2', at an angle tried, is the bend with those edges as numbers."""
    from simba.Modules.Matching.backends import backends
    from simba.Modules.Matching.transverse import parse

    block = {"variables": ["BEND1.angle"], "targets": {"M3": {"R43": 0.0}}}

    def r43(edge, angle):
        b = bend(0.05)
        b.magnetic.entrance_edge_angle = b.magnetic.exit_edge_angle = edge
        framework, _ = track(tmp_path / str(edge), "ocelot", seed_beam, {**block, "enable": False}, middle=[b])
        line = framework["FODO"]
        return dict(backends()["ocelot"](line, parse(line, block)).twiss([angle]))["M3"]["R43"]

    chained = r43("angle/2", 0.3)
    assert chained == pytest.approx(r43(0.15, 0.3), rel=1e-9)
    assert chained != pytest.approx(r43(0.0, 0.3), rel=1e-3), "the edges must focus"
