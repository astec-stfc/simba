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


def track(tmp_path, code, seed_beam, match, middle=None, definitions=None, groups=None, quiet=True):
    """The FODO cell, its input beam rematched to INPUT_TWISS, which the match must start from.
    ``quiet`` silences warnings."""
    skip_missing(code)
    machine, _, section = fodo_machine(tmp_path, middle=middle, functional_definitions=definitions)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {"twiss": INPUT_TWISS},
            "output": {"start_element": "M1", "end_element": "M3"},
            "match": match,
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
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        framework.track()
    return framework, read_beam(framework.subdirectory, "M3")


BETA_5 = {"variables": ["QUAD1F", "QUAD1D"], "targets": {"M3": {"beta_x": 5.0, "beta_y": 5.0}}}


MATCHERS = ["ocelot", "xsuite", "madx", "cheetah"]


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
    """elegant has no matcher here; each code matches for it, and elegant agrees."""
    skip_missing(matcher)
    _, end = track(tmp_path, "elegant", seed_beam, {**BETA_5, "code": matcher})
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)
    assert end.twiss.beta_y.val == pytest.approx(5.0, rel=1e-2)


def test_a_line_whose_code_cannot_match_falls_back(tmp_path, seed_beam):
    """elegant has no backend, so the first that can (Ocelot) matches, without a word."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, end = track(tmp_path, "elegant", seed_beam, BETA_5, quiet=False)
    assert not [w for w in caught if "match" in str(w.message)]
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)


DISPERSION = {"variables": ["QUAD1F", "QUAD1D"],
              "targets": {"M3": {"beta_x": 5.0, "eta_x": ["a<", 1.0]}}}


def test_a_code_that_cannot_target_it_hands_over(tmp_path, seed_beam):
    """Cheetah's Twiss has no dispersion: its line is matched elsewhere, with a warning."""
    with pytest.warns(UserWarning, match="cheetah cannot match this.*eta_x.*matching in ocelot"):
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


@pytest.mark.parametrize("matcher,checker", [("ocelot", "madx"), ("xsuite", "ocelot"), ("madx", "xsuite")])
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
    end = dict(backends()[checker](line, parse(line, IMAGING)).twiss(values))["M3"]
    assert end["R12"] == pytest.approx(0.0, abs=1e-6)
    assert end["R34"] == pytest.approx(0.0, abs=1e-6)


def bend(angle):
    """A 1 m sector bend from 1.5 m to 2.5 m: fodo_machine ends the line 0.5 m after it."""
    return Dipole(
        name="BEND1", machine_area="FODO", magnetic={"length": 1.0, "angle": angle},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )


def test_every_code_gives_the_same_canonical_matrix(tmp_path, seed_beam):
    """Ocelot's (tau, p) and MAD-X's (t, pt) read into Xsuite's (zeta, delta)."""
    from simba.Modules.Matching.backends import MATRIX, backends
    from simba.Modules.Matching.transverse import parse

    block = {"variables": ["QUAD1F"], "targets": {"M3": {"R16": 0.0, "R56": 0.0, "R51": 0.0}}}
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**BETA_5, "enable": False},
                         middle=fodo_quads()[:1] + [bend(0.2)])
    line = framework["FODO"]
    matrices = {}
    for code in ("ocelot", "xsuite", "madx"):
        skip_missing(code)
        end = dict(backends()[code](line, parse(line, block)).twiss([-1.0]))["M3"]
        matrices[code] = [end[q] for q in sorted(MATRIX)]
    assert abs(matrices["xsuite"][sorted(MATRIX).index("R16")]) > 1e-2, "the bend must disperse"
    for code in ("ocelot", "madx"):
        assert matrices[code] == pytest.approx(matrices["xsuite"], abs=1e-5), code


def test_rebuilding_leaves_the_elements_as_they_were(tmp_path, seed_beam):
    from simba.Modules.Matching.backends import backends
    from simba.Modules.Matching.transverse import parse

    block = {"variables": ["BEND1.angle"], "targets": {"M3": {"R16": 0.1}}}
    framework, _ = track(tmp_path, "ocelot", seed_beam, {**block, "enable": False}, middle=[bend(0.05)])
    line = framework["FODO"]
    backend = backends()["ocelot"](line, parse(line, block))
    assert dict(backend.twiss([0.3]))["M3"]["R16"] != pytest.approx(dict(backend.twiss([0.05]))["M3"]["R16"])
    assert line.elements["BEND1"].magnetic.angle == pytest.approx(0.05)


@pytest.mark.parametrize("matcher", ["ocelot", "xsuite", "madx"])
def test_a_dipole_angle_is_matched_by_rebuilding(tmp_path, matcher, seed_beam):
    """Any field LAURA exports: here a sector bend's angle, for R16 = L (1 - cos a) / a
    plus 0.5 m of drift at the slope sin a."""
    from scipy.optimize import brentq

    skip_missing(matcher)
    framework, _ = track(tmp_path, "ocelot", seed_beam,
                         {"variables": ["BEND1.angle"], "targets": {"M3": {"R16": [">", 0.08]}},
                          "code": matcher},
                         middle=[bend(0.05)])
    angle = framework["FODO"].elements["BEND1"].magnetic.angle
    expected = brentq(lambda a: (1 - np.cos(a)) / a + 0.5 * np.sin(a) - 0.08, 0.01, 1.0)
    assert angle == pytest.approx(expected, rel=1e-3)


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
