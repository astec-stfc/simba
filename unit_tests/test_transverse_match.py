"""``match``: quadrupole strengths for target Twiss, through :mod:`simba.Modules.Matching`."""

import os
import shutil
import warnings

import pytest

import simba.Framework as fw
from helpers import fodo_machine, fodo_quads, read_beam, skip_missing
from laura.models.element import Sextupole
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


@pytest.mark.parametrize("code", ["ocelot", "elegant", "xsuite"])
def test_the_matched_line_tracks_to_the_targets(tmp_path, code, seed_beam):
    """The old match started from the beam file, not the rematched input beam: it
    gave beta (2.85, 4.28) here, not (5, 5)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, end = track(tmp_path, code, seed_beam, BETA_5, quiet=False)
    assert not [w for w in caught if "match missed" in str(w.message)]
    assert end.twiss.beta_x.val == pytest.approx(5.0, rel=1e-2)
    assert end.twiss.beta_y.val == pytest.approx(5.0, rel=1e-2)


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


def test_a_sextupole_is_refused(tmp_path, seed_beam):
    """Ocelot never varies one: the old match wrote back wherever the minimiser left it."""
    middle = fodo_quads() + [Sextupole(
        name="SEXT1", machine_area="FODO", magnetic={"length": 0.2, "k2l": 1.0},
        physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": 4.0}},
    )]
    with pytest.raises(ValueError, match="SEXT1"):
        track(tmp_path, "ocelot", seed_beam, {**BETA_5, "variables": ["QUAD1F", "SEXT1"]}, middle=middle)


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
