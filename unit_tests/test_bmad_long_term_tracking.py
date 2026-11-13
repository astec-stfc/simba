"""Bmad's turns through ``long_term_tracking``: what the ring, ramp and RF tests leave out."""

import os
import shutil
import warnings

import numpy as np
import pytest

import simba.Framework as fw
from helpers import BMAD_SO, fodo_machine, fodo_quads, read_beam, skip_missing, use_local_bmad
from laura.models.element import HorizontalCorrector, Sextupole, VerticalCorrector
from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Generators import frameworkGenerator

TURNS = 4


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


def _tracked(tmp_path, code, tracking, seed_beam, k2l=0.0):
    """The FODO ring, with two correctors and a sextupole of `k2l` between its quadrupoles, tracked by ``code``."""
    skip_missing(code)
    correctors = [
        cls(name=name, machine_area="FODO", physical={"length": 0.0, "middle": {"x": 0.0, "y": 0.0, "z": z}})
        for cls, name, z in ((HorizontalCorrector, "HCOR", 4.0), (VerticalCorrector, "VCOR", 4.1))
    ]
    quad1f, quad1d = fodo_quads()
    sextupole = Sextupole(
        name="SEXT", machine_area="FODO", magnetic={"length": 0.2, "k2l": k2l},
        physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )
    machine, _, section = fodo_machine(tmp_path, closed=True, middle=[quad1f, sextupole, quad1d, *correctors])
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
    if code == "bmad":
        use_local_bmad(framework)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return framework


def _means(tmp_path, code, tracking, seed_beam):
    """Mean ``x``, ``y``, ``x'`` and ``y'`` at the ring's end, turn by turn."""
    framework = _tracked(tmp_path, code, tracking, seed_beam)
    beams = [read_beam(framework.subdirectory, "M3", turn) for turn in range(1, tracking["turns"] + 1)]
    return np.array([
        [np.mean(b.x.val), np.mean(b.y.val), np.mean(b.cpx.val / b.cp.val), np.mean(b.cpy.val / b.cp.val)]
        for b in beams
    ])


def test_a_program_kicks_on_its_turn_as_xsuites_does(tmp_path, seed_beam):
    """The turn-2 pulse is a ``ramper`` in ``long_term_tracking``, the turn-4 one Tao's ``set element``.

    Against Xsuite: elegant programs only a ``bumper``, and skips a corrector with a warning.
    """
    programs = [
        {"element": "HCOR", "turns": [1, 2, 3], "values": [0.0, 1e-4, 0.0]},
        {"element": "VCOR", "turns": [3, 4], "values": [0.0, -2e-4]},
    ]
    tracking = {"turns": TURNS, "write_turns": True, "programs": programs}
    bmad, xsuite = (_means(tmp_path / code, code, tracking, seed_beam) for code in ("bmad", "xsuite"))
    unkicked = _means(tmp_path / "none", "bmad", {"turns": TURNS, "write_turns": True}, seed_beam)
    kick = bmad - unkicked
    # HCOR is 0.1 m before M3; VCOR is at M3, so shows in y' alone
    np.testing.assert_allclose(kick[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(kick[1, [0, 2]], [1e-5, 1e-4], rtol=1e-2)
    np.testing.assert_allclose(kick[:3, [1, 3]], 0.0, atol=1e-10)
    assert kick[3, 3] == pytest.approx(-2e-4, rel=1e-2)
    # a few parts per million of the kicks: the codes, not the programs
    np.testing.assert_allclose(bmad, xsuite, rtol=1e-5, atol=1e-9)


def test_the_frequency_map_is_xsuites_under_a_program(tmp_path, seed_beam):
    """The grid's turns are ``long_term_tracking``'s, with the beam's rampers, not Tao's last-turn settings."""
    program = {"element": "QUAD1F", "parameter": "k1", "turns": [1, 64], "values": [-1.1, -1.1]}
    grid = {"nx": 2, "ny": 2, "x_max": 2e-4, "y_max": 2e-4}

    def footprint(code, **extra):
        tracking = {"turns": 64, "dynamic_aperture": grid, **extra}
        framework = _tracked(tmp_path / f"{code}{len(extra)}", code, tracking, seed_beam)
        return np.array(framework["FODO"].run_frequency_map())[:, :4]

    bmad, xsuite = (footprint(code, programs=[program]) for code in ("bmad", "xsuite"))
    unprogrammed = footprint("bmad")
    assert len(bmad) == 4
    assert np.min(np.abs(bmad[:, 2:] - unprogrammed[:, 2:])) > 1e-3
    np.testing.assert_allclose(bmad, xsuite, rtol=0, atol=1e-5)


def test_the_dynamic_aperture_is_xsuites(tmp_path, seed_beam):
    """The same losses on the same turns, from the same rays, behind a sextupole."""
    tracking = {"turns": 64, "dynamic_aperture": {"nx": 6, "n_lines": 3, "x_max": 0.02, "y_max": 0.02}}
    bmad, xsuite = (
        _tracked(tmp_path / code, code, tracking, seed_beam, k2l=20.0)["FODO"].run_dynamic_aperture()
        for code in ("bmad", "xsuite")
    )
    assert 0 < sum(turns == 64 for *_, turns in bmad) < len(bmad), "the rays must straddle the aperture"
    assert bmad == xsuite


def test_a_long_ramper_is_read_whole(tmp_path):
    """Two thousand knots, across many lines: Tao reads off the last of them."""
    if not os.path.exists(BMAD_SO):
        pytest.skip("Bmad is not built")
    from pytao import Tao

    times = np.arange(2000) * 1e-8
    values = np.linspace(1.0, 2.0, 2000)
    lattice = tmp_path / "ring.bmad"
    lattice.write_text(
        "parameter[particle] = electron\nparameter[geometry] = open\nbeginning[p0c] = 5e6\n"
        "beginning[beta_a] = 1\nbeginning[beta_b] = 1\n"
        "d: drift, l = 1\nq: quadrupole, l = 0.5, k1 = 1\nring: line = (d, q)\nuse, ring\n"
        + bmadLattice._ramper("r1", "q[k1]", times, values)
    )
    tao = Tao(lattice_file=str(lattice), noplot=True, so_lib=BMAD_SO)
    tao.cmd(f"set element r1 time = {float(times[-2])!r}")
    assert tao.ele_gen_attribs("q")["K1"] == pytest.approx(values[-2], rel=1e-12)
