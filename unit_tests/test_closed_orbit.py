"""The orbit that closes on itself."""

import math
import tempfile
from functools import cache

import numpy as np
import pytest

from simba.Framework_objects import frameworkLattice

NCELL = 8
QUAD_L, QUAD_K1, BEND_L, DRIFT_L = 0.3, 1.2, 1.0, 0.5
ANGLE = 2 * math.pi / NCELL
KICK = 1e-4
CELL_L = 2 * QUAD_L + BEND_L + 3 * DRIFT_L
CIRCUMFERENCE = NCELL * CELL_L


@cache
def xsuite_orbit(kick):
    xt = pytest.importorskip("xtrack")
    els, nms = [], []
    for i in range(NCELL):
        for nm, el in (
            (f"qf{i}", xt.Quadrupole(length=QUAD_L, k1=QUAD_K1)),
            (f"d{i}a", xt.Drift(length=DRIFT_L)),
            (f"b{i}", xt.Bend(length=BEND_L, angle=ANGLE)),
            (f"d{i}b", xt.Drift(length=DRIFT_L)),
            (f"qd{i}", xt.Quadrupole(length=QUAD_L, k1=-QUAD_K1)),
            (f"d{i}c", xt.Drift(length=DRIFT_L)),
        ):
            nms.append(nm)
            els.append(el)
    if kick:
        nms.append("kick")
        els.append(xt.Multipole(knl=[-kick], length=0))
    line = xt.Line(elements=els, element_names=nms)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    co = line.find_closed_orbit()
    return np.array(
        [float(np.atleast_1d(getattr(co, k))[0]) for k in ("x", "px", "y", "py")]
    )


@cache
def madx_orbit(kick):
    madx_module = pytest.importorskip("cpymad.madx")
    body = []
    for i in range(NCELL):
        s = i * CELL_L
        body.append(
            f"qf{i}: quadrupole, at={s + QUAD_L / 2}, l={QUAD_L}, k1={QUAD_K1};"
        )
        body.append(
            f"b{i}: sbend, at={s + QUAD_L + DRIFT_L + BEND_L / 2}, "
            f"l={BEND_L}, angle={ANGLE};"
        )
        body.append(
            f"qd{i}: quadrupole, "
            f"at={s + QUAD_L + 2 * DRIFT_L + BEND_L + QUAD_L / 2}, "
            f"l={QUAD_L}, k1={-QUAD_K1};"
        )
    if kick:
        body.append(f"kk: hkicker, at={CIRCUMFERENCE - 1e-6}, l=0, kick={kick};")
    madx = madx_module.Madx(stdout=False, cwd=tempfile.mkdtemp())
    madx.input(
        f"beam, particle=electron, pc=1.0;\n"
        f"ring: sequence, l={CIRCUMFERENCE};\n" + "\n".join(body) + "\nendsequence;\n"
        "use, sequence=ring;\ntwiss;"
    )
    tw = madx.table.twiss
    return np.array([float(tw[k][0]) for k in ("x", "px", "y", "py")])


def test_a_perfect_ring_has_a_zero_closed_orbit():
    """Xsuite searches for the orbit, so stops at ~1e-9; MAD-X gives exact zeros."""
    assert np.allclose(madx_orbit(0.0), 0.0, atol=1e-15)
    assert np.allclose(xsuite_orbit(0.0), 0.0, atol=1e-8)


def test_a_kick_produces_a_closed_orbit():
    assert abs(xsuite_orbit(KICK)[0]) > 1e-5
    assert abs(madx_orbit(KICK)[0]) > 1e-5


def test_the_orbit_stays_in_the_kicked_plane():
    for orbit in (xsuite_orbit(KICK), madx_orbit(KICK)):
        assert orbit[2] == pytest.approx(0.0, abs=1e-12)
        assert orbit[3] == pytest.approx(0.0, abs=1e-12)


def test_the_orbit_scales_almost_linearly_with_the_kick():
    """To 0.03%: the displaced orbit samples the sector bends off-axis."""
    one = xsuite_orbit(KICK)[0]
    two = xsuite_orbit(2 * KICK)[0]
    assert two == pytest.approx(2 * one, rel=1e-3)
    assert two / one != pytest.approx(2.0, rel=1e-6)


def test_the_two_codes_agree_on_the_size_of_the_orbit():
    """To about 5%; `hkicker kick` and `Multipole knl[0]` deflect opposite ways."""
    xsuite = abs(xsuite_orbit(KICK)[0])
    madx = abs(madx_orbit(KICK)[0])
    assert madx == pytest.approx(xsuite, rel=0.06)


class FakeRing:
    """A lattice stub exposing what `ring_parameters` reads."""

    otm_longitudinal_sign = 1
    otm_longitudinal_scale = 0

    def __init__(self, orbit):
        self.one_turn_map = np.eye(6)
        self.optics_summary = None
        self.closed_orbit = orbit
        self.elements = {}
        self.global_parameters = {"beam": type("B", (), {"BetaGamma": 1957.0})()}

    one_turn_map_canonical = frameworkLattice.one_turn_map_canonical
    ring_parameters = frameworkLattice.ring_parameters


def test_the_base_class_has_no_closed_orbit():
    assert frameworkLattice.read_closed_orbit(frameworkLattice) is None


def test_no_orbit_means_no_orbit_keys():
    assert not [k for k in FakeRing(None).ring_parameters() if "closed_orbit" in k]


def test_the_orbit_is_reported_componentwise():
    got = FakeRing([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).ring_parameters()
    assert got["closed_orbit_x"] == 1.0
    assert got["closed_orbit_px"] == 2.0
    assert got["closed_orbit_delta"] == 6.0


def test_a_short_orbit_vector_is_not_padded():
    """Ocelot gives four components; naming two more would invent numbers."""
    got = FakeRing([1.0, 2.0, 3.0, 4.0]).ring_parameters()
    assert got["closed_orbit_py"] == 4.0
    assert "closed_orbit_zeta" not in got
