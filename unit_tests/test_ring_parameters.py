"""Tune, periodic Twiss and momentum compaction, derived in one place."""

import math

import numpy as np
import pytest

from simba.Framework_objects import frameworkLattice
from simba.Modules.Matrices import (
    fractional_tune,
    is_stable,
    momentum_compaction,
    periodic_twiss,
    phase_advance,
    slip_factor,
)


def rotation(mu, beta=1.0, alpha=0.0, plane="x"):
    """A one-turn map with phase advance `mu` and known Twiss, in textbook form."""
    index = {"x": 0, "y": 2}[plane]
    gamma = (1 + alpha**2) / beta
    block = np.array(
        [
            [math.cos(mu) + alpha * math.sin(mu), beta * math.sin(mu)],
            [-gamma * math.sin(mu), math.cos(mu) - alpha * math.sin(mu)],
        ]
    )
    matrix = np.eye(6)
    matrix[index : index + 2, index : index + 2] = block
    return matrix


def xsuite_drift():
    """Xsuite's R matrix of a 2 m drift at 5 MeV/c, and its gamma."""
    xt = pytest.importorskip("xtrack")
    line = xt.Line(elements=[xt.Drift(length=2.0)])
    line.particle_ref = xt.Particles(p0c=5e6, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    matrix = np.asarray(
        line.compute_R_matrix(particle_on_co=line.particle_ref.copy())["R_matrix"]
    )
    return matrix, float(line.particle_ref.gamma0[0])


def test_an_identity_map_is_stable():
    assert is_stable(np.eye(6), "x")


def test_a_drifting_map_is_marginally_stable():
    """|trace/2| == 1 exactly: the boundary belongs to the stable side."""
    matrix = np.eye(6)
    matrix[0, 1] = 2.0
    assert is_stable(matrix, "x")


def test_a_diverging_map_is_unstable():
    matrix = np.eye(6)
    matrix[0, 0], matrix[1, 1] = 3.0, 1 / 3.0
    assert not is_stable(matrix, "x")


def test_an_unstable_plane_has_no_tune_or_beta():
    matrix = np.eye(6)
    matrix[0, 0], matrix[1, 1] = 3.0, 1 / 3.0
    assert math.isnan(fractional_tune(matrix, "x"))
    assert math.isnan(periodic_twiss(matrix, "x")["beta"])


@pytest.mark.parametrize("q", [0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 0.95])
def test_the_tune_round_trips(q):
    """Above 0.5 the trace alone matches the reflection; the R12 sign separates them."""
    assert fractional_tune(rotation(2 * math.pi * q), "x") == pytest.approx(q)


def test_an_identity_map_has_zero_tune():
    assert fractional_tune(np.eye(6), "x") == pytest.approx(0.0)


def test_the_planes_are_read_separately():
    matrix = rotation(2 * math.pi * 0.3, plane="x") @ rotation(
        2 * math.pi * 0.8, plane="y"
    )
    assert fractional_tune(matrix, "x") == pytest.approx(0.3)
    assert fractional_tune(matrix, "y") == pytest.approx(0.8)


def test_the_phase_advance_stays_in_one_turn():
    for q in (0.1, 0.9):
        assert 0 <= phase_advance(rotation(2 * math.pi * q), "x") < 2 * math.pi


@pytest.mark.parametrize("beta,alpha", [(1.0, 0.0), (12.5, -2.3), (0.4, 1.7)])
def test_the_periodic_twiss_round_trips(beta, alpha):
    got = periodic_twiss(rotation(2 * math.pi * 0.37, beta, alpha), "x")
    assert got["beta"] == pytest.approx(beta)
    assert got["alpha"] == pytest.approx(alpha)


def test_gamma_follows_the_twiss_identity():
    got = periodic_twiss(rotation(2 * math.pi * 0.37, 12.5, -2.3), "x")
    assert got["gamma"] == pytest.approx((1 + got["alpha"] ** 2) / got["beta"])


def test_a_circle_has_unit_momentum_compaction():
    """A ring of pure bends has alpha_c exactly 1: the anchor for the sign and
    the 1/gamma**2 term."""
    xt = pytest.importorskip("xtrack")
    n, rho = 64, 1.5
    line = xt.Line(
        elements=[xt.Bend(length=2 * np.pi * rho / n, angle=2 * np.pi / n) for _ in range(n)],
        element_names=[f"b{i}" for i in range(n)],
    )
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    matrix = np.asarray(
        line.compute_R_matrix(particle_on_co=line.particle_ref.copy())["R_matrix"]
    )
    gamma0 = float(line.particle_ref.gamma0[0])
    assert momentum_compaction(
        matrix, line.get_length(), gamma0
    ) == pytest.approx(1.0, abs=1e-6)


def test_a_straight_line_has_no_momentum_compaction():
    matrix, gamma0 = xsuite_drift()
    assert momentum_compaction(matrix, 2.0, gamma0) == pytest.approx(0.0, abs=1e-9)


def test_the_slip_factor_is_negative_below_transition_for_a_drift():
    """alpha_c = 0, so eta = -1/gamma**2."""
    matrix, gamma0 = xsuite_drift()
    assert slip_factor(matrix, 2.0) == pytest.approx(-1 / gamma0**2, rel=1e-6)


@pytest.fixture(scope="module")
def fodo():
    xt = pytest.importorskip("xtrack")
    els, nms = [], []
    for i in range(8):
        for nm, el in (
            (f"qf{i}", xt.Quadrupole(length=0.3, k1=1.2)),
            (f"d{i}a", xt.Drift(length=0.5)),
            (f"b{i}", xt.Bend(length=1.0, angle=2 * np.pi / 8)),
            (f"d{i}b", xt.Drift(length=0.5)),
            (f"qd{i}", xt.Quadrupole(length=0.3, k1=-1.2)),
            (f"d{i}c", xt.Drift(length=0.5)),
        ):
            nms.append(nm)
            els.append(el)
    line = xt.Line(elements=els, element_names=nms)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV)
    line.build_tracker()
    tw = line.twiss(method="4d")
    matrix = np.asarray(
        line.compute_R_matrix(particle_on_co=tw.particle_on_co)["R_matrix"]
    )
    return tw, matrix


def test_the_slip_factor_matches_xsuite_on_a_ring_with_dispersion(fodo):
    """The discriminating case: drift and circle have no dispersion correction.
    The R56-only answer is 4.7% out here, however weak the bending."""
    tw, matrix = fodo
    assert slip_factor(matrix, tw.s[-1]) == pytest.approx(tw.slip_factor, rel=1e-8)


def test_the_r56_only_shortcut_would_have_been_wrong(fodo):
    """So the correction cannot quietly be undone as a simplification."""
    tw, matrix = fodo
    naive = -matrix[4, 5] / tw.s[-1]
    assert not np.isclose(naive, tw.slip_factor, rtol=1e-3)


def test_the_momentum_compaction_matches_xsuite(fodo):
    tw, matrix = fodo
    gamma0 = math.sqrt(1 + float(tw.particle_on_co.beta0[0] * tw.particle_on_co.gamma0[0]) ** 2)
    assert momentum_compaction(matrix, tw.s[-1], gamma0) == pytest.approx(
        tw.momentum_compaction_factor, rel=1e-8
    )


def test_the_fractional_tune_matches_xsuite(fodo):
    tw, matrix = fodo
    assert fractional_tune(matrix, "x") == pytest.approx(tw.qx % 1.0, abs=1e-6)
    assert fractional_tune(matrix, "y") == pytest.approx(tw.qy % 1.0, abs=1e-6)


def test_the_integer_part_is_not_recoverable(fodo):
    """Not a bug: the map cannot know how many times the phase wrapped."""
    tw, matrix = fodo
    assert tw.qx > 1.0
    assert fractional_tune(matrix, "x") < 1.0


def test_the_periodic_twiss_matches_xsuite(fodo):
    tw, matrix = fodo
    got = periodic_twiss(matrix, "x")
    assert got["beta"] == pytest.approx(tw.betx[0], rel=1e-6)
    assert got["alpha"] == pytest.approx(tw.alfx[0], rel=1e-6)


class FakeElement:
    def __init__(self, length):
        self.physical = type("P", (), {"length": length})()


class FakeRing:
    """A lattice stub exposing what `ring_parameters` reads."""

    otm_longitudinal_sign = 1
    otm_longitudinal_scale = 0

    def __init__(self, matrix, length=10.0, betagamma=1957.0, summary=None):
        self.one_turn_map = matrix
        self.optics_summary = summary
        self.closed_orbit = None
        self.elements = {"a": FakeElement(length)}
        self.global_parameters = {"beam": type("B", (), {"BetaGamma": betagamma})()}

    one_turn_map_canonical = frameworkLattice.one_turn_map_canonical
    ring_parameters = frameworkLattice.ring_parameters


def test_no_map_means_no_parameters():
    assert FakeRing(None).ring_parameters() == {}


def test_the_accessor_reports_both_planes():
    matrix = rotation(2 * math.pi * 0.3, 4.0, plane="x") @ rotation(
        2 * math.pi * 0.8, 9.0, plane="y"
    )
    got = FakeRing(matrix).ring_parameters()
    assert got["tune_x"] == pytest.approx(0.3)
    assert got["tune_y"] == pytest.approx(0.8)
    assert got["beta_x"] == pytest.approx(4.0)
    assert got["beta_y"] == pytest.approx(9.0)
    assert got["stable_x"] and got["stable_y"]


def test_the_accessor_includes_compaction_when_the_convention_allows():
    got = FakeRing(np.eye(6)).ring_parameters()
    assert "momentum_compaction" in got
    assert "slip_factor" in got


def test_the_accessor_omits_compaction_for_an_unconvertible_code():
    """elegant: tune and beta still come back, compaction does not."""
    ring = FakeRing(np.eye(6))
    ring.otm_longitudinal_scale = None
    got = ring.ring_parameters()
    assert "tune_x" in got
    assert "momentum_compaction" not in got
    assert "slip_factor" not in got


def test_chromaticity_is_not_invented_from_the_map():
    """dQ/ddelta needs two momenta, so it is not in a one-turn map."""
    assert not [k for k in FakeRing(np.eye(6)).ring_parameters() if "chrom" in k]


def test_chromaticity_appears_when_the_code_reports_it():
    got = FakeRing(np.eye(6), summary={"chromaticity_x": -1.4}).ring_parameters()
    assert got["chromaticity_x"] == -1.4


def test_the_total_tune_comes_from_the_code_not_the_map():
    got = FakeRing(np.eye(6), summary={"tune_x_total": 1.93}).ring_parameters()
    assert got["tune_x_total"] == 1.93
    assert got["tune_x"] == pytest.approx(0.0)
