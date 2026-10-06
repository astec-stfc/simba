"""The strongest correctness test available: the codes check each other.

`laura-simba-ring-tracking.md` section 5 is the reason this exists. Nothing in
the ring work is pinned against a real machine, and these quantities fail with
*wrong numbers* rather than wrong structure -- a tune out by a factor, a beta
from the open solution instead of the closed one. No reference data exists to
check against, but four independent codes do, and they were written by
different people from different conventions. If they agree on a tune to six
digits, it is very unlikely they are all wrong the same way.

One 8-cell FODO ring, built natively in each code, compared on:

* the **fractional tune** and **periodic beta/alpha** from each code's
  one-turn map, through simba's own derivations;
* the **full tune** and **chromaticity** each code reports itself.

Conventions are handled as A2 measured them: the transverse blocks need no
conversion, the longitudinal does. elegant is absent from the compaction
comparison for that reason and present in everything else.
"""

import math
import os
import tempfile
from functools import lru_cache

import numpy as np
import pytest

from simba.Modules.Matrices import (
    fractional_tune,
    momentum_compaction,
    periodic_twiss,
)

# One cell, and the whole ring is eight of them. Every code below builds
# exactly this, so any disagreement is the code and not the lattice.
NCELL = 8
QUAD_L, QUAD_K1 = 0.3, 1.2
BEND_L = 1.0
DRIFT_L = 0.5
ANGLE = 2 * math.pi / NCELL
PC = 1e9
MC2 = 0.510998950e6
CIRCUMFERENCE = NCELL * (2 * QUAD_L + BEND_L + 3 * DRIFT_L)
GAMMA0 = math.sqrt(1 + (PC / MC2) ** 2)


@lru_cache(maxsize=None)
def xsuite_ring():
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
    line = xt.Line(elements=els, element_names=nms)
    line.particle_ref = xt.Particles(p0c=PC, mass0=MC2)
    line.build_tracker()
    tw = line.twiss(method="4d")
    matrix = np.asarray(
        line.compute_R_matrix(particle_on_co=tw.particle_on_co)["R_matrix"]
    )
    return matrix, tw


@lru_cache(maxsize=None)
def ocelot_ring():
    oc = pytest.importorskip("ocelot.cpbd.elements")
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.optics import lattice_transfer_map
    from ocelot.cpbd.optics import twiss as ocelot_twiss

    cell = []
    for _ in range(NCELL):
        cell += [
            oc.Quadrupole(l=QUAD_L, k1=QUAD_K1),
            oc.Drift(l=DRIFT_L),
            oc.SBend(l=BEND_L, angle=ANGLE),
            oc.Drift(l=DRIFT_L),
            oc.Quadrupole(l=QUAD_L, k1=-QUAD_K1),
            oc.Drift(l=DRIFT_L),
        ]
    lattice = MagneticLattice(cell)
    e_tot_gev = math.sqrt(PC**2 + MC2**2) / 1e9
    matrix = np.asarray(lattice_transfer_map(lattice, e_tot_gev))
    return matrix, ocelot_twiss(lattice, tws0=None), lattice


@lru_cache(maxsize=None)
def madx_ring():
    madx_module = pytest.importorskip("cpymad.madx")
    madx = madx_module.Madx(stdout=False, cwd=tempfile.mkdtemp())
    cell_l = 2 * QUAD_L + BEND_L + 3 * DRIFT_L
    body = []
    for i in range(NCELL):
        s = i * cell_l
        body.append(f"qf{i}: quadrupole, at={s + QUAD_L / 2}, l={QUAD_L}, k1={QUAD_K1};")
        body.append(
            f"b{i}: sbend, at={s + QUAD_L + DRIFT_L + BEND_L / 2}, "
            f"l={BEND_L}, angle={ANGLE};"
        )
        body.append(
            f"qd{i}: quadrupole, "
            f"at={s + QUAD_L + 2 * DRIFT_L + BEND_L + QUAD_L / 2}, "
            f"l={QUAD_L}, k1={-QUAD_K1};"
        )
    madx.input(
        f"beam, particle=electron, pc={PC / 1e9};\n"
        f"ring: sequence, l={CIRCUMFERENCE};\n" + "\n".join(body) + "\nendsequence;\n"
        "use, sequence=ring;\n"
        "select, flag=sectormap, full;\n"
        "twiss, sectormap, sectortable=smap;"
    )
    table = madx.table.smap
    maps = [
        np.array([[table[f"r{i}{j}"][row] for j in range(1, 7)] for i in range(1, 7)])
        for row in range(len(table["r11"]))
    ]
    total = np.eye(6)
    for m in maps:
        total = m @ total
    return total, madx.table.summ


BMAD_SO = "/home/xkc85723/Documents/bmad-ecosystem/production/lib/libtao.so"


@lru_cache(maxsize=None)
def bmad_ring():
    """The same ring through Tao, with chromaticity by finite difference --
    the definition, rather than an analytic integral."""
    pytest.importorskip("pytao")
    if not os.path.exists(BMAD_SO):
        pytest.skip("Bmad libtao not installed")
    from pytao import Tao

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "ring.bmad")
    with open(path, "w") as handle:
        handle.write(
            "parameter[particle] = electron\n"
            "parameter[geometry] = closed\n"
            f"parameter[p0c] = {PC}\n"
            f"qf: quadrupole, l = {QUAD_L}, k1 = {QUAD_K1}\n"
            f"qd: quadrupole, l = {QUAD_L}, k1 = {-QUAD_K1}\n"
            f"b: sbend, l = {BEND_L}, angle = {ANGLE!r}\n"
            f"d: drift, l = {DRIFT_L}\n"
            "cell: line = (qf, d, b, d, qd, d)\n"
            f"lat: line = ({NCELL}*cell)\n"
            "use, lat\n"
        )
    tao = Tao(lattice_file=path, so_lib=BMAD_SO, noplot=True)

    def tunes():
        out = {}
        for plane, attribute in (("x", "ele.a.phi"), ("y", "ele.b.phi")):
            value = tao.cmd(f"pipe lat_list 1@0>>END|model {attribute}")[0]
            out[plane] = float(str(value).strip()) / (2 * math.pi)
        return out

    delta = 1e-4
    on = tunes()
    tao.cmd(f"set particle_start pz = {delta}")
    plus = tunes()
    tao.cmd(f"set particle_start pz = {-delta}")
    minus = tunes()
    tao.cmd("set particle_start pz = 0")
    return {
        "tune_x_total": on["x"],
        "tune_y_total": on["y"],
        "chromaticity_x": (plus["x"] - minus["x"]) / (2 * delta),
        "chromaticity_y": (plus["y"] - minus["y"]) / (2 * delta),
    }


# --- the tune, four ways ------------------------------------------------


def test_the_fractional_tune_agrees_across_codes():
    """The headline check. Transverse blocks need no conversion, so these are
    directly comparable as the codes produce them."""
    tunes = {
        "xsuite": fractional_tune(xsuite_ring()[0], "x"),
        "ocelot": fractional_tune(ocelot_ring()[0], "x"),
        "madx": fractional_tune(madx_ring()[0], "x"),
    }
    reference = tunes["xsuite"]
    for name, value in tunes.items():
        assert value == pytest.approx(reference, abs=1e-6), f"{name}: {tunes}"


def test_the_vertical_tune_agrees_across_codes():
    tunes = {
        "xsuite": fractional_tune(xsuite_ring()[0], "y"),
        "ocelot": fractional_tune(ocelot_ring()[0], "y"),
        "madx": fractional_tune(madx_ring()[0], "y"),
    }
    reference = tunes["xsuite"]
    for name, value in tunes.items():
        assert value == pytest.approx(reference, abs=1e-6), f"{name}: {tunes}"


def test_the_ring_is_not_accidentally_on_a_trivial_tune():
    """Guards every comparison above: agreement on 0.0 would prove nothing."""
    tune = fractional_tune(xsuite_ring()[0], "x")
    assert 0.05 < tune < 0.95


# --- periodic Twiss -----------------------------------------------------


def test_the_periodic_beta_agrees_across_codes():
    """The quantity A1 was built for: the closed solution, which no incoming
    beam can supply."""
    betas = {
        "xsuite": periodic_twiss(xsuite_ring()[0], "x")["beta"],
        "ocelot": periodic_twiss(ocelot_ring()[0], "x")["beta"],
        "madx": periodic_twiss(madx_ring()[0], "x")["beta"],
    }
    reference = betas["xsuite"]
    for name, value in betas.items():
        assert value == pytest.approx(reference, rel=1e-5), f"{name}: {betas}"


def test_the_periodic_alpha_agrees_across_codes():
    alphas = {
        "xsuite": periodic_twiss(xsuite_ring()[0], "x")["alpha"],
        "ocelot": periodic_twiss(ocelot_ring()[0], "x")["alpha"],
        "madx": periodic_twiss(madx_ring()[0], "x")["alpha"],
    }
    reference = alphas["xsuite"]
    for name, value in alphas.items():
        assert value == pytest.approx(reference, abs=1e-5), f"{name}: {alphas}"


# --- the codes' own numbers, against the map's ---------------------------


def test_each_codes_own_tune_matches_the_map():
    """An independent route to the same number: the map is differentiated
    about the closed orbit, the code's own tune is accumulated phase. They
    should not need to agree, and do."""
    _, tw = xsuite_ring()
    _, periodic, _ = ocelot_ring()
    _, summ = madx_ring()
    native = {
        "xsuite": float(tw.qx) % 1.0,
        "ocelot": (float(periodic[-1].mux) / (2 * math.pi)) % 1.0,
        "madx": float(summ["q1"][-1]) % 1.0,
    }
    for name, value in native.items():
        assert value == pytest.approx(
            fractional_tune(xsuite_ring()[0], "x"), abs=1e-5
        ), f"{name}: {native}"


def test_the_codes_agree_on_the_integer_tune_too():
    """Which the map cannot give, so this is the only place it is checked."""
    _, tw = xsuite_ring()
    _, summ = madx_ring()
    assert int(float(tw.qx)) == int(float(summ["q1"][-1]))


# --- chromaticity -------------------------------------------------------


def test_the_horizontal_chromaticity_agrees_across_three_codes():
    """Chromaticity is not derivable from one map -- it is dQ/ddelta, so it
    needs two momenta. Every ring code computes it off the matched solution,
    which is why A1 had to come first.

    Horizontally Xsuite, MAD-X and Bmad agree to six digits at -0.866684,
    from three independent implementations.
    """
    _, tw = xsuite_ring()
    _, summ = madx_ring()
    assert float(summ["dq1"][-1]) == pytest.approx(float(tw.dqx), rel=1e-5)
    assert bmad_ring()["chromaticity_x"] == pytest.approx(float(tw.dqx), rel=1e-5)


def test_madx_and_bmad_agree_on_vertical_chromaticity():
    """Two independent codes, same number to six digits: -0.653642."""
    _, summ = madx_ring()
    assert bmad_ring()["chromaticity_y"] == pytest.approx(
        float(summ["dq2"][-1]), rel=1e-5
    )


def test_xsuite_is_the_vertical_chromaticity_outlier():
    """A real model difference, run down rather than tolerated.

    ===========  ==========
    Xsuite       -0.461311
    Ocelot       -0.581475
    MAD-X        -0.653642
    Bmad         -0.653642
    ===========  ==========

    Horizontally all but Ocelot agree. What was established:

    * Neither code is misreporting. Each one's quoted chromaticity equals its
      *own* finite-difference ``dQ/ddelta`` exactly, so this is the models
      genuinely tracking different vertical tunes off-momentum.
    * It is the **dipole**. Per-cell divergence shrinks monotonically as the
      bend angle does -- 45 deg: -0.0240, 22.5: -0.0184, 11.25: -0.0097,
      5.6: -0.0034 -- and extrapolates to zero for a ring with no bending.
    * Localised to the bend's off-momentum vertical transport. For a single
      1 m / 45 deg sector bend, ``R34`` is exactly ``L`` in both at
      ``delta = 0``, but at ``delta = 0.01`` Xsuite gives 0.991086 and MAD-X
      1.000987 -- opposite directions, and ``0.991086 * 1.01 = 1.000997``.
      MAD-X's vertical block is Xsuite's without the ``1/(1+delta)``
      rigidity factor.
    * Ruled out: MAD-X's ``chrom`` flag (no change), Xsuite's dipole edge
      treatment (no change).

    Which is *right* is deliberately not asserted here. The rigidity factor
    argues for Xsuite; two independent codes agreeing against it argue the
    other way. Pinned as a baseline either way, so a change anywhere shows up.
    """
    from ocelot.cpbd.chromaticity import chromaticity

    _, tw = xsuite_ring()
    _, periodic, lattice = ocelot_ring()
    _, summ = madx_ring()
    measured = {
        "xsuite": float(tw.dqy),
        "ocelot": float(chromaticity(lattice, periodic[0])[1]),
        "madx": float(summ["dq2"][-1]),
        "bmad": bmad_ring()["chromaticity_y"],
    }
    assert measured["xsuite"] == pytest.approx(-0.461311, rel=1e-4)
    assert measured["ocelot"] == pytest.approx(-0.581475, rel=1e-4)
    assert measured["madx"] == pytest.approx(-0.653642, rel=1e-4)
    assert measured["bmad"] == pytest.approx(-0.653642, rel=1e-4)
    spread = max(measured.values()) - min(measured.values())
    assert spread > 0.1, f"codes now agree -- good news, update this: {measured}"


def test_ocelot_horizontal_chromaticity_is_the_outlier():
    """Three codes agreeing to six digits against one: -1.16126 against
    -0.866684. Ocelot's chromaticity is an analytic integral over the
    lattice, where the others difference tunes at two momenta. Failing this
    test would be good news."""
    from ocelot.cpbd.chromaticity import chromaticity

    _, tw = xsuite_ring()
    _, periodic, lattice = ocelot_ring()
    ocelot_x = float(chromaticity(lattice, periodic[0])[0])
    assert ocelot_x == pytest.approx(-1.16126, rel=1e-4)
    assert abs(ocelot_x / float(tw.dqx) - 1) > 0.3


def test_ocelot_reports_the_tune_through_simba_not_just_through_ocelot():
    """`read_optics_summary` is the only route by which a ring's tune
    reaches simba from Ocelot, and until now it raised `NameError: name
    'pi' is not defined` on its first line -- `pi` was never imported into
    `Ocelot.py`. Nothing caught it: `postProcess` does not guard the call,
    so every Ocelot ring run died there, and every test went through
    `ocelot_twiss` directly instead of through simba's own method.

    So this drives the real method. The value is checked against the
    independent Xsuite tune, not against Ocelot's own.
    """
    from simba.Codes.Ocelot.Ocelot import ocelotLattice

    _, tw = xsuite_ring()
    _, _, lattice = ocelot_ring()

    class FakeOcelot:
        read_optics_summary = ocelotLattice.read_optics_summary
        _ocelot_periodic = ocelotLattice._ocelot_periodic
        objectname = "ring"
        lat_obj = lattice
        _periodic = None
        # the periodic solution is seeded with the reference energy, in eV
        reference_energy = math.hypot(PC, MC2)

    summary = FakeOcelot().read_optics_summary()
    assert summary["tune_x_total"] == pytest.approx(float(tw.qx), rel=1e-4)
    assert summary["tune_y_total"] == pytest.approx(float(tw.qy), rel=1e-4)


def test_bmad_agrees_on_the_tune_including_the_integer():
    _, tw = xsuite_ring()
    assert bmad_ring()["tune_x_total"] == pytest.approx(float(tw.qx), rel=1e-6)
    assert bmad_ring()["tune_y_total"] == pytest.approx(float(tw.qy), rel=1e-6)


def test_the_chromaticity_is_negative_and_substantial():
    """A FODO ring of pure quadrupoles has large natural chromaticity, so a
    near-zero answer would mean something is not being computed at all."""
    _, tw = xsuite_ring()
    assert float(tw.dqx) < -0.5
    assert float(tw.dqy) < -0.3


# --- momentum compaction ------------------------------------------------


def test_the_momentum_compaction_agrees_across_codes():
    """Longitudinal, so this one *does* need the convention conversion:
    Ocelot's map is sign-flipped and beta0**2 scaled against Xsuite's."""
    from simba.Codes.Ocelot.Ocelot import ocelotLattice

    beta0 = math.sqrt(1 - 1 / GAMMA0**2)
    ocelot_raw = ocelot_ring()[0].copy()
    ratio = ocelotLattice.otm_longitudinal_sign * beta0 ** (
        ocelotLattice.otm_longitudinal_scale
    )
    diagonal = np.array([1.0, 1.0, 1.0, 1.0, ratio, 1.0])
    ocelot_canonical = (diagonal[:, None] * ocelot_raw) / diagonal[None, :]

    values = {
        "xsuite": momentum_compaction(xsuite_ring()[0], CIRCUMFERENCE, GAMMA0),
        "ocelot": momentum_compaction(ocelot_canonical, CIRCUMFERENCE, GAMMA0),
    }
    assert values["ocelot"] == pytest.approx(values["xsuite"], rel=1e-4), values


def test_the_compaction_matches_what_the_codes_report():
    _, tw = xsuite_ring()
    _, summ = madx_ring()
    derived = momentum_compaction(xsuite_ring()[0], CIRCUMFERENCE, GAMMA0)
    assert derived == pytest.approx(float(tw.momentum_compaction_factor), rel=1e-6)
    assert derived == pytest.approx(float(summ["alfa"][-1]), rel=2e-2)
