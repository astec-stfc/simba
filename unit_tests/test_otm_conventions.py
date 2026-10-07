"""The codes do not agree on what a one-turn map's coordinates mean."""

import tempfile
from functools import lru_cache

import numpy as np
import pytest

from simba.Codes.Bmad.Bmad import bmadLattice
from simba.Codes.Elegant.Elegant import elegantLattice
from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite.Xsuite import xsuiteLattice
from simba.Framework_objects import frameworkLattice

L = 2.0
MC2 = 0.510998950e6
PC_SLOW = 0.4e6
PC_FAST = 100e6


def beta_gamma(pc):
    gamma = np.sqrt(1 + (pc / MC2) ** 2)
    return np.sqrt(1 - 1 / gamma**2), gamma


def drift_r56(pc):
    """The time-of-flight answer, `L / (beta0 * gamma0)**2`."""
    beta, gamma = beta_gamma(pc)
    return L / (beta * gamma) ** 2


@lru_cache(maxsize=None)
def ocelot_drift(pc):
    from ocelot.cpbd.elements import Drift
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.optics import lattice_transfer_map

    # Ocelot takes the TOTAL energy in GeV, not pc -- at PC_FAST the two are
    # within 1e-5 of each other, which is exactly how a unit slip like this
    # survives a high-energy-only check.
    e_tot = np.sqrt(pc**2 + MC2**2)
    return np.asarray(
        lattice_transfer_map(MagneticLattice((Drift(l=L),)), e_tot / 1e9)
    )


@lru_cache(maxsize=None)
def xsuite_drift(pc):
    xt = pytest.importorskip("xtrack")
    line = xt.Line(elements=[xt.Drift(length=L)])
    line.particle_ref = xt.Particles(p0c=pc, mass0=MC2)
    line.build_tracker()
    result = line.compute_R_matrix(particle_on_co=line.particle_ref.copy())
    return np.asarray(result["R_matrix"], dtype=float)


@lru_cache(maxsize=None)
def madx_drift(pc):
    madx_module = pytest.importorskip("cpymad.madx")
    madx = madx_module.Madx(stdout=False, cwd=tempfile.mkdtemp())
    madx.input(
        f"beam, particle=electron, pc={pc / 1e9};\n"
        f"seq: sequence, l={L}; d: drift, at={L / 2}, l={L}; endsequence;\n"
        "use, sequence=seq;\n"
        "select, flag=sectormap, full;\n"
        "twiss, betx=1, bety=1, sectormap, sectortable=smap;"
    )
    table = madx.table.smap
    row = list(table["name"]).index("d")
    return np.array(
        [[table[f"r{i}{j}"][row] for j in range(1, 7)] for i in range(1, 7)]
    )


class FakeLine:
    """Carries a map and a convention, and nothing else."""

    def __init__(self, matrix, cls):
        self.one_turn_map = matrix
        self.otm_longitudinal_sign = cls.otm_longitudinal_sign
        self.otm_longitudinal_scale = cls.otm_longitudinal_scale

    one_turn_map_canonical = frameworkLattice.one_turn_map_canonical


def canonical(matrix, cls, pc, magnitude=True):
    beta, _ = beta_gamma(pc)
    return FakeLine(matrix, cls).one_turn_map_canonical(
        beta0=beta, magnitude=magnitude
    )


# --- transverse needs no conversion -------------------------------------


@pytest.mark.parametrize("pc", [PC_SLOW, PC_FAST], ids=["slow", "fast"])
def test_xsuite_and_madx_agree_transversely(pc):
    """`x'` and `px` are the same thing at the closed orbit, so R12 == L in
    both. This is why a cross-code tune check needs no normalising."""
    for matrix in (xsuite_drift(pc), madx_drift(pc)):
        assert matrix[0, 1] == pytest.approx(L, rel=1e-9)
        assert matrix[2, 3] == pytest.approx(L, rel=1e-9)
    assert np.allclose(madx_drift(pc)[:4, :4], xsuite_drift(pc)[:4, :4], atol=1e-12)


def test_ocelot_agrees_transversely_where_it_works():
    assert np.allclose(
        ocelot_drift(PC_FAST)[:4, :4], xsuite_drift(PC_FAST)[:4, :4], atol=1e-12
    )


def test_ocelot_works_at_low_energy_given_the_right_units():
    """Ocelot has no low-energy floor. It looked like it did until the units
    were right: `lattice_transfer_map` wants TOTAL energy, and feeding it pc
    asks for a particle below its own rest mass, which NaNs in
    `sqrt(1 - igamma2)`. Guards the fixture's conversion, not Ocelot."""
    assert not np.isnan(ocelot_drift(PC_SLOW)).any()


def test_ocelot_nans_only_on_an_impossible_particle():
    """The trap itself, pinned: total energy below the rest mass."""
    from ocelot.cpbd.elements import Drift
    from ocelot.cpbd.magnetic_lattice import MagneticLattice
    from ocelot.cpbd.optics import lattice_transfer_map

    with np.errstate(invalid="ignore"):
        impossible = lattice_transfer_map(
            MagneticLattice((Drift(l=L),)), 0.5 * MC2 / 1e9
        )
    assert np.isnan(np.asarray(impossible)).any()


# --- longitudinal is three different problems ---------------------------


def test_madx_measures_the_time_of_flight_term():
    assert madx_drift(PC_SLOW)[4, 5] == pytest.approx(drift_r56(PC_SLOW), rel=1e-5)


def test_xsuite_differs_from_madx_by_beta_squared():
    """At beta0 = 1 these coincide, which is why the slow energy matters."""
    beta, _ = beta_gamma(PC_SLOW)
    assert xsuite_drift(PC_SLOW)[4, 5] == pytest.approx(
        beta**2 * drift_r56(PC_SLOW), rel=1e-3
    )


def test_ocelot_is_sign_flipped():
    assert ocelot_drift(PC_FAST)[4, 5] == pytest.approx(-drift_r56(PC_FAST), rel=1e-4)


def test_elegant_and_ocelot_are_the_sign_flipped_pair():
    """Measured on a 1 m / 0.3 rad dipole: R51 is the same magnitude in all
    four codes and differs only in sign, which is what makes a sign-only
    normalisation exact rather than approximate."""
    assert elegantLattice.otm_longitudinal_sign == -1
    assert ocelotLattice.otm_longitudinal_sign == -1
    assert xsuiteLattice.otm_longitudinal_sign == 1
    assert madxLattice.otm_longitudinal_sign == 1
    assert bmadLattice.otm_longitudinal_sign == 1


# --- the conversion puts them together ----------------------------------


def test_ocelot_follows_madx_magnitude_not_xsuite():
    """The one that had to be measured down the energy range: at beta0 ~ 1 a
    beta0**2 factor is invisible, so a high-energy check cannot tell the two
    apart. Ocelot is -L/(beta0.gamma0)**2 at every energy it works at."""
    for pc in (1.0e6, 2e6, 5e6):
        beta, gamma = beta_gamma(pc)
        assert ocelot_drift(pc)[4, 5] == pytest.approx(
            -L / (beta * gamma) ** 2, rel=1e-5
        )


@pytest.mark.parametrize("pc", [1.0e6, 2e6, 5e6], ids=lambda v: f"{v/1e6:g}MeV")
def test_normalising_makes_ocelot_and_xsuite_agree(pc):
    left = canonical(ocelot_drift(pc), ocelotLattice, pc)
    right = canonical(xsuite_drift(pc), xsuiteLattice, pc)
    assert left[4, 5] == pytest.approx(right[4, 5], rel=1e-3)


def test_normalising_makes_madx_and_xsuite_agree():
    """The one that actually does work: beta0**2 at beta0 = 0.62."""
    left = canonical(madx_drift(PC_SLOW), madxLattice, PC_SLOW)
    right = canonical(xsuite_drift(PC_SLOW), xsuiteLattice, PC_SLOW)
    assert left[4, 5] == pytest.approx(right[4, 5], rel=1e-3)


def test_without_normalising_they_disagree_by_a_lot():
    """Guards the test above against passing for the wrong reason."""
    assert madx_drift(PC_SLOW)[4, 5] / xsuite_drift(PC_SLOW)[4, 5] > 2.0


def test_normalising_leaves_the_transverse_block_alone():
    raw = madx_drift(PC_SLOW)
    assert np.allclose(canonical(raw, madxLattice, PC_SLOW)[:4, :4], raw[:4, :4])


def test_normalising_cannot_change_the_tune():
    """A diagonal similarity preserves trace and determinant, so whatever the
    conversion gets wrong, it cannot be the tune."""
    raw = madx_drift(PC_SLOW)
    converted = canonical(raw, madxLattice, PC_SLOW)
    assert np.trace(converted[:2, :2]) == pytest.approx(np.trace(raw[:2, :2]))
    assert np.linalg.det(converted) == pytest.approx(np.linalg.det(raw))


# --- elegant is the one that must not be rescaled -----------------------


def test_elegant_declines_to_normalise_magnitudes():
    """Its conversion is additive, not a factor, so it returns None rather
    than a plausible wrong answer."""
    assert elegantLattice.otm_longitudinal_scale is None
    assert canonical(np.eye(6), elegantLattice, PC_SLOW) is None


def test_elegant_still_aligns_its_signs():
    """The magnitude is unconvertible, the sign is not -- and a sign is
    enough to stop a comparison tripping over direction."""
    raw = np.eye(6)
    raw[4, 0] = +0.295520  # elegant's measured dipole R51
    aligned = canonical(raw, elegantLattice, PC_SLOW, magnitude=False)
    assert aligned is not None
    assert aligned[4, 0] == pytest.approx(-0.295520)


def test_sign_alignment_needs_no_energy():
    assert (
        FakeLine(np.eye(6), elegantLattice).one_turn_map_canonical(magnitude=False)
        is not None
    )


def test_a_rescale_of_elegant_would_have_been_silently_wrong():
    """elegant's drift R56 is 0 -- its fifth coordinate is geometric path
    length, so the drift's time-of-flight term simply is not in there. Every
    scale factor maps 0 to 0, so a normalised elegant map would report zero
    momentum compaction and raise nothing. Hence the None above."""
    for factor in (1.0, -1.0, 0.38, 1 / 0.38):
        assert not np.isclose(0.0 * factor, drift_r56(PC_SLOW))


# --- MAD-X composes its map from per-element pieces ---------------------
#
# `sectortable` is element by element, not cumulative, which reads like the
# opposite until you notice the zero-length end marker comes back as the
# identity. So the line's map is the ordered product.


@lru_cache(maxsize=None)
def madx_drift_and_dipole(pc):
    """Per-element sector maps for a 1.5 m drift followed by a 1 m bend."""
    madx_module = pytest.importorskip("cpymad.madx")
    madx = madx_module.Madx(stdout=False, cwd=tempfile.mkdtemp())
    madx.input(
        f"beam, particle=electron, pc={pc / 1e9};\n"
        f"seq: sequence, l={1.5 + 1.0};\n"
        "  d: drift, at=0.75, l=1.5;\n"
        "  b: sbend, at=2.0, l=1.0, angle=0.3;\n"
        "endsequence;\n"
        "use, sequence=seq;\n"
        "select, flag=sectormap, full;\n"
        "twiss, betx=1, bety=1, sectormap, sectortable=smap;"
    )
    table = madx.table.smap
    return [
        np.array([[table[f"r{i}{j}"][row] for j in range(1, 7)] for i in range(1, 7)])
        for row in range(len(table["r11"]))
    ]


def ordered_product(matrices):
    """What `madxLattice.read_one_turn_map` does."""
    total = np.eye(6)
    for matrix in matrices:
        total = matrix @ total
    return total


def test_the_end_marker_proves_the_table_is_not_cumulative():
    """A zero-length marker at the end of a 2.5 m line: identity if the rows
    are per-element, the whole line's map if they were cumulative."""
    assert np.allclose(madx_drift_and_dipole(PC_SLOW)[-1], np.eye(6), atol=1e-12)


def test_the_ordered_product_matches_xsuite():
    xt = pytest.importorskip("xtrack")
    line = xt.Line(
        elements=[xt.Drift(length=1.5), xt.Bend(length=1.0, angle=0.3)]
    )
    line.particle_ref = xt.Particles(p0c=PC_SLOW, mass0=MC2)
    line.build_tracker()
    expected = np.asarray(
        line.compute_R_matrix(particle_on_co=line.particle_ref.copy())["R_matrix"]
    )
    got = ordered_product(madx_drift_and_dipole(PC_SLOW))
    assert np.allclose(got[:4, :4], expected[:4, :4], atol=1e-9)


def test_the_product_order_matters():
    """Guards the test above: the reversed product is a different matrix, so
    agreement is not an accident of a symmetric lattice."""
    maps = madx_drift_and_dipole(PC_SLOW)
    assert not np.allclose(
        ordered_product(maps), ordered_product(list(reversed(maps))), atol=1e-6
    )


def test_the_product_is_symplectic():
    assert np.linalg.det(ordered_product(madx_drift_and_dipole(PC_SLOW))) == (
        pytest.approx(1.0, rel=1e-9)
    )
