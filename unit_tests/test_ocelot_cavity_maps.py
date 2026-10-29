"""Ocelot's cavity maps at small energy gain; see :mod:`simba.Codes.Ocelot.cavitymaps`.

The oracle is Ocelot's own formula, evaluated with enough digits to survive its
cancellation: the patched forms are the same expressions in exact arithmetic.
"""

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
pytest.importorskip("ocelot")

from simba.Codes.Ocelot.cavitymaps import _gain_ratios, stable_cavity_maps  # noqa: E402

FREQ = 1.3e9


def ocelot_ratios(g0, g1):
    """Ocelot's ratios as written, at 150 digits."""
    mp.mp.dps = 150
    g0, g1 = mp.mpf(g0), mp.mpf(g1)
    b0, b1 = mp.sqrt(1 - 1 / g0**2), mp.sqrt(1 - 1 / g1**2)
    d = g0 - g1
    return (
        (g0 * g1 * (b0 * b1 - 1) + 1) / d**2,
        (b0**3 * g0**3 - b1**3 * g1**3) / d,
        (b1**3 * g1**3 + b0 * (g0 - g1**3)) / d**2,
        (2 * g0 * g1**3 * (b0 * b1**3 - 1) + g0**2 + 3 * g1**2 - 2) / d**3,
    )


@pytest.mark.parametrize("g0", [3.0, 70.0, 1957.0, 5597.0])
@pytest.mark.parametrize("gain", [1e-9, 1e-5, 0.1, 40.0])
def test_the_gain_ratios_are_ocelots(g0, gain):
    """At 1 GeV and a gain of 1e-5 in gamma, Ocelot's float arithmetic keeps
    no correct digit of these; the rewritten forms keep them all."""
    ours = _gain_ratios(g0, g0 + gain)
    for mine, exact in zip(ours, ocelot_ratios(g0, g0 + gain)):
        assert mine == pytest.approx(float(exact), rel=1e-9)


def cavity(phi, v=1e-4, length=0.1):
    import ocelot.cpbd.elements as oc

    stable_cavity_maps()
    return oc.Cavity(l=length, v=v, freq=FREQ, phi=phi).element


@pytest.mark.parametrize("phi", [90.0, -90.0, 89.99, 30.0])
def test_r55_is_ocelots_formula(phi):
    """90 degrees is the zero crossing, where Ocelot gave inf; 89.99 is where
    it gave 4.8e-5 for -3.6e-11."""
    from ocelot.common.globals import speed_of_light
    from ocelot.cpbd.high_order import m_e_GeV

    energy, v, length = 1.0, 1e-4, 0.1
    R = cavity(phi, v, length)._R_main_matrix(energy, length)

    mp.mp.dps = 150
    m, ph = mp.mpf(m_e_GeV), mp.mpf(phi * np.pi / 180.0)
    g0 = mp.mpf(energy) / m
    g1 = (mp.mpf(energy) + v * mp.cos(ph)) / m
    b0, b1 = mp.sqrt(1 - 1 / g0**2), mp.sqrt(1 - 1 / g1**2)
    k = 2 * mp.pi * FREQ / mp.mpf(speed_of_light)
    r55 = k * length * b0 * v / m * mp.sin(ph) * (g0 * g1 * (b0 * b1 - 1) + 1) / (
        b1 * g1 * (g0 - g1) ** 2
    )
    assert R[4, 4] - 1 == pytest.approx(float(r55), rel=1e-6)
    assert np.isfinite(R).all()


@pytest.mark.parametrize("phi", [90.0, -90.0])
def test_a_particle_crosses_a_cavity_at_zero_crossing(phi):
    from ocelot.cpbd.transformations.cavity import CavityTM

    element = cavity(phi)
    X = np.array([[1e-4], [1e-5], [-1e-4], [2e-5], [1e-3], [1e-3]])
    out = CavityTM.from_element(element).map4cav(X.copy(), 1.0, None, element.l)
    assert np.isfinite(out).all()
    assert out[4, 0] == pytest.approx(1e-3, rel=1e-6)
