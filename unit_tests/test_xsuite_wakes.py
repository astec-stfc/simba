"""Structure wakes in Xsuite, by xwakes: a per-cell table, scaled by ``n_cells``."""

import numpy as np
import pytest

pytest.importorskip("xwakes")

import xtrack as xt
from laura.translator.utils.fields import FieldMap
from laura.translator.utils.fields.field_parameter import FieldParameter
from laura.translator.utils.units import UnitValue
from scipy.constants import e

from simba.Codes.Xsuite.Xsuite import _Wake

S = np.linspace(0.0, 0.01, 2001)
W0, DECAY = 1e13, 3e-3
CELLS, CHARGE, N, P0C, SIGMA = 10, 250e-12, 100000, 1e9, 3e-4


def _field(**columns):
    units = {"z": "m", "Wz": "V/C", "Wx": "V/C/m"}
    return FieldMap(
        field_type="3DWake" if "Wx" in columns else "LongitudinalWake",
        **{
            k: FieldParameter(name=k, value=UnitValue(v, units=units[k]))
            for k, v in {"z": -S[::-1], **columns}.items()
        },
    )


def _bunch(**coords):
    zeta = np.random.default_rng(3).normal(0.0, SIGMA, N)
    return xt.Particles(
        p0c=P0C, mass0=xt.ELECTRON_MASS_EV, zeta=zeta, weight=CHARGE / e / N, **coords
    )


def test_energy_loss_is_the_convolution_of_the_bunch_with_n_cells_of_the_table():
    wake = W0 * np.exp(-S / DECAY)
    p = _bunch()
    _Wake(_field(Wz=wake[::-1]), CELLS, 1000).track(p)
    edges = np.linspace(p.zeta.min(), p.zeta.max(), 401)
    mid = (edges[1:] + edges[:-1]) / 2
    lam = np.histogram(p.zeta, edges)[0] / N
    volts = [
        -CHARGE * CELLS * np.sum(lam[mid >= z] * W0 * np.exp(-(mid[mid >= z] - z) / DECAY))
        for z in mid
    ]
    expected = np.interp(p.zeta, mid, volts) / P0C
    assert p.delta.mean() == pytest.approx(expected.mean(), rel=0.02)
    assert p.delta.std() == pytest.approx(expected.std(), rel=0.01)


def test_a_transverse_wake_kicks_the_tail_towards_the_offset():
    wake = W0 * np.exp(-S / DECAY)
    p = _bunch(x=np.full(N, 1e-3))
    _Wake(_field(Wz=0 * wake, Wx=wake[::-1]), CELLS, 1000).track(p)
    tail, head = p.zeta < -SIGMA, p.zeta > SIGMA
    assert p.px[tail].mean() > 5 * p.px[head].mean() > 0
