from types import SimpleNamespace as NS

import pytest

pytest.importorskip("ocelot")
from simba.Codes.Ocelot.mbi import MBI


def test_csr_impedance_negative_bend():
    pos = MBI.csrimpedance(None, 1e-5, NS(l=0.2, angle=0.1))
    assert pos != 0 and MBI.csrimpedance(None, 1e-5, NS(l=0.2, angle=-0.1)) == pos
