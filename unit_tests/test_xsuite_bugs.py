from types import SimpleNamespace as NS

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
from simba.Codes.Xsuite.Xsuite import xsuiteLattice


def test_bunch_statistics_skip_lost_particles():
    p = xt.Particles(p0c=1e9, x=[-1e-3, 1e-3, 5.0], px=[1e-5, -1e-5, 0.0], y=[1e-3, -1e-3, 0.0])
    p.state[2] = 0
    fake = NS()
    fake.compute_norm_emit = lambda *a: xsuiteLattice.compute_norm_emit(fake, *a)
    fake.compute_norm_emit_corrected = lambda *a: xsuiteLattice.compute_norm_emit_corrected(fake, *a)
    stats = xsuiteLattice.bunch_statistics(fake, p)
    assert stats["mean_x"] == 0 and np.isclose(stats["sigma_x"], 1e-3)
