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


@pytest.mark.parametrize("orjson", [True, False])
def test_a_written_xsuite_json_reads_back_the_same(orjson, tmp_path, monkeypatch):
    """With orjson, or without it as the json fallback."""
    import json
    import sys
    from simba.Modules.Beams.xsuite import write_xsuite_json

    if orjson:
        pytest.importorskip("orjson")
    else:
        monkeypatch.setitem(sys.modules, "orjson", None)
    p = xt.Particles(p0c=575.9e6, x=np.random.normal(0, 1e-4, 100), delta=np.random.normal(0, 1e-3, 100))
    path = tmp_path / "beam.xsuite.json"
    write_xsuite_json(p.to_dict(), path)
    back = xt.Particles.from_dict(json.loads(path.read_text()))
    for name in ("x", "delta", "p0c", "state", "particle_id"):
        assert np.array_equal(getattr(back, name), getattr(p, name)), name
