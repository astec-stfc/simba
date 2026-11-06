"""GPT runs: a failed ``gpt`` is reported (simba#58), and the aperture time windows."""

import stat
from types import SimpleNamespace

import numpy as np
import pytest

from simba import Framework_objects
from simba.Codes.GPT.GPT import gptLattice
from simba.Modules.constants import speed_of_light


def test_a_failed_gpt_run_says_why(tmp_path):
    """gpt's exit status and stderr were dropped, so a rejected input file surfaced as
    `'gdf_emit' object has no attribute 'avgG'` in the Twiss reader."""
    gpt = tmp_path / "gpt"
    gpt.write_text("#!/bin/sh\necho 'T.in(6): Error: syntax error' >&2\nexit 1\n")
    gpt.chmod(gpt.stat().st_mode | stat.S_IEXEC)
    stub = SimpleNamespace(
        code="gpt",
        objectname="T",
        auto_phase=False,
        executables={"gpt": [str(gpt)]},
        global_parameters={"GPTLICENSE": "1", "master_subdir": str(tmp_path)},
    )
    stub.run_command = lambda *a, **k: Framework_objects.frameworkLattice.run_command(stub, *a, **k)
    with pytest.raises(RuntimeError, match="syntax error"):
        gptLattice.run(stub)
    assert "syntax error" in (tmp_path / "T.log").read_text()


def _timing(definition="beam", **beam):
    start = SimpleNamespace(s=10.0, s_point="middle", length=0.0, middle=SimpleNamespace(z=4.0))
    stub = SimpleNamespace(
        global_parameters={"beam": SimpleNamespace(**beam)},
        startObject=SimpleNamespace(physical=start),
        local_z=lambda point: point.z,
        particle_definition=definition,
    )
    lines = gptLattice._bunch_timing(stub).splitlines()
    return {k: float(v.rstrip(";")) for k, v in (line.split(" = ") for line in lines)}


def test_the_aperture_window_spans_the_bunch_in_s():
    """s = s of the start element + z from it; at t = 0 the head would pass s = 0 at c
    and the slowest tail at bunch_v_min."""
    t = _timing(z=np.array([3.999, 4.0, 4.001]), Bz=np.array([0.99, 0.999, 0.999]))
    assert t["bunch_v_min"] == pytest.approx(0.99 * speed_of_light)
    assert t["bunch_t_head"] == pytest.approx(-10.001 / speed_of_light)
    assert t["bunch_t_tail"] == pytest.approx(-9.999 / (0.99 * speed_of_light))


def test_from_a_cathode_the_window_opens_at_the_first_emission():
    t = _timing(z=np.zeros(2), Bz=np.array([1e-3, 1e-3]), t=np.array([1e-12, 5e-12]), definition="laser")
    assert t["bunch_t_head"] == pytest.approx(1e-12 - 6.0 / speed_of_light)
    assert t["bunch_t_tail"] == pytest.approx(5e-12 - 6.0 / (1e-3 * speed_of_light))
