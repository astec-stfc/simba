"""Guards found by running a real ring (Diamond-I, the BR14 booster) through every code.

Each is tested on a stub carrying only what the method reads, so none needs the
code it guards installed.
"""

import types
import warnings

import pytest

from simba.Codes.MADX.MADX import madxLattice
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Xsuite import Xsuite as xsuite_module
from simba.Codes.Xsuite.Xsuite import xsuiteLattice


# -- Xsuite: GPU or CPU context ---------------------------------------------------


class XsuiteStub:
    use_gpu = xsuiteLattice.use_gpu

    def __init__(self, tracking=None):
        self.file_block = {"tracking": tracking} if tracking is not None else {}


@pytest.fixture
def cupy(monkeypatch):
    monkeypatch.setattr(xsuite_module, "has_cupy", True)
    monkeypatch.delenv("SIMBA_XSUITE_CONTEXT", raising=False)


def test_xsuite_takes_the_gpu_when_cupy_is_there(cupy):
    assert XsuiteStub().use_gpu() is True


def test_xsuite_can_be_held_to_the_cpu_by_the_tracking_block(cupy):
    """A CUDA kernel for a few hundred elements took minutes and >19 GB to compile."""
    assert XsuiteStub({"context": "cpu"}).use_gpu() is False


def test_xsuite_can_be_held_to_the_cpu_by_the_environment(cupy, monkeypatch):
    monkeypatch.setenv("SIMBA_XSUITE_CONTEXT", "cpu")
    assert XsuiteStub().use_gpu() is False
    assert XsuiteStub({"context": "gpu"}).use_gpu() is True  # the study wins


def test_xsuite_without_cupy_warns_when_asked_for_the_gpu(monkeypatch):
    monkeypatch.setattr(xsuite_module, "has_cupy", False)
    with pytest.warns(UserWarning, match="cupy is not installed"):
        assert XsuiteStub({"context": "gpu"}).use_gpu() is False


def test_xsuite_ignores_a_nonsense_context_with_a_warning(cupy):
    with pytest.warns(UserWarning, match="not one of auto, cpu, gpu"):
        assert XsuiteStub({"context": "tpu"}).use_gpu() is True


# -- Xsuite: periodic twiss when the RF has no 6D solution -------------------------


class FakeLine:
    def __init__(self, fail_6d):
        self.fail_6d = fail_6d
        self.calls = []

    def twiss(self, **kwargs):
        self.calls.append(kwargs["method"])
        if kwargs["method"] == "6d" and self.fail_6d:
            raise ValueError("Invalid n3")
        return kwargs["method"]


class TwissStub:
    _twiss_now = xsuiteLattice._twiss_now
    periodic = True
    rf_voltage = 1e6
    objectname = "RING"

    def __init__(self, fail_6d):
        self.line = FakeLine(fail_6d)


def test_an_unstable_rf_falls_back_to_the_4d_solution_with_a_warning():
    """Source cavity at an unstable phase: Xsuite raised ``Invalid n3`` and the
    exception took the transverse optics down with it."""
    ring = TwissStub(fail_6d=True)
    with pytest.warns(UserWarning, match="not longitudinally stable"):
        assert ring._twiss_now() == "4d"
    assert ring.line.calls == ["6d", "4d"]


def test_a_stable_rf_keeps_the_6d_solution():
    ring = TwissStub(fail_6d=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ring._twiss_now() == "6d"


# -- MAD-X: slices behind the periodic optics --------------------------------------


class MadxStub:
    slices = madxLattice.slices
    warn_if_coarsely_sliced = madxLattice.warn_if_coarsely_sliced
    nslice_quadrupole, nslice_sbend, nslice_sextupole, nslice_solenoid = 4, 8, 4, 4
    objectname = "RING"

    def __init__(self, periodic=True, tracking=None):
        self.periodic = periodic
        self.file_block = {"tracking": tracking} if tracking is not None else {}


def test_slice_counts_can_be_set_in_the_tracking_block():
    ring = MadxStub(tracking={"nslice_quadrupole": 32})
    assert ring.slices("nslice_quadrupole") == 32
    assert ring.slices("nslice_sbend") == 8


def test_default_slicing_warns_that_periodic_optics_carry_the_error():
    """4 slices shifted a Diamond-I tune by 0.010; 64 left 3e-5."""
    with pytest.warns(UserWarning, match="slicing error"):
        MadxStub().warn_if_coarsely_sliced()


def test_fine_slicing_does_not_warn():
    fine = {"nslice_quadrupole": 16, "nslice_sbend": 16, "nslice_sextupole": 16}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MadxStub(tracking=fine).warn_if_coarsely_sliced()


def test_an_open_line_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MadxStub(periodic=False).warn_if_coarsely_sliced()


# -- Ocelot: why there is no chromaticity -------------------------------------------


class OcelotStub:
    read_optics_summary = ocelotLattice.read_optics_summary
    objectname = "RING"
    lat_obj = None

    def _ocelot_periodic(self):
        return [types.SimpleNamespace(mux=2 * 3.141592653589793 * 1.25, muy=2 * 3.141592653589793 * 0.5)]


def test_the_ocelot_chromaticity_warning_says_why(monkeypatch):
    pytest.importorskip("ocelot")
    import ocelot.cpbd.chromaticity as chromaticity

    def refuse(*args, **kwargs):
        raise TypeError("unsupported operand type(s) for *: 'SecondTM' and 'Twiss'")

    monkeypatch.setattr(chromaticity, "chromaticity", refuse)
    with pytest.warns(UserWarning, match="natural chromaticity without the sextupoles"):
        summary = OcelotStub().read_optics_summary()
    assert summary["tune_x_total"] == pytest.approx(1.25)
    assert "chromaticity_x" not in summary
