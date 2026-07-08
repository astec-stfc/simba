"""
Tests for RFTrackGenerator. RF_Track is not on PyPI and not installed in CI, so
these fake it: a recording ``Bunch6dT_Generator`` verifies the generic ->
RF-Track option mapping and unit multipliers without a real install.
"""
import os
import types

import numpy as np
import pytest

from laura.translator.conversion_rules.codes import rftrack_conversion

from simba.Codes.Generators import RFTrackGenerator


class _FakeGen:
    """Records attributes set on it (stands in for Bunch6dT_Generator)."""


class _FakeBunch6dT:
    def __init__(self, gen, npart):
        self.gen = gen
        self.npart = npart

    def get_phase_space(self, fmt):
        # X Y Z Px Py Pz t0 m Q N, one particle
        return np.array([[1.0, 2.0, 0.0, 0.01, 0.02, 5.0, 3.0, 0.511, -1.0, 1e5]])


@pytest.fixture
def fake_rft(monkeypatch):
    fake = types.SimpleNamespace(
        Bunch6dT_Generator=_FakeGen,
        Bunch6dT=_FakeBunch6dT,
    )
    monkeypatch.setattr(rftrack_conversion, "get_rftrack", lambda: fake)
    return fake


@pytest.fixture
def generator():
    return RFTrackGenerator(
        global_parameters={
            "master_subdir": f"{os.path.dirname(os.path.abspath(__file__))}"
        },
        species="electron",
        cathode=True,
        charge=100e-12,
        sigma_x=1e-4,
        sigma_y=1e-4,
        sigma_t=3e-12,
        sigma_pz=1e3,
        gaussian_cutoff_x=3,
        gaussian_cutoff_y=3,
        gaussian_cutoff_z=3,
        distribution_type_x="g",
        distribution_type_pz="fd_300",
        number_of_particles=1000,
        e_photon=4.73,  # native RF-Track option, forwarded verbatim
    )


def test_code(generator):
    assert generator.code == "rftrack"


def test_write_maps_options_and_units(fake_rft, generator):
    generator.write()
    G = generator.gen_obj
    assert G.species == "electrons"
    assert G.cathode is True
    assert G.q_total == pytest.approx(100e-12 * 1e9)   # C -> nC
    assert G.sig_x == pytest.approx(1e-4 * 1e3)         # m -> mm
    assert G.sig_t == pytest.approx(3e-12 * 1e9)        # s -> ns
    assert G.c_sig_x == 3
    assert G.c_sig_t == 3                               # cathode longitudinal cutoff
    assert G.dist_x == "gaussian"                       # letter code translated
    assert G.dist_pz == "fd_300"                        # unknown string passed through
    assert G.e_photon == 4.73                           # native option forwarded


def test_run_emits_bunch(fake_rft, generator):
    generator.write()
    generator.run()
    assert isinstance(generator.bunch, _FakeBunch6dT)
    assert generator.bunch.gen is generator.gen_obj
    assert generator.bunch.npart == 1000


def test_postprocess_writes_beam(fake_rft, generator):
    generator.write()
    generator.run()
    generator.postProcess()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laser.openpmd.hdf5")
    assert os.path.isfile(out)
    os.remove(out)
