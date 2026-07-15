"""
Tests for RF-Track ``Volume`` (Bunch6dT / time-integration) tracking and the
Lattice/Volume switch. RF_Track is faked (not installed in CI). Covers:

- ``beam_to_bunch6dt`` full-matrix conversion (carries emission time T0),
- ``rftrackLattice._tracking_type`` resolution (auto/cathode/explicit),
- ``_setup_space_charge`` Volume branch (sc_dt_mm + emission + mirror),
- ``to_rftrack_volume`` wrapping the lattice in a Volume.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from laura.translator.conversion_rules.codes import rftrack_conversion
from simba.Modules import constants
from simba.Codes.RFTrack.RFTrack import rftrackLattice


# --- beam_to_bunch6dt --------------------------------------------------------

class _U:
    """Minimal stand-in for a UnitValue holding an array."""

    def __init__(self, arr):
        self.val = np.asarray(arr, dtype=float)

    def in_units_of(self, unit):
        assert unit == "milli"
        return self.val * 1e3  # m -> mm


class _FakeBunch6dT:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)


@pytest.fixture
def fake_rft_beams(monkeypatch):
    fake = SimpleNamespace(Bunch6dT=_FakeBunch6dT)
    monkeypatch.setattr(rftrack_conversion, "get_rftrack", lambda: fake)
    return fake


def _fake_beam(n=4):
    return SimpleNamespace(
        x=_U(np.full(n, 1e-3)),   # 1 mm
        y=_U(np.full(n, 2e-3)),
        z=_U(np.zeros(n)),        # cathode plane
        px=_U(np.zeros(n)),
        py=_U(np.zeros(n)),
        pz=_U(np.full(n, 5e-19)),  # some SI momentum
        t=_U(np.full(n, 1e-12)),   # 1 ps emission time
        Q=_U(np.full(n, -1e-11)),  # total charge samples
        particle_mass=np.full(n, constants.m_e),
    )


def test_beam_to_bunch6dt_matrix(fake_rft_beams):
    from simba.Modules.Beams import rftrack as rbf_rftrack

    b = rbf_rftrack.beam_to_bunch6dt(_fake_beam(4))
    assert isinstance(b, _FakeBunch6dT)
    m = b.matrix
    assert m.shape == (4, 10)                          # full [X Px Y Py Z Pz M Q N T0]
    np.testing.assert_allclose(m[:, 0], 1.0)           # X in mm
    np.testing.assert_allclose(m[:, 2], 2.0)           # Y in mm
    # T0 (last col) = t[s] * c * 1e3  (mm/c)
    np.testing.assert_allclose(m[:, 9], 1e-12 * constants.speed_of_light * 1e3)
    np.testing.assert_allclose(m[:, 7], -1.0)          # Q = charge state [e]


# --- _tracking_type ----------------------------------------------------------

def _lattice(charge=None, tracking_type="auto", input_block=None):
    return SimpleNamespace(
        tracking_type=tracking_type,
        file_block={
            **({"charge": charge} if charge is not None else {}),
            **({"input": input_block} if input_block is not None else {}),
        },
        globalSettings={"charge": None},
        global_parameters={"beam": SimpleNamespace(x=list(range(512)))},
        startObject=SimpleNamespace(
            physical=SimpleNamespace(start=SimpleNamespace(z=0.0))
        ),
        lat_obj=SimpleNamespace(),
    )


def _tt(latt):
    latt._space_charge_settings = lambda: rftrackLattice._space_charge_settings(latt)
    return rftrackLattice._tracking_type(latt)


def test_tracking_type_auto_lattice_for_plain_section():
    assert _tt(_lattice()) == "lattice"


def test_tracking_type_auto_volume_for_cathode():
    assert _tt(_lattice(charge={"cathode": True})) == "volume"


def test_tracking_type_explicit_field_override():
    # Explicit field beats the auto/cathode default.
    assert _tt(_lattice(charge={"cathode": True}, tracking_type="lattice")) == "lattice"
    assert _tt(_lattice(tracking_type="volume")) == "volume"


def test_tracking_type_charge_block_override():
    assert _tt(_lattice(charge={"tracking": "volume"})) == "volume"
    assert _tt(_lattice(charge={"tracking": "bunch6d"})) == "lattice"


# --- _setup_space_charge (Volume branch) -------------------------------------

class _FakeSC:
    def __init__(self, nx, ny, nz):
        self.mesh = (nx, ny, nz)
        self.mirror_z = None

    def set_mirror(self, z):
        self.mirror_z = z


@pytest.fixture
def fake_rft_sc(monkeypatch):
    cvars = SimpleNamespace(SC_engine=None)
    fake = SimpleNamespace(SpaceCharge_PIC_FreeSpace=_FakeSC, cvar=cvars)
    monkeypatch.setattr(rftrack_conversion, "get_rftrack", lambda: fake)
    return fake


def _bind(latt):
    latt._space_charge_settings = lambda: rftrackLattice._space_charge_settings(latt)
    latt._tracking_type = lambda: rftrackLattice._tracking_type(latt)
    return latt


def test_setup_volume_sc_and_emission(fake_rft_sc):
    latt = _lattice(charge={"space_charge_mode": "3D", "cathode": True})
    latt.lat_obj = SimpleNamespace(sc_dt_mm=None, emission_nsteps=None, emission_range=None)
    rftrackLattice._setup_space_charge(_bind(latt))
    engine = fake_rft_sc.cvar.SC_engine
    assert isinstance(engine, _FakeSC)
    assert engine.mirror_z == 0.0            # cathode mirror (§7.5)
    assert latt.lat_obj.sc_dt_mm == 1.0      # Volume SC kick interval (§5.1.1)
    assert latt.lat_obj.emission_nsteps == 10
    assert latt.lat_obj.emission_range == 2.0


# --- to_rftrack_volume -------------------------------------------------------

class _FakeVolume:
    def __init__(self):
        self.added = []

    def add(self, obj, x, y, z):
        self.added.append((obj, x, y, z))


def test_to_rftrack_volume_wraps_lattice(monkeypatch):
    from laura.translator.converters.section import SectionLatticeTranslator

    fake = SimpleNamespace(Volume=_FakeVolume)
    monkeypatch.setattr(
        "laura.translator.conversion_rules.codes.rftrack_conversion.get_rftrack",
        lambda: fake,
    )
    sentinel_lattice = object()
    dummy = SimpleNamespace(
        to_rftrack=lambda P_Q, save, sc_nsteps: sentinel_lattice
    )
    vol = SectionLatticeTranslator.to_rftrack_volume(dummy, P_Q=1.0, save=False)
    assert isinstance(vol, _FakeVolume)
    assert vol.added == [(sentinel_lattice, 0.0, 0.0, 0.0)]
