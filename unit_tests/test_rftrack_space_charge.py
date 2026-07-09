"""
Tests for RF-Track space-charge / cathode-mirror wiring. RF_Track is not
installed in CI, so these fake it. They cover:

- ``space_charge_engine`` grid sizing (same as ASTRA) + mirror activation,
- ``rftrackLattice._space_charge_settings`` reading the ASTRA ``charge`` block,
- ``rftrackLattice._setup_space_charge`` installing the engine / mirror.
"""
from types import SimpleNamespace

import pytest

from laura.translator.conversion_rules.codes import rftrack_conversion
from simba.Codes.RFTrack.RFTrack import rftrackLattice


class _FakeSC:
    def __init__(self, nx, ny, nz):
        self.mesh = (nx, ny, nz)
        self.mirror_z = None

    def set_mirror(self, z):
        self.mirror_z = z


@pytest.fixture
def fake_rft(monkeypatch):
    cvars = SimpleNamespace(SC_engine=None)
    fake = SimpleNamespace(SpaceCharge_PIC_FreeSpace=_FakeSC, cvars=cvars)
    monkeypatch.setattr(rftrack_conversion, "get_rftrack", lambda: fake)
    return fake


# --- space_charge_engine -----------------------------------------------------

def test_engine_grid_matches_astra(fake_rft):
    # 512 particles -> cube root 8 -> nearest power of two 8 (same as ASTRA).
    sc = rftrack_conversion.space_charge_engine(512)
    assert sc.mesh == (8, 8, 8)
    assert sc.mirror_z is None


def test_engine_sample_interval_shrinks_grid(fake_rft):
    # 512 / 8 = 64 -> cube root 4 -> grid 4.
    sc = rftrack_conversion.space_charge_engine(512, sample_interval=8)
    assert sc.mesh == (4, 4, 4)


def test_engine_mirror_activated(fake_rft):
    sc = rftrack_conversion.space_charge_engine(512, mirror_z=0.0)
    assert sc.mirror_z == 0.0


# --- _space_charge_settings --------------------------------------------------

def _lattice(charge=None, input_block=None, npart=512, start_z=0.0):
    return SimpleNamespace(
        file_block={
            **({"charge": charge} if charge is not None else {}),
            **({"input": input_block} if input_block is not None else {}),
        },
        globalSettings={"charge": None},
        global_parameters={"beam": SimpleNamespace(x=list(range(npart)))},
        startObject=SimpleNamespace(
            physical=SimpleNamespace(start=SimpleNamespace(z=start_z))
        ),
        lat_obj=SimpleNamespace(),
    )


def test_settings_disabled_by_default():
    s = rftrackLattice._space_charge_settings(_lattice())
    assert s["enabled"] is False
    assert s["cathode"] is False


def test_settings_enabled_and_cathode_mirror():
    s = rftrackLattice._space_charge_settings(
        _lattice(charge={"space_charge_mode": "2D", "cathode": True})
    )
    assert s["enabled"] is True
    assert s["cathode"] is True
    assert s["mirror"] is True  # defaults on for a cathode section


def test_settings_cathode_from_initial_distribution():
    s = rftrackLattice._space_charge_settings(
        _lattice(input_block={"particle_definition": "initial_distribution"})
    )
    assert s["cathode"] is True


def test_settings_mode_false_string_disabled():
    s = rftrackLattice._space_charge_settings(
        _lattice(charge={"space_charge_mode": "False"})
    )
    assert s["enabled"] is False


# --- _setup_space_charge -----------------------------------------------------

def test_setup_noop_when_disabled(fake_rft):
    latt = _lattice()
    rftrackLattice._setup_space_charge(_bind(latt))
    assert fake_rft.cvars.SC_engine is None


def test_setup_installs_engine_and_mirror(fake_rft):
    latt = _lattice(
        charge={"space_charge_mode": "3D", "cathode": True}, start_z=0.0
    )
    latt.lat_obj = SimpleNamespace(emission_nsteps=0, emission_range=0.0)
    rftrackLattice._setup_space_charge(_bind(latt))
    engine = fake_rft.cvars.SC_engine
    assert isinstance(engine, _FakeSC)
    assert engine.mirror_z == 0.0                 # mirror at cathode (§7.5)
    assert latt.lat_obj.emission_nsteps == 10     # emission options (§7.4)
    assert latt.lat_obj.emission_range == 2.0


def test_setup_no_mirror_for_non_cathode(fake_rft):
    latt = _lattice(charge={"space_charge_mode": "3D"})
    rftrackLattice._setup_space_charge(_bind(latt))
    assert fake_rft.cvars.SC_engine.mirror_z is None


def _bind(latt):
    """Attach the bound _space_charge_settings method used by _setup_space_charge."""
    latt._space_charge_settings = lambda: rftrackLattice._space_charge_settings(latt)
    return latt
