from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
from scipy.constants import c

from simba.Codes.Genesis.Genesis import (
    genesis_alter_beam_command,
    genesis_profile_file_command,
    genesis_profile_polynom_command,
    genesis_setup_command,
    genesis_sort_command,
    genesis_sponrad_command,
    genesis_track_command,
    genesisLattice,
)


def _el(name, start, end):
    return NS(name=name, physical=NS(start=NS(z=start), end=NS(z=end)), magnetic=NS(length=end - start))


def test_chicane_uses_physical_positions():
    group = NS(elementObjects=[_el("D1", 0, 0.2), _el("D2", 1.2, 1.4), _el("D4", 3.0, 3.2)], r56=1e-4)
    fake = NS(
        chicanes="CH", groupObjects={"CH": group}, split_element=None,
        section=NS(to_genesis=lambda split_element, chicanes: chicanes),
    )
    ch = genesisLattice.writeElements(fake)["CH"]
    assert ch["drift_length"] == 1.0 and ch["length"] == 3.2


def _seeded(tmp_path, steady_state, **laser):
    from laura.models.laser import LaserElement
    fake = NS(
        wigglers=[NS(name="MOD", laser=LaserElement(wavelength=250e-9, waist=6e-4, pulse_energy=2e-5, **laser))],
        steady_state=steady_state, commandFiles={}, files=[], field_power=1e3,
        global_parameters={"master_subdir": str(tmp_path)}, bunch_centre_s=lambda: 1e-4,
    )
    return fake, genesisLattice.get_field_power(fake)


def test_seed_laser_centred_on_the_bunch(tmp_path):
    """initial_position is from the bunch centroid, towards the head (larger Genesis s)."""
    fake, label = _seeded(tmp_path, False, pulse_duration_fwhm=1e-12, initial_position=2e-5)
    gauss = fake.commandFiles["profile_gauss"]
    assert label == "@MOD_laser_profile" and gauss.c0 == pytest.approx(fake.wigglers[0].laser.peak_power)
    assert gauss.s0 == pytest.approx(1.2e-4) and gauss.sig == pytest.approx(1e-12 * c / 2.35482, rel=1e-5)
    _, power = _seeded(tmp_path, True, pulse_duration_fwhm=1e-12, initial_position=gauss.sig)
    assert power == pytest.approx(gauss.c0 * np.exp(-0.5))  # steady state: the power at the centroid


def test_seed_laser_time_profile(tmp_path):
    """Ex(t) (t towards the tail) becomes a power table, zero beyond its ends."""
    from laura.translator.utils.fields import FieldMap
    from laura.translator.utils.fields.field_parameter import set_field
    t = np.linspace(-3e-12, 3e-12, 61)
    profile = FieldMap()
    profile.reset_dicts()
    profile.field_type, profile.read = "LaserTimeProfile", True
    set_field(profile, "t", t, "s")
    set_field(profile, "Ex", np.exp(-((t - 1e-12) ** 2) / (2e-24)), "")
    fake, label = _seeded(tmp_path, False, field_definition=profile)
    with h5py.File(tmp_path / "MOD_laser_profile.h5") as f:
        s, power = f["s"][()], f["power"][()]
    assert fake.commandFiles["profile_file"][0].ydata == "MOD_laser_profile.h5/power"
    assert power[0] == power[-1] == 0 and np.all(np.diff(s) > 0)
    assert s[np.argmax(power)] == pytest.approx(1e-4 - c * 1e-12)  # a later t is further towards the tail
    assert power.max() == pytest.approx(fake.wigglers[0].laser.peak_power)


def test_lambda_is_written():
    assert "lambda = 2e-09" in genesis_alter_beam_command(**{"lambda": 2e-9}).write_Genesis()


def test_setup_seed_per_instance_and_bunchharm_one():
    kw = {"rootname": "a", "lattice": "b", "beamline": "c", "gamma0": 1, "lambda0": 1, "delz": 1, "npart": 1, "nbins": 1}
    seeds = {genesis_setup_command(**kw).seed for _ in range(5)}
    assert len(seeds) > 1
    assert "seed = 7" in genesis_setup_command(seed=7, **kw).write_Genesis()
    assert genesis_track_command(bunchharm=1).bunchharm == 1


def test_camelcase_and_profile_keywords_are_written():
    out = genesis_sponrad_command(doLoss=True).write_Genesis()
    assert "doLoss = 1" in out and "doSpread = 0" in out
    out = genesis_profile_file_command(label="p", xdata="f/x", ydata="f/y", isTime=True).write_Genesis()
    assert "isTime = 1" in out and "reverse = 0" in out
    assert "c3 = 2.0" in genesis_profile_polynom_command(label="p", c0=1.0, c3=2.0).write_Genesis()
    assert genesis_sort_command().write_Genesis() == "&sort\n&end\n"
