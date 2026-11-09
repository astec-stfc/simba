"""``longitudinal_match``: RF settings for a target energy and chirp (#26)."""

import os
import shutil
import warnings

import numpy as np
import pytest
from scipy.constants import c

import simba.Framework as fw
from helpers import read_beam, skip_missing
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, RFCavity
from simba.Codes.Generators import frameworkGenerator
from simba.Modules.Matching import rf_settings

F = 2998.5e6
K = 2 * np.pi * F / c


def forward(energy_in, systems):
    """The energy polynomial after ``systems``, ``[(k, voltage, phase [deg])]``, to third order."""
    out = np.array(energy_in, dtype=float)
    for k, voltage, phase in systems:
        x, y = voltage * np.cos(np.radians(phase)), voltage * np.sin(np.radians(phase))
        out += [x, -k * y, -(k**2) * x / 2, k**3 * y / 6]
    return out


@pytest.mark.parametrize("harmonic", [None, 3])
def test_rf_settings_inverts_the_rf(harmonic):
    energy_in = [100e6, 2e6, -3e8, 1e10]
    systems = [(K, 40e6, 17.0)] + ([(harmonic * K, 4e6, -170.0)] if harmonic else [])
    out = forward(energy_in, systems)
    h = out[1:] / out[0]
    solved = rf_settings(energy_in, F, out[0], *(h if harmonic else h[:1]), harmonic=harmonic)
    assert solved["main"] == pytest.approx((40e6, 17.0))
    if harmonic:
        assert solved["harmonic"] == pytest.approx((4e6, -170.0))


def test_on_crest_stays_on_crest():
    """The old code wrote 180 - phase: an on-crest match decelerated."""
    solved = rf_settings([100e6, 0, 0, 0], F, 120e6, 0.0)
    assert solved["main"] == pytest.approx((20e6, 0.0), abs=1e-6)


def test_curvature_needs_a_harmonic():
    with pytest.raises(ValueError, match="harmonic"):
        rf_settings([100e6, 0], F, 120e6, -5.0, curvature=10.0)


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    directory = str(tmp_path_factory.mktemp("seed"))
    frameworkGenerator(
        global_parameters={"master_subdir": directory},
        filename="M1.openpmd.hdf5", initial_momentum=100e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        # little uncorrelated spread: it is noise in the fitted skewness
        sigma_z=1e-3, sigma_pz=10,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=2000,
    ).write()
    return os.path.join(directory, "M1.openpmd.hdf5")


def cavity(name, z, length, phase, field_amplitude, frequency=F, **extra):
    return RFCavity(
        name=name, machine_area="L",
        physical={"length": length, "middle": {"x": 0.0, "y": 0.0, "z": z}},
        cavity={"frequency": frequency, "phase": phase, **extra},
        simulation={"field_amplitude": field_amplitude},
    )


def track(tmp_path, code, cavities, seed_beam, match=None, definitions=None, groups=None):
    """A fresh Framework: the match must load the beam itself (#26's first fault).
    ``definitions`` are the functional definitions, ``groups`` the settings' groups."""
    skip_missing(code)
    end = max(cv.physical.middle.z + cv.physical.length / 2 for cv in cavities) + 0.5
    elements = [
        Marker(name="M1", machine_area="L", hardware_class="Marker",
               physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}}),
        *cavities,
        Marker(name="M3", machine_area="L", hardware_class="Marker",
               physical={"middle": {"x": 0.0, "y": 0.0, "z": end}}),
    ]
    section = {"sections": {"L": [e.name for e in elements]}}
    machine = LAURA(
        element_list=elements,
        layout={"default_layout": "line1", "layouts": {"line1": ["L"]}},
        section=section,
        functional_definitions=dict(definitions or {}),
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    settings = fw.FrameworkSettings()
    settings.functional_definitions = dict(definitions or {})
    settings.groups = dict(groups or {})
    settings.files = {
        "L": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            **({"longitudinal_match": match} if match else {}),
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    framework["L"].lsc_enable = False
    framework["L"].csr_enable = False
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return framework, read_beam(framework.subdirectory, "M1"), read_beam(framework.subdirectory, "M3")


def shape(beam):
    """``(energy at the centre, [chirp, curvature, skewness])``, fitted to fifth order so
    the RF's own higher orders do not leak into them."""
    t, energy = np.asarray(beam.t.val), np.asarray(beam.energy.val)
    e = np.polyfit(c * (t.mean() - t), energy, 5)[::-1]
    return e[0], e[1:4] / e[0]


@pytest.mark.parametrize("code", ["ocelot", "xsuite", "elegant", "cheetah"])
def test_one_cavity_keeps_its_gain_and_takes_the_chirp(tmp_path, code, seed_beam):
    """The old code put degrees into cos and set the trough: -8.16 MeV, not +18.79."""
    _, before, after = track(
        tmp_path, code, [cavity("CAV", 1.0, 1.0, 20.0, 20e6)], seed_beam,
        {"cavities": ["CAV"], "chirp": -5.0},
    )
    (e0, _), (e1, h) = shape(before), shape(after)
    assert e1 - e0 == pytest.approx(20e6 * np.cos(np.radians(20.0)), rel=1e-3)
    assert h[0] == pytest.approx(-5.0, rel=1e-3)


def test_a_harmonic_linearises(tmp_path, seed_beam):
    cavities = [
        cavity("CAV", 1.0, 1.0, 20.0, 40e6),
        cavity("H3", 2.25, 0.5, 180.0, 5e6, frequency=3 * F),
    ]
    match = {"cavities": ["CAV"], "harmonics": ["H3"], "chirp": -5.0}
    _, before, after = track(tmp_path / "match", "ocelot", cavities, seed_beam, match)
    _, _, plain = track(tmp_path / "plain", "ocelot", cavities, seed_beam)
    (e0, _), (e1, h), (_, unmatched) = shape(before), shape(after), shape(plain)
    gain = 40e6 * np.cos(np.radians(20.0)) - 5e6
    assert e1 - e0 == pytest.approx(gain, rel=1e-3)
    assert h[0] == pytest.approx(-5.0, rel=1e-3)
    assert abs(h[1]) < 1e-2 * abs(unmatched[1])
    assert abs(h[2]) < 1e-2 * abs(unmatched[2])


@pytest.mark.parametrize("code", ["ocelot", "elegant"])
def test_a_travelling_wave_cavity_is_set_by_its_gradient(tmp_path, code, seed_beam):
    """field_amplitude is a gradient there: the old code took it as the voltage.
    elegant tracks it with its own travelling-wave element."""
    tw = {"structure_type": "TravellingWave", "n_cells": 30, "cell_length": 0.0333,
          "mode_numerator": 2, "mode_denominator": 3}
    cavities = [cavity("CAV", 1.0, 1.0, 20.0, 20e6, **tw)]
    _, before, plain = track(tmp_path / "plain", code, cavities, seed_beam)
    _, _, after = track(tmp_path / "match", code,
                        [cavity("CAV", 1.0, 1.0, 20.0, 20e6, **tw)], seed_beam,
                        {"cavities": ["CAV"], "chirp": -5.0})
    (e0, _), (e_plain, _), (e1, h) = shape(before), shape(plain), shape(after)
    assert e1 - e0 == pytest.approx(e_plain - e0, rel=1e-3)
    assert h[0] == pytest.approx(-5.0, rel=1e-3)


def test_harmonics_must_be_a_harmonic(tmp_path, seed_beam):
    cavities = [cavity("CAV", 1.0, 1.0, 20.0, 20e6), cavity("H", 2.25, 0.5, 0.0, 1e6, frequency=2.5 * F)]
    with pytest.raises(ValueError, match="not a harmonic"):
        track(tmp_path, "ocelot", cavities, seed_beam, {"cavities": ["CAV"], "harmonics": ["H"], "chirp": 0.0})


def test_an_unknown_cavity_is_refused(tmp_path, seed_beam):
    with pytest.raises(KeyError, match="NOPE"):
        track(tmp_path, "ocelot", [cavity("CAV", 1.0, 1.0, 20.0, 20e6)], seed_beam,
              {"cavities": ["NOPE"], "chirp": 0.0})


def test_linked_cavities_keep_their_definitions(tmp_path, seed_beam):
    """One phase and one gradient for a linac: the definitions are matched, not overwritten."""
    cavities = [cavity(f"C{i}", 0.75 + i, 0.5, "linac_phase", "linac_gradient") for i in range(2)]
    definitions = {"linac_phase": 20.0, "linac_gradient": 20e6}
    framework, before, after = track(
        tmp_path, "ocelot", cavities, seed_beam,
        {"cavities": ["C0", "C1"], "chirp": -5.0}, definitions,
    )
    for name in ("C0", "C1"):
        element = framework["L"].elements[name]
        assert (element.cavity.phase, element.simulation.field_amplitude) == ("linac_phase", "linac_gradient")
    matched = framework.settings["functional_definitions"]
    assert matched["linac_phase"] != pytest.approx(20.0)
    (e0, _), (e1, h) = shape(before), shape(after)
    assert e1 - e0 == pytest.approx(2 * 20e6 * np.cos(np.radians(20.0)), rel=1e-3)
    assert h[0] == pytest.approx(-5.0, rel=1e-3)


def test_a_definition_shared_outside_the_match_is_refused(tmp_path, seed_beam):
    cavities = [cavity(f"C{i}", 0.75 + i, 0.5, "linac_phase", 20e6) for i in range(3)]
    with pytest.raises(ValueError, match="C2"):
        track(tmp_path, "ocelot", cavities, seed_beam,
              {"cavities": ["C0", "C1"], "chirp": -5.0}, {"linac_phase": 20.0})


def test_a_definition_cannot_take_two_values(tmp_path, seed_beam):
    cavities = [cavity("CAV", 1.0, 1.0, "phase", 40e6),
                cavity("H3", 2.25, 0.5, "phase", 5e6, frequency=3 * F)]
    with pytest.raises(ValueError, match="both"):
        track(tmp_path, "ocelot", cavities, seed_beam,
              {"cavities": ["CAV"], "harmonics": ["H3"], "chirp": -5.0}, {"phase": 20.0})


def test_a_group_names_the_cavities(tmp_path, seed_beam):
    """Groups from the settings, nested, in place of the cavity names. C2 is outside
    the group: the match leaves it be, and allows for it."""
    cavities = [cavity(f"C{i}", 0.75 + i, 0.5, 20.0, 20e6) for i in range(3)]
    groups = {
        "half1": {"type": "element_group", "elements": ["C0"]},
        "linac": {"type": "element_group", "elements": ["half1", "C1"]},
    }
    framework, before, after = track(
        tmp_path, "ocelot", cavities, seed_beam, {"cavities": "linac", "chirp": -5.0}, groups=groups,
    )
    phases = [framework["L"].elements[f"C{i}"].cavity.phase for i in range(3)]
    assert phases[0] == phases[1] != pytest.approx(20.0)
    assert phases[2] == pytest.approx(20.0)  # not in the group
    (e0, _), (e1, h) = shape(before), shape(after)
    assert e1 - e0 == pytest.approx(3 * 20e6 * np.cos(np.radians(20.0)), rel=1e-3)
    assert h[0] == pytest.approx(-5.0, rel=1e-3)


@pytest.mark.parametrize("where", ["before", "after"])
def test_a_fixed_cavity_is_allowed_for_wherever_it_is(tmp_path, seed_beam, where):
    """A cavity left out of the match, upstream or downstream of the matched one."""
    z = {"before": (2.0, 1.0), "after": (1.0, 2.0)}[where]
    cavities = [cavity("CAV", z[0], 0.5, 20.0, 20e6), cavity("FIXED", z[1], 0.5, -30.0, 10e6)]
    framework, before, after = track(tmp_path, "ocelot", cavities, seed_beam,
                                     {"cavities": ["CAV"], "chirp": -5.0, "energy_gain": 25e6})
    assert framework["L"].elements["FIXED"].cavity.phase == pytest.approx(-30.0)
    (e0, _), (e1, h) = shape(before), shape(after)
    assert e1 - e0 == pytest.approx(25e6, rel=1e-3)
    assert h[0] == pytest.approx(-5.0, rel=1e-3)


def test_a_group_member_that_is_not_a_cavity_is_refused(tmp_path, seed_beam):
    groups = {"linac": {"type": "element_group", "elements": ["CAV", "M3"]}}
    with pytest.raises(ValueError, match="M3"):
        track(tmp_path, "ocelot", [cavity("CAV", 1.0, 1.0, 20.0, 20e6)], seed_beam,
              {"cavities": "linac", "chirp": -5.0}, groups=groups)
