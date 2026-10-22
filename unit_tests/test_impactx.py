"""ImpactX, tracked in-process: against Xsuite on the same lines, and crested field maps."""

import os
import shutil
import warnings
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("impactx")

from helpers import fodo_machine, fodo_quads, read_beam, skip_missing
from laura.models.element import Dipole, HorizontalCorrector, Marker, Solenoid, VerticalCorrector
from laura.translator.utils.fields import FieldMap
from laura.translator.utils.fields.field_parameter import FieldParameter
from laura.translator.utils.units import UnitValue

import simba.Framework as fw
from simba.Codes.Generators import frameworkGenerator
from simba.Modules.Beams import impactx as rbf_impactx
from simba.Modules.Twiss import twiss

CAVITY_LENGTH = 1.5 * 299792458.0 / 1.3e9


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    """One bunch for every run: ``frameworkGenerator`` is unseeded."""
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3, sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=5000,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def field_map(tmp_path):
    """A 3-cell pi-mode standing-wave map at 1.3 GHz, as LAURA's own test writes it."""
    z = np.linspace(0, CAVITY_LENGTH, 301)
    field = FieldMap()
    field.filename = str(tmp_path / "cavity.hdf5")
    field.field_type, field.cavity_type, field.frequency = "1DElectroDynamic", "StandingWave", 1.3e9
    field.z = FieldParameter(name="z", value=UnitValue(z, "m"))
    field.Ez = FieldParameter(name="Ez", value=UnitValue(np.sin(3 * np.pi * z / z[-1]), "V/m"))
    field.read = True
    return field.write_field_file(code="hdf5")


def run(tmp_path, seed_beam, code="impactx", charge="False", cavity=False, tracking=None, middle=None, closed=False):
    machine, _, section = fodo_machine(
        tmp_path, cavity=cavity, cavity_length=CAVITY_LENGTH if cavity else 0.0, middle=middle,
        closed=closed,
    )
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": charge},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            "tracking": tracking or {},
            "lsc_enable": False,
            "csr_enable": False,
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return framework, read_beam(framework.subdirectory, "M3")


def test_a_beam_survives_the_round_trip_less_its_lost_particles(seed_beam):
    """Particle 3 lost: the rest keep their order, and the reference its place."""
    import pandas as pd

    beam = read_beam(os.path.dirname(seed_beam), "M1")
    p0c, t0 = 5e6, float(np.mean(beam.t.val))
    coords = rbf_impactx.write_impactx_beam(beam, p0c, t0)
    ids = np.arange(len(beam.x.val)) + 100
    df = pd.DataFrame({
        "idcpu": ids,
        **{f"position_{c}": coords[c] for c in "xyt"},
        **{f"momentum_{c}": coords[f"p{c}"] for c in "xyt"},
    }).drop(index=3)
    out = deepcopy(beam)
    rbf_impactx.read_impactx_beam(out, df, ids, p0c, t0, ref_index=5)
    keep = np.arange(len(ids)) != 3
    for coord in ("x", "y", "t", "cpx", "cpy", "cpz"):
        np.testing.assert_allclose(getattr(out, coord).val, getattr(beam, coord).val[keep], rtol=1e-12, atol=1e-18)
    assert out.reference_particle_index == 4
    assert float(out._beam.total_charge) == pytest.approx(float(beam._beam.total_charge) * keep.mean())


def test_impactx_and_xsuite_agree_particle_by_particle(tmp_path, seed_beam):
    """To 1e-5 of the beam size, their quadrupoles' difference. Xsuite's ``t`` is
    ``s / c`` on; ImpactX's is the reference's time of flight, as elegant's."""
    skip_missing("xsuite")
    _, impactx = run(tmp_path / "impactx", seed_beam)
    _, xsuite = run(tmp_path / "xsuite", seed_beam, code="xsuite")
    for coord, atol in (("x", 5e-9), ("y", 5e-9), ("z", 1e-8), ("cpx", 0.05), ("cp", 1e-6)):
        np.testing.assert_allclose(
            getattr(impactx, coord).val, getattr(xsuite, coord).val, rtol=0, atol=atol
        )
    beta0 = 5e6 / np.hypot(5e6, 510998.95)
    flight = np.mean(impactx.t.val) - np.mean(read_beam(os.path.dirname(seed_beam), "M1").t.val)
    assert flight == pytest.approx(4.25 / (beta0 * 299792458.0), rel=1e-6)


def test_space_charge_grows_a_round_bunch_as_the_envelope_equation_says(tmp_path):
    """1 mm, cold, 100 pC at 5 MeV/c over 2 m becomes 1.496 mm (see
    test_xsuite_space_charge); ImpactX's grid gives 1.467, Xsuite 1.439."""
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-3, sigma_px=1, sigma_y=1e-3, sigma_py=1, sigma_z=1e-3, sigma_pz=1,
        gaussian_cutoff_x=5, gaussian_cutoff_y=5, gaussian_cutoff_z=5,
        gaussian_cutoff_px=5, gaussian_cutoff_py=5, gaussian_cutoff_pz=5,
        charge=100e-12, number_of_particles=20000,
    ).write()
    seed = str(tmp_path / "M1.openpmd.hdf5")
    beam = read_beam(str(tmp_path), "M1")
    drift = [Marker(name="MID", machine_area="FODO", hardware_class="Marker", physical={"middle": {"x": 0.0, "y": 0.0, "z": 2.0}})]
    growth = {}
    for charge in ("3d", "False"):
        _, out = run(tmp_path / charge, seed, charge=charge, middle=drift)
        growth[charge] = np.std(out.x.val) / np.std(beam.x.val)
    assert growth["3d"] == pytest.approx(1.496, rel=0.03)
    assert growth["False"] == pytest.approx(1, abs=1e-3)


def test_its_twiss_file_is_read_with_the_beam_optics(tmp_path, seed_beam):
    skip_missing("xsuite")
    framework, _ = run(tmp_path / "impactx", seed_beam)
    run(tmp_path / "xsuite", seed_beam, code="xsuite")
    impactx = twiss.initialise_directory(framework.subdirectory)
    xsuite = twiss.initialise_directory(str(tmp_path / "xsuite"))
    assert impactx.element_name.val[-1] == "M3"
    for key in ("beta_x", "beta_y", "alpha_x", "alpha_y", "enx", "sigma_x"):
        assert getattr(impactx, key).val[-1] == pytest.approx(getattr(xsuite, key).val[-1], rel=0.02)


def test_a_cavity_is_thin_unless_field_maps_are_asked_for(tmp_path, seed_beam):
    cavity = {"cavity": {"frequency": 1.3e9}, "simulation": {"field_amplitude": 20e6, "field_definition": field_map(tmp_path)}}
    thin, _ = run(tmp_path / "thin", seed_beam, cavity=cavity)
    thick, _ = run(tmp_path / "thick", seed_beam, cavity=cavity, tracking={"field_maps": True})
    assert "ShortRF" in [d["type"] for d in thin["FODO"].dicts]
    assert "RFCavity" in [d["type"] for d in thick["FODO"].dicts]


def test_autophase_puts_a_field_map_cavity_on_crest(tmp_path, seed_beam):
    """Its phase is absolute, so the crest comes from the reference's time of
    flight: off it either way, the beam gains less."""
    path = field_map(tmp_path)
    energies = {}
    for phase in (-10.0, 0.0, 10.0):
        cavity = {
            "cavity": {"frequency": 1.3e9, "phase": phase},
            "simulation": {"field_amplitude": 20e6, "field_definition": path},
        }
        _, beam = run(tmp_path / str(phase), seed_beam, cavity=cavity, tracking={"field_maps": True})
        energies[phase] = np.mean(beam.cp.val)
    # 20 MV/m times |int sin(kz) exp(ikz) dz| = L / 2, less a little slip at 5 MeV/c
    assert energies[0.0] - 5e6 == pytest.approx(20e6 * CAVITY_LENGTH / 2, rel=0.01)
    assert energies[0.0] > energies[-10.0] and energies[0.0] > energies[10.0]
    for phase in (-10.0, 10.0):
        assert (energies[phase] - 5e6) / (energies[0.0] - 5e6) == pytest.approx(np.cos(np.radians(10)), rel=2e-3)


def test_a_corrector_program_kicks_as_xsuites(tmp_path, seed_beam):
    """``xkick``/``ykick`` with Xsuite's signs: a pulse on turn 2 in each plane."""
    skip_missing("xsuite")
    correctors = [
        cls(name=name, machine_area="FODO", physical={"length": 0.0, "middle": {"x": 0.0, "y": 0.0, "z": z}})
        for cls, name, z in ((HorizontalCorrector, "HCOR", 4.0), (VerticalCorrector, "VCOR", 4.1))
    ]
    programs = [
        {"element": "HCOR", "turns": [1, 2, 3], "values": [0.0, 1e-4, 0.0]},
        {"element": "VCOR", "turns": [1, 2, 3], "values": [0.0, -2e-4, 0.0]},
    ]
    means = {}
    for code in ("impactx", "xsuite"):
        framework, _ = run(
            tmp_path / code, seed_beam, code=code, closed=True,
            middle=fodo_quads() + deepcopy(correctors),
            tracking={"turns": 4, "write_turns": True, "programs": programs},
        )
        means[code] = np.array([
            [np.mean(getattr(read_beam(framework.subdirectory, "M3", turn), u).val) for u in "xy"]
            for turn in range(1, 5)
        ])
    assert np.abs(means["xsuite"][2] - means["xsuite"][0]).min() > 1e-5
    # measured 1.2e-9 m on 4e-4 m
    np.testing.assert_allclose(means["impactx"], means["xsuite"], rtol=0, atol=1e-8)


def test_a_solenoid_follows_a_ramp_as_xsuites_does(tmp_path, seed_beam):
    """ImpactX's exact solenoid is a ``ChrAcc``, whose ``bz`` is a field, not a strength."""
    skip_missing("xsuite")
    solenoid = Solenoid(
        name="SOL1", machine_area="FODO",
        magnetic={"length": 0.5, "fields": {"S0L": 0.5}},
        physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 1.75}},
    )
    quads = fodo_quads()
    final = {}
    for code in ("impactx", "xsuite"):
        _, beam = run(
            tmp_path / code, seed_beam, code=code, closed=True,
            middle=[quads[0], deepcopy(solenoid), quads[1]],
            tracking={"turns": 10, "ramp": {"turns": [1, 10], "momentum": [5e6, 5.1e6]}},
        )
        final[code] = np.array([beam.x.val, beam.y.val])
    np.testing.assert_allclose(final["impactx"], final["xsuite"], rtol=0, atol=1e-7)


def test_the_reference_trajectory_is_xsuites(tmp_path, seed_beam):
    skip_missing("xsuite")
    trajectory = {
        code: run(tmp_path / code, seed_beam, code=code, closed=True, tracking={"turns": 8})[0][
            "FODO"
        ].track_reference_particle()
        for code in ("impactx", "xsuite")
    }
    for name in ("x", "px", "y", "py"):
        assert len(trajectory["impactx"][name]) == 8
        np.testing.assert_allclose(trajectory["impactx"][name], trajectory["xsuite"][name], rtol=0, atol=1e-9)


def test_a_rings_optics_are_xsuites(tmp_path, seed_beam):
    """Its one-turn map, closed orbit, tunes and periodic twiss, all from probes."""
    import pandas as pd

    skip_missing("xsuite")
    lines, optics = {}, {}
    for code, csv in (("impactx", "FODO_twiss.impactx.csv"), ("xsuite", "FODO_twiss.csv")):
        framework, _ = run(tmp_path / code, seed_beam, code=code, closed=True)
        lines[code] = framework["FODO"]
        # by s: Xsuite names a row by the element it enters, ImpactX by the one it leaves
        optics[code] = pd.read_csv(os.path.join(framework.subdirectory, csv)).drop_duplicates("s").set_index("s")
    ours, theirs = lines["impactx"], lines["xsuite"]
    np.testing.assert_allclose(ours.one_turn_map, theirs.one_turn_map, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ours.closed_orbit, 0, rtol=0, atol=1e-15)
    for u in "xy":
        assert ours.optics_summary[f"tune_{u}_total"] == pytest.approx(
            theirs.optics_summary[f"tune_{u}_total"], rel=1e-9
        )
        assert ours.optics_summary[f"chromaticity_{u}"] == pytest.approx(
            theirs.optics_summary[f"chromaticity_{u}"], rel=1e-4
        )
        for column in (f"bet{u}", f"alf{u}", f"mu{u}"):
            np.testing.assert_allclose(optics["impactx"][column], optics["xsuite"][column], rtol=1e-6, atol=1e-9)


def test_quantum_radiation_takes_a_bends_energy_loss(tmp_path):
    """5 GeV through 1 m bent 0.1 rad loses ``C_gamma E^4 theta / (2 pi rho)``, 88 keV, on average."""
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
        filename="M1.openpmd.hdf5", initial_momentum=5e9,
        sigma_x=1e-5, sigma_px=1e3, sigma_y=1e-5, sigma_py=1e3, sigma_z=1e-4, sigma_pz=1e3,
        charge=1e-12, number_of_particles=2000,
    ).write()
    seed = str(tmp_path / "M1.openpmd.hdf5")
    bend = [Dipole(
        name="B1", machine_area="FODO", hardware_class="Magnet",
        magnetic={"magnetic_length": 1.0, "k0l": 0.1},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.5}},
    )]
    start = np.mean(read_beam(str(tmp_path), "M1").cp.val)
    loss = {
        model: start - np.mean(run(tmp_path / model, seed, middle=deepcopy(bend), tracking={"radiation": model})[1].cp.val)
        for model in ("quantum", "off")
    }
    energy = np.hypot(5.0, 0.51099895e-3)
    assert loss["quantum"] == pytest.approx(8.846e-5 * energy**4 * 0.1 / (2 * np.pi * 10.0) * 1e9, rel=0.05)
    assert loss["off"] == pytest.approx(0, abs=1.0)
