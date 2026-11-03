"""Electrons and positrons tracked together, end to end through Framework.track, in each
code that can (Bmad, ASTRA, GPT, Xsuite)."""

import importlib.util
import os
import shutil
import warnings

import numpy as np
import pytest
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole

import simba.Framework as fw
import simba.Modules.Beams as rbf
from simba import exceptions
from simba.Codes.Generators import frameworkGenerator

BMAD_SO = os.path.expanduser("~/Documents/bmad-ecosystem/production/lib/libtao.so")

needs_bmad = pytest.mark.skipif(not os.path.exists(BMAD_SO), reason="Bmad libtao not installed")

CODES = [
    pytest.param("xsuite", marks=pytest.mark.skipif(
        importlib.util.find_spec("xtrack") is None, reason="xtrack not installed")),
    pytest.param("bmad", marks=needs_bmad),
    pytest.param("astra", marks=pytest.mark.needs_container),
    pytest.param("gpt", marks=pytest.mark.skipif(
        "GPTLICENSE" not in os.environ, reason="requires a local GPT install and GPTLICENSE env var")),
]

TOLERANCE = {"xsuite": 1e-6, "bmad": 1e-3, "gpt": 3e-2, "astra": 1.5e-1}
"""How far each code's x at M3 may be from the thick-quadrupole linear map, as a fraction of
the beam size: Xsuite and Bmad are that map; GPT's quadrupoles have fringe fields; ASTRA's
are hard-edged and it tracks in time (measured: 1.4%, 6%)."""

K1 = 10.0
"""Strength of the first quadrupole [m^-2]; 0.2 m long, so k1l = 2."""


def _machine(tmp_path):
    """M1, 0.4 m drift, Q1 (0.2 m), 0.4 m drift, MID, 0.4 m, Q2 = -Q1, 0.4 m, M3."""
    def marker(name, z):
        return Marker(name=name, machine_area="LINE", hardware_class="Marker",
                      physical={"middle": {"x": 0.0, "y": 0.0, "z": z}})

    def quad(name, z, k1l):
        return Quadrupole(name=name, machine_area="LINE", magnetic={"length": 0.2, "k1l": k1l},
                          physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": z}})

    elements = [marker("M1", 0.0), quad("Q1", 0.5, K1 * 0.2), marker("MID", 1.0),
                quad("Q2", 1.5, -K1 * 0.2), marker("M3", 2.0)]
    section = {"sections": {"LINE": {"elements": [e.name for e in elements]}}}
    machine = LAURA(element_list=elements,
                    layout={"default_layout": "line1", "layouts": {"line1": ["LINE"]}},
                    section=section)
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, section


@pytest.fixture(scope="module")
def electrons(tmp_path_factory):
    """Electrons at 5 MeV/c; generated once, as ``frameworkGenerator`` is unseeded."""
    directory = tmp_path_factory.mktemp("electrons")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="electrons.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-3, sigma_px=1e3, sigma_y=1e-3, sigma_py=1e3,
        sigma_z=1e-4, sigma_pz=1e4, charge=1e-15, number_of_particles=50,
    ).write()
    return os.path.join(str(directory), "electrons.openpmd.hdf5")


def _seed(electrons, directory, others=()):
    """The electrons, with copies of them as the given species alongside."""
    os.makedirs(directory, exist_ok=True)
    beam = _read(electrons)
    beam.set_species("electron")
    for species in others:
        other = _read(electrons)
        other.set_species(species)
        beam.others[species] = other
    path = os.path.join(str(directory), "seed.openpmd.hdf5")
    beam.write_openpmd_beam_file(path)
    return path


def _read(path):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, path)
    return beam


def _run(tmp_path, code, seed, inputs=None):
    """Track ``seed`` from M1 to M3; the run's subdirectory."""
    machine, section = _machine(tmp_path)
    settings = fw.FrameworkSettings()
    settings.files = {"LINE": {
        "code": code, "charge": {"space_charge_mode": "False"}, "input": inputs or {},
        "output": {"start_element": "M1", "end_element": "M3"},
        "tracking": {"context": "cpu"}, "lsc_enable": False, "csr_enable": False,
    }}
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(
        machine=machine, directory=str(tmp_path), clean=True, verbose=False,
        container_runtime="docker" if code == "astra" else None,
    )
    framework.loadSettings(settings=settings)
    shutil.copy(seed, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    if code == "bmad":
        framework["LINE"].libtao = BMAD_SO
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return framework.subdirectory


def _x_linear(beam, k1, to="MID"):
    """
    Linear x at ``to`` (MID or M3) from ``beam`` at M1, with Q1 of strength ``k1`` (and
    Q2 of -k1) at the mean momentum.
    """
    def quad(x, xp, k):  # 0.2 m thick quadrupole
        root = np.sqrt(np.abs(k))
        c = np.where(k > 0, np.cos(root * 0.2), np.cosh(root * 0.2))
        s = np.where(k > 0, np.sin(root * 0.2), np.sinh(root * 0.2)) / root
        sp = np.where(k > 0, -root * np.sin(root * 0.2), root * np.sinh(root * 0.2))
        return c * x + s * xp, sp * x + c * xp

    k = k1 * np.mean(beam.cp.val) / beam.cp.val  # chromatic: k1 scales as p0 / p
    x, xp = quad(beam.x.val + 0.4 * beam.xp.val, beam.xp.val, k)
    x = x + 0.4 * xp  # at MID
    if to == "M3":
        x, xp = quad(x + 0.4 * xp, xp, -k)
        x = x + 0.4 * xp
    return x


@pytest.mark.parametrize("code", CODES)
def test_each_species_follows_its_own_linear_map(tmp_path, electrons, code):
    # the positrons start on the electrons' coordinates; Q1 focuses one in x and
    # defocuses the other, so each must match the linear map with its own sign of k1
    seed = _seed(electrons, tmp_path / "seed", ["positron"])
    end = _read(os.path.join(_run(tmp_path / "run", code, seed), "M3.openpmd.hdf5"))

    assert end.species == "electron" and list(end.others) == ["positron"]
    positrons = end.others["positron"]
    assert len(end.x) == len(positrons.x) == 50
    assert float(end.Particles.total_charge) == pytest.approx(-1e-15)
    assert float(positrons.Particles.total_charge) == pytest.approx(1e-15)

    start = _read(seed)
    # ASTRA hands back the particles rolled (its writer moves the one nearest the
    # mean to the reference row when the beam has none), so compare it unordered
    order = np.sort if code == "astra" else (lambda x: x)
    atol = TOLERANCE[code] * np.std(start.x.val)
    np.testing.assert_allclose(order(end.x.val), order(_x_linear(start, K1, "M3")), rtol=0, atol=atol)
    np.testing.assert_allclose(
        order(positrons.x.val), order(_x_linear(start, -K1, "M3")), rtol=0, atol=atol
    )


@needs_bmad
def test_bmad_tracks_the_electrons_as_if_alone(tmp_path, electrons):
    seed = _seed(electrons, tmp_path / "seed_mixed", ["positron"])
    mixed = _run(tmp_path / "mixed", "bmad", seed)
    alone = _run(tmp_path / "alone", "bmad", _seed(electrons, tmp_path / "seed_alone"))

    end = _read(os.path.join(mixed, "M3.openpmd.hdf5"))
    end_alone = _read(os.path.join(alone, "M3.openpmd.hdf5"))
    np.testing.assert_allclose(end.x.val, end_alone.x.val, rtol=0, atol=1e-15)
    np.testing.assert_allclose(end.cpx.val, end_alone.cpx.val, rtol=1e-12, atol=1e-6)

    # and the species come apart already at MID, after Q1 alone
    start, mid = _read(seed), _read(os.path.join(mixed, "MID.openpmd.hdf5"))
    atol = 1e-3 * np.std(start.x.val)
    np.testing.assert_allclose(mid.x.val, _x_linear(start, K1), rtol=0, atol=atol)
    np.testing.assert_allclose(mid.others["positron"].x.val, _x_linear(start, -K1), rtol=0, atol=atol)


@needs_bmad
def test_photons_ride_through_and_cannot_be_tracked(tmp_path, electrons):
    seed = _read(electrons)
    photons = _read(electrons)
    photons.set_species("photon")
    photons.Particles.nmacro = np.full(len(photons.x), 3.0)
    seed.set_species("electron")
    seed.others["photon"] = photons
    path = os.path.join(str(tmp_path), "seed.openpmd.hdf5")
    seed.write_openpmd_beam_file(path)

    end = _read(os.path.join(_run(tmp_path / "run", "bmad", path), "M3.openpmd.hdf5"))
    assert list(end.others) == ["photon"]
    np.testing.assert_allclose(end.others["photon"].Particles.nmacro, 3.0)

    selected = path.replace("seed.", "selected.")
    shutil.copy(path, selected)
    with pytest.raises(exceptions.NeutralSpeciesError):
        _run(tmp_path / "photons", "bmad", selected, inputs={"species": "photon"})


@needs_bmad
def test_bmad_refuses_a_species_of_another_mass(tmp_path, electrons):
    with pytest.raises(exceptions.MixedMassError, match="proton"):
        _run(tmp_path / "run", "bmad", _seed(electrons, tmp_path / "seed", ["positron", "proton"]))
