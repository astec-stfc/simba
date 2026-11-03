"""BDSIM's physics list and options, and material that does nothing without them."""

import os
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from simba import exceptions
from simba.Modules import constants

pytest.importorskip("pybdsim")
from simba.Codes.BDSIM.BDSIM import bdsimLattice  # noqa: E402

needs_bdsim = pytest.mark.skipif(shutil.which("bdsim") is None, reason="BDSIM not installed")


@pytest.mark.parametrize(
    "tracking, physics_list, options",
    [
        (None, "", {}),
        ({"physics_list": "em"}, "em", {"physicsList": '"em"'}),
        (
            {"physics_list": "em", "options": {"defaultRangeCut": 1e-3, "physicsVerbose": "1"}},
            "em",
            {"defaultRangeCut": 1e-3, "physicsVerbose": '"1"', "physicsList": '"em"'},
        ),
    ],
)
def test_the_physics_list_and_options_come_from_the_tracking_block(tracking, physics_list, options):
    line = SimpleNamespace(file_block={"tracking": tracking})
    line.physics_list = bdsimLattice.physics_list.fget(line)
    assert line.physics_list == physics_list
    assert bdsimLattice.bdsim_options.fget(line) == options


def _collimator_line(tmp_path, tracking, npart=400):
    """A 0.5 mm-radius copper collimator in a round 1 mm beam, through Framework.track."""
    from laura import LAURA
    from laura.exporters.yaml_exporter import export_machine
    from laura.models.element import Collimator, Marker

    import simba.Framework as fw
    import simba.Modules.Beams as rbf
    from simba.Codes.Generators import frameworkGenerator

    def marker(name, z):
        return Marker(name=name, machine_area="LINE", hardware_class="Marker",
                      physical={"middle": {"x": 0.0, "y": 0.0, "z": z}})

    col = Collimator(name="COL", machine_area="LINE", material="copper",
                     aperture={"shape": "circular", "radius": 5e-4},
                     physical={"length": 0.1, "middle": {"x": 0.0, "y": 0.0, "z": 0.5}})
    elements = [marker("M1", 0.0), col, marker("M3", 1.0)]
    section = {"sections": {"LINE": {"elements": [e.name for e in elements]}}}
    machine = LAURA(element_list=elements, section=section,
                    layout={"default_layout": "l", "layouts": {"l": ["LINE"]}})
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)}, filename="seed.openpmd.hdf5",
        initial_momentum=100e6, sigma_x=1e-3, sigma_px=1e3, sigma_y=1e-3, sigma_py=1e3,
        sigma_z=1e-4, sigma_pz=1e4, charge=1e-12, number_of_particles=npart,
    ).write()
    settings = fw.FrameworkSettings()
    settings.files = {"LINE": {
        "code": "bdsim", "charge": {"space_charge_mode": "False"}, "tracking": tracking,
        "output": {"start_element": "M1", "end_element": "M3"},
    }}
    settings.layout, settings.section = machine.layout, section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    shutil.copy(tmp_path / "seed.openpmd.hdf5", os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        framework.track()
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    beam.root_file = os.path.join(framework.subdirectory, "LINE.root")
    return beam, [w.message for w in caught]


@needs_bdsim
def test_without_a_physics_list_material_does_nothing_and_says_so(tmp_path):
    beam, caught = _collimator_line(tmp_path, {}, npart=100)
    assert len(beam.x) == 100
    assert any(isinstance(m, exceptions.MaterialWithoutPhysicsWarning) for m in caught)


@needs_bdsim
def test_a_round_collimator_passes_the_core_of_the_beam(tmp_path):
    # inside r = 0.5 mm of a round 1 mm Gaussian: 1 - exp(-r^2 / 2 sigma^2) = 11.75%
    npart = 400
    beam, caught = _collimator_line(tmp_path, {"physics_list": "em"}, npart=npart)
    expected = 1 - np.exp(-0.5**2 / 2)
    spread = np.sqrt(expected * (1 - expected) / npart)
    assert abs(len(beam.x) / npart - expected) < 4 * spread
    assert float(beam.Particles.total_charge) == pytest.approx(-1e-12 * len(beam.x) / npart)
    assert not any(isinstance(m, exceptions.MaterialWithoutPhysicsWarning) for m in caught)


@needs_bdsim
def test_the_secondaries_asked_for_come_out_as_their_own_species(tmp_path):
    from simba.Modules.Beams import bdsim

    npart = 200
    beam, _ = _collimator_line(
        tmp_path, {"physics_list": "em", "secondaries": ["photon", "positron"]}, npart=npart
    )
    # what BDSIM itself recorded at the end sampler
    raw = bdsim.read_bdsim_sampler_arrays(beam.root_file, samplers=["M3"])["M3"]
    secondary = raw["parentID"] != 0
    assert len(beam.x) == np.count_nonzero(~secondary)  # secondary electrons not asked for
    assert beam.species == "electron"
    per_hit = 1e-12 / (constants.elementary_charge * npart)  # real particles per hit
    for species, pdg, sign in (("photon", 22, 0), ("positron", -11, 1)):
        count = np.count_nonzero(secondary & (raw["partID"] == pdg))
        if not count:
            assert species not in beam.others
            continue
        group = beam.others[species]
        assert len(group.x) == count
        np.testing.assert_allclose(group.Particles.nmacro, per_hit, rtol=1e-12)
        assert float(group.Particles.total_charge) == pytest.approx(sign * constants.elementary_charge * per_hit * count)
        assert np.all(group.Particles.energy.val < 100.1e6)
    assert "photon" in beam.others  # 10 cm of copper always showers


def _sampler(*hits):
    """Sampler arrays from ``(eventID, parentID, partID, T)`` hits, at 100 MeV along z."""
    columns = np.array(hits, dtype=float).T
    n = len(hits)
    return {
        "eventID": columns[0].astype(int), "parentID": columns[1].astype(int),
        "partID": columns[2].astype(int), "T": columns[3], "x": np.arange(n, dtype=float),
        "y": np.zeros(n), "xp": np.zeros(n), "yp": np.zeros(n), "zp": np.ones(n),
        "energy": np.full(n, 0.1), "S": np.ones(n), "weight": np.ones(n),
    }


def test_the_reference_particle_is_found_by_its_event():
    import simba.Modules.Beams as rbf
    from simba.Modules.Beams import bdsim

    # event 1 was lost, so the reference (event 2) is the second primary, not the third
    arrays = _sampler((0, 0, 11, 1.0), (2, 0, 11, 1.1), (3, 0, 11, 1.2), (2, 1, 22, 1.1))
    beam = rbf.beam()
    with pytest.warns(UserWarning, match="1 of 4 particles were lost"):
        bdsim.interpret_bdsim_sampler(beam, arrays, charge=-4e-15, ref_index=2, ninjected=4,
                                      secondaries=["photon"])
    assert beam.reference_particle_index == 1
    assert beam.x.val[beam.reference_particle_index] == 1.0
    assert list(beam.particle_species) == ["electron", "electron", "electron", "photon"]
    assert float(beam.Particles.total_charge) == pytest.approx(-3e-15)


def test_a_lost_reference_and_unknown_secondaries_are_reported():
    import simba.Modules.Beams as rbf
    from simba.Modules.Beams import bdsim

    arrays = _sampler((0, 0, 11, 1.0), (0, 1, 2112, 1.0), (0, 1, 22, 1.0))  # a neutron
    beam = rbf.beam()
    with pytest.warns(UserWarning) as caught:
        bdsim.interpret_bdsim_sampler(beam, arrays, charge=-2e-15, ref_index=1, ninjected=2,
                                      secondaries="all")
    messages = " ".join(str(w.message) for w in caught)
    assert "reference particle was lost" in messages and "2112" in messages
    assert list(beam.particle_species) == ["electron", "photon"]


def test_unknown_secondaries_are_refused():
    line = SimpleNamespace(file_block={"tracking": {"secondaries": ["photons", "muon"]}})
    with pytest.raises(ValueError, match="muon"):
        bdsimLattice.secondaries.fget(line)
    line.file_block["tracking"]["secondaries"] = ["photons", "positron"]
    assert bdsimLattice.secondaries.fget(line) == ("photon", "positron")
