"""Several species in one openPMD beam file: photons, the primary species, and carrying the rest through a line."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from h5py import File
from scipy.constants import c, e

constants_c = c

import simba.Modules.Beams as rbf
from simba import exceptions
from simba.Framework_objects import frameworkLattice
from simba.Modules.units import UnitValue


def make_beam(species, n=50, cpz=1e8, charge=-10e-12, nmacro=None, seed=0):
    rng = np.random.default_rng(seed)
    q_over_c = e / c
    beam = rbf.beam()
    p = beam.Particles
    p.x = UnitValue(rng.normal(0, 1e-4, n), "m")
    p.y = UnitValue(rng.normal(0, 1e-4, n), "m")
    p.z = UnitValue(rng.normal(0, 1e-5, n), "m")
    p.t = UnitValue(-p.z.val / c, "s")
    p.px = UnitValue(rng.normal(0, 1e3, n) * q_over_c, "kg*m/s")
    p.py = UnitValue(rng.normal(0, 1e3, n) * q_over_c, "kg*m/s")
    p.pz = UnitValue(np.full(n, cpz) * q_over_c, "kg*m/s")
    p.status = UnitValue(np.ones(n))
    p.set_total_charge(charge)
    if nmacro is not None:
        p.nmacro = UnitValue(np.full(n, float(nmacro)))
    beam.set_species(species)
    return beam


def test_photon_species_has_finite_energy_and_no_charge():
    photons = make_beam("photons", cpz=5e6, nmacro=1000)
    assert photons.species == "photon"
    assert np.all(photons.Particles.charge.val == 0)
    assert np.allclose(photons.Particles.energy.val, photons.Particles.cp.val)
    assert photons.Particles.get_particle_index(0.0, 0) == 5


def test_three_species_round_trip(tmp_path):
    electrons = make_beam("electron", charge=-100e-12)
    electrons.others["photon"] = make_beam("photon", n=80, cpz=5e6, nmacro=1000, seed=1)
    electrons.others["positron"] = make_beam("positron", n=20, charge=1e-12, seed=2)
    filename = str(tmp_path / "three.openpmd.hdf5")
    electrons.write_openpmd_beam_file(filename)

    back = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(back, filename)
    assert back.species == "electron"
    assert np.isclose(back.Particles.total_charge.val, -100e-12)
    assert set(back.others) == {"photon", "positron"}

    photons = back.others["photon"]
    assert len(photons.x) == 80
    assert np.allclose(photons.Particles.nmacro.val, 1000)
    assert np.all(photons.Particles.charge.val == 0)
    assert np.allclose(photons.Particles.cpz.val, 5e6)
    assert np.isclose(back.others["positron"].Particles.total_charge.val, 1e-12)

    # and the original beam, re-read, is unchanged by carrying the others
    assert np.allclose(back.x.val, electrons.x.val)


def test_species_picks_the_tracked_group(tmp_path):
    electrons = make_beam("electron")
    electrons.others["photon"] = make_beam("photon", nmacro=7, seed=1)
    filename = str(tmp_path / "pick.openpmd.hdf5")
    electrons.write_openpmd_beam_file(filename)

    back = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(back, filename, species="photon")
    assert back.species == "photon"
    assert list(back.others) == ["electron"]
    with pytest.raises(ValueError, match="not proton"):
        rbf.openpmd.read_openpmd_beam_file(rbf.beam(), filename, species="proton")


def test_a_neutral_beam_needs_nmacro_to_be_written(tmp_path):
    photons = make_beam("photon")
    with pytest.raises(ValueError, match="nmacro"):
        photons.write_openpmd_beam_file(str(tmp_path / "bad.openpmd.hdf5"))


def test_several_species_and_no_primary_must_be_chosen(tmp_path):
    filename = str(tmp_path / "noprimary.openpmd.hdf5")
    electrons = make_beam("electron")
    electrons.others["photon"] = make_beam("photon", nmacro=1, seed=1)
    electrons.write_openpmd_beam_file(filename)
    with File(filename, "a") as h5:
        del h5["particles"].attrs[rbf.openpmd.PRIMARY_SPECIES_ATTR]
    with pytest.raises(ValueError, match="say which one"):
        rbf.openpmd.read_openpmd_beam_file(rbf.beam(), filename)


def test_append_skips_a_species_already_there(tmp_path):
    filename = str(tmp_path / "end.openpmd.hdf5")
    make_beam("electron").write_openpmd_beam_file(filename)
    assert rbf.openpmd.append_species(filename, [make_beam("photon", nmacro=3, seed=1)]) == ["photon"]
    with pytest.warns(UserWarning, match="already holds photon"):
        assert rbf.openpmd.append_species(filename, [make_beam("photon", nmacro=3)]) == []
    back = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(back, filename)
    assert len(back.others["photon"].x) == 50


def test_append_keeps_the_existing_species_primary_and_merges_a_species_already_there(tmp_path):
    filename = str(tmp_path / "end.openpmd.hdf5")
    make_beam("electron").write_openpmd_beam_file(filename)
    added = rbf.openpmd.append_species(filename, [make_beam("photon", n=50, nmacro=3, seed=1)])
    assert added == ["photon"]
    # photons the code made itself, and photons carried to it from upstream
    assert rbf.openpmd.append_species(filename, [make_beam("photon", n=20, nmacro=5, seed=2)], merge=True) == ["photon"]

    back = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(back, filename)
    assert back.species == "electron"
    assert list(back.others) == ["photon"]
    photons = back.others["photon"]
    assert len(photons.x) == 70
    np.testing.assert_allclose(photons.Particles.nmacro, [3.0] * 50 + [5.0] * 20)


def _line_given(others, species=None, multiple=False):
    beam = make_beam("electron")  # read_input_file has moved its others to the line
    return SimpleNamespace(
        _carried_species=dict(others),
        merges_carried_species=False,
        global_parameters={"beam": beam},
        file_block={"input": {"species": species}} if species else {},
        objectname="LINE",
        code="astra" if multiple else "elegant",
        tracks_multiple_species=multiple,
    )


def test_absorb_then_split_gives_back_each_species():
    electrons = make_beam("electron", n=30, charge=-30e-12)
    electrons.reference_particle_index = 4
    positrons = make_beam("positron", n=10, charge=5e-12, seed=1)
    protons = make_beam("proton", n=5, charge=1e-12, cpz=2e9, seed=2)
    x_e, x_p = electrons.x.val.copy(), positrons.x.val.copy()
    electrons.absorb([positrons, protons])

    assert len(electrons.x) == 45 and electrons.is_mixed
    assert electrons.species == "electron"
    assert np.isclose(electrons.Particles.total_charge.val, -24e-12)
    assert np.allclose(electrons.Particles.energy.val[-5:], np.hypot(938.272e6, 2e9), rtol=1e-5)

    groups = electrons.split_species()
    assert list(groups) == ["electron", "positron", "proton"]
    assert np.allclose(groups["electron"].x.val, x_e)
    assert np.allclose(groups["positron"].x.val, x_p)
    assert np.isclose(groups["positron"].Particles.total_charge.val, 5e-12)
    assert groups["electron"].reference_particle_index == 4
    assert groups["proton"].reference_particle_index is None
    assert not groups["proton"].is_mixed


def test_a_mixed_beam_is_written_as_one_group_per_species(tmp_path):
    electrons = make_beam("electron", n=30)
    electrons.absorb([make_beam("positron", n=10, charge=5e-12, seed=1)])
    electrons.others["photon"] = make_beam("photon", n=7, nmacro=1, seed=2)
    filename = str(tmp_path / "mixed.openpmd.hdf5")
    electrons.write_openpmd_beam_file(filename)

    back = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(back, filename)
    assert back.species == "electron" and len(back.x) == 30
    assert {k: len(b.x) for k, b in back.others.items()} == {"positron": 10, "photon": 7}


def test_a_multi_species_code_keeps_and_merges_the_charged_species():
    line = _line_given(
        {"positron": make_beam("positron", n=10, charge=5e-12), "photon": make_beam("photon", nmacro=1)},
        multiple=True,
    )
    frameworkLattice.check_other_species(line)
    assert frameworkLattice.merge_other_species(line) == ["positron"]
    beam = line.global_parameters["beam"]
    assert len(beam.x) == 60 and beam.is_mixed
    assert list(line._carried_species) == ["photon"] and not beam.others
    # the line's reference is the tracked species, not an average over both
    assert frameworkLattice.reference_charge.fget(line) == -1
    assert np.isclose(frameworkLattice.rest_energy.fget(line), 0.51099895e6)


@pytest.mark.parametrize("with_reference", [True, False])
def test_astra_round_trips_a_mixed_beam(tmp_path, with_reference):
    from simba.Codes.ASTRA.ASTRA import astraLattice

    assert astraLattice.tracks_multiple_species
    # more positrons, all on axis: without a reference, one of them is nearest the
    # bunch mean, but the reference (ASTRA's first row) must be an electron
    positrons = make_beam("positron", n=40, charge=4e-12, seed=1)
    for coord in ("x", "y", "z", "px", "py", "t"):
        setattr(positrons.Particles, coord, UnitValue(np.zeros(40), getattr(positrons.Particles, coord).units))
    electrons = make_beam("electron", n=10, charge=-10e-12)
    electrons.reference_particle = None
    electrons.reference_particle_index = None
    electrons.absorb([positrons])
    if with_reference:
        electrons.reference_particle_index = 3
        electrons.reference_particle = [np.asarray(getattr(electrons, c))[3] for c in electrons.reference_particle_coords]
    filename = str(tmp_path / "mixed.astra")
    rbf.astra.write_astra_beam_file(electrons, filename)

    back = rbf.beam()
    rbf.astra.read_astra_beam_file(back, filename)
    assert back.species == "electron" and back.is_mixed
    groups = back.split_species()
    assert {k: len(b.x) for k, b in groups.items()} == {"electron": 10, "positron": 40}
    assert np.isclose(groups["positron"].Particles.total_charge.val, 4e-12)
    assert np.isclose(groups["electron"].Particles.total_charge.val, -10e-12)


def test_gpt_round_trips_a_mixed_beam(tmp_path):
    from simba.Codes.GPT.GPT import gptLattice

    assert gptLattice.tracks_multiple_species
    electrons = make_beam("electron", n=30, charge=-30e-12)
    electrons.absorb([
        make_beam("positron", n=10, charge=5e-12, seed=1),
        make_beam("proton", n=5, charge=1e-12, cpz=2e9, seed=2),
    ])
    filename = str(tmp_path / "mixed.gdf")
    rbf.gdf.write_gdf_beam_file(electrons, filename)

    back = rbf.beam()
    rbf.gdf.read_gdf_beam_file(back, filename, position=None)
    back.species = "electron"  # as gptLattice's screen conversion sets it
    groups = back.split_species()
    assert {k: len(b.x) for k, b in groups.items()} == {"electron": 30, "positron": 10, "proton": 5}
    assert np.isclose(groups["positron"].Particles.total_charge.val, 5e-12)
    assert np.allclose(groups["proton"].Particles.cp.val, electrons.Particles.cp.val[-5:], rtol=1e-6)


def _mixed_for_xsuite():
    electrons = make_beam("electron", n=30, charge=-30e-12)
    electrons.Particles.px = UnitValue(np.zeros(30), "kg*m/s")
    positrons = make_beam("positron", n=10, charge=5e-12, seed=1)
    positrons.Particles.px = UnitValue(np.zeros(10), "kg*m/s")
    protons = make_beam("proton", n=5, charge=1e-12, seed=2)  # same 100 MeV/c: slow
    protons.Particles.px = UnitValue(np.zeros(5), "kg*m/s")
    electrons.absorb([positrons, protons])
    return electrons


def test_xsuite_round_trips_a_mixed_beam():
    import xobjects as xo
    from simba.Codes.Xsuite.Xsuite import xsuiteLattice

    assert xsuiteLattice.tracks_multiple_species
    beam = _mixed_for_xsuite()
    context = xo.ContextCpu()
    particles = rbf.xsuite.write_xsuite_beam_file(
        beam, "unused.xsuite.json", write=False, p0c=1e8, t0=0.0, context=context
    )
    assert np.allclose(particles.energy, beam.Particles.energy.val, rtol=1e-9)

    back = copy.deepcopy(beam)
    rbf.xsuite.read_xsuite_beam_file(back, particles, t_reference=0.0, context=context)
    groups = back.split_species()
    assert {k: len(b.x) for k, b in groups.items()} == {"electron": 30, "positron": 10, "proton": 5}
    assert np.isclose(groups["positron"].Particles.total_charge.val, 5e-12)
    assert np.allclose(back.Particles.cp.val, beam.Particles.cp.val, rtol=1e-9)
    assert np.allclose(back.t.val, beam.t.val, atol=1e-18)


def test_xsuite_steers_and_times_each_species_as_its_own():
    import xobjects as xo
    import xtrack as xt

    beam = _mixed_for_xsuite()
    context = xo.ContextCpu()
    particles = rbf.xsuite.write_xsuite_beam_file(
        beam, "unused.xsuite.json", write=False, p0c=1e8, t0=0.0, context=context
    )
    line = xt.Line(elements=[xt.Bend(length=1.0, k0=0.01, angle=0.0), xt.Drift(length=1.0)])
    line.particle_ref = xt.Particles(mass0=0.51099895e6, q0=-1, p0c=1e8)
    line.build_tracker(_context=context)
    line.track(particles)
    rbf.xsuite.read_xsuite_beam_file(beam, particles, t_reference=0.0, context=context)

    groups = beam.split_species()
    kick = {k: np.mean(b.Particles.cpx.val) for k, b in groups.items()}
    assert kick["electron"] < 0 < kick["positron"]
    assert np.isclose(kick["positron"], -kick["electron"], rtol=1e-6)
    assert np.isclose(kick["proton"], kick["positron"], rtol=1e-6)  # same charge, same field integral
    # the 100 MeV/c protons (beta ~ 0.106) take far longer over the 2 m than the electrons
    lag = np.mean(groups["proton"].t.val) - np.mean(groups["electron"].t.val)
    expected = 2.0 / constants_c * (1 / 0.10597869 - 1 / 0.99998694)
    assert np.isclose(lag, expected, rtol=1e-3)


def test_bmad_warns_of_other_species_on_the_reference_orbit():
    # Bmad's bend tracks particles within 1e-9 of the zero orbit with the reference's map
    from simba.Codes.Bmad.Bmad import bmadLattice

    line = SimpleNamespace(objectname="LINE")
    rows = np.zeros((3, 8))
    rows[2, 5] = 1e-6  # off the reference momentum: tracked properly
    with pytest.warns(exceptions.NearAxisSpeciesWarning, match="1 particles that are not electron"):
        bmadLattice._warn_near_axis(line, rows, ["electron", "positron", "positron"], "electron")


@pytest.mark.parametrize(
    "csr, space_charge, warns",
    [(False, "False", False), (True, "False", True), (False, "3D", True)],
)
def test_bmad_warns_that_its_collective_effects_take_one_species(csr, space_charge, warns):
    import warnings

    from simba.Codes.Bmad.Bmad import bmadLattice

    line = SimpleNamespace(
        objectname="LINE", _input_species=np.array(["electron", "positron"], dtype=object),
        csr_enable=csr, lsc_enable=False,
        space_charge_mode="3d" if space_charge == "3D" else "",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bmadLattice._warn_collective_species(line)
    assert any(isinstance(w.message, exceptions.CollectiveSpeciesWarning) for w in caught) == warns


@pytest.mark.parametrize("electrons_only", [False, True])
def test_no_code_is_given_photons_to_track(electrons_only):
    line = SimpleNamespace(
        global_parameters={"beam": make_beam("photon", nmacro=1)},
        objectname="LINE", code="xsuite", electrons_only=electrons_only,
    )
    with pytest.raises(exceptions.NeutralSpeciesError, match="photon"):
        frameworkLattice.check_species(line)


def test_a_single_species_code_merges_nothing():
    line = _line_given({"photon": make_beam("photon", nmacro=1)})
    assert frameworkLattice.merge_other_species(line) == []


def test_photons_alone_ride_along_with_any_code():
    line = _line_given({"photon": make_beam("photon", nmacro=1)})
    frameworkLattice.check_other_species(line)
    assert list(line._carried_species) == ["photon"]


def test_a_second_charged_species_is_refused():
    line = _line_given({"positron": make_beam("positron"), "photon": make_beam("photon", nmacro=1)})
    with pytest.raises(exceptions.MultipleSpeciesError, match="positron"):
        frameworkLattice.check_other_species(line)


def test_naming_the_tracked_species_drops_the_other_charged_ones():
    line = _line_given(
        {"positron": make_beam("positron"), "photon": make_beam("photon", nmacro=1)},
        species="electron",
    )
    with pytest.warns(exceptions.SpeciesDroppedWarning, match="positron"):
        frameworkLattice.check_other_species(line)
    assert list(line._carried_species) == ["photon"]


def test_reading_the_input_hands_the_other_species_to_the_line(tmp_path):
    # so that no code writes them into its screen files (some write a copy of
    # the global beam), and only carry_other_species adds them to the end file
    electrons = make_beam("electron")
    electrons.others["photon"] = make_beam("photon", nmacro=1, seed=1)
    electrons.write_openpmd_beam_file(str(tmp_path / "START.openpmd.hdf5"))
    beam = rbf.beam()
    line = SimpleNamespace(
        global_parameters={"beam": beam, "master_lattice": str(tmp_path), "master_subdir": str(tmp_path)},
        file_block={},
        getInitialTwiss=lambda: None,
    )
    frameworkLattice.read_input_file(line, str(tmp_path) + "/", "START")
    assert list(line._carried_species) == ["photon"]
    assert beam.species == "electron" and not beam.others


def test_an_absolute_input_path_is_read_as_given(tmp_path):
    # not prefixed with the output directory, as a $master_lattice$ path isn't
    (tmp_path / "beams").mkdir()
    (tmp_path / "out").mkdir()
    make_beam("electron").write_openpmd_beam_file(str(tmp_path / "beams" / "IN.openpmd.hdf5"))
    line = SimpleNamespace(
        global_parameters={"beam": rbf.beam(), "master_lattice": str(tmp_path), "master_subdir": str(tmp_path / "out")},
        file_block={},
        getInitialTwiss=lambda: None,
    )
    path = frameworkLattice.read_input_file(line, str(tmp_path / "out") + "/", str(tmp_path / "beams" / "IN"))
    assert path == str(tmp_path / "beams" / "IN.openpmd.hdf5")
    assert line.global_parameters["beam"].species == "electron"


def test_a_line_carries_the_species_it_did_not_track(tmp_path):
    # the line's code wrote only electrons to its end file; the photons it was
    # given come along to the next line
    end = str(tmp_path / "END.openpmd.hdf5")
    make_beam("electron").write_openpmd_beam_file(end)
    line = SimpleNamespace(
        _carried_species={"photon": make_beam("photon", nmacro=2, seed=1)},
        output_beam_file=lambda name: str(tmp_path / f"{name}.openpmd.hdf5"),
        end="END",
        objectname="LINE",
        merges_carried_species=False,
    )
    assert frameworkLattice.carry_other_species(line) == ["photon"]
    back = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(back, end)
    assert back.species == "electron" and list(back.others) == ["photon"]

    line._carried_species = {}
    assert frameworkLattice.carry_other_species(line) == []

    line._carried_species = {"photon": make_beam("photon", nmacro=2)}
    line.end = "NOWHERE"
    with pytest.warns(exceptions.SpeciesNotCarriedWarning):
        assert frameworkLattice.carry_other_species(line) == []
