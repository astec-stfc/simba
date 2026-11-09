"""Solenoids and alignment errors through simba, against Xsuite: MAD-X's
solenoid ``ks`` needs the segment's rigidity, and its MAKETHIN drops EALIGN."""

import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
from helpers import fodo_machine, fodo_quads, read_beam, skip_missing
from laura.models.element import Dipole, MatrixTransform, Quadrupole, Sextupole, Solenoid
from simba.Codes.Generators import frameworkGenerator


def solenoid():
    return Solenoid(
        name="SOL1", machine_area="FODO",
        magnetic={"length": 0.5, "fields": {"S0L": 0.02}},
        physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
    )


def misaligned_quad():
    return Quadrupole(
        name="QUAD1", machine_area="FODO",
        magnetic={"length": 1.0, "k1l": 1.0, "tilt": 0.1},
        physical={
            "length": 1.0,
            "middle": {"x": 0.0, "y": 0.0, "z": 2.5},
            "error": {
                "position": {"x": 1e-3, "y": -2e-3, "z": 3e-3},
                "rotation": {"theta": 4e-3, "phi": 5e-3, "psi": 6e-3},
            },
        },
    )


def misaligned_bend(error):
    """Untilted: MAD-X's MAKETHIN models a tilted strong bend 1e-4 off even
    without errors."""
    return Dipole(
        name="BEND1", machine_area="FODO",
        magnetic={"length": 1.0, "angle": 0.5},
        physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 2.5}, "error": error},
    )


@pytest.fixture(scope="module")
def seed_beam(tmp_path_factory):
    directory = tmp_path_factory.mktemp("seed")
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="M1.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3,
        sigma_z=1e-3, sigma_pz=1e3,
        gaussian_cutoff_x=3, gaussian_cutoff_y=3, gaussian_cutoff_z=3,
        gaussian_cutoff_px=3, gaussian_cutoff_py=3, gaussian_cutoff_pz=3,
        charge=100e-12, number_of_particles=32,
    ).write()
    return os.path.join(str(directory), "M1.openpmd.hdf5")


def track(tmp_path, code, middle, seed_beam):
    skip_missing(code)
    machine, _, section = fodo_machine(tmp_path, middle=middle)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(
        machine=machine, directory=str(tmp_path), clean=True, verbose=False
    )
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    beam = read_beam(framework.subdirectory, "M3")
    return np.array([beam.x.val, beam.y.val])


@pytest.mark.parametrize("code", ["madx", "ocelot", "cheetah"])
def test_a_solenoid_turns_the_beam_as_xsuite_does(tmp_path, code, seed_beam):
    """One MAD-X slice is 2.4% off at this kL of 0.6; four are 5e-4."""
    expected = track(tmp_path / "x", "xsuite", [solenoid()], seed_beam)
    final = track(tmp_path / code, code, [solenoid()], seed_beam)
    assert close(final, expected, 1e-3)


def test_a_matrix_transform_tracks_in_madx_as_in_xsuite(tmp_path, seed_beam):
    """MAD-X's (t, pt) differ from Bmad's (z, pz) by beta, 0.5% at 5 MeV/c."""
    r = np.eye(6)
    r[0, 4], r[2, 5] = 1.0, 0.5
    matrix = MatrixTransform(
        name="MAT1", machine_area="FODO",
        simulation={"r_matrix": r},
        physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
    )
    expected = track(tmp_path / "x", "xsuite", [matrix], seed_beam)
    final = track(tmp_path / "m", "madx", [matrix], seed_beam)
    assert close(final, expected, 1e-3)


def test_madx_keeps_alignment_errors_through_makethin(tmp_path, seed_beam):
    """The errors move x by 60% of its spread; four quadrupole slices are 3%
    off that, and 64 are 3e-4."""
    expected = track(tmp_path / "x", "xsuite", [misaligned_quad()], seed_beam)
    final = track(tmp_path / "m", "madx", [misaligned_quad()], seed_beam)
    assert close(final, expected, 5e-2)


@pytest.mark.parametrize(
    "error",
    [
        {"position": {"x": 1e-3}},
        {"position": {"z": 1e-3}},
        {"rotation": {"theta": 1e-3}},
        {"rotation": {"phi": 1e-3}},
        {"rotation": {"psi": 1e-2}},
    ],
    ids=["dx", "dz", "dx_rot", "dy_rot", "dz_rot"],
)
def test_madx_slices_follow_a_bends_arc(tmp_path, seed_beam, error):
    """Each slice is misaligned where it sits on the arc. Straight-line levers
    were 30% off on dx_rot and 5x on dy_rot; a thin slice cannot take the
    shift along the arc, which leaves dx 5% off."""
    def effect(code):
        bare = track(tmp_path / f"{code}0", code, [misaligned_bend({})], seed_beam)
        moved = track(tmp_path / code, code, [misaligned_bend(error)], seed_beam)
        return moved - bare

    got, want = effect("madx"), effect("xsuite")
    assert np.abs(got - want).max() < 0.1 * np.abs(want).max()


def madx_ring_optics(tmp_path, slices, seed_beam, code="madx",
                     error={"position": {"x": 1e-3}, "rotation": {"theta": 1e-3}}, k2l=0.0):
    """``(optics summary, closed orbit)`` of a one-cell FODO ring with a
    misaligned quadrupole, from MAD-X (or ``code``) slicing quadrupoles ``slices`` times.
    ``k2l`` adds a sextupole between the quadrupoles."""
    skip_missing(code)
    quads = fodo_quads()
    for quad, k1l in zip(quads, (-0.5, 0.5)):
        quad.magnetic.k1l = k1l
    quads[0].physical.error = error
    if k2l:
        quads.insert(1, Sextupole(
            name="SEXT1", machine_area="FODO", magnetic={"length": 0.5, "k2l": k2l},
            physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
        ))
    machine, _, section = fodo_machine(tmp_path, closed=True, middle=quads)
    settings = fw.FrameworkSettings()
    settings.files = {
        "FODO": {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": "M1", "end_element": "M3"},
            "tracking": {"periodic": True, "turns": 1, "nslice_quadrupole": slices},
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(
        machine=machine, directory=str(tmp_path), clean=True, verbose=False
    )
    framework.loadSettings(settings=settings)
    shutil.copy(seed_beam, os.path.join(framework.subdirectory, "M1.openpmd.hdf5"))
    framework.track()
    ring = framework.latticeObjects["FODO"]
    return ring.read_optics_summary(), ring.read_closed_orbit()


def test_madx_periodic_optics_come_from_the_thick_ring(tmp_path, seed_beam):
    """The periodic TWISS ran on the line sliced for tracking, so 4 slices put
    a Diamond-I tune 0.010 out; on the thick sequence, misaligned through its
    own EALIGN, the slicing no longer reaches the optics or the orbit."""
    coarse, coarse_orbit = madx_ring_optics(tmp_path / "1", 1, seed_beam)
    fine, fine_orbit = madx_ring_optics(tmp_path / "64", 64, seed_beam)

    assert coarse == pytest.approx(fine, rel=1e-9)
    assert abs(coarse_orbit[0]) > 1e-5, "the error must move the orbit"
    assert coarse_orbit == pytest.approx(fine_orbit, abs=1e-12)


def test_ocelot_closed_orbit_sees_the_misalignment(tmp_path, seed_beam):
    """Displaced only: Ocelot's map has no yaw (``rotation.theta``)."""
    error = {"position": {"x": 1e-3}}
    _, madx = madx_ring_optics(tmp_path / "madx", 1, seed_beam, error=error)
    _, ocelot = madx_ring_optics(tmp_path / "ocelot", 1, seed_beam, code="ocelot", error=error)
    assert abs(madx[0]) > 1e-5, "the error must move the orbit"
    assert ocelot[:4] == pytest.approx(madx[:4], abs=1e-9)


def test_ocelot_tune_sees_a_sextupole_off_axis(tmp_path, seed_beam):
    """The orbit is through each element's map in turn: one composed second-order
    map drops enough to put it 2% out. (At k2l = 50 there is no closed orbit.)"""
    error, k2l = {"position": {"x": 1e-3}}, 5.0
    on_axis, _ = madx_ring_optics(tmp_path / "axis", 1, seed_beam, error={}, k2l=k2l)
    madx, _ = madx_ring_optics(tmp_path / "madx", 1, seed_beam, error=error, k2l=k2l)
    ocelot, _ = madx_ring_optics(tmp_path / "ocelot", 1, seed_beam, code="ocelot", error=error, k2l=k2l)
    shift = madx["tune_x_total"] - on_axis["tune_x_total"]
    assert abs(shift) > 1e-3, "the orbit through the sextupole must move the tune"
    for plane in "xy":
        assert ocelot[f"tune_{plane}_total"] == pytest.approx(madx[f"tune_{plane}_total"], abs=1e-3 * abs(shift))


def close(final, expected, rtol):
    return all(
        np.abs(got - want).max() < rtol * np.abs(want).max()
        for got, want in zip(final, expected)
    )
