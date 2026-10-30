"""Space charge in Xsuite: ``charge: space_charge_mode: 3d``, as Ocelot reads it."""

import os
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("xfields")

import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole
from simba import exceptions
from simba.Codes.Generators import frameworkGenerator
from simba.Codes.Xsuite.Xsuite import xsuiteLattice

LENGTH = 2.0
ENVELOPE_GROWTH = 1.496


def _machine(tmp_path, quadrupole=False):
    """A 2 m drift, or with a quadrupole in the middle of it."""
    elements = [
        Marker(
            name="START", machine_area="D", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
        ),
        Marker(
            name="END", machine_area="D", hardware_class="Marker",
            physical={"middle": {"x": 0.0, "y": 0.0, "z": LENGTH}},
        ),
    ]
    if quadrupole:
        elements.insert(1, Quadrupole(
            name="QUAD", machine_area="D",
            magnetic={"length": 0.5, "k1l": 0.5},
            physical={"length": 0.5, "middle": {"x": 0.0, "y": 0.0, "z": 1.0}},
        ))
    names = [e.name for e in elements]
    section = {"sections": {"D": {"elements": names}}}
    machine = LAURA(
        element_list=elements,
        layout={"default_layout": "line1", "layouts": {"line1": ["D"]}},
        section=section,
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, section


def _beam(directory, charge, particles):
    """Cold and round. Generated once per module: ``frameworkGenerator`` is
    unseeded, so both arms of a comparison must be handed the same file."""
    frameworkGenerator(
        global_parameters={"master_subdir": str(directory)},
        filename="START.openpmd.hdf5", initial_momentum=5e6,
        sigma_x=1e-3, sigma_px=1, sigma_y=1e-3, sigma_py=1,
        sigma_z=1e-3, sigma_pz=1,
        gaussian_cutoff_x=5, gaussian_cutoff_y=5, gaussian_cutoff_z=5,
        gaussian_cutoff_px=5, gaussian_cutoff_py=5, gaussian_cutoff_pz=5,
        charge=charge, number_of_particles=particles,
    ).write()
    return os.path.join(str(directory), "START.openpmd.hdf5")


@pytest.fixture(scope="module")
def bunch(tmp_path_factory):
    return _beam(tmp_path_factory.mktemp("bunch"), 100e-12, 20000)


@pytest.fixture(scope="module")
def faint(tmp_path_factory):
    """1e-19 C: space charge moves a particle about 4e-13 m (4e-9 m at 1 fC)."""
    return _beam(tmp_path_factory.mktemp("faint"), 1e-19, 2000)


def _framework(tmp_path, beam, code="xsuite", mode="3d", tracking=None, quadrupole=False):
    machine, section = _machine(tmp_path, quadrupole)
    settings = fw.FrameworkSettings()
    settings.files = {
        "D": {
            "code": code,
            "charge": {"space_charge_mode": mode},
            "input": {},
            "output": {"start_element": "START", "end_element": "END"},
            "tracking": tracking or {},
            "lsc_enable": False,
            "csr_enable": False,
        }
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(
        machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False
    )
    framework.loadSettings(settings=settings)
    shutil.copy(beam, os.path.join(framework.subdirectory, "START.openpmd.hdf5"))
    return framework


def _track(framework):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    return _read(os.path.join(framework.subdirectory, "END.openpmd.hdf5"))


def _read(path):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, path)
    return beam


def _growth(beam, out):
    return float(np.std(out.x.val) / np.std(beam.x.val))


@pytest.mark.parametrize(
    "mode, on", [("3d", True), ("3D", True), ("False", False), ("", False), ("off", False)]
)
def test_the_mode_is_read_as_ocelot_reads_it(mode, on):
    line = SimpleNamespace(file_block={"charge": {"space_charge_mode": mode}}, trackBeam=True)
    line.space_charge_mode = xsuiteLattice.space_charge_mode.fget(line)
    assert xsuiteLattice.space_charge.fget(line) is on


def test_no_charge_block_is_no_space_charge():
    line = SimpleNamespace(file_block={}, trackBeam=True)
    assert xsuiteLattice.space_charge_mode.fget(line) == ""


@pytest.mark.parametrize("mode, warns", [("2d", True), ("3D", False), ("False", False)])
def test_a_mode_xsuite_has_no_model_for_is_warned_about(mode, warns, tmp_path, faint):
    """Not tracked without space charge in silence, as every mode was."""
    line = _framework(tmp_path, faint, mode=mode)["D"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line.preProcess()
    said = any(isinstance(w.message, exceptions.SpaceChargeModeWarning) for w in caught)
    assert said is warns


def test_a_kick_every_step_at_its_middle(tmp_path, faint):
    framework = _framework(tmp_path, faint)
    _track(framework)
    table = framework["D"].line.get_table()
    s = [float(v) for v, name in zip(table.s, table.name) if name.startswith("simba_space_charge")]
    np.testing.assert_allclose(s, (np.arange(20) + 0.5) * 0.1, atol=1e-12)


def test_a_bunch_without_charge_is_tracked_as_without_space_charge(tmp_path, faint):
    """The kicks and their slices change nothing else. ``atol`` clears the faint
    bunch's own kicks, which reached 1.04e-12 m on a 5-sigma particle."""
    with_sc = _track(_framework(tmp_path / "on", faint, quadrupole=True))
    without = _track(_framework(tmp_path / "off", faint, mode="False", quadrupole=True))
    for coord in ("x", "y", "z"):
        np.testing.assert_allclose(
            getattr(with_sc, coord).val, getattr(without, coord).val, rtol=0, atol=1e-11
        )
    np.testing.assert_allclose(with_sc.cpx.val, without.cpx.val, rtol=0, atol=1e-3)


def test_the_bunch_grows_as_the_envelope_equation_says(tmp_path, bunch):
    """sigma'' = K / (4 sigma) slice by slice, K = 2 I / (I_A beta^3 gamma^3):
    1 mm becomes 1.496 mm over the 2 m. Without space charge it stays 1 mm."""
    beam = _read(bunch)
    growth = _growth(beam, _track(_framework(tmp_path, bunch)))
    assert growth == pytest.approx(ENVELOPE_GROWTH, rel=0.05)
    assert _growth(beam, _track(_framework(tmp_path / "off", bunch, mode="False"))) < 1.001


def test_xsuite_and_ocelot_agree(tmp_path, bunch):
    """The same bunch through Ocelot's 3D space charge: 1.2% apart."""
    pytest.importorskip("ocelot")
    beam = _read(bunch)
    xsuite = _growth(beam, _track(_framework(tmp_path / "xsuite", bunch)))
    ocelot = _growth(beam, _track(_framework(tmp_path / "ocelot", bunch, code="ocelot")))
    assert xsuite == pytest.approx(ocelot, rel=0.03)


@pytest.mark.parametrize("solver", ["FFTSolver3D", "FFTSolver2p5D", "FFTSolver2p5DAveraged"])
def test_every_solver_runs_on_cpu(solver, tmp_path, faint):
    """With pyFFTW installed the first kick raised an ``AssertionError``; simba
    now gives each solver a numpy plan of its own shape."""
    framework = _framework(tmp_path, faint)
    framework["D"].pic_solver = solver
    _track(framework)
    assert framework["D"].line.iscollective


def test_space_charge_over_several_turns(tmp_path, faint):
    framework = _framework(tmp_path, faint, tracking={"turns": 3})
    out = _track(framework)
    assert len(out.x.val) == 2000


def test_single_particle_studies_see_no_space_charge(tmp_path, faint):
    """The reference orbit, DA and the frequency map track probes, not a bunch."""
    framework = _framework(tmp_path, faint)
    _track(framework)
    line = framework["D"]
    assert line.line.iscollective
    assert not line.single_particle_line.iscollective
    names = line.single_particle_line.element_names
    assert "simba_space_charge_0" in names


def test_without_space_charge_the_line_is_its_own(tmp_path, faint):
    framework = _framework(tmp_path, faint, mode="False")
    _track(framework)
    line = framework["D"]
    assert line.single_particle_line is line.line


def _grid_and_particles():
    """An 8-cell grid over +-1 mm, and three particles, the last 5 mm out."""
    import xtrack as xt

    solver = SimpleNamespace(pic_solver="FFTSolver3D")
    kick = _space_charge_kick(xsuiteLattice._space_charge_fftplan(solver, 8))
    line = SimpleNamespace(line=SimpleNamespace(elements=[kick]), objectname="D")
    particles = xt.Particles(
        p0c=5e6, mass0=xt.ELECTRON_MASS_EV, x=[0.0, 0.0, 5e-3], y=[0, 0, 0], zeta=[0, 0, 0]
    )
    return line, particles


def test_a_beam_that_outgrows_its_grid_is_warned_about():
    """Grids are sized on the first pass, and a particle outside feels no field."""
    line, particles = _grid_and_particles()
    with pytest.warns(exceptions.SpaceChargeOffGridWarning, match="33%"):
        xsuiteLattice.check_space_charge_grids(line, particles)


@pytest.mark.filterwarnings("error")
def test_a_lost_particle_outside_the_grid_is_not_counted():
    line, particles = _grid_and_particles()
    particles.state[2] = -1
    xsuiteLattice.check_space_charge_grids(line, particles)


@pytest.mark.parametrize(
    "value, on", [(True, True), ("true", True), ("yes", True), (False, False), ("off", False)]
)
def test_the_resize_setting(value, on):
    line = SimpleNamespace(file_block={"charge": {"space_charge_resize": value}})
    assert xsuiteLattice.space_charge_resize.fget(line) is on


def test_the_grids_are_not_resized_unless_asked():
    line = SimpleNamespace(file_block={"charge": {"space_charge_mode": "3d"}})
    assert xsuiteLattice.space_charge_resize.fget(line) is False


@pytest.fixture(scope="module")
def strong(tmp_path_factory):
    """3 nC: grows 8-fold over 2 m, and 57% leaves grids sized without space charge."""
    return _beam(tmp_path_factory.mktemp("strong"), 3e-9, 5000)


def _off_grid(framework):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        framework.track()
    return [w for w in caught if isinstance(w.message, exceptions.SpaceChargeOffGridWarning)]


def _last_grid(framework):
    import xfields as xf

    kicks = [e for e in framework["D"].line.elements if isinstance(e, xf.SpaceCharge3D)]
    return float(kicks[-1].fieldmap.x_grid[-1])


def test_a_resized_grid_holds_a_bunch_that_outgrew_the_first(tmp_path, strong):
    as_sized = _framework(tmp_path / "as_sized", strong)
    resized = _framework(tmp_path / "resized", strong)
    resized["D"].file_block["charge"]["space_charge_resize"] = True
    assert _off_grid(as_sized)
    assert not _off_grid(resized)
    assert _last_grid(resized) > 4 * _last_grid(as_sized)


def _space_charge_kick(fftplan):
    import xfields as xf

    return xf.SpaceCharge3D(
        length=0.1, x_range=(-1e-3, 1e-3), y_range=(-1e-3, 1e-3),
        z_range=(-1e-3, 1e-3), nx=8, ny=8, nz=8, solver="FFTSolver3D",
        gamma0=10.0, fftplan=fftplan,
    )
