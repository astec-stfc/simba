"""WarpX lines. Set WARPX_EXE to an RZ build (``warpx.rz``) to run the real binary."""

import os
import shutil

import h5py
import numpy as np
import pytest
from scipy.constants import c, e

from laura.models.element import BeamBeam, Quadrupole, Wiggler
from simba.Modules import Beams as rbf
from simba.Modules.Beams.fbpic import read_fbpic_beam_file

from test_hipace import _framework, _inputs, _lens, _stage

WARPX = os.environ.get("WARPX_EXE") or shutil.which("warpx.rz")
WARPX_3D = os.environ.get("WARPX_3D_EXE") or shutil.which("warpx.3d")
needs_warpx = pytest.mark.skipif(not WARPX, reason="no WarpX RZ binary (set WARPX_EXE)")
needs_warpx_3d = pytest.mark.skipif(not WARPX_3D, reason="no WarpX 3D binary (set WARPX_3D_EXE)")
# For the binary runs: AMReX takes 3/4 of the GPU up front by default, more than a small card
# shared with a display has; and shape 2-3 with 2 RZ modes crashes single-precision CUDA builds
# (WarpX's own RZ example too; double precision is fine), so the RZ runs use shape 1.
RUN_INPUTS = {"amrex.the_arena_init_size": 2**28, "algo.particle_shape": 1}


def test_inputs_carry_the_beam_the_run_length_and_overrides(tmp_path):
    framework = _framework(tmp_path, _stage(), offset_z=0.003, code="warpx",
                           hipace_settings={"inputs": {"warpx.verbose": 1}, "bunch_z_position": 1e-5,
                                            "single_precision": True})
    framework.track(track=False, postprocess=False)
    params = _inputs(framework)
    assert params["particles.species_names"] == "electrons beam"
    assert params["beam.injection_style"] == "external_file"
    assert params["beam.injection_file"] == "L_beam.h5"
    # The reference particle starts 3 mm in and leaves at the exit.
    assert float(params["stop_time"]) == pytest.approx((0.012 - 0.003) / c, rel=1e-3, abs=0)
    # The stage entrance is 3 mm behind the beam, at box z = 1e-5 - 0.003.
    assert float(params["electrons.zmin"]) == pytest.approx(1e-5 - 0.003, abs=1e-6)
    assert params["diag.species"] == "beam"
    assert params["warpx.verbose"] == "1"
    assert params["warpx.n_rz_azimuthal_modes"] == "1"  # no laser: m = 0, and WarpX can then...
    assert params["beam.initialize_self_fields"] == "1"  # ...start the beam from its own field
    path = os.path.join(framework.subdirectory, "L_beam.h5")
    with h5py.File(path) as f:
        species = f["data/0/particles/beam"]
        assert species["position/x"].dtype == np.float32  # loaded as WarpX's particle type
        assert species["positionOffset/z"].attrs["value"] == 0.0
    beam = rbf.beam()
    read_fbpic_beam_file(beam, path, species="beam")
    assert np.average(beam.z.val) == pytest.approx(1e-5, abs=1e-6)


@pytest.mark.parametrize("solver", ["yee", "psatd"])
def test_time_step_is_explicit_so_dumps_can_be_spaced(tmp_path, solver):
    framework = _framework(tmp_path, _stage(), code="warpx",
                           hipace_settings={"n_out": 4, "inputs": {"algo.maxwell_solver": solver}})
    framework.track(track=False, postprocess=False)
    params = _inputs(framework)
    dr, dz = 1e-4 / 16, 1e-4 / 32  # the test stage's box
    expected = (min(dr, dz) / c if solver == "psatd"
                else 1 / (c * np.sqrt((1 + 0.2105) / dr**2 + 1 / dz**2)))  # Yee, 1 mode: alpha = 0.2105
    dt = float(params["warpx.const_dt"])
    assert dt == pytest.approx(expected, rel=1e-9, abs=0)  # approx's default abs 1e-12 would pass anything
    assert int(params["diag.intervals"]) == int(float(params["stop_time"]) / dt // 4)


def _quad(k1=2.0, length=0.2):
    return Quadrupole(name="Q", machine_area="L", physical={"length": length, "middle": {"z": length / 2}},
                      magnetic={"length": length, "k1l": k1 * length})


@pytest.mark.parametrize("space_charge", ["off", "3d"])
def test_quad_line_runs_in_3d_about_the_beam(tmp_path, space_charge):
    framework = _framework(tmp_path, _quad(), code="warpx", hipace_settings={"single_precision": False})
    line = framework.latticeObjects["L"]
    line.space_charge_mode = space_charge
    framework.track(track=False, postprocess=False)
    params = _inputs(framework)
    assert line.executable == "warpx_3d"
    assert params["geometry.dims"] == "3"
    assert params["lattice.elements"] == "Q"
    brho = 1e9 / c  # 1 GeV/c electrons
    assert float(params["Q.dBdx"]) == pytest.approx(-2.0 * brho, rel=1e-6)
    beta = 1e9 / np.hypot(1e9, 0.51099895e6)
    assert float(params["warpx.moving_window_v"]) == pytest.approx(beta)
    travel = 0.2 - line.plasma_box_origin  # from the incoming reference particle (or centroid)
    assert float(params["stop_time"]) == pytest.approx(travel / (beta * c), rel=1e-6, abs=0)
    lo = [float(v) for v in params["geometry.prob_lo"].split()]
    assert lo[0] == pytest.approx(-10 * 2e-6, rel=0.2)  # 10 sigma_x of the incoming beam
    assert "beam.initialize_self_fields" not in params
    if space_charge == "3d":
        assert params["warpx.do_electrostatic"] == "relativistic"
        assert params["boundary.field_lo"] == "open open open"
    else:
        assert params["algo.maxwell_solver"] == "none"
        assert "warpx.do_electrostatic" not in params


def test_beam_beam_line_is_a_fixed_box_collision(tmp_path):
    bb = BeamBeam(name="BB", machine_area="L", physical={"middle": {"z": 0.0}},
                  simulation={"charge": 1.0, "n_particles": 1e10, "horizontal_sigma": 3e-6,
                              "vertical_sigma": 3e-6})
    framework = _framework(tmp_path, bb, code="warpx",
                           hipace_settings={"single_precision": False, "opposing_p0c": 2e9})
    framework.track(track=False, postprocess=False)
    line = framework.latticeObjects["L"]
    params = _inputs(framework)
    assert line.executable == "warpx_3d"
    assert params["particles.species_names"] == "beam opposing"
    assert "warpx.do_moving_window" not in params
    assert params["warpx.do_electrostatic"] == "relativistic"
    assert params["warpx.use_2d_slices_fft_solver"] == "1"
    d = 4 * 5e-6  # 4 sigma_z of the incoming beam; the element has no width
    assert float(params["opposing.z_m"]) == pytest.approx(d, rel=0.1)
    assert float(params["opposing.uz"]) == pytest.approx(-2e9 / 0.51099895e6)  # the override
    assert float(params["stop_time"]) == pytest.approx(2 * float(params["opposing.z_m"]) / c, rel=1e-9, abs=0)
    beam = rbf.beam()
    read_fbpic_beam_file(beam, os.path.join(framework.subdirectory, "L_beam.h5"), species="beam")
    assert np.mean(beam.z.val) == pytest.approx(-float(params["opposing.z_m"]), rel=1e-3)  # d before the IP
    assert line._drift_after == pytest.approx(-float(params["opposing.z_m"]))  # back to the IP, the line's end


def _undulator(length=0.3):
    return Wiggler(name="U", machine_area="L", physical={"length": length, "middle": {"z": length / 2}},
                   magnetic={"length": length, "period": 0.03, "strength": 1.4})


def test_undulator_line_is_a_boosted_fel(tmp_path):
    framework = _framework(tmp_path, _undulator(), code="warpx",
                           hipace_settings={"single_precision": False, "n_out": 3})
    framework.track(track=False, postprocess=False)
    params = _inputs(framework)
    gamma = np.hypot(1e9, 0.51099895e6) / 0.51099895e6
    factor = 1 + 1.4**2 / 2
    assert float(params["warpx.gamma_boost"]) == pytest.approx(gamma / np.sqrt(factor), rel=1e-4)
    assert params["particles.rigid_injected_species"] == "beam"
    line = framework.latticeObjects["L"]
    # WarpX z = 0 at the box front, where back-transformed snapshot times start
    assert float(params["geometry.prob_hi"].split()[2]) == pytest.approx(0.0, abs=1e-15)
    assert float(params["beam.zinject_plane"]) == pytest.approx(0.0 - line.zstart, rel=1e-9)  # the entrance
    assert params["diag.diag_type"] == "BackTransformed"
    assert params["diag.num_snapshots_lab"] == "4"
    assert "stop_time" not in params and "diag.intervals" not in params
    wavelength = 0.03 * factor / (2 * gamma**2)
    lo, hi = (float(params[f"geometry.prob_{s}"].split()[2]) for s in ("lo", "hi"))
    assert hi - lo > 10 * wavelength  # the slippage over 10 periods, at least
    nz = int(params["amr.n_cell"].split()[2])
    assert (hi - lo) / nz == pytest.approx(wavelength / 16, rel=0.05)
    assert "Q.type" not in params


def test_undulator_without_fel_is_a_field_in_a_transport_line(tmp_path):
    framework = _framework(tmp_path, _undulator(), code="warpx",
                           hipace_settings={"single_precision": False, "fel": False})
    framework.track(track=False, postprocess=False)
    params = _inputs(framework)
    assert "warpx.gamma_boost" not in params
    assert params["particles.B_ext_particle_init_style"] == "parse_B_ext_particle_function"


@needs_warpx
def test_lens_focuses_and_moves_the_beam_to_its_exit(tmp_path, monkeypatch):
    """Thin lens: px changes by -e g L x, and the beam leaves at the lens exit."""
    monkeypatch.setenv("WARPX_EXE", WARPX)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    framework = _framework(tmp_path, _lens(), code="warpx", hipace_settings={"inputs": RUN_INPUTS})
    framework.track()
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    assert len(out.x.val) == 1000
    assert out.Particles.total_charge.val == pytest.approx(-10e-12, rel=1e-6)
    assert np.mean(out.z.val) == pytest.approx(0.02, abs=1e-5)
    slope = np.polyfit(out.x.val, out.Particles.px.val, 1)[0]
    assert slope == pytest.approx(-e * 500.0 * 0.02, rel=2e-2)


# --- 3D lines in the binary, each against an independent number. A cold beam (no angular
# spread) makes the linear optics exact; a coarse z grid keeps the step count down.
COLD = {"sigma_px": 1e-3, "sigma_py": 1e-3}  # eV/c; the generator refuses exactly zero
COARSE = {"inputs": RUN_INPUTS | {"algo.particle_shape": 3}, "n_cell": [16, 16, 16]}


def _run(tmp_path, element, monkeypatch, cp=1e9, beam=COLD, **settings):
    monkeypatch.setenv("WARPX_EXE", WARPX or "")
    monkeypatch.setenv("WARPX_3D_EXE", WARPX_3D)
    framework = _framework(tmp_path, element, code="warpx", cp=cp, beam=beam,
                           hipace_settings=COARSE | settings)
    framework.track()
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    return out, framework


@needs_warpx_3d
def test_quad_matches_the_thick_lens_matrix(tmp_path, monkeypatch):
    """px/x at the exit is p M21 / M11 = -p sqrt(k) tan(sqrt(k) L) for a cold beam: the
    sign checks charge_sign Brho k1 (k1 > 0 focuses the electrons) and the size dBdx."""
    k1, length = 20.0, 0.05
    out, _ = _run(tmp_path, _quad(k1, length), monkeypatch)
    p = 1e9 * e / c
    slope = np.polyfit(out.x.val, out.Particles.px.val, 1)[0]
    assert slope == pytest.approx(-p * np.sqrt(k1) * np.tan(np.sqrt(k1) * length), rel=1e-2)
    assert len(out.x.val) == 1000
    assert np.mean(out.z.val) == pytest.approx(length, abs=1e-6)


@needs_warpx_3d
def test_beam_beam_kick_is_the_round_gaussian_lens(tmp_path, monkeypatch):
    """A cold beam through an opposing bunch of the same size: the linear fit of the kick
    is half the small-amplitude -N r_e m c / sigma^2 (averaging over the beam's own size)."""
    from scipy.constants import m_e, physical_constants

    n, sigma = 1e10, 2e-6
    bb = BeamBeam(name="BB", machine_area="L", physical={"middle": {"z": 0.0}},
                  simulation={"charge": 1.0, "n_particles": n, "horizontal_sigma": sigma,
                              "vertical_sigma": sigma})
    out, _ = _run(tmp_path, bb, monkeypatch, opposing_macroparticles=4000)
    r_e = physical_constants["classical electron radius"][0]
    slope = np.polyfit(out.x.val, out.Particles.px.val, 1)[0]
    assert slope == pytest.approx(-0.5 * n * r_e * m_e * c / sigma**2, rel=0.15)
    assert len(out.x.val) == 1000
    assert np.mean(out.z.val) == pytest.approx(0.0, abs=1e-6)  # drifted back to the IP


@needs_warpx_3d
def test_cavity_gains_its_transit_time_integral_on_crest(tmp_path, monkeypatch):
    """A standing-wave map, auto-phased by LAURA's reference tracking: the beam gains
    LAURA's transit-time integral (cavity_energy_gain), through WarpX's field expressions."""
    from laura.models.element import RFCavity
    from laura.translator.converters.converter import translate_elements
    from laura.translator.utils.fields import FieldMap
    from laura.translator.utils.fields.field_parameter import FieldParameter
    from laura.translator.utils.functions import cavity_energy_gain
    from laura.translator.utils.units import UnitValue

    frequency = 2.998e9
    wavelength = c / frequency
    z = np.linspace(0, 2 * wavelength, 2001)
    field = FieldMap()
    field.z = FieldParameter(name="z", value=UnitValue(z, "m"))
    field.Ez = FieldParameter(name="Ez", value=UnitValue(np.cos(2 * np.pi * z / wavelength), "V/m"))
    field.field_type, field.frequency, field.read = "1DElectroDynamic", frequency, True
    field.filename = str(tmp_path.parent / f"{tmp_path.name}_rf.hdf5")  # the framework cleans tmp_path
    path = field.write_field_file("hdf5")
    rf = RFCavity(name="C", machine_area="L",
                  physical={"length": 2 * wavelength, "middle": {"z": wavelength}},
                  cavity={"structure_type": "StandingWave", "frequency": frequency, "phase": 0.0},
                  simulation={"field_amplitude": 1e7, "field_definition": os.path.abspath(path)})
    expected = cavity_energy_gain(translate_elements([rf])["C"]) * 1e6  # eV
    out, _ = _run(tmp_path, rf, monkeypatch)
    gain = np.mean(np.hypot(out.Particles.cp.val, 0.51099895e6)) - np.hypot(1e9, 0.51099895e6)
    assert gain == pytest.approx(expected, rel=5e-3)
    assert np.mean(out.z.val) == pytest.approx(2 * wavelength, abs=1e-6)


@needs_warpx_3d
def test_space_charge_runs_at_a_gev(tmp_path, monkeypatch):
    """The relativistic Poisson solve asserts beta < 1, which single precision rounds away at
    a GeV: this needs a double-precision WarpX. At 1 GeV the beam barely grows over 5 cm
    (perveance ~4e-12): the self-forces cancel to 1/gamma^2, which needs a collocated grid."""
    framework = None
    quad = _quad(0.0, 0.05)
    monkeypatch.setenv("WARPX_3D_EXE", WARPX_3D)
    framework = _framework(tmp_path, quad, code="warpx", beam=COLD, hipace_settings=COARSE)
    framework.latticeObjects["L"].space_charge_mode = "3d"
    if framework.latticeObjects["L"]._particle_dtype() == np.float32:
        pytest.skip("needs a double-precision WarpX (beta rounds to 1 in single precision)")
    framework.track()
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    assert len(out.x.val) == 1000
    assert np.mean(out.z.val) == pytest.approx(0.05, abs=1e-6)
    assert np.std(out.x.val) == pytest.approx(2e-6, rel=0.05)  # no growth to speak of


@needs_warpx_3d
@pytest.mark.xfail(strict=True, reason="~8% of the beam is dropped at injection in the boosted frame (open)")
def test_fel_line_keeps_the_beam(tmp_path, monkeypatch):
    out, _ = _run(tmp_path, _undulator(0.06), monkeypatch, cp=50e6, beam={},
                  n_cell=[16, 16, 0], inputs=RUN_INPUTS)
    assert np.mean(out.z.val) == pytest.approx(0.06, abs=1e-5)  # the last lab snapshot, at the line end
    assert len(out.x.val) == 1000


@needs_warpx_3d
def test_fel_line_ends_at_the_line_end(tmp_path, monkeypatch):
    """Boosted frame and back-transformed snapshots: the beam comes out where the line ends.
    Needs double precision (single-precision BTD fails, WarpX's own FEL example too)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, framework = _run(tmp_path, _undulator(0.06), monkeypatch, cp=50e6, beam={},
                              n_cell=[16, 16, 0], inputs=RUN_INPUTS)
    assert np.mean(out.z.val) == pytest.approx(0.06, abs=1e-5)
    assert len(out.x.val) > 800
