"""HiPACE++ lines. Set HIPACE_EXE (or put ``hipace`` on PATH) to run the real binary."""

import os
import shutil

import numpy as np
import pytest
from scipy.constants import c, e

import simba.Framework as fw
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Plasma, PlasmaLens
from simba.Codes.Generators import frameworkGenerator
from simba.Modules import Beams as rbf
from simba.Modules.Beams.fbpic import read_fbpic_beam_file
from simba.Modules.Beams.hipace import write_hipace_beam_file

HIPACE = os.environ.get("HIPACE_EXE") or shutil.which("hipace")
needs_hipace = pytest.mark.skipif(not HIPACE, reason="no hipace binary (set HIPACE_EXE)")

BOX = {"r_max": 1e-4, "n_longitudinal": 32, "n_radial": 16,
       "min_longitudinal_position": -5e-5, "max_longitudinal_position": 5e-5}


def _lens(length=0.02):
    return PlasmaLens(name="LENS", machine_area="L", physical={"length": length, "middle": {"z": length / 2}},
                      simulation=BOX | {"dt_bunch": "1e-12"}, focusing_strength=500.0)


def _stage(length=0.012):
    return Plasma(name="STAGE", machine_area="L", physical={"length": length, "middle": {"z": length / 2}},
                  plasma={"density": 1e23, "density_profile": False}, simulation=BOX | {"dt_bunch": "1e-12"})


def _framework(tmp_path, element, offset_z=0.0, cp=1e9, hipace_settings=None, code="hipace", beam=None):
    end = element.physical.length
    m1 = Marker(name="M1", machine_area="L", hardware_class="Marker", physical={"middle": {"z": 0.0}})
    m3 = Marker(name="M3", machine_area="L", hardware_class="Marker", physical={"middle": {"z": end}})
    names = ["M1", element.name, "M3"]
    machine = LAURA(element_list=[m1, element, m3], section={"sections": {"L": names}},
                    layout={"default_layout": "line1", "layouts": {"line1": ["L"]}})
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    settings = fw.FrameworkSettings()
    settings.files = {"L": {"code": code, "charge": {"space_charge_mode": "False"}, "input": {},
                            "output": {"start_element": "M1", "end_element": "M3"}}}
    settings.layout = machine.layout
    settings.section = {"sections": {"L": names}}
    settings.element_list = f"{tmp_path}/lattice"
    if hipace_settings:
        settings["global"] = {**(settings.get("global") or {}), f"{code.upper()}settings": hipace_settings}
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    frameworkGenerator(**{
        "global_parameters": {"master_subdir": framework.subdirectory}, "filename": "M1.openpmd.hdf5",
        "number_of_particles": 1000, "initial_momentum": cp, "offset_z": offset_z,
        "sigma_x": 2e-6, "sigma_px": 10, "sigma_y": 2e-6, "sigma_py": 10, "sigma_z": 5e-6, "sigma_pz": 1e3,
        "gaussian_cutoff_x": 3, "gaussian_cutoff_y": 3, "gaussian_cutoff_z": 3,
        "gaussian_cutoff_px": 3, "gaussian_cutoff_py": 3, "gaussian_cutoff_pz": 3, "charge": 10e-12,
    } | (beam or {})).write()
    return framework


def _inputs(framework):
    with open(os.path.join(framework.subdirectory, "L.in")) as f:
        return dict(line.split(" = ", 1) for line in f.read().splitlines() if " = " in line)


def test_beam_file_round_trips_through_the_openpmd_reader(tmp_path):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, _write_beam(tmp_path))
    path = str(tmp_path / "beam.h5")
    charge, mass = write_hipace_beam_file(beam, path, zstart=0.5)
    assert charge == pytest.approx(-e)
    back = rbf.beam()
    read_fbpic_beam_file(back, path, z_offset=0.5, species="beam")
    for coord in ("x", "y", "z", "cpx", "cpy", "cpz"):
        np.testing.assert_allclose(getattr(back, coord).val, getattr(beam, coord).val, rtol=1e-12, atol=1e-18)
    assert back.Particles.total_charge.val == pytest.approx(beam.Particles.total_charge.val)


def _write_beam(tmp_path):
    frameworkGenerator(global_parameters={"master_subdir": str(tmp_path)}, filename="in.openpmd.hdf5",
                       number_of_particles=100, initial_momentum=1e8, offset_z=0.5,
                       sigma_x=1e-6, sigma_y=1e-6, sigma_z=1e-6, sigma_px=1e3, sigma_py=1e3, sigma_pz=1e5,
                       charge=1e-12).write()
    return str(tmp_path / "in.openpmd.hdf5")


def test_mixed_species_are_refused(tmp_path):
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(beam, _write_beam(tmp_path))
    beam.Particles.particle_charge.val[0] *= -1
    with pytest.raises(ValueError, match="species"):
        write_hipace_beam_file(beam, str(tmp_path / "beam.h5"))


def test_inputs_carry_the_beam_the_start_time_and_overrides(tmp_path):
    framework = _framework(tmp_path, _stage(), offset_z=0.003,
                           hipace_settings={"inputs": {"hipace.verbose": 1}, "bunch_z_position": 1e-5})
    framework.track(track=False, postprocess=False)
    params = _inputs(framework)
    assert params["beams.names"] == "beam"
    assert params["beam.injection_type"] == "from_file"
    assert params["beam.input_file"] == "L_beam.h5"
    assert float(params["hipace.initial_time"]) == pytest.approx(0.003 / c, rel=1e-3)
    assert int(params["max_step"]) == int(np.ceil((0.012 - 0.003) / c / 1e-12))
    assert params["hipace.verbose"] == "1"
    assert params["boundary.field"] == "Dirichlet"
    assert os.path.isfile(os.path.join(framework.subdirectory, "L_beam.h5"))
    # The beam sits at bunch_z_position in the box.
    beam = rbf.beam()
    read_fbpic_beam_file(beam, os.path.join(framework.subdirectory, "L_beam.h5"), species="beam")
    assert np.average(beam.z.val) == pytest.approx(1e-5, abs=1e-6)


@needs_hipace
def test_lens_focuses_and_moves_the_beam_to_its_exit(tmp_path, monkeypatch):
    """Thin lens: px changes by -e g L x, and the beam leaves at the lens exit."""
    monkeypatch.setenv("HIPACE_EXE", HIPACE)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    framework = _framework(tmp_path, _lens())
    framework.track()
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    assert len(out.x.val) == 1000
    assert out.Particles.total_charge.val == pytest.approx(-10e-12)
    assert np.mean(out.z.val) == pytest.approx(0.02, abs=1e-6)
    slope = np.polyfit(out.x.val, out.Particles.px.val, 1)[0]
    assert slope == pytest.approx(-e * 500.0 * 0.02, rel=1e-2)


@needs_hipace
def test_beam_starting_inside_a_stage_runs_from_there(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPACE_EXE", HIPACE)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    framework = _framework(tmp_path, _stage(), offset_z=0.009)
    framework.track()
    out = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(out, os.path.join(framework.subdirectory, "M3.openpmd.hdf5"))
    assert out.Particles.total_charge.val == pytest.approx(-10e-12)
    assert np.mean(out.z.val) == pytest.approx(0.012, abs=1e-6)
