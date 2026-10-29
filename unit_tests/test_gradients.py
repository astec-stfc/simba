"""Forward-mode gradients from Cheetah, checked against finite differences of the tracking."""

import os
import warnings
from functools import reduce

import h5py
import numpy as np
import pytest

pytest.importorskip("cheetah")

import simba.Framework as fw
from simba import exceptions
from simba.Codes.Generators.Generators import frameworkGenerator
from simba.Modules.gradients import read_gradients
from laura.models.element import Marker, RFCavity, Screen
from helpers import fodo_machine, fodo_quads, read_beam, skip_missing

COORDINATES = ("x", "cpx", "y", "cpy", "t", "cpz")


def _framework(tmp_path, middle, lines, parameters, finite_difference=()):
    """
    The FODO cell with the ``middle`` elements between the quads, tracked as ``lines``
    (name -> code, start, end), with the ``finite_difference`` lines asking for it.
    """
    quads = fodo_quads()
    machine, names, section = fodo_machine(tmp_path, middle=[quads[0], *middle, quads[1]])
    settings = fw.FrameworkSettings()
    settings["global"] = {"gradients": {"parameters": parameters}}
    settings.files = {
        name: {
            "code": code,
            "charge": {"space_charge_mode": "False"},
            "input": {},
            "output": {"start_element": start, "end_element": end},
            **({"gradients": {"finite_difference": True}} if name in finite_difference else {}),
        }
        for name, (code, start, end) in lines.items()
    }
    settings.layout = machine.layout
    settings.section = section
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    # one beam for every run: the generator is unseeded
    frameworkGenerator(
        global_parameters={"master_subdir": framework.subdirectory},
        filename="M1.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4, sigma_px=1e3, sigma_y=1e-4, sigma_py=1e3, sigma_z=1e-3, sigma_pz=1e3,
        charge=100e-12,
    ).write()
    return framework


def _value(framework, parameter):
    name, key = parameter.split(":")
    return reduce(getattr, key.split("."), framework.getElement(name))


def _track(framework, line, changes={}):
    """
    Track with ``changes`` (ELEMENT:keyword -> value), then put them back; the end particles
    and ``line``'s beta_x, if it is a Cheetah line.
    """
    original = {parameter: _value(framework, parameter) for parameter in changes}
    for parameter, value in changes.items():
        framework.modifyElement(*parameter.split(":"), value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        framework.track()
    for parameter, value in original.items():
        framework.modifyElement(*parameter.split(":"), value)
    subdir = framework.subdirectory
    beam = read_beam(subdir, "M3")
    particles = np.stack([getattr(beam, c).val for c in COORDINATES], -1)
    twiss = os.path.join(subdir, f"{line}_twiss.cheetah.hdf5")
    if not os.path.isfile(twiss):
        return particles, None
    with h5py.File(twiss) as f:
        beta_x = f["Twiss"]["beta_x"][()]
    return particles, beta_x


def _check_against_finite_differences(framework, line, grads, column, step=1e-4):
    parameter = grads["parameters"][column]
    value = _value(framework, parameter)
    plus, beta_plus = _track(framework, line, {parameter: value + step})
    minus, beta_minus = _track(framework, line, {parameter: value - step})
    fd = (plus - minus) / (2 * step)
    end = grads["Beams"]["M3"][..., column]
    assert end.shape == fd.shape
    for c in range(6):
        # plus the round-off a finite difference sees in coordinates this large
        noise = 1e-10 * np.abs(plus[:, c]).max() / step
        np.testing.assert_allclose(end[:, c], fd[:, c], rtol=0, atol=1e-6 * np.abs(fd[:, c]).max() + noise)
    if beta_plus is None:
        return
    fd_beta = (beta_plus - beta_minus) / (2 * step)
    np.testing.assert_allclose(
        grads["Twiss"]["beta_x"][:, column], fd_beta, rtol=0, atol=1e-6 * np.abs(fd_beta).max(),
    )


def test_cheetah_gradients_match_finite_differences(tmp_path):
    screen = Screen(
        name="SCR", machine_area="FODO",
        physical={"length": 0.1, "middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )
    framework = _framework(tmp_path, [screen], {"FODO": ("cheetah", "M1", "M3")}, ["QUAD1F:k1l"])
    _track(framework, "FODO")

    grads = read_gradients(os.path.join(framework.subdirectory, "FODO_gradients.cheetah.hdf5"))
    assert grads["parameters"] == ["QUAD1F:k1l"]
    np.testing.assert_allclose(grads["parameter_values"], [-1.0])
    assert grads["attrs"]["coordinates"].tolist() == list(COORDINATES)
    assert any("SCR" in name for name in grads["Beams"])
    assert grads["Twiss"]["beta_x"].shape == (len(grads["Twiss"]["s"]), 1)
    _check_against_finite_differences(framework, "FODO", grads, 0)


def _split_fodo(tmp_path, code_b="cheetah", cavity=False, finite_difference=()):
    """
    The FODO cell as two lines, A (M1 to M2, holding QUAD1F) and B (M2 to M3, holding QUAD1D);
    ``cavity`` puts an off-crest cavity in A, whose phase moves B's reference energy and time.
    """
    middle = [Marker(
        name="M2", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 2.0}},
    )]
    parameters = ["QUAD1F:k1l", "QUAD1D:k1l"]
    if cavity:
        middle.insert(0, RFCavity(
            name="CAV1", machine_area="FODO",
            physical={"length": 0.2, "middle": {"x": 0.0, "y": 0.0, "z": 1.6}},
            cavity={"frequency": 3e9, "phase": 20.0},
            simulation={"field_amplitude": 1e6},
        ))
        parameters = ["CAV1:cavity.phase", "QUAD1D:k1l"]
    lines = {"A": ("cheetah", "M1", "M2"), "B": (code_b, "M2", "M3")}
    return _framework(tmp_path, middle, lines, parameters, finite_difference)


def test_gradients_chain_across_cheetah_lines(tmp_path):
    framework = _split_fodo(tmp_path)
    _track(framework, "B")

    grads = read_gradients(os.path.join(framework.subdirectory, "B_gradients.cheetah.hdf5"))
    # the upstream parameter comes first, carried on from line A
    assert grads["parameters"] == ["QUAD1F:k1l", "QUAD1D:k1l"]
    np.testing.assert_allclose(grads["parameter_values"], [-1.0, 1.0])
    _check_against_finite_differences(framework, "B", grads, 0)
    _check_against_finite_differences(framework, "B", grads, 1)


def test_an_upstream_cavity_moves_the_next_lines_reference(tmp_path):
    framework = _split_fodo(tmp_path, cavity=True)
    _track(framework, "B")

    grads = read_gradients(os.path.join(framework.subdirectory, "B_gradients.cheetah.hdf5"))
    assert grads["parameters"] == ["CAV1:cavity.phase", "QUAD1D:k1l"]
    assert np.abs(grads["Twiss"]["energy"][:, 0]).max() > 0
    _check_against_finite_differences(framework, "B", grads, 0, step=1e-3)


def test_gradients_stop_at_a_code_that_cannot_differentiate(tmp_path):
    skip_missing("ocelot")
    framework = _split_fodo(tmp_path, code_b="ocelot")
    with pytest.warns(exceptions.GradientsUnsupportedWarning):
        framework.track()


@pytest.mark.parametrize("cavity", [False, True])
def test_finite_differences_carry_gradients_through_ocelot(tmp_path, cavity):
    skip_missing("ocelot")
    framework = _split_fodo(tmp_path, code_b="ocelot", cavity=cavity, finite_difference=["B"])
    _track(framework, "B")

    grads = read_gradients(os.path.join(framework.subdirectory, "B_gradients.ocelot.hdf5"))
    assert grads["parameters"] == (["CAV1:cavity.phase"] if cavity else ["QUAD1F:k1l"]) + ["QUAD1D:k1l"]
    assert grads["attrs"]["method"] == "finite_difference"
    assert os.path.isfile(os.path.join(framework.subdirectory, "B_jacobian.ocelot.hdf5"))
    _check_against_finite_differences(framework, "B", grads, 0, step=1e-3 if cavity else 1e-4)
    _check_against_finite_differences(framework, "B", grads, 1)


def test_rematching_breaks_the_chain(tmp_path):
    framework = _split_fodo(tmp_path)
    framework.latticeObjects["B"].file_block["input"]["twiss"] = {"beta_x": 5.0, "alpha_x": 0.0}
    with warnings.catch_warnings(), pytest.raises(exceptions.GradientChainError):
        warnings.simplefilter("ignore")
        framework.track()


@pytest.mark.parametrize("p0", [30e6, 25e6])  # no energy change, and a cavity's
def test_the_differentiable_reference_time_is_the_written_one(p0):
    """The tangents' reference time (torch) is the forward run's (numpy), and differentiates where p == p0."""
    import torch
    from torch.func import jvp
    from simba.FrameworkHelperFunctions import reference_time as written
    from simba.Modules.Beams.cheetah import reference_time

    mass, start = 0.511e6, (1e-9, 2.0, p0)
    p = torch.tensor(30e6, dtype=torch.float64)
    assert float(reference_time(torch.tensor(7.0, dtype=torch.float64), p, mass, start)) == pytest.approx(
        written(7.0, 30e6, mass, start), rel=1e-12
    )
    _, dt = jvp(lambda q: reference_time(torch.tensor(7.0, dtype=torch.float64), q, mass, start), (p,), (torch.ones_like(p),))
    assert torch.isfinite(dt)
