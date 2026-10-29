"""does Xsuite's `energy_program` rescale element `k`?"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ramped_quadrupole():
    """One quadrupole whose reference energy doubles over the program."""
    xt = pytest.importorskip("xtrack")
    line = xt.Line(elements=[xt.Quadrupole(length=0.3, k1=1.2)], element_names=["q"])
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV)
    mass0 = float(line.particle_ref.mass0)
    energy0 = float(line.particle_ref.energy0[0])
    line.energy_program = xt.EnergyProgram(
        t_s=np.array([0.0, 1e-3]),
        kinetic_energy0=np.array([energy0 - mass0, 2 * energy0 - mass0]),
    )
    line.build_tracker()
    line.enable_time_dependent_vars = True
    return line


def sample(line, t):
    """`(p0c, k1, px)` after one pass at program time `t`."""
    line.vars["t_turn_s"] = t
    particles = line.build_particles(x=1e-3, px=0, y=0, py=0)
    line.track(particles)
    return (
        float(line.particle_ref.p0c[0]),
        float(line["q"].k1),
        float(particles.px[0]),
    )


def test_the_program_doubles_the_reference_momentum(ramped_quadrupole):
    start = sample(ramped_quadrupole, 0.0)[0]
    end = sample(ramped_quadrupole, 1e-3)[0]
    assert end == pytest.approx(2 * start, rel=1e-6)


def test_k1_is_not_rescaled(ramped_quadrupole):
    """The question R10 was asked to answer."""
    for t in (0.0, 5e-4, 1e-3):
        assert sample(ramped_quadrupole, t)[1] == pytest.approx(1.2, rel=1e-12)


def test_the_normalised_optics_are_unchanged_through_the_ramp(ramped_quadrupole):
    """`px` is normalised to the reference momentum, so an unchanged `px`
    through a doubling of `p0c` means the focusing seen by the beam is the
    same -- the magnet kept its `k` and its field rose with the energy."""
    deflections = [sample(ramped_quadrupole, t)[2] for t in (0.0, 5e-4, 1e-3)]
    assert deflections[1] == pytest.approx(deflections[0], rel=1e-9)
    assert deflections[2] == pytest.approx(deflections[0], rel=1e-9)


def test_the_absolute_kick_grows_with_the_energy(ramped_quadrupole):
    """The other side of the same coin, and the thing that makes it a ramp:
    the same normalised kick at twice the momentum is twice the field."""
    first = sample(ramped_quadrupole, 0.0)
    last = sample(ramped_quadrupole, 1e-3)
    assert last[2] * last[0] == pytest.approx(2 * first[2] * first[0], rel=1e-6)
