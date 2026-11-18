"""Where a line's output beams are placed on a multipass path, and Ocelot's
synchrotron radiation for its dipoles."""

from types import SimpleNamespace

import numpy as np
import pytest

from laura.models.element import Dipole
from ocelot import ParticleArray
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Codes.Ocelot.radiation import DipoleRadiation
from simba.Framework_objects import frameworkLattice
from test_path_arc_lengths import ERL, machine


class Line:
    """A lattice stub carrying only what `output_s` and the start/end names read."""

    def __init__(self, machine, start="INJ_Q", end="LIN_Q#2"):
        self.machine = machine
        self.elementObjects = dict(machine.elements)
        self.elements = {"LIN_Q.2": machine.lattices["ERL"].element_on_pass("LIN_Q#2")}
        self.start, self.end = start, end

    output_s = frameworkLattice.output_s
    _named_element = frameworkLattice._named_element
    start_name = frameworkLattice.start_name
    end_name = frameworkLattice.end_name
    startObject = frameworkLattice.startObject
    endObject = frameworkLattice.endObject


def test_a_line_ending_on_one_pass_names_it_as_the_line_does():
    line = Line(machine(ERL))
    assert (line.start_name, line.end_name) == ("INJ_Q", "LIN_Q.2")
    assert line.endObject is line.elements["LIN_Q.2"]
    assert line.startObject is line.elementObjects["INJ_Q"]


def test_a_later_pass_is_placed_at_its_arc_length_along_the_path():
    line = Line(machine(ERL))
    arc = line.machine.lattices["ERL"].arc_lengths()
    assert line.output_s("LIN_Q.2") == pytest.approx(arc["LIN_Q#2"])
    assert line.output_s("LIN_Q.2") > line.output_s("LIN_Q.1")


def test_an_element_entered_once_keeps_its_own_s():
    line = Line(machine(ERL))
    assert line.output_s("DMP_Q") == line.elementObjects["DMP_Q"].physical.s


class Bends:
    """An Ocelot lattice stub carrying only what `physproc_radiation` reads."""

    radiation = "quantum"

    def __init__(self):
        bend = Dipole(name="B", hardware_class="Magnet", machine_area="A",
                      magnetic={"length": 0.5, "angle": 0.25}, physical={"length": 0.5})
        self.names = ["B", "M"]
        self.elementObjects = {"B": bend}
        self.lat_obj = SimpleNamespace(sequence=[SimpleNamespace(angle=0.25, l=0.5), object()])

    physproc_radiation = ocelotLattice.physproc_radiation


def test_each_dipole_radiates_at_its_own_radius():
    ((proc, _, _, radius),) = Bends().physproc_radiation()
    assert radius == pytest.approx(2.0)
    assert (proc.curvature, proc.quantum) == (pytest.approx(0.5), True)


def _beam(n=20000, energy=2.0, x=0.0):
    beam = ParticleArray(n)
    beam.E = energy
    beam.rparticles[:] = 0.0
    beam.rparticles[0] = x
    beam.rparticles[1] = 1e-4
    return beam


def test_a_full_circle_loses_u0():
    """U0 = C_gamma E^4 / rho, 88.46 keV at 1 GeV and 1 m; per metre it scales as E^4 / rho^2."""
    rho, beam = 5.0, _beam()
    DipoleRadiation(1 / rho, quantum=False).apply(beam, 2 * np.pi * rho)
    u0 = 88.46e3 * 2.0**4 / rho
    p0c = np.sqrt((beam.E * 1e9) ** 2 - 0.511e6**2)
    assert -beam.rparticles[5].mean() * p0c == pytest.approx(u0, rel=1e-3)
    # the transverse momentum goes with the energy
    assert beam.rparticles[1].mean() == pytest.approx(1e-4 * (1 - u0 / 2e9), rel=1e-6)


def test_the_loss_follows_the_path_length():
    """Outside a positive bend the path, and the loss, is longer by (1 + h x): the damping partition."""
    rho = 5.0
    inside, outside = _beam(1, x=-0.01), _beam(1, x=0.01)
    for beam in (inside, outside):
        DipoleRadiation(1 / rho, quantum=False).apply(beam, 1.0)
    ratio = outside.rparticles[5][0] / inside.rparticles[5][0]
    assert ratio == pytest.approx((1 + 0.01 / rho) / (1 - 0.01 / rho), rel=1e-9)


def test_in_a_fixed_field_the_loss_goes_as_the_energy_squared():
    """An off-energy particle bends on rho p / p0, so loses as E^2 B^2, not E^4 / rho^2: Jz = 2 + D."""
    low, high = _beam(1), _beam(1)
    low.rparticles[5], high.rparticles[5] = -0.01, 0.01
    for beam in (low, high):
        DipoleRadiation(0.2, quantum=False).apply(beam, 1.0)
    ratio = (high.rparticles[5][0] - 0.01) / (low.rparticles[5][0] + 0.01)
    assert ratio == pytest.approx((1.01 / 0.99) ** 2, rel=1e-6)


def test_quantum_kicks_spread_the_energy_at_the_dipole_rate():
    """d<delta^2>/ds = 55/(24 sqrt 3) r_e lambda_c gamma^5 / rho^3."""
    rho, length, beam = 5.0, 1.0, _beam()
    np.random.seed(1)
    DipoleRadiation(1 / rho, quantum=True).apply(beam, length)
    gamma = 2e9 / 0.51099895e6
    rate = 55 / (24 * np.sqrt(3)) * 2.8179403e-15 * 3.8615927e-13 * gamma**5 / rho**3
    assert beam.rparticles[5].std() == pytest.approx(np.sqrt(rate * length), rel=0.02)


def test_a_dipole_radiates_along_its_length():
    """Ocelot gives a process starting and ending at one element dz = 0: no radiation."""
    bends = Bends()
    ((_, start, end, _),) = bends.physproc_radiation()
    assert (start, end) == tuple(bends.lat_obj.sequence)
