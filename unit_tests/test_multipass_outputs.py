"""Where a line's output beams are placed on a multipass path, and Ocelot's
synchrotron radiation for its dipoles."""

import pytest

from laura.models.element import Dipole
from simba.Codes.Ocelot.Ocelot import ocelotLattice
from simba.Framework_objects import frameworkLattice
from unit_tests.test_path_arc_lengths import ERL, machine


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
        self.names = ["B"]
        self.elementObjects = {"B": bend}
        self.lat_obj = type("Lat", (), {"sequence": [object()]})()

    physproc_radiation = ocelotLattice.physproc_radiation


def test_each_dipole_radiates_at_its_own_radius():
    ((proc, _, radius),) = Bends().physproc_radiation()
    assert radius == pytest.approx(2.0)
    assert (proc.radius, proc.energy_loss, proc.quant_diff) == (pytest.approx(2.0), True, True)
