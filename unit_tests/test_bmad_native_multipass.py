"""Bmad lines written as native Bmad multipass (`multipass: native`)."""

import pytest

from simba.Codes.Bmad.Bmad import bmadLattice
from unit_tests.test_path_arc_lengths import ERL, machine


class Line:
    """A Bmad lattice stub carrying only what these methods read."""

    def __init__(self, start="INJ_Q", end="DMP_Q", native=True):
        self.machine = machine(ERL)
        self.file_block = {"multipass": "native"} if native else {}
        self.start, self.end = start, end
        self.objectname = "ERL"

    native_multipass = bmadLattice.native_multipass
    _native_multipass_lattice = bmadLattice._native_multipass_lattice
    _tao_name = bmadLattice._tao_name


def test_one_pass_of_a_shared_element_is_its_slave():
    assert Line()._tao_name("LIN_Q.2") == "LIN_Q\\2"
    assert Line()._tao_name("ARC_Q") == "ARC_Q"


def test_flattened_lines_keep_their_own_names():
    assert Line(native=False)._tao_name("LIN_Q.2") == "LIN_Q.2"


def test_native_multipass_needs_the_whole_path():
    with pytest.raises(ValueError, match="whole beam path"):
        Line(end="LIN_Q#2")._native_multipass_lattice()


class Tao:
    """`ele_lord_slave` rows as Tao gives them for a multipass slave."""

    def ele_lord_slave(self, name):
        return [
            {"type": "Element", "location_name": "0>>9", "status": "Multipass_Lord"},
            {"type": "Slave", "location_name": "0>>2", "status": "Multipass_Slave"},
            {"type": "Slave", "location_name": "0>>7", "status": "Multipass_Slave"},
            {"type": "Lord", "location_name": "0>>9", "status": "Multipass_Lord"},
        ]


def test_a_multipass_slave_is_read_where_it_is_not_at_its_last_sibling():
    stub = type("Stub", (), {"tao": Tao(), "_exit_location": bmadLattice._exit_location})()
    assert stub._exit_location("LIN_Q\\1") == "LIN_Q\\1"
