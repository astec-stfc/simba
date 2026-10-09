"""Shared fixtures: the small FODO cell many tests track through."""

import pytest
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole


def _quads():
    return [
        Quadrupole(
            name="QUAD1F", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": -1},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 0.75}},
        ),
        Quadrupole(
            name="QUAD1D", machine_area="FODO",
            magnetic={"length": 1.0, "k1l": 1.0},
            physical={"length": 1.0, "middle": {"x": 0.0, "y": 0.0, "z": 3.25}},
        ),
    ]


def _build(tmp_path):
    middle = _quads()
    m1 = Marker(
        name="M1", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    end_z = middle[-1].physical.middle.z + middle[-1].physical.length
    m3 = Marker(
        name="M3", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": end_z}},
    )
    names = ["M1"] + [e.name for e in middle] + ["M3"]
    machine = LAURA(
        element_list=[m1, *middle, m3],
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section={"sections": {"FODO": names}},
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, names


@pytest.fixture
def fodo_elements():
    """A fresh QUAD1F / QUAD1D pair."""
    return _quads()


@pytest.fixture(scope="session")
def fodo_machine():
    """``(tmp_path) -> (machine, names)``: the quads between M1 and M3, exported
    to ``tmp_path/lattice``. Session-scoped, as it is stateless, so any scope can use it."""
    return _build
