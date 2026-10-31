"""GPT, OPAL and ASTRA decks are written in a frame local to the first element."""
import math

import pytest
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole

import simba.Framework as fw

THETA, X0, Z0 = 0.3, 10.0, 100.0


def _phys(s, length=0.0):
    # a positive theta heads the beam towards -x in LAURA
    middle = {"x": X0 - s * math.sin(THETA), "y": 0.0, "z": Z0 + s * math.cos(THETA)}
    return {"length": length, "middle": middle, "global_rotation": {"theta": THETA}}


def _lattice(tmp_path, code, **file_block):
    elements = [
        Marker(name="M1", machine_area="L", hardware_class="Marker", physical=_phys(0)),
        Quadrupole(name="Q1", machine_area="L", magnetic={"length": 1.0, "k1l": 1.0}, physical=_phys(1.5, 1.0)),
        Marker(name="M3", machine_area="L", hardware_class="Marker", physical=_phys(3)),
    ]
    names = [e.name for e in elements]
    machine = LAURA(
        element_list=elements,
        layout={"default_layout": "l", "layouts": {"l": ["L"]}},
        section={"sections": {"L": names}},
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    settings = fw.FrameworkSettings()
    settings.files = {"L": {
        "code": code,
        "charge": {"space_charge_mode": "False"},
        "input": {},
        "output": {"start_element": "M1", "end_element": "M3"},
        **file_block,
    }}
    settings.layout = machine.layout
    settings.section = {"sections": {"L": names}}
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    return framework.latticeObjects["L"]


@pytest.mark.parametrize("code", ["gpt", "opal", "astra"])
def test_rotated_line_is_written_on_axis(tmp_path, code):
    lattice = _lattice(tmp_path, code)
    assert lattice.in_local_frame
    assert lattice.local_z(lattice.endObject.physical.end) == pytest.approx(3.0)
    world = lattice.section.elements.elements.values()
    for s, element, original in zip((0, 1.5, 3), lattice.deck_section.elements.elements.values(), world):
        assert element.physical.s == pytest.approx(original.physical.s)  # OPAL places by s
        assert element.physical.middle.x == pytest.approx(0, abs=1e-9)
        assert element.physical.middle.z == pytest.approx(s)
        assert element.physical.global_rotation.theta == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("code", ["gpt", "opal"])
def test_local_frame_false_keeps_world_coordinates(tmp_path, code):
    lattice = _lattice(tmp_path, code, local_frame=False)
    assert not lattice.in_local_frame
    end = lattice.endObject.physical.end
    assert lattice.local_z(end) == pytest.approx(end.z)


def test_astra_starting_rotation_sets_the_frame(tmp_path):
    lattice = _lattice(tmp_path, "astra", starting_rotation="[0, 0, 0]")
    assert not lattice.in_local_frame
