"""GPT's csr1d is left out for a bunch that starts at the cathode."""
import types

import pytest
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Dipole, Marker

import simba.Framework as fw


@pytest.fixture
def bent_gpt_line(tmp_path):
    elements = [
        Marker(name="M1", machine_area="L", hardware_class="Marker", physical={"middle": {"z": 0.0}}),
        Dipole(name="B1", machine_area="L", magnetic={"length": 0.2, "angle": 0.1},
               physical={"length": 0.2, "middle": {"z": 0.6}}),
        Marker(name="M3", machine_area="L", hardware_class="Marker", physical={"middle": {"z": 1.2}}),
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
        "code": "gpt",
        "charge": {"space_charge_mode": "False"},
        "input": {},
        "output": {"start_element": "M1", "end_element": "M3"},
    }}
    settings.layout = machine.layout
    settings.section = {"sections": {"L": names}}
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    lattice = framework.latticeObjects["L"]
    lattice.global_parameters["beam"] = types.SimpleNamespace(x=[0.0] * 10, Brho=0.5)
    return lattice


def test_csr_is_written_for_a_tracked_beam(bent_gpt_line):
    bent_gpt_line.particle_definition = "M1"
    bent_gpt_line.writeElements()
    assert "csr1d" in bent_gpt_line.headers


def test_csr_is_left_out_from_the_cathode(bent_gpt_line):
    bent_gpt_line.particle_definition = "laser"
    with pytest.warns(UserWarning, match="cathode"):
        bent_gpt_line.writeElements()
    assert "csr1d" not in bent_gpt_line.headers
