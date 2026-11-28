"""Structure wakes in Ocelot: which elements get one, and how many tables each adds up to."""

import numpy as np
import pytest

pytest.importorskip("ocelot")

from laura import LAURA  # noqa: E402
from laura.exporters.yaml_exporter import export_machine  # noqa: E402
from laura.models.element import Marker, RFCavity, Wakefield  # noqa: E402
from laura.translator.utils.fields import FieldMap  # noqa: E402
from laura.translator.utils.fields.field_parameter import FieldParameter  # noqa: E402
from laura.translator.utils.units import UnitValue  # noqa: E402

import simba.Framework as fw  # noqa: E402
from simba.Codes.Ocelot.Ocelot import ocelotLattice  # noqa: E402


def _wake_file(path, reference_length):
    z = np.linspace(0, 1e-3, 11)
    wake = FieldMap(field_type="LongitudinalWake", reference_length=reference_length, read=True)
    wake.z = FieldParameter(name="z", value=UnitValue(z, "m"))
    wake.Wz = FieldParameter(name="Wz", value=UnitValue(1e12 * np.exp(-z / 1e-4), "V/C"))
    wake.filename = str(path)
    wake.write_field_file(code="hdf5")
    return wake.filename


def _lattice(tmp_path, monkeypatch, reference_length, wake_enable=True):
    path = _wake_file(tmp_path / "wk.hdf5", reference_length)

    def physical(z):
        return {"length": 0.3, "middle": {"x": 0, "y": 0, "z": z}}

    elements = [
        Marker(name="M1", machine_area="L", hardware_class="Marker", physical={"middle": {"z": 0}}),
        RFCavity(name="C", machine_area="L", physical=physical(1.0),
                 cavity={"structure_type": "StandingWave", "frequency": 1.3e9, "phase": 0.0, "n_cells": 3},
                 simulation={"field_amplitude": 0.0, "wakefield_definition": path}),
        Wakefield(name="W", machine_area="L", physical=physical(2.0), cavity={"n_cells": 3},
                  simulation={"wakefield_definition": path, "wakefield_enable": wake_enable}),
        Marker(name="M3", machine_area="L", hardware_class="Marker", physical={"middle": {"z": 3}}),
    ]
    names = [e.name for e in elements]
    machine = LAURA(
        element_list=elements,
        layout={"default_layout": "l", "layouts": {"l": ["L"]}},
        section={"sections": {"L": names}},
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    settings = fw.FrameworkSettings()
    settings.files = {"L": {"code": "ocelot", "charge": {"space_charge_mode": "False"}, "input": {},
                            "output": {"start_element": "M1", "end_element": "M3"}}}
    settings.layout = machine.layout
    settings.section = {"sections": {"L": names}}
    settings.element_list = f"{tmp_path}/lattice"
    framework = fw.Framework(machine=machine, directory=str(tmp_path / "run"), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    lattice = framework.latticeObjects["L"]
    monkeypatch.setattr(ocelotLattice, "reference_p0c", property(lambda self: 1e9))  # no beam
    lattice.tw_focusing = False
    lattice.writeElements()
    return lattice


def _wake_factors(tmp_path, monkeypatch, reference_length, wake_enable=True):
    navi = _lattice(tmp_path, monkeypatch, reference_length, wake_enable).navi_setup(write_beams=False)
    return [p.factor for p in navi.process_table.proc_list if type(p).__name__ == "Wake"]


@pytest.mark.parametrize("reference_length, weight", [(1.2, 0.25), (None, 3)])
def test_every_wake_adds_up_to_length_over_reference_length_tables(tmp_path, monkeypatch, reference_length, weight):
    """A per-module table on each cavity was applied n_cells times, as LAURA's ASTRA,
    Elegant and GPT no longer do, and a Wakefield element's wake was dropped."""
    assert _wake_factors(tmp_path, monkeypatch, reference_length) == pytest.approx([weight, weight])


def test_a_wake_switched_off_on_its_element_is_not_applied(tmp_path, monkeypatch):
    assert _wake_factors(tmp_path, monkeypatch, 1.2, wake_enable=False) == pytest.approx([0.25])


def test_each_wake_is_applied_over_its_whole_element_beside_a_finer_process(tmp_path, monkeypatch):
    """A Wake(step=100) beside a process at every step (as space charge) was applied
    only at its element's end, scaled by that last step: 1/30 of the wake here."""
    from ocelot import ParticleArray, track
    from ocelot.cpbd.physics_proc import PhysProc
    from ocelot.cpbd.wake3D import Wake

    lattice = _lattice(tmp_path, monkeypatch, 1.2)
    lattice.unit_step = 0.01
    navi = lattice.navi_setup(write_beams=False)
    navi.add_physics_proc(PhysProc(step=1), lattice.lat_obj.sequence[0], lattice.lat_obj.sequence[-1])
    applied = {}

    def fraction(self, p_array, dz):
        # as Wake.apply; the Wakefield element's is a kick (s_start == s_stop), applied whole
        length = self.s_stop - self.s_start
        applied[id(self)] = applied.get(id(self), 0) + (dz / length if length else 1)

    monkeypatch.setattr(Wake, "apply", fraction)
    p_array = ParticleArray(n=10)
    p_array.E = 1.0
    track(lattice.lat_obj, p_array, navi, calc_tws=False, print_progress=False)
    assert sorted(applied.values()) == pytest.approx([1, 1])
