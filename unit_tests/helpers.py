"""Builders shared between test files."""

import os
import shutil
import warnings

import pytest

import simba.Modules.Beams as rbf
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Dipole, Drift, Marker, Quadrupole, RFCavity
from laura.models.element_list import MachineModel
from simba.Framework_objects import frameworkLattice

ELEGANT = shutil.which("elegant")
needs_elegant = pytest.mark.skipif(ELEGANT is None, reason="elegant is not installed")


def skip_missing(code):
    """Skip unless ``code``'s executable or Python package is installed."""
    if code == "elegant" and ELEGANT is None:
        pytest.skip("elegant is not installed")
    module = {"xsuite": "xtrack", "madx": "cpymad", "ocelot": "ocelot", "bmad": "pytao"}
    if code in module:
        pytest.importorskip(module[code])


def fodo_quads():
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


def fodo_machine(tmp_path, cavity=False, closed=False, cavity_length=0.0):
    """The FODO cell between markers M1 and M3, exported to ``tmp_path/lattice``.

    ``cavity`` adds CAV1 at the end (a dict gives its fields; the bare one has
    no voltage). ``closed`` makes the section a ring. Ocelot's cavity divides
    by its length, so cannot be thin.
    """
    middle = fodo_quads()
    if cavity:
        middle.append(
            RFCavity(
                name="CAV1", machine_area="FODO",
                physical={"length": cavity_length,
                          "middle": {"x": 0.0, "y": 0.0, "z": 4.0 + cavity_length / 2}},
                **(cavity if isinstance(cavity, dict) else {}),
            )
        )
    m1 = Marker(
        name="M1", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    last = middle[-1]
    end_z = last.physical.middle.z + (last.physical.length or 0.0)
    m3 = Marker(
        name="M3", machine_area="FODO", hardware_class="Marker",
        physical={"middle": {"x": 0.0, "y": 0.0, "z": end_z}},
    )
    names = ["M1"] + [e.name for e in middle] + ["M3"]
    section = (
        {"sections": {"FODO": {"elements": names, "geometry": "closed"}}}
        if closed
        else {"sections": {"FODO": names}}
    )
    machine = LAURA(
        element_list=[m1, *middle, m3],
        layout={"default_layout": "line1", "layouts": {"line1": ["FODO"]}},
        section=section,
    )
    export_machine(path=f"{tmp_path}/lattice", machine=machine, overwrite=True)
    return machine, names, section


def read_beam(subdir, name, turn=None):
    """``name``'s beam in ``subdir``, on ``turn`` if given."""
    beam = rbf.beam()
    rbf.openpmd.read_openpmd_beam_file(
        beam, os.path.join(subdir, f"{name}.openpmd.hdf5"), turn=turn
    )
    return beam


class BentLine:
    """A stub line of ``nbend`` bends of ``angle``, each followed by a 1 m
    drift, exposing what the closure checks read off real geometry."""

    def __init__(self, nbend, angle, **tracking):
        elements, order = {}, []
        for i in range(nbend):
            bend, drift = f"B{i}", f"D{i}"
            elements[bend] = Dipole(
                name=bend,
                hardware_class="Magnet",
                machine_area="A",
                magnetic={"magnetic_length": 1.0, "k0l": angle},
                physical={"length": 1.0},
            )
            elements[drift] = Drift(
                name=drift,
                hardware_class="Drift",
                hardware_type="Drift",
                machine_area="A",
                physical={"length": 1.0},
            )
            order += [bend, drift]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MachineModel(
                elements=elements,
                section={"sections": {"RING": order}},
                layout={"layouts": {"M": ["RING"]}, "default_layout": "M"},
            )
        self.startObject = model[order[0]]
        self.endObject = model[order[-1]]
        self.elements = {name: model[name] for name in order}
        self.file_block = {"tracking": {"turns": 1000, **tracking}}
        self.objectname = "RING"
        self.code = "elegant"

    def _machine_geometry(self):
        """No layout behind this stub, so not closed unless ``periodic`` says."""

    turns = frameworkLattice.turns
    periodic = frameworkLattice.periodic
    closed_geometry = frameworkLattice.closed_geometry
    nsuperperiods = frameworkLattice.nsuperperiods
    net_bend_angle = frameworkLattice.net_bend_angle
    check_turns_closed = frameworkLattice.check_turns_closed
    check_superperiods_close = frameworkLattice.check_superperiods_close


def xtrack_sector(copies=1):
    """A FODO cell with sextupoles, repeated ``copies`` times."""
    import xtrack as xt

    elements, names = [], []
    for copy in range(copies):
        for i in range(4):
            elements += [
                xt.Drift(length=0.5),
                xt.Multipole(knl=[0.0, 0.3], length=0.0),
                xt.Drift(length=0.5),
                xt.Multipole(knl=[0.0, -0.3], length=0.0),
                xt.Multipole(knl=[0.0, 0.0, 2.0], length=0.0),
            ]
            names += [f"d{i}a_{copy}", f"qf{i}_{copy}", f"d{i}b_{copy}",
                      f"qd{i}_{copy}", f"sx{i}_{copy}"]
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
    line.build_tracker()
    return line
