import os
import h5py
import numpy as np
from .. import constants
from ..units import UnitValue


def rotate_beamXZ(self, theta, preOffset=[0, 0, 0], postOffset=[0, 0, 0]):
    preOffset = np.array(preOffset)
    postOffset = np.array(postOffset)

    rotation_matrix = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-1 * np.sin(theta), 0, np.cos(theta)],
        ]
    )
    beam = np.array([self.x, self.y, self.z]).transpose()
    self._beam.x, self._beam.y, self._beam.z = (
        np.dot(beam - preOffset, rotation_matrix) - postOffset
    ).transpose()

    beam = np.array([self.px, self.py, self.pz]).transpose()
    self._beam.px, self._beam.py, self._beam.pz = np.dot(
        beam, rotation_matrix
    ).transpose()

    if isinstance(self.reference_particle, np.ndarray):
        beam = np.array(
            [
                self.reference_particle[0],
                self.reference_particle[1],
                self.reference_particle[2],
            ]
        )
        (
            self.reference_particle[0],
            self.reference_particle[1],
            self.reference_particle[2],
        ) = (
            np.dot([beam - preOffset], rotation_matrix)[0] - postOffset
        )
        beam = np.array(
            [
                self.reference_particle[3],
                self.reference_particle[4],
                self.reference_particle[5],
            ]
        )
        (
            self.reference_particle[3],
            self.reference_particle[4],
            self.reference_particle[5],
        ) = np.dot([beam], rotation_matrix)[0]

    self._beam.x = UnitValue(self._beam.x, "m")
    self._beam.y = UnitValue(self._beam.y, "m")
    self._beam.z = UnitValue(self._beam.z, "m")
    self._beam.px = UnitValue(self._beam.px, "kg*m/s")
    self._beam.py = UnitValue(self._beam.py, "kg*m/s")
    self._beam.pz = UnitValue(self._beam.pz, "kg*m/s")

    self._beam.theta += theta
    self._beam.offset = +preOffset
    self.theta += float(theta)
    self.offset = +preOffset


def unrotate_beamXZ(self):
    if abs(self.theta) > 0:
        self.rotate_beamXZ(-1 * self.theta, -1 * self._beam.offset)


def write_HDF5_beam_file(
    self,
    filename,
    centered=False,
    mass=constants.m_e,
    sourcefilename=None,
    pos=None,
    rotation=None,
    longitudinal_reference="t",
    xoffset=0,
    yoffset=0,
    zoffset=0,
    toffset=0,
    cathode=False,
):
    if isinstance(zoffset, (list, np.ndarray)) and len(zoffset) == 3:
        xoffset = zoffset[0]
        yoffset = zoffset[1]
        zoffset = zoffset[2]
    with h5py.File(filename, "w", rdcc_nbytes=1024**3) as f:
        inputgrp = f.create_group("Parameters")
        if self._beam.total_charge == 0:
            self._beam.total_charge = np.sum(self._beam.charge)
        if sourcefilename is not None:
            inputgrp["Source"] = sourcefilename
        if pos is not None:
            inputgrp["Starting_Position"] = pos
        else:
            inputgrp["Starting_Position"] = [0, 0, 0]
        if rotation is not None:
            inputgrp["Rotation"] = rotation
        else:
            inputgrp["Rotation"] = 0
        inputgrp["total_charge"] = self._beam.total_charge
        inputgrp["npart"] = len(self.x)
        inputgrp["centered"] = centered
        inputgrp["code"] = self.code
        inputgrp["particle_mass"] = mass
        inputgrp["toffset"] = toffset
        beamgrp = f.create_group("beam")
        if self.reference_particle is not None:
            beamgrp["reference_particle"] = self.reference_particle
        if self._beam.status is not None:
            beamgrp["status"] = self._beam.status
        beamgrp["longitudinal_reference"] = longitudinal_reference
        beamgrp["cathode"] = cathode
        if len(self._beam.charge) == len(self.x):
            chargevector = self._beam.charge
        else:
            chargevector = np.full(len(self.x), self.charge / len(self.x))
        if len(self._beam.particle_index) == len(self.x):
            massvector = self._beam.particle_mass
        else:
            massvector = np.full(len(self.x), constants.electron_mass)
        array = np.array(
            [
                self.x + UnitValue(xoffset, units="m"),
                self.y + UnitValue(yoffset, units="m"),
                self.z + UnitValue(zoffset, units="m"),
                self.cpx,
                self.cpy,
                self.cpz,
                self.t + UnitValue(toffset, units="s"),
                massvector,
                chargevector,
                self.nmacro,
            ]
        ).transpose()
        beamgrp["columns"] = np.array(
            ["x", "y", "z", "cpx", "cpy", "cpz", "t", "particle", "q", "w"], dtype="S"
        )
        beamgrp["units"] = np.array(
            ["m", "m", "m", "eV", "eV", "eV", "s", "", "e", ""], dtype="S"
        )
        beamgrp.create_dataset("beam", data=array, rdcc_nbytes=1024**3)


def write_HDF5_summary_file(filename, beams=[], clean=False):
    if isinstance(beams, str):
        beams = [beams]
    mode = "a" if not clean else "w"
    basedirectory = os.path.dirname(filename)
    with h5py.File(filename, mode) as f:
        for name in beams:
            pre, ext = os.path.splitext(os.path.basename(name))
            try:
                f[pre] = h5py.ExternalLink(os.path.relpath(name, basedirectory), "/")
            except RuntimeError:
                pass


def read_HDF5_beam_file(self, filename, local=False):
    self.reset_dicts()
    self.filename = filename
    self.code = "SIMBA"
    with h5py.File(filename, "r") as h5file:
        if h5file.get("beam/reference_particle") is not None:
            self.reference_particle = np.array(
                h5file.get("beam/reference_particle")
            )
        if h5file.get("beam/longitudinal_reference") is not None:
            self.longitudinal_reference = np.array(
                h5file.get("beam/longitudinal_reference")
            )
        else:
            self.longitudinal_reference = "t"

        hdf5beam = np.array(h5file.get("beam/beam")).transpose()
        columns = [c.decode("utf-8") for c in h5file.get("/beam/columns")]
        if "particle" in columns and len(columns) == 10:
            x, y, z, cpx, cpy, cpz, t, mass, charge, nmacro = hdf5beam
            if np.mean(mass) > 1e-10:
                mass = [self.mass_index[m] for m in mass]
        elif len(columns) == 9:
            x, y, z, cpx, cpy, cpz, t, charge, nmacro = hdf5beam
            mass = constants.m_e
        elif len(columns) == 8:
            x, y, z, cpx, cpy, cpz, t, charge = hdf5beam
            nmacro = charge / constants.elementary_charge
            mass = constants.m_e
        else:
            raise ValueError(f"HDF5 columns unknown: {columns}")

        self._beam.charge = UnitValue(charge, "C")
        self.set_mass_and_charge(
            mass,
            constants.elementary_charge * np.full(len(x), self._beam.chargesign),
            len(x),
        )

        self._beam.x = UnitValue(x, "m")
        self._beam.y = UnitValue(y, "m")
        self._beam.z = UnitValue(z, "m")
        self.set_momenta(cpx, cpy, cpz)
        self._beam.clock = UnitValue(np.full(len(x), 0), "s")
        self._beam.t = UnitValue(t, "s")
        self._beam.set_total_charge(np.sum(self._beam.charge))
        if h5file.get("beam/status") is not None:
            self._beam.status = UnitValue(np.array(h5file.get("beam/status")), "")
        elif bool(np.array(h5file.get("beam/cathode"))):
            self._beam.status = UnitValue(np.full(len(self._beam.t), -1), "")
        self._beam.nmacro = UnitValue(nmacro, "")
        startposition = h5file.get("/Parameters/Starting_Position")
        startposition = np.array(startposition) if startposition is not None else [0, 0, 0]
        self.starting_position = startposition
        theta = h5file.get("/Parameters/Rotation")
        theta = np.array(theta) if theta is not None else 0
        self.theta = float(theta)
        if local is True:
            rotate_beamXZ(self, self.theta, preOffset=self.starting_position)
