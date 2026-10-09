import os
import numpy as np
from .. import constants
from ..units import UnitValue
from ..SDDSFile import SDDSFile, SDDS_Types


def count_SDDS_pages(fileName, ascii=False) -> int:
    """Number of pages in ``fileName``; for a watch point, the number of passes."""
    return SDDSFile(index=0, ascii=ascii).count_pages(fileName)


def read_SDDS_beam_file(
    self, fileName, charge=None, ascii=False, page=-1, xyzoffset=[0, 0, 0], ref_index=None,
    sdds_file=None,
):
    """Read one page of an SDDS beam file.

    Pass ``sdds_file`` (a :class:`~laura.translator.utils.elegant.sdds_file.SDDSFile`)
    to load a multi-page file once rather than once per page.
    """
    self.reset_dicts()
    if sdds_file is None:
        self.sddsindex += 1
        sdds_file = SDDSFile(index=self.sddsindex, ascii=ascii)
    elegantObject = sdds_file
    elegantObject.read_file(fileName, page=page)
    elegantData = elegantObject.data
    required_keys = ["x", "y", "t", "xp", "yp", "p"]
    beamprops = {}
    for k, v in elegantData.items():
        # case handling for multiple ELEGANT runs per file
        # only extract the first run (in ELEGANT this is the fiducial run)
        if isinstance(v, np.ndarray):
            if v.ndim > 1:
                beamprops.update({k: v[0]})
            else:
                try:
                    beamprops.update({k: v})
                except Exception:
                    pass
        else:
            try:
                beamprops.update({k: np.array(v)})
            except Exception:
                pass
    for k in required_keys:
        if k not in beamprops:
            raise ValueError(f"Could not find column {k} in SDDS file")
    self.filename = fileName
    self.set_mass_and_charge(constants.m_e, -constants.elementary_charge, len(beamprops["x"]))

    self.code = "SDDS"
    self._beam.x = UnitValue(beamprops["x"] + xyzoffset[0], units="m")
    self._beam.y = UnitValue(beamprops["y"] + xyzoffset[1], units="m")
    self._beam.t = UnitValue(beamprops["t"], units="s")
    cp = beamprops["p"] * self.particle_rest_energy_eV
    cpz = cp / np.sqrt(beamprops["xp"] ** 2 + beamprops["yp"] ** 2 + 1)
    cpx = beamprops["xp"] * cpz
    cpy = beamprops["yp"] * cpz
    self.set_momenta(cpx, cpy, cpz)
    if "Charge" in elegantData and len(elegantData["Charge"]) > 0:
        self._beam.set_total_charge(elegantData["Charge"][0])
    elif charge is None:
        self._beam.set_total_charge(self._beam.total_charge)
    else:
        self._beam.set_total_charge(charge)
    self._beam.nmacro = UnitValue(
        np.abs(self._beam.charge / self._beam.particle_charge)
    )
    self._beam.status = UnitValue(np.full(len(self._beam.x), 5))
    self.set_z_from_t(xyzoffset[2], ref_index)
    if "s" not in beamprops:
        beamprops["s"] = 0
    self._beam.s = UnitValue(beamprops["s"], units="m")


def write_SDDS_file(self, filename: str = None, ascii=False, xyzoffset=[0, 0, 0]):
    """Write the beam to an SDDS file."""
    if filename is None:
        fn = os.path.splitext(self.filename)
        filename = fn[0].removesuffix(".ocelot").removesuffix(".openpmd") + ".sdds"
    xoffset = xyzoffset[0]
    yoffset = xyzoffset[1]
    self.sddsindex += 1
    try:
        x = SDDSFile(index=self.sddsindex, ascii=ascii)
    except ValueError:
        self.sddsindex += 1
        x = SDDSFile(index=self.sddsindex, ascii=ascii)

    Cnames = ["x", "xp", "y", "yp", "t", "p"]
    Ctypes = [
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
        SDDS_Types.SDDS_DOUBLE,
    ]
    Csymbols = ["", "x'", "", "y'", "", ""]
    Cunits = ["m", "", "m", "", "s", "m$be$nc"]
    Ccolumns = [
        np.array(self.x) - float(xoffset),
        self.xp,
        np.array(self.y) - float(yoffset),
        self.yp,
        self.t,
        self.cp / self.particle_rest_energy_eV,
    ]
    x.add_columns(Cnames, Ccolumns, Ctypes, Cunits, Csymbols)

    Pnames = ["pCentral", "Charge", "Particles"]
    Ptypes = [SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE, SDDS_Types.SDDS_DOUBLE]
    Psymbols = ["p$bcen$n", "", ""]
    Punits = ["m$be$nc", "C", ""]
    parameterData = [
        np.mean(self.BetaGamma),
        abs(self._beam.total_charge),
        len(self.x),
    ]
    x.add_parameters(Pnames, parameterData, Ptypes, Punits, Psymbols)

    x.write_file(filename)
