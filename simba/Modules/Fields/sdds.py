import numpy as np

from ..units import UnitValue
from ..constants import speed_of_light

from .FieldParameter import FieldParameter
from warnings import warn
from ..SDDSFile import SDDSFile, SDDS_Types


def write_SDDS_field_file(self, sddsindex: int = 0, ascii: bool = False) -> str:
    """
    Generate the field data in a format that is suitable for SDDS, based on the
    :class:`~SimulationFramework.Modules.Fields.field` object provided.
    This is then written to an SDDS file.
    The `field_type` parameter determines the format of the file.

    A warning is raised if the field type is not supported (perhaps elevate to a `NotImplementedError`?

    Parameters
    ----------
    self: :class:`~SimulationFramework.Modules.Fields.field`
        The field object
    sddsindex: int
        Must be provided for :class:`~SimulationFramework.Modules.SDDSFile.SddsFile` class
    ascii: bool, optional
        Convert to ascii?

    Returns
    -------
    str:
        The name of the SDDS field file.
    """
    sdds_filename = self._output_filename(extension=".sdds")
    sddsfile = SDDSFile(index=sddsindex, ascii=ascii)
    zdata = self.z_values
    tdata = self.t_values
    if self.field_type == "LongitudinalWake":
        wzdata = self.Wz.value.val
        cnames = ["z", "t", "Wz"]
        cunits = ["m", "s", "V/C"]
        ccolumns = [
            zdata,
            tdata,
            wzdata,
        ]
    elif self.field_type == "TransverseWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        ccolumns = np.array(
            [
                zdata,
                tdata,
                wxdata,
                wydata,
            ]
        )
        cnames = ["z", "t", "Wx", "Wy"]
        cunits = ["m", "s", "V/C/m", "V/C/m"]
    elif self.field_type == "3DWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        ccolumns = np.array(
            [
                zdata,
                tdata,
                wxdata,
                wydata,
                wzdata,
            ]
        )
        cnames = ["z", "t", "Wx", "Wy", "Wz"]
        cunits = ["m", "s", "V/C/m", "V/C/m", "V/C"]
    elif self.field_type == "1DElectroDynamic":
        ezdata = self.Ez.value.val
        cnames = ["z", "Ez"]
        cunits = ["m", "V"]
        ccolumns = [
            zdata,
            ezdata,
        ]
    else:
        warn(f"Field type {self.field_type} not supported for SDDS")
        return
    if ccolumns is not None:
        ctypes = [SDDS_Types.SDDS_DOUBLE for _ in ccolumns]
        csymbols = ["" for _ in ccolumns]
        sddsfile.add_columns(cnames, ccolumns, ctypes, cunits, csymbols)
        sddsfile.write_file(sdds_filename)
    return sdds_filename


def read_SDDS_field_file(self, filename: str, field_type: str):
    """
    Read an SDDS field file and convert it into a :class:`SimulationFramework.Modules.Fields.field` object.
    Only works for wakefield files.

    Parameters
    ----------
    self: :class:`~SimulationFramework.Modules.Fields.field`
        The field object to be updated.
    filename: str
        The path to the SDDS field file
    field_type: str
        The name of the field, see :attr:`~SimulationFramework.Modules.Fields.allowed_fields`

    Returns
    -------
    None

    Raises
    ------
    NotImplementedError:
        if a given `field_type` is not implemented
    """
    self.reset_dicts()
    setattr(self, "field_type", field_type)
    try:
        elegantObject = SDDSFile(index=1, ascii=True)
    except Exception:
        elegantObject = SDDSFile(index=1, ascii=False)
    elegantObject.read_file(filename, page=-1)
    if field_type in ["LongitudinalWake", "3DWake", "TransverseWake"]:
        for key, val in elegantObject._columns.items():
            data = np.array(val.data)
            if val.unit == "m":
                setattr(
                    self,
                    "z",
                    FieldParameter(name="z", value=UnitValue(data, units=val.unit)),
                )
            elif val.unit == "s":
                setattr(
                    self,
                    "t",
                    FieldParameter(
                        name="t", value=UnitValue(data, units="m")
                    ),
                )
            elif val.unit == "V/C":
                setattr(
                    self,
                    "Wz",
                    FieldParameter(name="Wz", value=UnitValue(data, units="V/C")),
                )
            elif val.unit == "V/C/m":
                setattr(
                    self,
                    "Wx",
                    FieldParameter(name="Wx", value=UnitValue(data, units="V/C/m")),
                )
                setattr(
                    self,
                    "Wy",
                    FieldParameter(name="Wy", value=UnitValue(data, units="V/C/m")),
                )
            else:
                raise ValueError(f"Unit {val.unit} not recognised in {filename}")
    else:
        raise NotImplementedError(
            f"{field_type} loading not implemented for SDDS files"
        )
