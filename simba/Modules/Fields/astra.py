import numpy as np
import re
from warnings import warn
from .FieldParameter import FieldParameter
from ..units import UnitValue

d = ",!?/&-:;@'\n \t"


def generate_astra_field_data(self) -> np.ndarray:
    """
    Generate the field data in a format that is suitable for ASTRA, based on the
    :class:`~simba.Modules.Fields.field` object provided.
    The `field_type` parameter determines the format of the file.
    See the `ASTRA manual`_ for more details.

    A warning is raised if the field type is not supported (perhaps elevate to a `NotImplementedError`?

    .. _ASTRA manual: https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf

    Parameters
    ----------
    self: :class:`~simba.Modules.Fields.field`
        The field object

    Returns
    -------
    np.ndarray:
        The formatted field data.
    """
    length = str(self.length)
    data = None
    zdata = self.z_values
    if isinstance(self.field_type, bytes):
        self.field_type = self.field_type.decode("utf-8")
    if self.field_type == "LongitudinalWake":
        wzdata = self.Wz.value.val
        preamble = np.array(
            [
                [1, 0],
                [length, 0],
                [0, 0],
                [0, 0],
            ]
        )
        data = np.concatenate([preamble, np.transpose([zdata, wzdata])])
    elif self.field_type == "TransverseWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        data = np.concatenate(
            [np.array([[length, ""]]), np.transpose([zdata, wxdata, wydata])]
        )
    elif self.field_type == "3DWake":
        zpreamble = np.array(
            [
                [3, 0],
                [length, 0],
                [0, 0],
                [0, 0],
            ]
        )
        xpreamble = np.array(
            [
                [length, 0],
                [0, 0],
                [0, 13],
            ]
        )
        ypreamble = np.array(
            [
                [length, 0],
                [0, 0],
                [0, 24],
            ]
        )
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        zvals = np.concatenate([zpreamble, np.transpose([zdata, wzdata])])
        xvals = np.concatenate([xpreamble, np.transpose([zdata, wxdata])])
        yvals = np.concatenate([ypreamble, np.transpose([zdata, wydata])])
        data = np.concatenate([zvals, xvals, yvals])
    elif self.field_type == "1DMagnetoStatic":
        bzdata = self.Bz.value.val
        data = np.transpose([zdata, bzdata])
    elif self.field_type == "3DMagnetoStatic":
        xdata = self.x.value.val
        ydata = self.y.value.val
        bzdata = self.Bz.value.val
        data = np.array(
            [
                [z, Bz]
                for x, y, z, Bz in zip(xdata, ydata, zdata, bzdata)
                if x == 0.0 and y == 0.0
            ]
        )
    elif self.field_type == "1DElectroDynamic":
        ezdata = self.Ez.value.val
        if self.cavity_type == "TravellingWave":
            spdata = ["" for _ in range(self.length)]
            if any(
                    [
                        getattr(self, param) is None for param in [
                            "start_cell_z",
                            "end_cell_z",
                            "mode_numerator",
                            "mode_denominator"
                        ]
                    ]
            ):
                raise ValueError(
                    "start_cell_z, end_cell_z", "mode_numerator", "mode_denominator"
                    "must be defined for TravellingWave cavities"
                )
            preamble = np.array(
                [
                    [
                        self.start_cell_z,
                        self.end_cell_z,
                        self.mode_numerator,
                        self.mode_denominator,
                    ]
                ]
            )
            data = np.concatenate(
                [preamble, np.transpose([zdata, ezdata, spdata, spdata])]
            )
        else:
            data = np.transpose([zdata, ezdata])
    elif self.field_type == "1DQuadrupole":
        gdata = self.G.value.val
        data = np.transpose([zdata, gdata])
    else:
        warn(f"Field type {self.field_type} not supported for ASTRA")
    return data


def write_astra_field_file(self) -> str:
    """
    Write the field data in an ASTRA-compatible format, based on the
    :class:`~simba.Modules.Fields.field` object provided.
    The absolute location of the file to be written is generated using
    :func:`~simba.Modules.Fields.field._output_filename`, which is parsed from the Master Lattice.

    Parameters
    ----------
    self: :class:`~simba.Modules.Fields.field`
        The field object

    Returns
    -------
    str:
        The converted filename
    """
    astra_file = self._output_filename(extension=".astra")
    data = generate_astra_field_data(self)
    if data is not None:
        with open(f"{astra_file}", "w") as f:
            for d in data:
                f.write(" ".join([str(x) for x in d]) + "\n")
    return astra_file


def read_astra_field_file(
    self,
    filename: str,
    field_type: str,
    cavity_type: str | None = None,
    frequency: float | None = None,
):
    """
    Read a field file from ASTRA format and convert it into a
    :class:`~simba.Modules.Fields.field` object (self).
    Certain parameters must be included, particularly for RF cavities.

    See the `ASTRA manual`_ for more details.

    .. _ASTRA manual: https://www.desy.de/~mpyflo/Astra_manual/Astra-Manual_V3.2.pdf

    Parameters
    ----------
    self: :class:`~simba.Modules.Fields.field`
        The field object to be updated.
    filename: str
        The path to the ASTRA field file
    field_type: str
        The name of the field, see :attr:`~simba.Modules.Fields.allowed_fields`
    cavity_type: str, optional
        The type of RF cavity, see :attr:`~simba.Modules.Fields.allowed_cavities`
    frequency: float, optional
        The frequency of the RF cavity.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        if the cavity `field_type` contains the string `Electro` and `cavity_type` is not provided
    ValueError:
        if the cavity `field_type` contains the string `Electro` and `frequency` is not provided
    NotImplementedError:
        if a given `field_type` is not implemented
    """
    self.reset_dicts()
    setattr(self, "field_type", field_type)
    try:
        if "Electro" in field_type:
            if cavity_type is None:
                raise ValueError(f"cavity_type must be provided for {field_type}")
            else:
                setattr(self, "cavity_type", cavity_type)
            if frequency is None:
                raise ValueError(f"frequency must be provided for {field_type}")
            else:
                setattr(self, "frequency", frequency)
    except Exception:
        raise ValueError(
            f"Fields read_astra_field_file error: {filename}, {field_type}, {cavity_type}, {frequency}"
        )
    if field_type == "1DMagnetoStatic":
        fdat = np.loadtxt(filename)
        setattr(
            self, "z", FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m"))
        )
        setattr(
            self,
            "Bz",
            FieldParameter(
                name="Bz", value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="T")
            ),
        )
    elif field_type == "1DElectroDynamic":
        if cavity_type == "StandingWave":
            fdat = np.loadtxt(filename)
            setattr(
                self,
                "z",
                FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m")),
            )
            setattr(
                self,
                "Ez",
                FieldParameter(
                    name="Ez",
                    value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="V/m"),
                ),
            )
        elif cavity_type == "TravellingWave":
            with open(filename) as f:
                rl = f.readlines()[0]
                twdat = re.split("[" + "\\".join(d) + "]", rl)
                setattr(self, "start_cell_z", float(twdat[0]))
                setattr(self, "end_cell_z", float(twdat[1]))
                setattr(self, "mode_numerator", float(twdat[2]))
                setattr(self, "mode_denominator", float(twdat[3]))
            fdat = np.loadtxt(filename, skiprows=1)
            setattr(
                self,
                "z",
                FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m")),
            )
            setattr(
                self,
                "Ez",
                FieldParameter(
                    name="Ez",
                    value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="V/m"),
                ),
            )
    elif field_type == "LongitudinalWake":
        try:
            fdat = np.loadtxt(filename)
            numrows = int(fdat[1, 0])
            setattr(
                self,
                "z",
                FieldParameter(
                    name="z", value=UnitValue(fdat[4 : 4 + numrows][::, 0], units="m")
                ),
            )
            setattr(
                self,
                "Wz",
                FieldParameter(
                    name="Wz",
                    value=UnitValue(fdat[4 : 4 + numrows][::, 1]),
                    units="V/C",
                ),
            )
        except Exception:
            fdat = np.loadtxt(filename, skiprows=1)
            setattr(
                self,
                "z",
                FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m")),
            )
            setattr(
                self,
                "Wz",
                FieldParameter(name="Wz", value=UnitValue(fdat[::, 1]), units="V/C"),
            )
    elif field_type == "3DWake":
        fdat = np.loadtxt(filename)
        numrows = int(fdat[1, 0])
        setattr(
            self,
            "z",
            FieldParameter(
                name="z", value=UnitValue(fdat[4 : 4 + numrows][::, 0], units="m")
            ),
        )
        setattr(
            self,
            "Wz",
            FieldParameter(
                name="Wz", value=UnitValue(fdat[4 : 4 + numrows][::, 1]), units="V/C"
            ),
        )
        setattr(
            self,
            "Wx",
            FieldParameter(
                name="Wx",
                value=UnitValue(fdat[numrows + 7 : (2 * numrows) + 7][::, 1]),
                units="V/C/m",
            ),
        )
        setattr(
            self,
            "Wy",
            FieldParameter(
                name="Wy",
                value=UnitValue(fdat[(2 * numrows) + 10 :][::, 1]),
                units="V/C/m",
            ),
        )
    elif field_type == "1DQuadrupole":
        fdat = np.loadtxt(filename)
        setattr(
            self, "z", FieldParameter(name="z", value=UnitValue(fdat[::, 0], units="m"))
        )
        setattr(
            self,
            "G",
            FieldParameter(
                name="G",
                value=UnitValue(fdat[::, 1] / np.max(fdat[::, 1]), units="T/m"),
            ),
        )
    else:
        raise NotImplementedError(
            f"{field_type} loading not implemented for ASTRA files"
        )
