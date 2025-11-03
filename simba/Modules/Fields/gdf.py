from typing import List
import numpy as np
import easygdf
from warnings import warn
from .FieldParameter import FieldParameter
from ..units import UnitValue
from ..constants import speed_of_light


def write_gdf_field_file(self) -> str:
    """
    Generate the field data in a format that is suitable for GPT, based on the
    :class:`~simba.Modules.Fields.field` object provided.
    This is then written to a GDF file.
    The `field_type` parameter determines the format of the file.

    A warning is raised if the field type is not supported (perhaps elevate to a `NotImplementedError`?

    Parameters
    ----------
    self: :class:`~simba.Modules.Fields.field`
        The field object

    Returns
    -------
    str:
        The name of the GDF field file.
    """
    gdf_file = self._output_filename(extension=".gdf")
    blocks = None
    zdata = self.z.value.val
    if self.field_type == "LongitudinalWake":
        wzdata = self.Wz.value.val
        blocks = union(
            [
                {"name": "z", "value": zdata},
                {"name": "Wz", "value": wzdata},
            ]
        )
    elif self.field_type == "TransverseWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wx", "value": wxdata},
            {"name": "Wy", "value": wydata},
        ]
    elif self.field_type == "3DWake":
        wxdata = self.Wx.value.val
        wydata = self.Wy.value.val
        wzdata = self.Wz.value.val
        blocks = [
            {"name": "z", "value": zdata},
            {"name": "Wx", "value": wxdata},
            {"name": "Wy", "value": wydata},
            {"name": "Wz", "value": wzdata},
        ]
    elif self.field_type == "1DMagnetoStatic":
        bzdata = self.Bz.value.val
        blocks = union(
            [
                {"name": "z", "value": zdata},
                {"name": "Bz", "value": bzdata},
            ]
        )
    elif self.field_type == "3DMagnetoStatic":
        xdata = self.x.value.val
        ydata = self.y.value.val
        bxdata = self.Bx.value.val
        bydata = self.By.value.val
        bzdata = self.Bz.value.val
        blocks = [
            {"name": "x", "value": xdata},
            {"name": "y", "value": ydata},
            {"name": "z", "value": zdata},
            {"name": "Bx", "value": bxdata},
            {"name": "By", "value": bydata},
            {"name": "Bz", "value": bzdata},
        ]
    elif self.field_type == "1DElectroDynamic":
        ezdata = self.Ez.value.val
        fielddata = np.array([zdata, ezdata]).transpose()
        if self.cavity_type == "TravellingWave":
            startpos = list(zdata).index(self.start_cell_z)
            # stoppos = list(zdata).index(self.end_cell_z)
            halfcell1 = 1.0 * fielddata[:startpos]
            halfcell2 = 1.0 * halfcell1[::-1]
            halfcell1[:, 1] /= max(halfcell1[:, 1])
            halfcell2[:, 0] = halfcell2[:, 0][::-1]
            halfcell2[:, 1] /= max(halfcell2[:, 1])
            halfcell1end = halfcell1[-1, 0]
            zstep = zdata[1] - zdata[0]
            lambdaRF = speed_of_light / self.frequency
            ncells = (self.n_cells - 1) * self.mode_numerator / self.mode_denominator
            nsteps = int(np.floor((ncells) * lambdaRF / zstep))
            middleRF = np.array(
                [
                    [
                        (x * zstep + halfcell1end + zstep),
                        np.cos((2 * np.pi / lambdaRF) * (x * zstep)),
                    ]
                    for x in range(0, nsteps + 1)
                ]
            )
            halfcell2[:, 0] += middleRF[-1, 0] + zstep
            zdata, ezdata = np.concatenate([halfcell1, middleRF, halfcell2]).transpose()
        blocks = union(
            [
                {"name": "z", "value": zdata},
                {"name": "Ez", "value": ezdata},
            ]
        )
    else:
        warn(f"Field type {self.field_type} not supported for GPT")
    if blocks is not None:
        # print(blocks)
        # print(union(blocks))
        easygdf.save(gdf_file, blocks)
    return gdf_file


def union(blocks: List) -> List:
    """
    Update the field data into a format compatible for easyGDF

    Parameters
    ----------
    blocks: List[Dict]
        The field parameters, keyed by name

    Returns
    -------
    List:
        A list of easyGDF-compatible dictionaries
    """
    names = [b["name"] for b in blocks]
    if "z" in names:
        zidx = names.index("z")
        _, indices = np.unique(
            np.round(blocks[zidx]["value"], decimals=6), return_index=True
        )
        blocks = [{"name": b["name"], "value": b["value"][indices]} for b in blocks]
        return blocks
    return blocks


def read_gdf_field_file(
    self,
    filename: str,
    field_type: str,
    cavity_type: str | None = None,
    frequency: float | None = None,
    normalize_b: bool = True
):
    """
    Read a GDF field file and convert it into a :class:`simba.Modules.Fields.field` object

    Parameters
    ----------
    self: :class:`~simba.Modules.Fields.field`
        The field object to be updated.
    filename: str
        The path to the GDF field file
    field_type: str
        The name of the field, see :attr:`~simba.Modules.Fields.allowed_fields`
    cavity_type: str, optional
        The type of RF cavity, see :attr:`~simba.Modules.Fields.allowed_cavities`
    frequency: float, optional
        The frequency of the RF cavity.
    normalize_b: bool, optional
        Normalize Bx and By with respect to Bz (True by default)

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
    if "Electro" in field_type:
        if cavity_type is None:
            raise ValueError(f"cavity_type must be provided for {field_type}")
        else:
            setattr(self, "cavity_type", cavity_type)
        if frequency is None:
            raise ValueError(f"frequency must be provided for {field_type}")
        else:
            setattr(self, "frequency", frequency)
    fdat = easygdf.load(filename)["blocks"]
    try:
        zval = [k["value"] for k in fdat if k["name"].lower() == "z"][0]
    except Exception:
        zval = [k["value"] * speed_of_light for k in fdat if k["name"].lower() == "t"][
            0
        ]
    if field_type == "1DMagnetoStatic":
        bzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Bz"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(
            self,
            "Bz",
            FieldParameter(
                name="Bz", value=UnitValue(bzval / np.max(bzval), units="T")
            ),
        )
    elif field_type == "3DMagnetoStatic":
        xval = [k["value"] for k in fdat if k["name"].lower() == "x"][0]
        yval = [k["value"] for k in fdat if k["name"].lower() == "y"][0]
        bxval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Bx"][0]
        byval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "By"][0]
        bzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Bz"][0]
        # Normalise by the maximum *on-axis* Bz field
        if normalize_b:
            normBz = max(
                [abs(Bz) for x, y, Bz in zip(xval, yval, bzval) if x == 0.0 and y == 0.0]
            )
        else:
            normBz = 1
        setattr(self, "x", FieldParameter(name="x", value=UnitValue(xval, units="m")))
        setattr(self, "y", FieldParameter(name="y", value=UnitValue(yval, units="m")))
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(
            self,
            "Bx",
            FieldParameter(name="Bx", value=UnitValue(bxval / normBz, units="T")),
        )
        setattr(
            self,
            "By",
            FieldParameter(name="By", value=UnitValue(byval / normBz, units="T")),
        )
        setattr(
            self,
            "Bz",
            FieldParameter(name="Bz", value=UnitValue(bzval / normBz, units="T")),
        )
    elif field_type == "1DElectroDynamic":
        if cavity_type == "StandingWave":
            ezval = [
                k["value"] for k in fdat if k["name"].lower().capitalize() == "Ez"
            ][0]
            setattr(
                self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m"))
            )
            setattr(
                self,
                "Ez",
                FieldParameter(
                    name="Ez", value=UnitValue(ezval / np.max(ezval), units="V/m")
                ),
            )
        elif cavity_type == "TravellingWave":
            raise NotImplementedError(f"{cavity_type} not implemented for GDF files")
    elif field_type == "LongitudinalWake":
        wzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wz"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(
            self, "Wz", FieldParameter(name="Wz", value=UnitValue(wzval, units="V/C"))
        )
    elif field_type == "TransverseWake":
        wxval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wx"][0]
        wyval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wy"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(
            self, "Wx", FieldParameter(name="Wx", value=UnitValue(wxval, units="V/C/m"))
        )
        setattr(
            self, "Wy", FieldParameter(name="Wy", value=UnitValue(wyval, units="V/C/m"))
        )
    elif field_type == "3DWake":
        wxval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wx"][0]
        wyval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wy"][0]
        wzval = [k["value"] for k in fdat if k["name"].lower().capitalize() == "Wz"][0]
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zval, units="m")))
        setattr(
            self, "Wx", FieldParameter(name="Wx", value=UnitValue(wxval, units="V/C/m"))
        )
        setattr(
            self, "Wy", FieldParameter(name="Wy", value=UnitValue(wyval, units="V/C/m"))
        )
        setattr(
            self, "Wz", FieldParameter(name="Wz", value=UnitValue(wzval, units="V/C"))
        )
    else:
        raise NotImplementedError(f"{field_type} loading not implemented for GDF files")
