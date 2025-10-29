import numpy as np
from .sdds import write_SDDS_field_file
from warnings import warn
from collections import Counter
from . import UnitValue, FieldParameter
import re

d = ",!?/&-:;@'\n \t"

def write_opal_field_file(
    self,
    frequency: float = None,
    radius: float = None,
    fourier: int = 100,
    orientation: str = None,
):
    length = self.length
    opal_file = self._output_filename(extension=".opal")
    data = None
    header = None
    fourier_ratio = fourier / length
    if isinstance(self.field_type, bytes):
        self.field_type = self.field_type.decode("utf-8")
    if self.field_type != "2DElectroDynamic":
        if fourier_ratio > 1.0:
            warn("Too many fourier components provided; defaulting to 1/100")
            fourier = int(length / 100)
        elif fourier_ratio < 0.01:
            warn("Not enough fourier components provided; defaulting to 1/100")
            fourier = int(length / 100)
    if self.field_type == "1DMagnetoStatic":
        zmin = min(self.z.value.val) * 1e2
        zmax = max(self.z.value.val) * 1e2
        data = self.Bz.value.val
        head = ["1DMagnetoStatic", str(fourier)]
        if radius is None:
            warn("Magnet radius not provided; defaulting to 10cm")
            radius = 0.1
        zvals = [str(zmin), str(zmax), str(length-1)]
        rvals = [str(0), str(radius * 100), fourier]
        header = [head, zvals, rvals]
    elif self.field_type == "1DElectroDynamic":
        if frequency is None:
            warn("RF Frequency not provided to field class")
            return
        head = ["AstraDynamic", str(fourier)]
        freq = [str(frequency * 1e-6)]
        zdata = self.z.value.val
        ezdata = self.Ez.value.val
        data = np.transpose([zdata, ezdata])
        header = [head, freq]
    elif self.field_type == "2DElectroDynamic":
        if frequency is None:
            warn("RF Frequency not provided to field class")
            return
        if radius is None:
            warn("Radius not provided to field class")
            return
        if orientation is None:
            warn("Orientation not provided to field class")
            return
        orient = self.orientation
        ezvals = self.Ez.value.val
        ervals = self.Er.value.val
        eabsvals = self.Ex.value.val
        brvals = self.Br.value.val
        shape = len(ezvals)
        count = Counter(ezvals)
        repeating_floats = {key: value for key, value in count.items() if value > 1}
        zlen = int(len(repeating_floats) - 1)
        rlen = int((shape / len(repeating_floats)) - 1)
        head = ["2DDynamic", orient]
        leng = ["0.0", str(self.length * 10), str(zlen)]
        freq = [str(frequency * 1e-6)]
        rad = ["0.0", str(self.radius * 10), str(rlen)]
        data = np.transpose([ezvals, ervals, eabsvals, brvals])
        header = [head, leng, freq, rad]
    elif "wake" in self.field_type:
        write_SDDS_field_file(self)
        warn(f"Field type {self.field_type} defaulting to SDDS type; use with caution")
    else:
        warn(f"Field type {self.field_type} not supported for OPAL")
    if data is not None:
        with open(f"{opal_file}", "w") as f:
            for h in header:
                f.write(" ".join([str(x) for x in h]) + "\n")
            if len(data.shape) > 1:
                for dat in data:
                    f.write(" ".join([str(x) for x in dat]) + "\n")
            else:
                for dat in data:
                    f.write(str(dat) + "\n")
    return opal_file

def read_opal_field_file(self, filename: str, field_type: str, cavity_type: str | None =  None, frequency: float | None = None):
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
    if field_type == "2DElectroDynamic":
        with open(filename) as f:
            rl = f.readlines()
            setattr(self, "length", float(rl[1].split(' ')[1]) * 1e-2)
            setattr(self, "frequency", float(rl[2]) * 1e6)
            setattr(self, "radius", float(rl[3].split(' ')[1]) * 1e-2)
            setattr(self, "orientation", re.split("[" + "\\".join(d) + "]", rl[0]))
        fdat = np.loadtxt(filename, skiprows=4)
        setattr(self, "Ez", FieldParameter(name="Ez", value=UnitValue(fdat[::, 0], units="V/m")))
        setattr(self, "Er", FieldParameter(name="Er", value=UnitValue(fdat[::, 1], units="V/m")))
        setattr(self, "Ex", FieldParameter(name="Ex", value=UnitValue(fdat[::, 2], units="V/m")))
        setattr(self, "Br", FieldParameter(name="Br", value=UnitValue(fdat[::, 3], units="T")))
    elif field_type == "1DMagnetoStatic":
        with open(filename) as f:
            rl = f.readlines()
            if rl[0].split(' ')[0] not in ["1DMagnetoStatic"]:
                raise NotImplementedError(f"{rl[0].split(' ')[0]} field type not implemented for OPAL fields")
            setattr(self, "fourier", int(rl[0].split(' ')[1]))
            setattr(self, "radius", float(rl[2].split(' ')[1]) * 1e-2)
            zstart = float(rl[1].split(' ')[0])
            zend = float(rl[1].split(' ')[1])
        fdat = np.loadtxt(filename, skiprows=3)
        zvals = np.linspace(zstart, zend, len(fdat))
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(zvals, units="m")))
        setattr(self, "Bz", FieldParameter(name="Bz", value=UnitValue(fdat / max(fdat), units="T")))
    elif field_type == "1DElectroDynamic":
        with open(filename) as f:
            rl = f.readlines()
            if rl[0].split(' ')[0] not in ["AstraDynamic"]:
                raise NotImplementedError(f"{rl[0].split(' ')[0]} field type not implemented for OPAL fields")
            setattr(self, "fourier", int(rl[0].split(' ')[1]))
            setattr(self, "frequency", float(rl[1]) * 1e6)
        fdat = np.loadtxt(filename, skiprows=2)
        setattr(self, "z", FieldParameter(name="z", value=UnitValue(fdat[::,0], units="m")))
        setattr(self, "Ez", FieldParameter(name="Ez", value=UnitValue(fdat[::,1] / max(fdat[::,1]), units="V/m")))