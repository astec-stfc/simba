import shlex
import numpy as np
from .. import constants

try:
    import sdds
except:
    print("sdds failed to load")
    pass

HEADER = "@"
NAMES = "*"
TYPES = "$"
COMMENTS = "#"
INDEX_ID = "INDEX&&&"
ID_TO_TYPE = {  # used when reading files
    "%s": np.str_,
    "%bpm_s": np.str_,
    "%le": np.float64,
    "%f": np.float64,
    "%hd": np.int64,
    "%d": np.int64,
}
DEFAULT_COLUMN_WIDTH = 20
MIN_COLUMN_WIDTH = 10


def read_tfs(tfs_file_path: str, index: str = None):
    """
    Parses the TFS table present in **tfs_file_path** and returns a dictionary.
    Args:
        tfs_file_path (str): Path object to the output TFS file.
        index (str): Name of the column to set as index. If not given, looks in **tfs_file_path**
            for a column starting with `INDEX&&&`.
    Returns:
        Dictionary object.
    """
    headers = OrderedDict()
    rows_list = []
    column_names = column_types = None

    with tfs_file_path.open("r") as tfs_data:
        for line in tfs_data:
            line_components = shlex.split(line)
            if not line_components:
                continue
            if line_components[0] == HEADER:
                name, value = _parse_header(line_components[1:])
                headers[name] = value
            elif line_components[0] == NAMES:
                column_names = np.array(line_components[1:])
            elif line_components[0] == TYPES:
                column_types = _compute_types(line_components[1:])
            elif line_components[0] == COMMENTS:
                continue
            else:
                if column_names is None:
                    raise Exception("Column names have not been set.")
                if column_types is None:
                    raise Exception("Column types have not been set.")
                line_components = [part.strip('"') for part in line_components]
                rows_list.append(line_components)
    return column_names, column_types, rows_list, headers


def read_tfs_beam_file(self, fileName, charge=None):
    self.reset_dicts()
    column_names, column_types, rows_list, headers = self.read_tfs(filename)
    print(column_names, column_types, rows_list, headers)
    self.filename = fileName
    self["code"] = "tfs"
    cp = (self._beam["p"]) * self.E0_eV
    cpz = cp / np.sqrt(self._beam["xp"] ** 2 + self._beam["yp"] ** 2 + 1)
    cpx = self._beam["xp"] * cpz
    cpy = self._beam["yp"] * cpz
    self._beam["px"] = cpx * self.q_over_c
    self._beam["py"] = cpy * self.q_over_c
    self._beam["pz"] = cpz * self.q_over_c
    # self._beam['t'] = self._beam['t']
    self._beam["z"] = (-1 * self._beam.Bz * constants.speed_of_light) * (
        self._beam.t - np.mean(self._beam.t)
    )  # np.full(len(self.t), 0)
    if "Charge" in SDDSparameters and len(SDDSparameters["Charge"]) > 0:
        self._beam["total_charge"] = SDDSparameters["Charge"][0]
        self._beam["charge"] = np.full(
            len(self._beam["z"]), self._beam["total_charge"] / len(self._beam["x"])
        )
    elif charge is None:
        self._beam["total_charge"] = 0
        self._beam["charge"] = np.full(
            len(self._beam["z"]), self._beam["total_charge"] / len(self._beam["x"])
        )
    else:
        self._beam["total_charge"] = charge
        self._beam["charge"] = np.full(
            len(self._beam["z"]), self._beam["total_charge"] / len(self._beam["x"])
        )
    # self._beam['charge'] = []


def write_mad8_beam_file(self, filename):
    """Save a mad8 beam file using multiple START commands."""
    Cnames = ["x", "xp", "y", "yp", "t", "p"]
    Ccolumns = ["x", "xp", "y", "yp", "t", "BetaGamma"]
    Ccolumns = np.array(
        [np.array(self.x), self.xp, np.array(self.y), self.yp, self.t, self.deltap]
    ).T
    with open(filename, "w") as file:
        for [x, px, y, py, t, deltap] in Ccolumns:
            file.write(
                """START, X="""
                + str(x)
                + """,&
            PX="""
                + str(px)
                + """,&
            Y="""
                + str(y)
                + """,&
            PY="""
                + str(py)
                + """,&
            T="""
                + str(-constants.speed_of_light * t)
                + """,&
            DELTAP="""
                + str(deltap)
                + """;\n"""
            )


def set_beam_charge(self, charge):
    self._beam["total_charge"] = charge
