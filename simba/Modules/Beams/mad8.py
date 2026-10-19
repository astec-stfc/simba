import numpy as np
from .. import constants


def write_mad8_beam_file(self, filename):
    """Save a mad8 beam file using multiple START commands."""
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
