import sys
import os
import numpy as np
from .. import constants
from ..units import UnitValue


def write_bdsim_beam_file(beam, filename):
    np.savetxt(
        filename,
        np.transpose(
            np.array(
                [
                    beam.x.val,
                    beam.y.val,
                    beam.z.val,
                    beam.xp.val,
                    beam.yp.val,
                    beam.energy.val,
                ]
            )
        ),
    )
