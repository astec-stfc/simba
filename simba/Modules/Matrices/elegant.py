import os
from ..SDDSFile import SDDSFile
import numpy as np
from .. import constants
from ..units import UnitValue


def read_elegant_matrix_files(self, filename, reset=True):
    if isinstance(filename, (list, tuple)):
        for f in filename:
            read_elegant_matrix_files(self, f, reset=False)
    elif os.path.isfile(filename):
        pre, ext = os.path.splitext(filename)
        self.sddsindex += 1
        elegantObject = SDDSFile(index=(self.sddsindex))
        elegantObject.read_file(pre + ".flr")
        elegantObject.read_file(pre + ".mat")
        elegantData = elegantObject.columns()
        update_arrays(self, elegantData, reset=reset)
        elegantData = elegantObject.parameters()
        update_arrays(self, elegantData, reset=reset)


def update_arrays(self, elegantData, reset=True):
    if reset:
        [delattr(self, k) for k in elegantData.keys() if hasattr(self, k)]
    [
        (
            self.append(k, v.data)
            if hasattr(self, k)
            else self.initialize_array(k, v.data, units=v.unit)
        )
        for k, v in elegantData.items()
    ]  # if not k == 's'
