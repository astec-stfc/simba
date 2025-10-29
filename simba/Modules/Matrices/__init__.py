import os
import math
import warnings
import numpy as np
from functools import reduce

try:
    from scipy import interpolate

    use_interpolate = True
except ImportError:
    use_interpolate = False
from .. import constants
import munch
import glob
from . import hdf5
from . import elegant

from ..units import UnitValue


class matrices(munch.Munch):
    """Class for dealing with R-matrices produced by Elegant.

    Usage:
    mat = matrices()
    mat.load(<filename>, reset=False, cumulative=True)
        Load sdds output file from the "matrix_output" command.
            reset: Reset all parameters to None
            cumulative: Are the R-matrices cumulative or element-by-element?
    mat.R:
        Return the nx6x6 R-matrices that have been loaded where "n" is the number of elements
    mat.cumulativeR:
        Return the cumulative R-matrices for the loaded R-matrices in order.
    mat.elementR:
        Return the element-by-element R-matrices for the loaded R-matrices in order.

    """

    def __init__(self):
        super().__init__()
        # self.reset_dicts()
        self.sddsindex = 0
        self._cumulative = {}
        self.codes = {
            "elegant": elegant.read_elegant_matrix_files,
        }
        self.code_signatures = [["elegant", ".mat"]]

    def read_elegant_matrix_files(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return elegant.read_elegant_matrix_files(self, *args, **kwargs)

    # def save_HDF5_twiss_file(self, *args, **kwargs):
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         return hdf5.write_HDF5_twiss_file(self, *args, **kwargs)

    def __repr__(self):
        return repr([k for k in self.keys()])

    def units(self, key):
        if key in self:
            return self[key].units

    def append(self, array, data):
        self[array].append(UnitValue(data, units=self[array][0].units))

    def initialize_array(self, array, data, units=None):
        self[array] = [UnitValue(data, units=units)]

    def _which_code(self, name):
        if name.lower() in self.codes.keys():
            return self.codes[name.lower()]
        return None

    def _determine_code(self, filename):
        for k, v in self.code_signatures:
            l = -len(v)
            if v == filename[l:]:
                return self.codes[k]
        return None

    def load(self, filename, reset=False, cumulative=True):
        self._cumulative[self.sddsindex] = cumulative
        if self._determine_code(filename) is not None:
            self._determine_code(filename)(self, filename, reset=reset)

    def generate_R_matrix(self, index):
        R = np.empty((len(self["R11"][index]), 6, 6))
        for k in range(len(self["R11"][index])):
            for i in range(1, 7):
                for j in range(1, 7):
                    mat = getattr(self, "R" + str(i) + str(j))[index]
                    R[k, i - 1, j - 1] = mat[k]
        return R

    @property
    def R(self, index=None):
        return [self.generate_R_matrix(i) for i in range(len(self.R11))]

    def flatten1(self, arr):
        newarr = []
        for ar in arr:
            for a in ar:
                newarr.append(a)
        return newarr

    def cumulativeR(self, combined=False):
        cR = []
        if combined:
            ir = list(reversed(self.flatten1(self.individualR())))
            r = ir[0]
            for mat in ir[1:]:
                r = np.dot(mat, r)
                cR.append(r)
        else:
            for i in range(len(self.R11)):
                if self._cumulative[i]:
                    cR.append(self.R[i])
                else:
                    ir = list(reversed(self.R[i]))
                    r = ir[0]
                    for mat in ir[1:]:
                        r = np.dot(mat, r)
                    cR.append(r)
        return cR

    def matrixsolve(self, A, b, elist):
        elist.append(np.linalg.solve(A.T, b.T).T)
        return b

    def individualR(self):
        iR = []
        for i in range(len(self.R11)):
            if self._cumulative:
                element_matrices = []
                reduce(
                    lambda A, b: self.matrixsolve(A, b, element_matrices),
                    self.R[i],
                    np.identity(6),
                )
                element_dict = dict()
                iR.append(element_matrices)
            else:
                iR.append(self.R[i])
        return iR


# def load_directory(directory='.', types={'elegant':'.twi', 'GPT': 'emit.gdf','ASTRA': 'Xemit.001'}, preglob='*', verbose=False, sortkey='z'):
#     t = twiss()
#     if verbose:
#         print('Directory:',directory)
#     for code, string in types.items():
#         twiss_files = glob.glob(directory+'/'+preglob+string)
#         if verbose:
#             print(code, [os.path.basename(t) for t in twiss_files])
#         if t._which_code(code) is not None and len(twiss_files) > 0:
#             t._which_code(code)(t, twiss_files, reset=False)
#     t.sort(key=sortkey)
#     return t
#
# def load_file(filename, *args, **kwargs):
#     twissobject = twiss()
#     code = twissobject._determine_code(filename)
#     if code is not None:
#         code(twissobject, filename, reset=False)
#     return twissobject
