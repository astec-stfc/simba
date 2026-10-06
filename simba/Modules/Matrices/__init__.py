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
try:
    import nafflib

    use_naff = True
except ImportError:  # optional: `tune_from_trajectory` falls back to an FFT
    use_naff = False
from .. import constants
import munch
import glob
from . import hdf5
from . import elegant

from ..units import UnitValue

PLANES = {"x": 0, "y": 2}
"""Row/column of the 6x6 map each transverse plane starts at."""


def _block(matrix, plane):
    index = PLANES[plane]
    return np.asarray(matrix, dtype=float)[index : index + 2, index : index + 2]


def tune_from_trajectory(positions, momenta=None) -> float:
    """
    Fractional tune from one particle's turn-by-turn motion.

    The code-independent half of a frequency map: any backend that can hand
    back a trajectory gets a tune from the same arithmetic, which is what
    makes a footprint comparable across codes.

    Uses NAFF (``nafflib``) when it is installed and momenta are given, and
    an interpolated FFT peak otherwise.

    With positions alone the spectrum of a real signal
    is symmetric. The analytic signal
    ``x - i*px`` breaks the degeneracy and gives the full ``[0, 1)``.

    Returns
    -------
    float
        The tune, in ``[0, 1)`` with momenta and ``[0, 0.5]`` without, or
        NaN for a trajectory too short, flat or not finite.
    """
    values = np.asarray(positions, dtype=float)
    if len(values) < 8 or not np.all(np.isfinite(values)):
        return float("nan")
    values = values - values.mean()
    if not np.any(values):
        return float("nan")
    slopes = None
    if momenta is not None:
        slopes = np.asarray(momenta, dtype=float)
        if len(slopes) != len(values) or not np.all(np.isfinite(slopes)):
            return float("nan")
        slopes = slopes - slopes.mean()
    if use_naff and slopes is not None:
        try:
            return float(nafflib.tune(values, slopes)) % 1.0
        except Exception:
            pass  # fall through to the FFT, which needs nothing installed
    window = np.hanning(len(values))
    if slopes is None:
        spectrum = np.abs(np.fft.rfft(values * window))
        spectrum[0] = 0.0
    else:
        spectrum = np.abs(np.fft.fft((values - 1j * slopes) * window))
    peak = int(np.argmax(spectrum))
    size = len(spectrum)
    left = spectrum[(peak - 1) % size]
    right = spectrum[(peak + 1) % size]
    denominator = left - 2 * spectrum[peak] + right
    refined = peak + (0.5 * (left - right) / denominator if denominator else 0.0)
    return float(refined % len(values)) / len(values)


def probe_grid(centroid, delta: float = 1e-6):
    """
    The 13 particles a single-particle run tracks instead of a bunch.

    The centroid, plus a pair straddling it along each of the six
    coordinates. Linear map is created by finite differences
    (:func:`map_from_probes`).

    Parameters
    ----------
    centroid: array-like
        Six coordinates of the reference particle.
    delta: float
        Finite-difference step.

    Returns
    -------
    numpy.ndarray
        ``6 x 13``, the centroid first.
    """
    centre = np.asarray(centroid, dtype=float)
    step = abs(float(delta))
    probes = [centre]
    for index in range(6):
        for sign in (1, -1):
            probe = centre.copy()
            probe[index] += sign * step
            probes.append(probe)
    return np.array(probes).T


def map_from_probes(tracked, delta: float = 1e-6):
    """
    Centroid and 6x6 map recovered from a tracked :func:`probe_grid`.

    Parameters
    ----------
    tracked: array-like
        ``6 x 13`` of the probes' coordinates at the observation point, in
        the order :func:`probe_grid` produced them.
    delta: float
        The same step the probes were built with.

    Returns
    -------
    tuple
        ``(centroid, R)``, or ``(None, None)`` if any probe is missing.
    """
    values = np.asarray(tracked, dtype=float)
    if values.shape != (6, 13) or not np.all(np.isfinite(values)):
        return (None, None)
    step = abs(float(delta))
    matrix = np.zeros((6, 6))
    for index in range(6):
        matrix[:, index] = (
            values[:, 1 + 2 * index] - values[:, 2 + 2 * index]
        ) / (2 * step)
    return (values[:, 0], matrix)


def transform_distribution(coordinates, centroid_in, centroid_out, matrix):
    """Push a full distribution through a map without tracking it.

    ``z_out = c_out + R (z_in - c_in)``.
    """
    values = np.asarray(coordinates, dtype=float)
    return np.asarray(centroid_out, dtype=float)[:, np.newaxis] + np.asarray(
        matrix, dtype=float
    ) @ (values - np.asarray(centroid_in, dtype=float)[:, np.newaxis])


def normalise_coordinates(positions, momenta, beta, alpha, orbit=(0.0, 0.0)):
    """
    Courant-Snyder normalisation of one plane's turn-by-turn motion.

    ``X = (x - x_co) / sqrt(beta)`` and
    ``PX = (alpha * (x - x_co) + beta * (px - px_co)) / sqrt(beta)``.

    Normalising turns the betatron ellipse into a circle, leaving a single
    line. Subtracting the closed orbit is the other half: on a ring with errors
    an amplitude measured from the axis is measured from the wrong centre.
    """
    offsets = np.asarray(positions, dtype=float) - orbit[0]
    slopes = np.asarray(momenta, dtype=float) - orbit[1]
    root = math.sqrt(abs(float(beta))) or 1.0
    return offsets / root, (float(alpha) * offsets + float(beta) * slopes) / root


def tune_diffusion(x, px, y, py, twiss=None) -> tuple:
    """
    Tunes and a diffusion index from one particle's turn-by-turn motion. The
    record is split into two consecutive halves and a tune taken from each;
    a particle on regular motion gives the same tune twice, one near a
    resonance does not. The index is

    ``D = log10(sqrt(dQx**2 + dQy**2))``

    so more negative is more regular. The tune difference is taken
    *circularly* -- ``(q2 - q1 + 0.5) % 1 - 0.5``
    Needs the momenta, and really wants NAFF.

    Parameters
    ----------
    twiss: dict | None
        ``beta_x``/``alpha_x``/``beta_y``/``alpha_y``, and optionally
        ``closed_orbit_x``/``_px``/``_y``/``_py``. Given these the motion is
        Courant-Snyder normalised first; see :func:`normalise_coordinates`
        for why that sharpens the line.

    Returns
    -------
    tuple
        ``(tune_x, tune_y, D)`` from the **first** window, or NaNs if either
        window has no usable tune.
    """
    arrays = [np.asarray(a, dtype=float) for a in (x, px, y, py)]
    if twiss:
        try:
            arrays[0], arrays[1] = normalise_coordinates(
                arrays[0],
                arrays[1],
                twiss["beta_x"],
                twiss["alpha_x"],
                (twiss.get("closed_orbit_x", 0.0), twiss.get("closed_orbit_px", 0.0)),
            )
            arrays[2], arrays[3] = normalise_coordinates(
                arrays[2],
                arrays[3],
                twiss["beta_y"],
                twiss["alpha_y"],
                (twiss.get("closed_orbit_y", 0.0), twiss.get("closed_orbit_py", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            pass  # un-normalised still gives a tune, just a blunter line
    half = len(arrays[0]) // 2
    if half < 8:
        return (float("nan"), float("nan"), float("nan"))
    first, second = slice(0, half), slice(half, 2 * half)
    tunes = []
    for window in (first, second):
        tunes.append(
            (
                tune_from_trajectory(arrays[0][window], arrays[1][window]),
                tune_from_trajectory(arrays[2][window], arrays[3][window]),
            )
        )
    (qx1, qy1), (qx2, qy2) = tunes
    if any(math.isnan(q) for q in (qx1, qy1, qx2, qy2)):
        return (float("nan"), float("nan"), float("nan"))
    shift_x = (qx2 - qx1 + 0.5) % 1.0 - 0.5
    shift_y = (qy2 - qy1 + 0.5) % 1.0 - 0.5
    drift = math.hypot(shift_x, shift_y)
    return (qx1, qy1, math.log10(max(drift, 1e-16)))


def is_stable(matrix, plane: str = "x") -> bool:
    """
    Whether motion in ``plane`` is bounded turn after turn.
    ``|trace / 2| <= 1`` is the condition.

    Parameters
    ----------
    matrix: Any
        Data containing matrix information
    plane: str
        Plane to check

    Returns
    -------
    bool
        True is the plane is stable.
    """
    block = _block(matrix, plane)
    return bool(abs((block[0, 0] + block[1, 1]) / 2.0) <= 1.0)


def phase_advance(matrix, plane: str = "x") -> float:
    """
    One turn's phase advance in ``plane``, radians in ``[0, 2*pi)``.

    Parameters
    ----------
    matrix: Any
        Data containing matrix information
    plane: str
        Plane to check

    Returns
    -------
    float
        The phase advance in radians.
    """
    block = _block(matrix, plane)
    if not is_stable(matrix, plane):
        return float("nan")
    mu = math.acos(min(1.0, max(-1.0, (block[0, 0] + block[1, 1]) / 2.0)))
    return 2.0 * math.pi - mu if block[0, 1] < 0 else mu


def fractional_tune(matrix, plane: str = "x") -> float:
    """
    The fractional tune in ``plane``.

    Parameters
    ----------
    matrix: Any
        Data containing matrix information
    plane: str
        Plane to check

    Returns
    -------
    float
        The fractional tune in ``plane``.
    """
    return phase_advance(matrix, plane) / (2.0 * math.pi)


def periodic_twiss(matrix, plane: str = "x") -> dict:
    """
    The periodic ``beta``, ``alpha`` and ``gamma`` in ``plane``.

    The Twiss the lattice itself determines, as opposed to whatever the
    incoming beam happened to have.

    Parameters
    ----------
    matrix: Any
        Data containing matrix information
    plane: str
        Plane to check

    Returns
    -------
    dict
        ``beta``, ``alpha``, ``gamma``, all NaN if the plane is unstable.
    """
    block = _block(matrix, plane)
    mu = phase_advance(matrix, plane)
    if math.isnan(mu) or math.sin(mu) == 0.0:
        return {"beta": float("nan"), "alpha": float("nan"), "gamma": float("nan")}
    sin_mu = math.sin(mu)
    beta = block[0, 1] / sin_mu
    alpha = (block[0, 0] - block[1, 1]) / (2.0 * sin_mu)
    return {"beta": beta, "alpha": alpha, "gamma": (1.0 + alpha**2) / beta}


def slip_factor(matrix, circumference: float, step: float = 1e-3) -> float:
    """
    ``eta``, the fractional change in revolution period per unit ``delta``.

    The map is in canonical coordinates, so pass
    :meth:`~simba.Framework_objects.frameworkLattice.one_turn_map_canonical`,
    not the raw map.

    Parameters
    ----------
    matrix: Any
        Data containing matrix information
    circumference: float
        Ring circumference
    step: float, optional
        Step size for finite difference, by default 1e-3

    Returns
    -------
    float
        The slip factor.
    """
    matrix = np.asarray(matrix, dtype=float)
    solve_matrix = matrix - np.eye(6)
    solve_matrix[4, :] = [0, 0, 0, 0, 1, 0]
    solve_matrix[5, :] = [0, 0, 0, 0, 0, 1]
    target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, step])
    orbit, *_ = np.linalg.lstsq(solve_matrix, target, rcond=None)
    dzeta = (matrix @ orbit)[4] - orbit[4]
    return -float(dzeta) / step / float(circumference)


def momentum_compaction(
    matrix, circumference: float, gamma0: float, step: float = 1e-3
) -> float:
    """
    ``alpha_c``, the fractional change in path length per unit ``delta``.

    ``alpha_c = eta + 1 / gamma0**2``; see :func:`slip_factor`.

    Parameters
    ----------
    matrix: Any
        Data containing matrix information
    circumference: float
        Ring circumference
    step: float, optional
        Step size for finite difference, by default 1e-3

    Returns
    -------
    float
        The momentum compaction factor.
    """
    return (
        slip_factor(matrix, circumference, step) + 1.0 / float(gamma0) ** 2
    )


class matrices(munch.Munch):
    """Class for dealing with R-matrices produced by Elegant.

    Usage::

        mat = matrices()
        mat.load(<filename>, reset=False, cumulative=True)

    ``load`` reads the sdds output file from the ``matrix_output`` command, with
    ``reset`` resetting all parameters to ``None`` and ``cumulative`` saying
    whether the R-matrices are cumulative or element-by-element.

    ``mat.R``
        The nx6x6 R-matrices that have been loaded, where ``n`` is the number of
        elements.

    ``mat.cumulativeR``
        The cumulative R-matrices for the loaded R-matrices in order.

    ``mat.elementR``
        The element-by-element R-matrices for the loaded R-matrices in order.
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
