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
import munch
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

    Uses NAFF (``nafflib``) when installed and momenta are given, else an
    interpolated FFT peak. Momenta resolve the ``q``/``1 - q`` ambiguity of a real signal.

    Parameters
    ----------
    positions: array-like
        Turn-by-turn position
    momenta: array-like, optional
        Turn-by-turn momentum

    Returns
    -------
    float
        Tune in ``[0, 1)`` with momenta, ``[0, 0.5]`` without; NaN for a
        trajectory too short, flat or not finite
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
    The centroid plus a pair of probes straddling it in each of the six coordinates.

    Track these instead of a bunch and recover the linear map with :func:`map_from_probes`.

    Parameters
    ----------
    centroid: array-like
        Six coordinates of the reference particle
    delta: float
        Finite-difference step

    Returns
    -------
    numpy.ndarray
        ``6 x 13``, the centroid first
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
        ``6 x 13`` probe coordinates at the observation point, in :func:`probe_grid` order
    delta: float
        Step the probes were built with

    Returns
    -------
    tuple
        ``(centroid, R)``, or ``(None, None)`` if any probe is missing
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
    """Push a distribution through a linear map: ``z_out = c_out + R (z_in - c_in)``."""
    values = np.asarray(coordinates, dtype=float)
    return np.asarray(centroid_out, dtype=float)[:, np.newaxis] + np.asarray(
        matrix, dtype=float
    ) @ (values - np.asarray(centroid_in, dtype=float)[:, np.newaxis])


def normalise_coordinates(positions, momenta, beta, alpha, orbit=(0.0, 0.0)):
    """
    Courant-Snyder normalisation of one plane's turn-by-turn motion.

    ``X = (x - x_co) / sqrt(beta)``,
    ``PX = (alpha * (x - x_co) + beta * (px - px_co)) / sqrt(beta)``.
    This turns the betatron ellipse into a circle, which sharpens the tune line.

    Parameters
    ----------
    positions, momenta: array-like
        Turn-by-turn motion
    beta, alpha: float
        Twiss parameters at the observation point
    orbit: tuple
        Closed orbit ``(x_co, px_co)``

    Returns
    -------
    tuple
        ``(X, PX)``
    """
    offsets = np.asarray(positions, dtype=float) - orbit[0]
    slopes = np.asarray(momenta, dtype=float) - orbit[1]
    root = math.sqrt(abs(float(beta))) or 1.0
    return offsets / root, (float(alpha) * offsets + float(beta) * slopes) / root


def tune_diffusion(x, px, y, py, twiss=None) -> tuple:
    """
    Tunes and diffusion index ``D = log10(sqrt(dQx**2 + dQy**2))`` from turn-by-turn motion.

    ``dQ`` is the (circular) tune change between the two halves of the record, so more
    negative ``D`` is more regular. Works best with NAFF installed.

    Parameters
    ----------
    x, px, y, py: array-like
        Turn-by-turn motion
    twiss: dict | None
        ``beta_x``/``alpha_x``/``beta_y``/``alpha_y``, and optionally
        ``closed_orbit_x``/``_px``/``_y``/``_py``; if given, the motion is
        normalised first (:func:`normalise_coordinates`)

    Returns
    -------
    tuple
        ``(tune_x, tune_y, D)`` with tunes from the first half, or NaNs if
        either half has no usable tune
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
    tunes = [
        (
            tune_from_trajectory(arrays[0][window], arrays[1][window]),
            tune_from_trajectory(arrays[2][window], arrays[3][window]),
        )
        for window in (first, second)
    ]
    (qx1, qy1), (qx2, qy2) = tunes
    if any(math.isnan(q) for q in (qx1, qy1, qx2, qy2)):
        return (float("nan"), float("nan"), float("nan"))
    shift_x = (qx2 - qx1 + 0.5) % 1.0 - 0.5
    shift_y = (qy2 - qy1 + 0.5) % 1.0 - 0.5
    drift = math.hypot(shift_x, shift_y)
    return (qx1, qy1, math.log10(max(drift, 1e-16)))


def is_stable(matrix, plane: str = "x") -> bool:
    """
    Whether motion in ``plane`` is bounded, i.e. ``|trace / 2| <= 1``.

    Parameters
    ----------
    matrix: array-like
        One-turn 6x6 map
    plane: str
        'x' or 'y'

    Returns
    -------
    bool
    """
    block = _block(matrix, plane)
    return bool(abs((block[0, 0] + block[1, 1]) / 2.0) <= 1.0)


def phase_advance(matrix, plane: str = "x") -> float:
    """
    One turn's phase advance in ``plane``.

    Parameters
    ----------
    matrix: array-like
        One-turn 6x6 map
    plane: str
        'x' or 'y'

    Returns
    -------
    float
        Phase advance in ``[0, 2*pi)`` rad; NaN if unstable
    """
    block = _block(matrix, plane)
    if not is_stable(matrix, plane):
        return float("nan")
    mu = math.acos(min(1.0, max(-1.0, (block[0, 0] + block[1, 1]) / 2.0)))
    return 2.0 * math.pi - mu if block[0, 1] < 0 else mu


def fractional_tune(matrix, plane: str = "x") -> float:
    """
    Fractional tune in ``plane``.

    Parameters
    ----------
    matrix: array-like
        One-turn 6x6 map
    plane: str
        'x' or 'y'

    Returns
    -------
    float
        Fractional tune; NaN if unstable
    """
    return phase_advance(matrix, plane) / (2.0 * math.pi)


def periodic_twiss(matrix, plane: str = "x") -> dict:
    """
    Periodic ``beta``, ``alpha`` and ``gamma`` in ``plane``, set by the lattice rather than the beam.

    Parameters
    ----------
    matrix: array-like
        One-turn 6x6 map
    plane: str
        'x' or 'y'

    Returns
    -------
    dict
        ``beta``, ``alpha``, ``gamma``; all NaN if the plane is unstable
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
    Slip factor ``eta``, the fractional change in revolution period per unit ``delta``.

    Parameters
    ----------
    matrix: array-like
        Canonical one-turn map, from
        :meth:`~simba.Framework_objects.frameworkLattice.one_turn_map_canonical`, not the raw map
    circumference: float
        Ring circumference [m]
    step: float, optional
        Finite-difference step in ``delta``

    Returns
    -------
    float
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
    Momentum compaction ``alpha_c = eta + 1 / gamma0**2``; see :func:`slip_factor`.

    Parameters
    ----------
    matrix: array-like
        Canonical one-turn map
    circumference: float
        Ring circumference [m]
    gamma0: float
        Reference Lorentz factor
    step: float, optional
        Finite-difference step in ``delta``

    Returns
    -------
    float
    """
    return (
        slip_factor(matrix, circumference, step) + 1.0 / float(gamma0) ** 2
    )


class matrices(munch.Munch):
    """R-matrices from ELEGANT's ``matrix_output``.

    Load with ``mat.load(filename, reset=False, cumulative=True)``, where ``cumulative``
    says whether the file holds cumulative or element-by-element matrices. ``mat.R`` is
    the list of loaded ``n x 6 x 6`` arrays; ``cumulativeR()`` and ``individualR()``
    convert between the two forms.
    """

    def __init__(self):
        super().__init__()
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

    def __repr__(self):
        return repr(list(self.keys()))

    def units(self, key):
        if key in self:
            return self[key].units

    def append(self, array, data):
        self[array].append(UnitValue(data, units=self[array][0].units))

    def initialize_array(self, array, data, units=None):
        self[array] = [UnitValue(data, units=units)]

    def _which_code(self, name):
        if name.lower() in self.codes:
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
        return [a for ar in arr for a in ar]

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
            if self._cumulative[i]:
                element_matrices = []
                reduce(
                    lambda A, b: self.matrixsolve(A, b, element_matrices),
                    self.R[i],
                    np.identity(6),
                )
                iR.append(element_matrices)
            else:
                iR.append(self.R[i])
        return iR
