"""
Shared RF-cavity transverse focusing matrices, used by both the MAD-X and
Ocelot backends (:mod:`~simba.Codes.MADX.MADX`,
:mod:`~simba.Codes.Ocelot.Ocelot`).
"""

import numpy as np

from . import constants

speed_of_light = constants.speed_of_light

TW1_N_SLICES = 50
"""Number of internal drift-kick-drift slices used by
:func:`tw1_focusing_matrix` to numerically build the travelling-wave
focusing matrix (ELEGANT's own ``BODY_FOCUS_MODEL=TW1`` requires
``N_KICKS >= 10`` for the same reason; the drift-kick-drift integration is
only first-order accurate, so a larger margin is used here since this is a
pure Python loop and the extra slices are practically free)."""


def tw1_focusing_matrix(
    volt: float,
    freq: float,
    phi: float,
    length: float,
    energy: float,
    m0: float,
    n_slices: int = TW1_N_SLICES,
    canonical_rescale: bool = True,
    end1_focus: bool = True,
    end2_focus: bool = True,
) -> tuple:
    """
    Build the transverse first-order transfer matrix for a travelling-wave
    cavity, reproducing ELEGANT's ``BODY_FOCUS_MODEL=TW1`` body kick *plus*
    its ``END1_FOCUS``/``END2_FOCUS`` entrance/exit RF focusing (see
    ``identifyRfcaBodyFocusModel``/the ``twFocusing1`` branch, and the
    end-focus kicks, of ``track_through_rf_cavity`` in ELEGANT's
    ``simple_rfca.c``). Unlike the Rosenzweig-Serafini standing-wave matrix
    (as used by MAD-X's ``rsmatrix`` cavity model and Ocelot's
    ``CavityAtom``, both of which assume a pure pi-mode standing wave and
    do not apply to a travelling structure), this integrates the actual
    radial-electric / azimuthal-magnetic field kick of a synchronous
    forward travelling wave. For an on-axis field
    ``Ez = E0 sin(kz - wt + phi)`` (phase velocity = c), the paraxial
    fields are ``Er = -(r/2) dEz/dz`` and ``Bphi = Er/c``, so the net
    radial Lorentz force on a co-propagating ultrarelativistic particle,
    ``F_r = q(Er - v_z*Bphi) = q*Er*(1 - v_z/c)``, is suppressed by
    ``(1 - beta) ~ 1/(2*gamma**2)`` relative to the naive electrostatic
    term -- this is why travelling-wave RF focusing is much weaker than
    standing-wave, and why re-using the SW matrix over-predicts it.

    The cavity body is sliced into ``n_slices`` drift-kick-drift steps,
    each applying ELEGANT's ``dpr = volt/(2*beta)*(omega/c)*(1-beta)*cos(phi)``
    kick (to both x and y, since the force is rotationally symmetric), and
    chain-multiplied in the beam's local (accelerating) momentum
    normalisation. ``end1_focus``/``end2_focus`` add ELEGANT's separate
    thin-lens entrance/exit RF-focusing kicks on top of the body matrix --
    the same "quasi-static" edge-field focusing present at the ends of any
    cavity (a much larger effect than the body kick itself, since it is
    *not* suppressed by ``(1-beta)``) -- using the standard
    ``k = -/+ (E0 sin(phi)) / (2 * E)`` thin-lens strength (``E0 = volt /
    length`` the average on-axis gradient, entrance evaluated at the
    entrance energy, exit at the exit energy, sign flipped between the
    two). Leaving both off reproduces the pre-existing body-only matrix.

    Parameters
    ----------
    volt: float
        Total cavity voltage [eV] (crest energy gain)
    freq: float
        RF frequency [Hz]
    phi: float
        RF phase [rad], such that the energy gain is ``volt * sin(phi)``
        (crest acceleration at ``phi = pi/2``, matching both ELEGANT's and
        MAD-X's ``de = volt * sin(2*pi*lag)`` phase convention)
    length: float
        Physical cavity length [m]
    energy: float
        Total beam (design) energy at the cavity entrance [eV]
    m0: float
        Particle rest energy [eV]
    n_slices: int
        Number of internal integration slices
    canonical_rescale: bool
        If True (default), rescale the output row by the ratio of the exit
        to entrance local momentum, converting from the local (energy-
        following) momentum normalisation used internally to a *fixed*
        reference-momentum canonical convention -- required by MAD-X, whose
        ``MATRIX``/track coordinates are normalised to a single reference
        momentum per segment (see :meth:`~simba.Codes.MADX.MADX.madxLattice.rs_matrix_cavity`).
        Ocelot re-normalises its canonical momentum locally at every
        element instead, so its caller should pass False.
    end1_focus: bool
        Apply the entrance RF-focusing kick (matches ELEGANT's
        ``END1_FOCUS``, default on there too).
    end2_focus: bool
        Apply the exit RF-focusing kick (matches ELEGANT's
        ``END2_FOCUS``, default on there too).

    Returns
    -------
    tuple
        (m11, m12, m21, m22, exit energy [eV]) for the transverse (x, x')
        and (y, y') blocks (identical to each other, since the focusing is
        rotationally symmetric)
    """
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    k_wave = 2.0 * np.pi * freq / speed_of_light
    volt_slice = volt / n_slices
    length_slice = length / n_slices
    de_slice = volt_slice * sin_phi

    M = np.eye(2)
    E_i = energy
    for _ in range(n_slices):
        gamma_i = E_i / m0
        P_i = np.sqrt(gamma_i**2 - 1.0)
        E_f = E_i + de_slice
        gamma_f = E_f / m0
        P_f = np.sqrt(gamma_f**2 - 1.0)
        beta_i = P_i / gamma_i
        one_minus_beta_i = 1.0 / (gamma_i * (gamma_i + P_i))
        dgamma_slice = de_slice / m0
        dpr = dgamma_slice / (2.0 * beta_i) * k_wave * one_minus_beta_i * cos_phi
        kappa = dpr / P_f if P_f > 0 else 0.0
        # Adiabatic damping: even with zero transverse kick, the geometric
        # slope u=px/pz shrinks by Pi/Pf across the slice because pz grows
        # while px is unchanged (u_after = u_before*(Pi/Pf) + kappa*x) --
        # dropping this (as an earlier version of this function did) breaks
        # symplecticity and lets beta functions blow up unphysically.
        rho = P_i / P_f if P_f > 0 else 1.0
        d = length_slice / 2.0
        # drift(d) . kick(kappa, rho) . drift(d), applied to the running product
        m11 = 1.0 + d * kappa
        m12 = d * (1.0 + d * kappa + rho)
        m21 = kappa
        m22 = d * kappa + rho
        M = np.array([[m11, m12], [m21, m22]]) @ M
        E_i = E_f

    if (end1_focus or end2_focus) and length > 0:
        E0 = volt / length
        if end1_focus:
            k1 = -(E0 * sin_phi) / (2.0 * energy)
            M = M @ np.array([[1.0, 0.0], [k1, 1.0]])
        if end2_focus:
            k2 = (E0 * sin_phi) / (2.0 * E_i)
            M = np.array([[1.0, 0.0], [k2, 1.0]]) @ M

    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]
    if canonical_rescale:
        P_i_total = np.sqrt((energy / m0) ** 2 - 1.0)
        P_f_total = np.sqrt((E_i / m0) ** 2 - 1.0)
        scale = P_f_total / P_i_total if P_i_total > 0 else 1.0
        m21, m22 = m21 * scale, m22 * scale
    return m11, m12, m21, m22, E_i
