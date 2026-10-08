"""
Ocelot's cavity maps, without the cancellation at small energy gain.

Ocelot's ``r55`` correction and second-order ``T566``, ``T556`` and ``T555``
divide by a power of ``g1 - g0``, the gain in gamma, against a numerator that
vanishes as fast. At the zero crossing, where a ring's cavity sits with
radiation off, that is 0/0: NaN, and the particle is lost.
"""

import numpy as np

_PATCHED = False


def _gain_ratios(g0: float, g1: float) -> tuple:
    """
    Ocelot's ``(g0 g1 (b0 b1 - 1) + 1) / (g0 - g1)^2`` and the ``T566``, ``T556`` and
    ``T555`` numerators over ``(g0 - g1)``, ``(g0 - g1)^2`` and ``(g0 - g1)^3``,
    with ``g0 - g1`` cancelled.
    """
    p0, p1 = np.sqrt(g0 * g0 - 1.0), np.sqrt(g1 * g1 - 1.0)
    u0, u1 = np.arccosh(g0), np.arccosh(g1)
    a, b = (u0 + u1) / 2.0, (u1 - u0) / 2.0
    sinh_a = np.sinh(a)
    r55 = -1.0 / (g0 * g1 + p0 * p1 - 1.0)
    t566 = (g0 + g1) * (p0 * p0 + p0 * p1 + p1 * p1) / (p0 + p1)
    t556 = (2.0 * np.sinh(2.0 * a + b) * np.cosh(b) + np.sinh(2.0 * a)) / (
        4.0 * g0 * sinh_a**2
    )
    t555 = np.sinh(2.0 * a + b) / (2.0 * sinh_a**3)
    return r55, t566, t556, t555


def stable_cavity_maps() -> None:
    """
    Patch Ocelot's ``CavityAtom`` first-order ``r55`` and ``CavityTM.map4cav``
    second-order terms with the forms above. Run before any Twiss or tracking
    """
    global _PATCHED
    if _PATCHED:
        return
    from ocelot.common.globals import speed_of_light
    from ocelot.cpbd.elements.cavity_atom import CavityAtom
    from ocelot.cpbd.high_order import m_e_GeV
    from ocelot.cpbd.transformations.cavity import CavityTM

    main_matrix = CavityAtom._R_main_matrix

    def _R_main_matrix(self, energy: float, length: float):
        R = main_matrix(self, energy, length)
        if self.v == 0.0 or energy == 0.0:
            return R
        V = self.v * length / self.l
        phi = self.phi * np.pi / 180.0
        g0 = energy / m_e_GeV
        g1 = (energy + V * np.cos(phi)) / m_e_GeV
        if g1 <= 1.0:
            return R
        beta0, beta1 = np.sqrt(1.0 - 1.0 / g0**2), np.sqrt(1.0 - 1.0 / g1**2)
        k = 2.0 * np.pi * self.freq / speed_of_light
        r55 = _gain_ratios(g0, g1)[0]
        R[4, 4] = 1.0 + k * length * beta0 * V / m_e_GeV * np.sin(phi) * r55 / (
            beta1 * g1
        )
        return R

    def map4cav(self, X, E, delta_length, length):
        # Ocelot 25.6's CavityTM.map4cav, with only T566, T556 and T555 changed
        params = self.get_params(E)
        if delta_length is not None:
            V = params.v * delta_length / length if length != 0 else params.v
            z = delta_length
        else:
            V = params.v
            z = length

        beta0 = 1
        igamma2 = 0
        g0 = 1e10
        if E != 0:
            g0 = E / m_e_GeV
            igamma2 = 1.0 / (g0 * g0)
            beta0 = np.sqrt(1.0 - igamma2)

        phi = params.phi * np.pi / 180.0

        X4 = np.copy(X[4])
        X5 = np.copy(X[5])
        X = self.mul_p_array(X, energy=E)
        delta_e = V * np.cos(phi)

        T566 = 1.5 * z * igamma2 / (beta0**3)
        T556 = 0.0
        T555 = 0.0
        if E + delta_e > 0:
            k = 2.0 * np.pi * params.freq / speed_of_light
            E1 = E + delta_e
            g1 = E1 / m_e_GeV
            beta1 = np.sqrt(1.0 - 1.0 / (g1 * g1))

            X[5] = X5 * E * beta0 / (E1 * beta1) + V * beta0 / (E1 * beta1) * (
                np.cos(-X4 * beta0 * k + phi) - np.cos(phi)
            )

            dgamma = V / m_e_GeV
            if delta_e > 0:
                r55, t566, t556, t555 = _gain_ratios(g0, g1)
                T566 = z * t566 / (2 * beta0 * beta1**3 * g0 * g1**3)
                T556 = beta0 * k * z * dgamma * g0 * t556 * np.sin(phi) / (
                    beta1**3 * g1**3
                )
                T555 = beta0**2 * k**2 * z * dgamma / 2.0 * (
                    dgamma * t555 / (beta1**3 * g1**3) * np.sin(phi) ** 2
                    - r55 / (beta1 * g1) * np.cos(phi)
                )
        X[4] += T566 * X5 * X5 + T556 * X4 * X5 + T555 * X4 * X4

        return X

    CavityAtom._R_main_matrix = _R_main_matrix
    CavityTM.map4cav = map4cav
    _PATCHED = True
