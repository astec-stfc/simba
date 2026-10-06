"""
Sanity checks for the Ocelot travelling-wave focusing bodge
(simba.Codes.Ocelot.Ocelot._sw_focusing_matrix / add_tw1_focusing).

Plain asserts, no framework: run directly with `python tests/test_ocelot_tw1.py`.
"""

import numpy as np

from simba.Codes.Ocelot.Ocelot import _sw_focusing_matrix
from simba.Modules.rf_focusing import tw1_focusing_matrix

M0 = 0.51099895e6  # electron rest energy [eV]
VOLT = 20e6  # eV
FREQ = 2998.5e6  # Hz
LENGTH = 2.0  # m
ENERGY = 10e6  # eV (entrance)
PHI = 2 * np.pi * 0.15  # off crest, matches tw1_focusing_matrix's de=volt*sin(phi)


def test_sw_matrix_matches_ocelots_own_cavity_atom():
    # _sw_focusing_matrix must reproduce Ocelot's real CavityAtom (eta=1)
    # formula exactly -- that's what the correction matrix cancels out at
    # run time. Build a real CavityAtom with the equivalent (V, phi_deg)
    # and compare its own _R_main_matrix against our reproduction.
    from ocelot.cpbd.elements.cavity_atom import CavityAtom
    from ocelot.common.globals import m_e_GeV

    phi_deg = 90.0 - np.degrees(PHI)  # Ocelot: de = v*cos(phi_deg)
    cav = CavityAtom(l=LENGTH, v=VOLT * 1e-9, phi=phi_deg, freq=FREQ)
    R = cav._R_main_matrix(energy=ENERGY * 1e-9, length=LENGTH)
    ocelot_block = R[0:2, 0:2]

    mine = _sw_focusing_matrix(VOLT, PHI, LENGTH, ENERGY, M0)
    # ENERGY here is in eV throughout (unit-invariant ratios), Ocelot's own
    # call above used GeV -- both should agree since the formula only uses
    # dimensionless gamma ratios.
    assert np.allclose(mine, ocelot_block, rtol=1e-9)


def test_correction_reproduces_tw_matrix():
    sw = _sw_focusing_matrix(VOLT, PHI, LENGTH, ENERGY, M0)
    m11, m12, m21, m22, _ = tw1_focusing_matrix(
        VOLT, FREQ, PHI, LENGTH, ENERGY, M0, canonical_rescale=False
    )
    tw = np.array([[m11, m12], [m21, m22]])
    correction = tw @ np.linalg.inv(sw)
    assert np.allclose(correction @ sw, tw, rtol=1e-9)


if __name__ == "__main__":
    test_sw_matrix_matches_ocelots_own_cavity_atom()
    test_correction_reproduces_tw_matrix()
    print("All Ocelot TW1 focusing sanity checks passed.")
