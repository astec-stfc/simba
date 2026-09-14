"""
Sanity checks for :func:`simba.Modules.rf_focusing.tw1_focusing_matrix`, the
travelling-wave RF focusing model (matching ELEGANT's BODY_FOCUS_MODEL=TW1),
shared by the MAD-X and Ocelot backends.

Plain asserts, no framework: run directly with `python tests/test_tw1_cavity.py`.
"""

import numpy as np

from simba.Modules.rf_focusing import tw1_focusing_matrix, TW1_N_SLICES

M0 = 0.51099895e6  # electron rest energy [eV]
VOLT = 20e6  # eV
FREQ = 2998.5e6  # Hz
CREST_PHI = np.pi / 2  # de = volt*sin(phi), crest at phi = pi/2
LENGTH = 2.0  # m
ENERGY = 10e6  # eV (entrance)


def test_zero_voltage_is_a_drift():
    m11, m12, m21, m22, e_out = tw1_focusing_matrix(0.0, FREQ, CREST_PHI, LENGTH, ENERGY, M0)
    assert np.isclose(m11, 1.0) and np.isclose(m22, 1.0)
    assert np.isclose(m12, LENGTH)
    assert np.isclose(m21, 0.0, atol=1e-12)
    assert np.isclose(e_out, ENERGY)


def test_energy_gain_matches_crest_voltage():
    _, _, _, _, e_out = tw1_focusing_matrix(VOLT, FREQ, CREST_PHI, LENGTH, ENERGY, M0)
    assert np.isclose(e_out, ENERGY + VOLT, rtol=1e-9)


def test_focusing_vanishes_exactly_on_crest():
    # dpr ~ cos(phi): at crest (phi=pi/2) the TW body kick is exactly zero,
    # unlike the SW matrix which still focuses on crest.
    _, _, r21, _, _ = tw1_focusing_matrix(VOLT, FREQ, CREST_PHI, LENGTH, ENERGY, M0)
    assert np.isclose(r21, 0.0, atol=1e-12)


def test_tw_focusing_is_much_weaker_than_standing_wave():
    # Off-crest, where both the TW (~cos phi) and SW kicks are nonzero.
    phi = 2 * np.pi * 0.15
    _, _, r21_tw, _, _ = tw1_focusing_matrix(VOLT, FREQ, phi, LENGTH, ENERGY, M0)

    cos_phi = np.sin(phi)  # MAD-X rs_matrix_cavity's own "cos_phi" convention
    de = VOLT * cos_phi
    Ei, Ef = ENERGY / M0, (ENERGY + de) / M0
    Ep = (Ef - Ei) / LENGTH
    alpha = np.sqrt(1.0 / 8.0) / cos_phi * np.log(Ef / Ei)
    r21_sw = -Ep / Ef * (cos_phi / np.sqrt(2.0) + np.sqrt(1.0 / 8.0) / cos_phi) * np.sin(alpha)
    r21_sw *= np.sqrt(Ef**2 - 1) / np.sqrt(Ei**2 - 1)  # MAD-X canonical rescale

    assert abs(r21_tw) < abs(r21_sw)
    print(f"|r21| travelling-wave = {abs(r21_tw):.3e}, standing-wave = {abs(r21_sw):.3e}")


def test_converged_with_slice_count():
    # The drift-kick-drift integration is first-order (error ~ 1/n_slices),
    # so check the error shrinks roughly as expected rather than requiring
    # tight agreement at a coarse resolution.
    phi = 2 * np.pi * 0.15
    reference = np.array(tw1_focusing_matrix(VOLT, FREQ, phi, LENGTH, ENERGY, M0, n_slices=4000))
    coarse = np.array(tw1_focusing_matrix(VOLT, FREQ, phi, LENGTH, ENERGY, M0, n_slices=TW1_N_SLICES))
    fine = np.array(tw1_focusing_matrix(VOLT, FREQ, phi, LENGTH, ENERGY, M0, n_slices=200))
    err_coarse = np.abs(coarse - reference)
    err_fine = np.abs(fine - reference)
    assert np.all(err_fine <= err_coarse + 1e-15)
    assert np.isclose(coarse[2], reference[2], rtol=0.05)  # r21 within 5% at the default slicing


def test_symplectic():
    # Determinant of the transverse block: 1 exactly for canonical_rescale
    # (MAD-X's fixed-reference-momentum convention), Pi/Pf for the raw
    # (Ocelot) convention -- dropping the adiabatic-damping term is exactly
    # the bug that broke this and let beta functions blow up unphysically.
    phi = 2 * np.pi * 0.15
    m11, m12, m21, m22, e_out = tw1_focusing_matrix(VOLT, FREQ, phi, LENGTH, ENERGY, M0)
    assert np.isclose(m11 * m22 - m12 * m21, 1.0, rtol=1e-9)

    m11, m12, m21, m22, e_out = tw1_focusing_matrix(
        VOLT, FREQ, phi, LENGTH, ENERGY, M0, canonical_rescale=False
    )
    Pi = np.sqrt((ENERGY / M0) ** 2 - 1.0)
    Pf = np.sqrt((e_out / M0) ** 2 - 1.0)
    assert np.isclose(m11 * m22 - m12 * m21, Pi / Pf, rtol=1e-9)


def test_canonical_rescale_toggle_changes_only_output_row():
    phi = 2 * np.pi * 0.15
    m11_c, m12_c, m21_c, m22_c, e_c = tw1_focusing_matrix(
        VOLT, FREQ, phi, LENGTH, ENERGY, M0, canonical_rescale=True
    )
    m11_l, m12_l, m21_l, m22_l, e_l = tw1_focusing_matrix(
        VOLT, FREQ, phi, LENGTH, ENERGY, M0, canonical_rescale=False
    )
    # input-row (position) terms and the exit energy don't depend on the
    # output normalisation convention
    assert np.isclose(m11_c, m11_l) and np.isclose(m12_c, m12_l)
    assert np.isclose(e_c, e_l)
    # the output-row (slope) terms do, since the cavity is accelerating
    assert not np.isclose(m21_c, m21_l)
    assert not np.isclose(m22_c, m22_l)


if __name__ == "__main__":
    test_zero_voltage_is_a_drift()
    test_energy_gain_matches_crest_voltage()
    test_focusing_vanishes_exactly_on_crest()
    test_tw_focusing_is_much_weaker_than_standing_wave()
    test_converged_with_slice_count()
    test_symplectic()
    test_canonical_rescale_toggle_changes_only_output_row()
    print("All tw1_focusing_matrix sanity checks passed.")
