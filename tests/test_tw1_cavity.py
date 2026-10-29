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


def test_body_focusing_vanishes_exactly_on_crest():
    # dpr ~ cos(phi): at crest (phi=pi/2) the TW body kick is exactly zero,
    # unlike the SW matrix which still focuses on crest. Isolate the body
    # kick by switching off the (independent) end1/end2 focus kicks, which
    # go as sin(phi) and so are instead *maximal* at crest.
    _, _, r21, _, _ = tw1_focusing_matrix(
        VOLT, FREQ, CREST_PHI, LENGTH, ENERGY, M0,
        end1_focus=False, end2_focus=False,
    )
    assert np.isclose(r21, 0.0, atol=1e-12)


def test_end_focus_kicks_are_off_by_default_symmetric_and_max_on_crest():
    # end1/end2 focus go as sin(phi): zero off... no -- maximal exactly on
    # crest (phi=pi/2, sin=1), the opposite of the body kick above.
    m11, m12, m21, m22, _ = tw1_focusing_matrix(
        VOLT, FREQ, CREST_PHI, LENGTH, ENERGY, M0,
        end1_focus=True, end2_focus=True,
    )
    assert not np.isclose(m21, 0.0, atol=1e-9)

    # switching both off exactly reproduces the pre-existing body-only matrix
    body_only = tw1_focusing_matrix(
        VOLT, FREQ, CREST_PHI, LENGTH, ENERGY, M0,
        end1_focus=False, end2_focus=False,
    )
    assert np.isclose(body_only[2], 0.0, atol=1e-12)


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


def test_end_focus_matches_elegant_reference_for_clara_l02_cavity():
    # Regression test for the bug this end-focus support fixes: MAD-X and
    # Ocelot's TW1 matrix (body-only) silently disagreed with ELEGANT's
    # Twiss through a real travelling-wave cavity, because ELEGANT applies
    # END1_FOCUS/END2_FOCUS (on by default) on top of BODY_FOCUS_MODEL=TW1
    # and the shared tw1_focusing_matrix only modelled the body kick.
    #
    # Reference values below are ELEGANT's own numbers for CLARA's
    # CLA-L02-LIN-CAV-01 (volt=87539819.51 V, freq=2998.5 MHz, phase=67 deg,
    # l=4.06667 m, entrance energy 35.174 MeV), obtained two ways that agree
    # with each other: (a) &matrix_output on a standalone single-element
    # rfca with end1_focus=end2_focus=1, giving R = [[0.40588, 2.11364],
    # [-0.08522, 0.30359]] entrance-only and [[1.00123, 2.11364],[0.08614,
    # 0.48482]] exit-only (composed: m11=0.40588, m12=2.11364, m21=-0.05040,
    # m22=0.48487); (b) &twiss_output on the same element with initial
    # (beta_x, alpha_x) = (75.40984, -0.09373681) giving beta_x=250.72 (no
    # end focus) and beta_x=41.6818 (with end focus, matching a full
    # framework run's Twiss_Summary to 8 significant figures).
    volt, freq = 87539819.51089449, 2998500000.0
    phase_deg = 67.0
    phi = np.radians(phase_deg)
    length = 4.06667
    m0 = 0.51099895e6
    p_central = 68.82667045970253
    energy = m0 * np.sqrt(p_central**2 + 1.0)

    m11, m12, m21, m22, _ = tw1_focusing_matrix(
        volt, freq, phi, length, energy, m0, canonical_rescale=False,
    )
    # n_slices=50 (default) vs. ELEGANT's much finer internal slicing
    # (n_kicks=360) -- a few tenths of a percent apart is expected.
    assert np.isclose(m11, 0.405883, rtol=5e-3)
    assert np.isclose(m12, 2.113645, rtol=5e-3)
    assert np.isclose(m21, -0.050403, rtol=2e-2)
    assert np.isclose(m22, 0.484869, rtol=5e-3)


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
    test_body_focusing_vanishes_exactly_on_crest()
    test_end_focus_kicks_are_off_by_default_symmetric_and_max_on_crest()
    test_tw_focusing_is_much_weaker_than_standing_wave()
    test_converged_with_slice_count()
    test_symplectic()
    test_end_focus_matches_elegant_reference_for_clara_l02_cavity()
    test_canonical_rescale_toggle_changes_only_output_row()
    print("All tw1_focusing_matrix sanity checks passed.")
