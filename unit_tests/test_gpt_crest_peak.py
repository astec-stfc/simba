"""The GPT crest scan ignores energies no cavity could have given."""
import numpy as np
import pytest

from simba.Codes.GPT.GPT import gptLattice

# LCLS L0A, GPT two-map TW model: (phi [deg], avgG at the exit). At 10 deg the
# bunch stalls and GPT returns gamma 374, beyond the ~170 that L0A could give.
L0A_SCAN = np.array([
    (0, 37.9), (10, 374.1), (50, 92.1), (90, 102.4), (180, 33.2),
    (240, 112.3), (260, 124.2), (270, 125.2), (280, 122.8), (350, 9.4),
])


def test_crest_skips_an_impossible_energy():
    phases, energies = L0A_SCAN.T
    assert gptLattice._crest_peak(phases, energies, gamma_max=170.0) == 270


def test_crest_scan_with_nothing_reachable_is_an_error():
    with pytest.raises(ValueError, match="reachable"):
        gptLattice._crest_peak(np.array([0.0, 10.0]), np.array([300.0, 400.0]), gamma_max=170.0)
