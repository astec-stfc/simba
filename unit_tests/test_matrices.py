import numpy as np

from simba.Modules.Matrices import matrices


def _identity_matrices(n_elements=2):
    """Build a matrices() instance whose R-matrix at every element is identity."""
    m = matrices()
    for i in range(1, 7):
        for j in range(1, 7):
            val = 1.0 if i == j else 0.0
            m.initialize_array(f"R{i}{j}", [val] * n_elements, units="m")
    return m


def test_init_defaults():
    m = matrices()
    assert "elegant" in m.codes
    assert m.code_signatures == [["elegant", ".mat"]]
    assert set(repr(m).strip("[]").replace("'", "").split(", ")) == {
        "sddsindex",
        "_cumulative",
        "codes",
        "code_signatures",
    }


def test_units_missing_key_returns_none():
    m = matrices()
    assert m.units("R11") is None


def test_units_present_key_returns_units():
    m = matrices()
    m.initialize_array("R11", [1.0, 2.0], units="m")
    assert m.units("R11") == "m"


def test_which_code():
    m = matrices()
    assert m._which_code("elegant") is m.codes["elegant"]
    assert m._which_code("ELEGANT") is m.codes["elegant"]
    assert m._which_code("missing") is None


def test_determine_code():
    m = matrices()
    assert m._determine_code("foo.mat") is m.codes["elegant"]
    assert m._determine_code("foo.txt") is None


def test_initialize_and_append():
    m = matrices()
    m.initialize_array("R11", [1.0, 2.0], units="m")
    assert len(m.R11) == 1
    assert list(m.R11[0]) == [1.0, 2.0]

    m.append("R11", [3.0])
    assert len(m.R11) == 2
    assert list(m.R11[1]) == [3.0]
    assert m.R11[1].units == "m"


def test_generate_r_matrix_identity():
    m = _identity_matrices(n_elements=2)
    R = m.generate_R_matrix(0)
    assert R.shape == (2, 6, 6)
    assert np.allclose(R[0], np.identity(6))
    assert np.allclose(R[1], np.identity(6))


def test_r_property():
    m = _identity_matrices(n_elements=3)
    R = m.R
    assert len(R) == 1
    assert R[0].shape == (3, 6, 6)


def test_cumulative_r_flagged_cumulative():
    m = _identity_matrices(n_elements=2)
    m._cumulative = {0: True}
    cr = m.cumulativeR(combined=False)
    assert len(cr) == 1
    assert cr[0].shape == (2, 6, 6)


def test_cumulative_r_not_cumulative_computes_products():
    m = _identity_matrices(n_elements=2)
    m._cumulative = {0: False}
    cr = m.cumulativeR(combined=False)
    assert len(cr) == 1
    assert np.allclose(cr[0], np.identity(6))


def test_individual_r():
    m = _identity_matrices(n_elements=2)
    m._cumulative = {0: True}
    ir = m.individualR()
    assert len(ir) == 1
    assert len(ir[0]) == 2
    for mat in ir[0]:
        assert np.allclose(mat, np.identity(6))
