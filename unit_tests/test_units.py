import pytest

from simba.Modules.units import UnitValue, unit_fraction


def test_unit_fraction_bracketed_divide():
    assert unit_fraction("kg/(m/s)") == (["kg", "s"], ["m"])
    unit_fraction("(m/s)")


@pytest.mark.parametrize("prefix", ["m", "milli", "milli-"])
def test_in_units_of_scales(prefix):
    assert UnitValue(0.002, units="m").in_units_of(prefix) == pytest.approx(2.0)


def test_math_parser_numbers_and_dunders():
    from simba.Modules.MathParser import MathParser

    p = MathParser({"x": 2.0})
    assert p.parse("x * 3 + sqrt(4)") == 8.0
    with pytest.raises(NameError):
        p.parse("x.__class__")
    with pytest.raises(TypeError):
        p.parse("'a'")


def test_individual_r_of_element_by_element_matrices():
    import numpy as np

    from simba.Modules.Matrices import matrices

    m = matrices()
    R = np.array([2 * np.eye(6), 6 * np.eye(6)])
    for i in range(6):
        for j in range(6):
            m[f"R{i + 1}{j + 1}"] = [R[:, i, j]]
    m._cumulative = {0: False}
    assert np.allclose(m.individualR()[0], R)
    m._cumulative = {0: True}
    assert np.allclose(m.individualR()[0][1], 3 * np.eye(6))


def test_wavefront_group_accepts_a_directory(tmp_path):
    from simba.Modules.Wavefronts import wavefrontGroup

    (tmp_path / "bad.fld.h5").write_text("not a wavefront")
    assert len(wavefrontGroup(filenames=str(tmp_path))) == 0
