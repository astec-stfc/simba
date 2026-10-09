import h5py
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from simba.Modules.Twiss import astra, cheetah, code_signatures, ocelot, opal, twiss


def test_every_signature_has_a_reader():
    t = twiss()
    for code, _ in code_signatures:
        assert code in t.codes
    assert t._determine_code("x_emit.gdf") is t.codes["gpt"]
    assert t._determine_code("x_twiss.oh5") is t.codes["ocelot"]


def test_longitudinal_emittance_labels():
    t = twiss()
    assert (t.ez.name, t.enz.name) == ("ez", "enz")


def test_append_and_lookup_at_z():
    t = twiss()
    t.append("z", [0.0, 1.0, 2.0])
    t.append("beta_x", [1.0, 2.0, 3.0])
    assert list(t.extract_values("beta_x", 0.0, 1.0)) == [1.0, 2.0]
    assert t.get_parameter_at_z("beta_x", 1.0) == 2.0
    assert t.get_parameter_at_z("beta_x", 1.0005) == 2.0
    assert t.get_parameter_at_z("beta_x", 1.5) == pytest.approx(2.5)
    assert t.interpolate(z=3.0) == 10**6


def test_lookup_at_element_takes_the_first_row():
    t = twiss()
    t.append("z", [0.0, 1.0, 2.0, 3.0])
    t.append("element_name", ["D", "A", "B", "A"])
    assert t.get_parameter_at_element("z", "A") == 1.0
    assert t.get_twiss_at_element("A", before=True)["z"] == 0.0
    assert t.get_parameter_at_element("z", "Q") is None
    assert t.get_twiss_at_element("Q") is None


@pytest.mark.parametrize("module", [astra, cheetah, ocelot, opal])
def test_cumtrapz_is_the_cumulative_integral(module):
    x = np.linspace(0, 2, 7)
    y = x**2
    assert np.allclose(module.cumtrapz(x=x, y=y), cumulative_trapezoid(y, x, initial=0))


def _write_cheetah(path, n, energy):
    s = np.linspace(0, 1, n)
    ones = np.ones(n)
    with h5py.File(path, "w") as f:
        g = f.create_group("Twiss")
        g["s"] = s
        g["energy"] = energy * ones
        for k in ["emittance_x", "emittance_y"]:
            g[k] = 1e-9 * ones
        for k in ["beta_x", "beta_y", "sigma_px", "sigma_py", "sigma_tau", "sigma_p"]:
            g[k] = ones
        for k in ["alpha_x", "alpha_y", "mu_x", "mu_y"]:
            g[k] = 0 * ones
        for k in ["sigma_x", "sigma_y"]:
            g[k] = np.sqrt(1e-9) * ones


def test_cheetah_reader_with_several_files(tmp_path):
    files = [str(tmp_path / f"{n}_twiss.cheetah.hdf5") for n in ("a", "b")]
    _write_cheetah(files[0], 3, 10e6)
    _write_cheetah(files[1], 4, 10e6)
    t = twiss()
    cheetah.read_cheetah_twiss_files(t, files)
    assert len(t.mux.val) == len(t.z.val) == 7
    assert t.mux.val[-1] == pytest.approx(1.0)  # beta = 1 m over 1 m
    assert t.kinetic_energy.val[0] == pytest.approx(10e6 - t.E0_eV)
    assert t.cp.val[0] == pytest.approx(np.sqrt(10e6**2 - t.E0_eV**2))
