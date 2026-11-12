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


@pytest.mark.parametrize("save", ["madx", "ocelot"])
def test_the_twiss_writers_do_not_drop_what_they_cannot_store(save, tmp_path):
    """An entry h5py could not store was skipped without a word."""
    from simba.Modules.Twiss.madx import save_madx_twiss_hdf
    writer = save_madx_twiss_hdf if save == "madx" else ocelot.save_ocelot_twiss_hdf
    filename = str(tmp_path / "twiss.hdf5")
    writer(None, filename, {"s": np.arange(3.0), "element_name": np.array([b"A", b"B", b"C"])})
    with h5py.File(filename, "r") as f:
        assert sorted(f) == ["element_name", "s"]
    with pytest.raises(TypeError):
        writer(None, filename, {"s": np.arange(3.0), "element_name": np.array(["A", "B", "C"])})


def test_astra_reader_momentum_spread_and_alpha(tmp_path):
    """One row at 35 MeV with alpha = -0.7 in both planes and 160 keV rms energy spread."""
    t = twiss()
    e_kin, rms_e, gamma = 35.0, 160.0, 1 + 35.0 / (t.E0_eV / 1e6)
    sx, enx = 0.5, 0.8  # mm, pi mm mrad
    ex = enx / gamma
    beta, alpha = sx**2 / ex, -0.7
    sxp = np.sqrt((1 + alpha**2) / beta * ex)
    col7 = -alpha * ex / sx  # <x x'>/sigma_x [mrad]
    row = [1.0, 1.0, 0.0, sx, sxp, enx, col7]
    for plane in "XY":
        np.savetxt(tmp_path / f"run.{plane}emit.001", [row])
    np.savetxt(tmp_path / "run.Zemit.001", [[1.0, 1.0, e_kin, 1.0, rms_e, 1.0, 0.0]])
    t.read_astra_twiss_files(str(tmp_path / "run.Zemit.001"))
    b = np.sqrt(1 - gamma**-2)
    for key in ["alpha_x", "alpha_y", "alpha_x_beam", "alpha_y_beam"]:
        assert getattr(t, key).val[0] == pytest.approx(alpha)
    assert t.sigma_cp.val[0] == pytest.approx(1e3 * rms_e / b)
    assert t.sigma_p.val[0] == pytest.approx(t.sigma_cp.val[0] / t.cp.val[0])


def test_astra_reader_says_when_astra_wrote_nothing(tmp_path):
    for plane in "XYZ":
        (tmp_path / f"run.{plane}emit.001").touch()
    with pytest.raises(ValueError, match="ASTRA wrote no Xemit statistics for run"):
        twiss().read_astra_twiss_files(str(tmp_path / "run.Zemit.001"))
