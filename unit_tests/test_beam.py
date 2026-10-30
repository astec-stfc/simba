import simba.Modules.Beams as rbf  # noqa E402
from simba.Modules.units import UnitValue
import pytest
import numpy as np
from scipy.constants import m_e, c, e, m_p
from typing import Dict
import os

@pytest.fixture
def simple_beam():
    np.random.seed(42)  # keep the randomly-generated beam reproducible across runs
    beam = rbf.beam()
    particle_mass = UnitValue(m_e, "kg")
    E0 = UnitValue(particle_mass * c ** 2, "J")
    beam.Particles.particle_rest_energy_eV = UnitValue(E0 / e, "eV/c")
    q_over_c = UnitValue(e / c, "C/(m/s)")

    beam_length = 1000
    bunch_charge = 100e-12
    beam.Particles.x = UnitValue(np.random.normal(0, 1e-2, beam_length), "m")
    beam.Particles.y = UnitValue(np.random.normal(0, 1e-2, beam_length), "m")
    beam.Particles.z = UnitValue(np.random.normal(0, 1e-3, beam_length), "m")
    beam.Particles.t = UnitValue(np.random.normal(0, 1e-3 / c, beam_length), "s")
    beam.Particles.px = UnitValue(np.random.normal(0, 1e3 * q_over_c, beam_length), "kg*m/s")
    beam.Particles.py = UnitValue(np.random.normal(0, 1e3 * q_over_c, beam_length), "kg*m/s")
    beam.Particles.pz = UnitValue(np.random.normal(1e9 * q_over_c, 1e3 * q_over_c, beam_length), "kg*m/s")
    beam.Particles.particle_mass = UnitValue(np.full(shape=beam_length, fill_value=m_e, dtype=np.float64), "kg")
    beam.Particles.set_total_charge(bunch_charge)
    beam.Particles.nmacro = UnitValue(np.full(shape=beam_length, fill_value=1, dtype=np.int64), "")
    beam.code = "simframe"
    beam.filename = "test.hdf5"
    beam.set_species("positron")
    return beam

def test_beam_matching(simple_beam):

    simple_beam.Particles.rematchXPlane(beta=10, alpha=-10, nEmit=1e-6)
    simple_beam.Particles.rematchYPlane(beta=5, alpha=10, nEmit=1e-6)

    assert all(
        np.isclose(
            [
                simple_beam.emittance.normalized_horizontal_emittance.val,
                simple_beam.emittance.normalized_vertical_emittance.val,
                simple_beam.twiss.beta_x,
                simple_beam.twiss.beta_y,
                simple_beam.twiss.alpha_x,
                simple_beam.twiss.alpha_y,
                simple_beam.sigmas.sigma_x,
                simple_beam.sigmas.sigma_y,
            ],
            [
                1e-6,
                1e-6,
                10,
                5,
                -10,
                10,
                np.sqrt(simple_beam.twiss.beta_x * simple_beam.emittance.horizontal_emittance),
                np.sqrt(simple_beam.twiss.beta_y * simple_beam.emittance.vertical_emittance)
            ]
        )
    )

    with pytest.warns(UserWarning, match="Both beta and alpha must be provided to rematch"):
        simple_beam.Particles.rematchXPlane(beta=1)
        simple_beam.Particles.rematchYPlane(beta=1)

    simple_beam.Particles.rematchXPlanePeakISlice(beta=10, alpha=-10, nEmit=1e-6)
    simple_beam.Particles.rematchYPlanePeakISlice(beta=5, alpha=10, nEmit=1e-6)
    assert all(
        np.isclose(
            [
                simple_beam.slice.slice_enx[simple_beam.slice.slice_max_peak_current_slice].val,
                simple_beam.slice.slice_beta_x[simple_beam.slice.slice_max_peak_current_slice].val,
                simple_beam.slice.slice_alpha_x[simple_beam.slice.slice_max_peak_current_slice].val,
                simple_beam.slice.slice_eny[simple_beam.slice.slice_max_peak_current_slice].val,
                simple_beam.slice.slice_beta_y[simple_beam.slice.slice_max_peak_current_slice].val,
                simple_beam.slice.slice_alpha_y[simple_beam.slice.slice_max_peak_current_slice].val,
            ],
            [
                1e-6,
                10,
                -10,
                1e-6,
                5,
                10,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    # initial_beam.write_HDF5_beam_file('./fodo/BEGIN.hdf5')
    assert isinstance(simple_beam.mve.slice_6D_Volume, np.ndarray)
    assert isinstance(simple_beam.mve.slice_density, np.ndarray)
    assert isinstance(simple_beam.mve.normalized_mve_horizontal_emittance, UnitValue)
    assert isinstance(simple_beam.mve.normalized_mve_vertical_emittance, UnitValue)
    assert isinstance(simple_beam.mve.horizontal_mve_emittance, UnitValue)
    assert isinstance(simple_beam.mve.vertical_mve_emittance, UnitValue)
    assert isinstance(simple_beam.mve.volume, float)

def test_beam_species(simple_beam):

    assert list(set(simple_beam.Particles.particle_index)) == [2]
    simple_beam.Particles.charge = UnitValue(
        np.full(
            shape=len(simple_beam.x),
            fill_value=-simple_beam.total_charge / len(simple_beam.x),
            dtype=np.float64
        ),
        "C"
    )
    assert list(set(simple_beam.Particles.particle_index)) == [1]
    simple_beam.Particles.particle_mass = UnitValue(
        np.full(
            shape=len(simple_beam.x),
            fill_value=m_p,
            dtype=np.float64
        ),
        "kg"
    )
    assert list(set(simple_beam.Particles.particle_index)) == [4]
    simple_beam.Particles.charge = UnitValue(
        np.full(
            shape=len(simple_beam.x),
            fill_value=simple_beam.total_charge / len(simple_beam.x),
            dtype=np.float64
        ),
        "C"
    )
    assert list(set(simple_beam.Particles.particle_index)) == [3]
    with pytest.raises(ValueError):
        simple_beam.Particles.get_particle_index(1, -1)

def test_model_dump(simple_beam):
    assert isinstance(simple_beam.model_dump(), Dict)

def test_other_derived_properties(simple_beam):
    for prop in [
        "xc",
        "xpc",
        "yc",
        "ypc",
        "deltap",
        "p",
        "Brho",
        "E0_eV",
        "BetaGamma",
        "Ex",
        "Ey",
        "Ez",
        "Bx",
        "By",
        "kinetic_energy",
        "mean_energy",
    ]:
        assert isinstance(getattr(simple_beam, prop), UnitValue)

def test_rotate(simple_beam):
    vals_to_change = [
        "sigma_x",
        "sigma_z",
        "mean_x",
        "mean_z"
    ]
    vals_to_check = [
        "sigma_y",
        "mean_y",
    ]
    initial_change_vals = [getattr(simple_beam, v) for v in vals_to_change]
    initial_check_vals = [getattr(simple_beam, v) for v in vals_to_check]
    simple_beam.write_astra_beam_file('test.astra')
    simple_beam.read_astra_beam_file('test.astra')
    simple_beam.rotate_beamXZ(1)
    rotate_change_vals = [getattr(simple_beam, v) for v in vals_to_change]
    rotate_check_vals = [getattr(simple_beam, v) for v in vals_to_check]
    assert initial_change_vals != rotate_change_vals
    assert initial_check_vals != rotate_check_vals
    assert simple_beam.theta == 1
    simple_beam.unrotate_beamXZ()
    unrotate_change_vals = [getattr(simple_beam, v) for v in vals_to_change]
    unrotate_check_vals = [getattr(simple_beam, v) for v in vals_to_check]
    assert all(np.isclose(initial_change_vals, unrotate_change_vals))
    assert all(np.isclose(initial_check_vals, unrotate_check_vals))
    assert simple_beam.theta == 0.0
    os.remove('test.astra')

def test_centroids(simple_beam):
    q_over_c = UnitValue(e / c, "C/(m/s)")
    simple_beam.Particles.x += 1
    simple_beam.Particles.y += 2
    simple_beam.Particles.z += 3
    simple_beam.Particles.t += 4
    simple_beam.Particles.px += 1e6 * q_over_c
    simple_beam.Particles.py += 2e6 * q_over_c
    simple_beam.Particles.pz += 3e6 * q_over_c

    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_x.val,
                simple_beam.centroids.mean_y.val,
                simple_beam.centroids.mean_z.val,
                simple_beam.centroids.mean_t.val,
            ],
            [
                1,
                2,
                3,
                4,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_cpx.val,
                simple_beam.centroids.mean_cpy.val,
                simple_beam.centroids.mean_cpz.val,
            ],
            [
                1e6,
                2e6,
                1.003e9,
            ],
            rtol=1e-2,
            atol=1e-2,
        )
    )

def test_astra_beam(simple_beam):
    simple_beam.write_astra_beam_file("test.astra")
    astra_beam = rbf.beam("test.astra")
    assert all(
        np.isclose(
            [
                simple_beam.emittance.normalized_horizontal_emittance.val,
                simple_beam.emittance.normalized_vertical_emittance.val,
                simple_beam.twiss.beta_x,
                simple_beam.twiss.beta_y,
                simple_beam.twiss.alpha_x,
                simple_beam.twiss.alpha_y,
                simple_beam.sigmas.sigma_x,
                simple_beam.sigmas.sigma_y,
                simple_beam.sigmas.linear_chirp_z,
                simple_beam.sigmas.momentum_spread,
            ],
            [
                astra_beam.emittance.normalized_horizontal_emittance.val,
                astra_beam.emittance.normalized_vertical_emittance.val,
                astra_beam.twiss.beta_x,
                astra_beam.twiss.beta_y,
                astra_beam.twiss.alpha_x,
                astra_beam.twiss.alpha_y,
                astra_beam.sigmas.sigma_x,
                astra_beam.sigmas.sigma_y,
                astra_beam.sigmas.linear_chirp_z,
                astra_beam.sigmas.momentum_spread,
            ]
        )
    )
    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_x.val,
                simple_beam.centroids.mean_y.val,
                simple_beam.centroids.mean_z.val,
                simple_beam.centroids.mean_t.val,
                simple_beam.centroids.mean_cpx.val,
                simple_beam.centroids.mean_cpy.val,
                simple_beam.centroids.mean_cpz.val,
            ],
            [
                astra_beam.centroids.mean_x.val,
                astra_beam.centroids.mean_y.val,
                astra_beam.centroids.mean_z.val,
                astra_beam.centroids.mean_t.val,
                astra_beam.centroids.mean_cpx.val,
                astra_beam.centroids.mean_cpy.val,
                astra_beam.centroids.mean_cpz.val,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    os.remove("test.astra")

def test_gdf_beam(simple_beam):
    simple_beam.write_gdf_beam_file("test.gdf")
    gdf_beam = rbf.beam("test.gdf")
    assert all(
        np.isclose(
            [
                simple_beam.emittance.normalized_horizontal_emittance.val,
                simple_beam.emittance.normalized_vertical_emittance.val,
                simple_beam.twiss.beta_x,
                simple_beam.twiss.beta_y,
                simple_beam.twiss.alpha_x,
                simple_beam.twiss.alpha_y,
                simple_beam.sigmas.sigma_x,
                simple_beam.sigmas.sigma_y,
                simple_beam.sigmas.linear_chirp_z,
                simple_beam.sigmas.momentum_spread,
            ],
            [
                gdf_beam.emittance.normalized_horizontal_emittance.val,
                gdf_beam.emittance.normalized_vertical_emittance.val,
                gdf_beam.twiss.beta_x,
                gdf_beam.twiss.beta_y,
                gdf_beam.twiss.alpha_x,
                gdf_beam.twiss.alpha_y,
                gdf_beam.sigmas.sigma_x,
                gdf_beam.sigmas.sigma_y,
                gdf_beam.sigmas.linear_chirp_z,
                gdf_beam.sigmas.momentum_spread,
            ]
        )
    )
    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_x.val,
                simple_beam.centroids.mean_y.val,
                simple_beam.centroids.mean_z.val,
                simple_beam.centroids.mean_t.val,
                simple_beam.centroids.mean_cpx.val,
                simple_beam.centroids.mean_cpy.val,
                simple_beam.centroids.mean_cpz.val,
            ],
            [
                gdf_beam.centroids.mean_x.val,
                gdf_beam.centroids.mean_y.val,
                gdf_beam.centroids.mean_z.val,
                gdf_beam.centroids.mean_t.val,
                gdf_beam.centroids.mean_cpx.val,
                gdf_beam.centroids.mean_cpy.val,
                gdf_beam.centroids.mean_cpz.val,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    os.remove("test.gdf")

def test_sdds_beam(simple_beam):
    simple_beam.write_SDDS_beam_file("test.sdds")
    sdds_beam = rbf.beam("test.sdds")
    assert all(
        np.isclose(
            [
                simple_beam.emittance.normalized_horizontal_emittance.val,
                simple_beam.emittance.normalized_vertical_emittance.val,
                simple_beam.twiss.beta_x,
                simple_beam.twiss.beta_y,
                simple_beam.twiss.alpha_x,
                simple_beam.twiss.alpha_y,
                simple_beam.sigmas.sigma_x,
                simple_beam.sigmas.sigma_y,
                simple_beam.sigmas.linear_chirp_z,
                simple_beam.sigmas.momentum_spread,
            ],
            [
                sdds_beam.emittance.normalized_horizontal_emittance.val,
                sdds_beam.emittance.normalized_vertical_emittance.val,
                sdds_beam.twiss.beta_x,
                sdds_beam.twiss.beta_y,
                sdds_beam.twiss.alpha_x,
                sdds_beam.twiss.alpha_y,
                sdds_beam.sigmas.sigma_x,
                sdds_beam.sigmas.sigma_y,
                sdds_beam.sigmas.linear_chirp_z,
                sdds_beam.sigmas.momentum_spread,
            ]
        )
    )
    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_x.val,
                simple_beam.centroids.mean_y.val,
                simple_beam.centroids.mean_z.val,
                simple_beam.centroids.mean_t.val,
                simple_beam.centroids.mean_cpx.val,
                simple_beam.centroids.mean_cpy.val,
                simple_beam.centroids.mean_cpz.val,
            ],
            [
                sdds_beam.centroids.mean_x.val,
                sdds_beam.centroids.mean_y.val,
                sdds_beam.centroids.mean_z.val,
                sdds_beam.centroids.mean_t.val,
                sdds_beam.centroids.mean_cpx.val,
                sdds_beam.centroids.mean_cpy.val,
                sdds_beam.centroids.mean_cpz.val,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    os.remove("test.sdds")

def test_ocelot_beam(simple_beam):
    simple_beam.write_ocelot_beam_file("test.ocelot.npz")
    ocelot_beam = rbf.beam("test.ocelot.npz")
    assert all(
        np.isclose(
            [
                simple_beam.emittance.normalized_horizontal_emittance.val,
                simple_beam.emittance.normalized_vertical_emittance.val,
                simple_beam.twiss.beta_x,
                simple_beam.twiss.beta_y,
                simple_beam.twiss.alpha_x,
                simple_beam.twiss.alpha_y,
                simple_beam.sigmas.sigma_x,
                simple_beam.sigmas.sigma_y,
                simple_beam.sigmas.linear_chirp_z,
                simple_beam.sigmas.momentum_spread,
            ],
            [
                ocelot_beam.emittance.normalized_horizontal_emittance.val,
                ocelot_beam.emittance.normalized_vertical_emittance.val,
                ocelot_beam.twiss.beta_x,
                ocelot_beam.twiss.beta_y,
                ocelot_beam.twiss.alpha_x,
                ocelot_beam.twiss.alpha_y,
                ocelot_beam.sigmas.sigma_x,
                ocelot_beam.sigmas.sigma_y,
                ocelot_beam.sigmas.linear_chirp_z,
                ocelot_beam.sigmas.momentum_spread,
            ]
        )
    )
    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_x.val,
                simple_beam.centroids.mean_y.val,
                simple_beam.centroids.mean_z.val,
                simple_beam.centroids.mean_t.val,
                simple_beam.centroids.mean_cpx.val,
                simple_beam.centroids.mean_cpy.val,
                simple_beam.centroids.mean_cpz.val,
            ],
            [
                ocelot_beam.centroids.mean_x.val,
                ocelot_beam.centroids.mean_y.val,
                ocelot_beam.centroids.mean_z.val,
                ocelot_beam.centroids.mean_t.val,
                ocelot_beam.centroids.mean_cpx.val,
                ocelot_beam.centroids.mean_cpy.val,
                ocelot_beam.centroids.mean_cpz.val,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    os.remove("test.ocelot.npz")

def test_sfhdf_beam(simple_beam):
    simple_beam.write_HDF5_beam_file("test.hdf5")
    sdhdf_beam = rbf.beam("test.hdf5")
    assert all(
        np.isclose(
            [
                simple_beam.emittance.normalized_horizontal_emittance.val,
                simple_beam.emittance.normalized_vertical_emittance.val,
                simple_beam.twiss.beta_x,
                simple_beam.twiss.beta_y,
                simple_beam.twiss.alpha_x,
                simple_beam.twiss.alpha_y,
                simple_beam.sigmas.sigma_x,
                simple_beam.sigmas.sigma_y,
                simple_beam.sigmas.linear_chirp_z,
                simple_beam.sigmas.momentum_spread,
            ],
            [
                sdhdf_beam.emittance.normalized_horizontal_emittance.val,
                sdhdf_beam.emittance.normalized_vertical_emittance.val,
                sdhdf_beam.twiss.beta_x,
                sdhdf_beam.twiss.beta_y,
                sdhdf_beam.twiss.alpha_x,
                sdhdf_beam.twiss.alpha_y,
                sdhdf_beam.sigmas.sigma_x,
                sdhdf_beam.sigmas.sigma_y,
                sdhdf_beam.sigmas.linear_chirp_z,
                sdhdf_beam.sigmas.momentum_spread,
            ]
        )
    )
    assert all(
        np.isclose(
            [
                simple_beam.centroids.mean_x.val,
                simple_beam.centroids.mean_y.val,
                simple_beam.centroids.mean_z.val,
                simple_beam.centroids.mean_t.val,
                simple_beam.centroids.mean_cpx.val,
                simple_beam.centroids.mean_cpy.val,
                simple_beam.centroids.mean_cpz.val,
            ],
            [
                sdhdf_beam.centroids.mean_x.val,
                sdhdf_beam.centroids.mean_y.val,
                sdhdf_beam.centroids.mean_z.val,
                sdhdf_beam.centroids.mean_t.val,
                sdhdf_beam.centroids.mean_cpx.val,
                sdhdf_beam.centroids.mean_cpy.val,
                sdhdf_beam.centroids.mean_cpz.val,
            ],
            rtol=1e-01,
            atol=1e-02,
        )
    )
    os.remove("test.hdf5")

def test_cheetah_beam_chirp_sign(simple_beam):
    """A chirped bunch must survive the cheetah round trip with its chirp intact.

    Every RMS moment is even under z -> -z, so a flipped longitudinal axis is
    invisible in the emittances and beam sizes. It only shows up in the sign of
    the chirp here -- and downstream as an RF chirp that subtracts from the
    incoming one instead of adding to it, once a cavity is tracked.
    """
    from simba.Modules.Beams.cheetah import interpret_cheetah_ParticleBeam

    # impose a known chirp: the later a particle arrives, the higher its momentum
    t = simple_beam.Particles.t.val
    pz = simple_beam.Particles.pz.val
    simple_beam.Particles.pz = UnitValue(pz * (1 + 10.0 * c * (t - t.mean())), "kg*m/s")

    def chirp(b):
        """d(dp/p)/dt -- signed, and immune to the sign convention used for z."""
        tt = b.Particles.t.val - b.Particles.t.val.mean()
        cp = b.Particles.cp.val
        return np.polyfit(tt, (cp - cp.mean()) / cp.mean(), 1)[0]

    chirp_in = chirp(simple_beam)
    assert chirp_in > 0, "test beam has no chirp to preserve"

    parray = simple_beam.write_cheetah_beam_file("test.cheetah.hdf5")
    roundtripped = rbf.beam()
    interpret_cheetah_ParticleBeam(roundtripped, parray)

    assert np.isclose(chirp(roundtripped), chirp_in, rtol=1e-3)

    # and the same again via the file, which is how beam() loads a cheetah dump
    from_file = rbf.beam("test.cheetah.hdf5")
    assert np.isclose(chirp(from_file), chirp_in, rtol=1e-3)
    os.remove("test.cheetah.hdf5")
    # sigma_z is left out: the fixture sets z and t as independent random arrays,
    # so a z reconstructed from t cannot reproduce it
    for moment in ("sigma_x", "sigma_y", "momentum_spread"):
        assert np.isclose(
            getattr(roundtripped.sigmas, moment), getattr(simple_beam.sigmas, moment), rtol=1e-3
        )


def test_resample(simple_beam):
    newlen = 10000
    newbeam = simple_beam.resample(newlen)
    for param in ["x", "y", "z", "px", "py", "pz"]:
        assert len(getattr(newbeam.Particles, param)) == newlen


def test_hdf5_round_trip_keeps_status_and_reference_particle(simple_beam, tmp_path):
    simple_beam.Particles.status = np.full(len(simple_beam.x), -1)
    simple_beam.reference_particle = np.arange(10.0)
    simple_beam.write_HDF5_beam_file(str(tmp_path / "b.hdf5"))
    newbeam = rbf.beam()
    newbeam.read_HDF5_beam_file(str(tmp_path / "b.hdf5"), local=True)
    assert set(np.array(newbeam.Particles.status)) == {-1}
    assert np.allclose(newbeam.reference_particle, np.arange(10.0))


def test_hdf5_cathode_flag_sets_status(simple_beam, tmp_path):
    simple_beam.write_HDF5_beam_file(str(tmp_path / "b.hdf5"), cathode=True)
    newbeam = rbf.beam(str(tmp_path / "b.hdf5"))
    assert set(np.array(newbeam.Particles.status)) == {-1}


def test_apply_mask(simple_beam):
    mask = np.array(simple_beam.x) > 0
    simple_beam.Particles.apply_mask(mask)
    assert len(simple_beam.x) == len(simple_beam.Particles.charge) == mask.sum()
    assert simple_beam.Particles.x.units == "m"


def test_energies(simple_beam):
    cp = np.array(simple_beam.Particles.cp)
    E0 = np.mean(simple_beam.Particles.particle_rest_energy_eV)
    energy = np.sqrt(cp**2 + E0**2)
    assert np.isclose(np.mean(simple_beam.Particles.kinetic_energy) / e, np.mean(energy - E0))
    assert np.isclose(simple_beam.centroids.CEn, np.mean(energy))


def test_slice_cpbins_keeps_time_bins(simple_beam):
    current = np.array(simple_beam.slice.slice_current)
    assert len(simple_beam.slice.slice_cpbins) > 0
    assert np.allclose(current, np.array(simple_beam.slice.slice_current))
    assert simple_beam.slice.sliceAnalysis(density=True)[-1] > 0


def test_unknown_attribute_raises(simple_beam):
    assert not hasattr(simple_beam, "not_a_beam_attribute")
    assert simple_beam.E0_eV is not None


def test_resample_sets_t(simple_beam):
    assert len(simple_beam.resample(100).Particles.t) == 100


def test_plot_single_key(simple_beam):
    import matplotlib
    matplotlib.use("Agg")
    from simba.Modules.Beams import plot
    plot.plot(simple_beam, keys="x")


def test_astra_write_with_index_array(simple_beam, tmp_path):
    simple_beam.filename = str(tmp_path / "run.openpmd.hdf5")
    simple_beam.write_astra_beam_file(index=np.full(len(simple_beam.x), 3))
    assert (tmp_path / "run.astra").exists()


def test_beam_group_add_directory(simple_beam, tmp_path):
    simple_beam.write_HDF5_beam_file(str(tmp_path / "a.hdf5"))
    simple_beam.write_HDF5_beam_file(str(tmp_path / "b.hdf5"))
    assert len(rbf.beamGroup(filenames=str(tmp_path))) == 2


def test_setting_particle_data_on_the_beam_sticks(simple_beam):
    simple_beam.z = UnitValue(np.zeros(len(simple_beam.x)), units="m")
    assert not np.any(simple_beam.z) and not np.any(simple_beam._beam.z)
    gamma = simple_beam.gamma
    with pytest.warns(UserWarning, match="cannot be set"):
        simple_beam.gamma = 0
    assert np.array_equal(simple_beam.gamma, gamma)


@pytest.mark.parametrize("mass", [m_e, m_p])
def test_opal_reader_detects_the_species(tmp_path, mass):
    import h5py

    n, charge = 4, -1e-12
    filename = str(tmp_path / "run_opal.h5")
    with h5py.File(filename, "w") as f:
        step = f.create_group("Step#0")
        # OPAL's TotalMass is labelled MeV but written in GeV
        step.attrs["TotalMass"] = [mass * c**2 / e * 1e-9 * abs(charge) / e]
        step.attrs["TotalCharge"] = [charge]
        for key in ["x", "y", "px", "py", "time"]:
            step[key] = np.zeros(n)
        step["pz"] = np.full(n, 10.0)
    beam = rbf.beam()
    rbf.opal.read_opal_beam_file(beam, filename)
    assert beam._beam.particle_mass.val[0] == pytest.approx(mass, rel=1e-6)
