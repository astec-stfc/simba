import simba.Modules.Beams as rbf  # noqa E402
from simba.Modules.units import UnitValue
import pytest
import numpy as np
from scipy.constants import m_e, c, e, m_p
from typing import Dict
import os

@pytest.fixture
def simple_beam():
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
    beam.Particles.charge = UnitValue(
        np.full(shape=beam_length, fill_value=bunch_charge / beam_length, dtype=np.float64), "C")
    beam.Particles.particle_mass = UnitValue(np.full(shape=beam_length, fill_value=m_e, dtype=np.float64), "kg")
    beam.Particles.total_charge = UnitValue(bunch_charge, "C")
    beam.Particles.nmacro = UnitValue(np.full(shape=beam_length, fill_value=1, dtype=np.int64), "")
    beam.code = "simframe"
    beam.filename = "test.hdf5"
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

def test_resample(simple_beam):
    newlen = 10000
    newbeam = simple_beam.resample(newlen)
    for param in ["x", "y", "z", "px", "py", "pz"]:
        assert len(getattr(newbeam.Particles, param)) == newlen
