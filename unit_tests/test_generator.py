from simba.Codes.Generators import (
    frameworkGenerator,
    ASTRAGenerator,
    GPTGenerator,
)
import numpy as np
import pytest

@pytest.fixture
def simple_generator(tmp_path):
    return frameworkGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
        filename="generator.openpmd.hdf5",
        initial_momentum=5e6,
        sigma_x=1e-4,
        sigma_px=1e3,
        sigma_y=1e-4,
        sigma_py=1e3,
        sigma_z=1e-3,
        sigma_pz=1e3,
        gaussian_cutoff_x=3,
        gaussian_cutoff_y=3,
        gaussian_cutoff_z=3,
        gaussian_cutoff_px=3,
        gaussian_cutoff_py=3,
        gaussian_cutoff_pz=3,
        charge=100e-12,
    )

def test_generator_write(simple_generator, tmp_path):
    simple_generator.write()
    assert (tmp_path / "generator.openpmd.hdf5").is_file()

def test_particles_property(simple_generator):
    gen = simple_generator
    assert gen.particles == 512
    gen.particles = 1000
    assert gen.particles == 1000

def test_thermal_kinetic_energy(simple_generator):
    gen = simple_generator
    energy = gen.thermal_kinetic_energy
    assert isinstance(energy, float)
    assert energy > 0

def test_generate_transverse_distribution(simple_generator):
    gen = simple_generator
    samples = gen.generate_transverse_distribution("x")
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (gen.particles, 2)

def test_generate_longitudinal_distribution(simple_generator):
    gen = simple_generator
    samples = gen.generate_longitudinal_distribution()
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (gen.particles, 2)

def test_t_is_the_same_bunch_as_z(simple_generator):
    """t was ``abs(-z / c)``: the bunch folded in half, an rms 0.60 of
    sigma_z and every particle behind the reference, for any code that
    tracks t (Xsuite's space charge saw 1.65 times the peak current)."""
    from simba.Modules import constants

    beam = simple_generator.generate()
    z, t = beam.z.val, beam.t.val
    np.testing.assert_allclose(
        t, -z / (beam.Bz.val * constants.speed_of_light), rtol=1e-12
    )
    assert np.std(t) * constants.speed_of_light == pytest.approx(np.std(z), rel=0.01)
    assert np.any(t > 0) and np.any(t < 0)


def test_load_defaults_dict(simple_generator):
    gen = simple_generator
    defaults = {"sigma_x": 2e-4, "sigma_y": 2e-4}
    gen.load_defaults(defaults)
    assert gen.sigma_x == 2e-4
    assert gen.sigma_y == 2e-4

@pytest.fixture
def astra_generator(tmp_path):
    return ASTRAGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
        filename="test_beam.txt",
        initial_momentum=5e6,
        sigma_x=1e-4,
        sigma_px=1e3,
        sigma_y=1e-4,
        sigma_py=1e3,
        sigma_z=1e-3,
        sigma_pz=1e3,
        gaussian_cutoff_x=3,
        gaussian_cutoff_y=3,
        gaussian_cutoff_z=3,
        gaussian_cutoff_px=3,
        gaussian_cutoff_py=3,
        gaussian_cutoff_pz=3,
        charge=100e-12,
        number_of_particles=1000
    )

def test_astra_generator_initialization(astra_generator):
    assert isinstance(astra_generator, ASTRAGenerator)
    assert astra_generator.code == "ASTRA"
    assert astra_generator.filename == "test_beam.txt"

def test_astra_generator_alias_application(astra_generator):
    assert hasattr(astra_generator, "FName")
    assert astra_generator.FName == "test_beam.txt"
    assert hasattr(astra_generator, "sig_x")
    assert astra_generator.sig_x == pytest.approx(1e-4 * 1000)

def test_astra_generator_write(astra_generator, tmp_path):
    astra_generator.write()
    content = (tmp_path / f"{astra_generator.objectname}.in").read_text()
    assert content.startswith("&INPUT")
    assert "FName = 'test_beam.txt'" in content

@pytest.fixture
def gpt_generator(tmp_path):
    return GPTGenerator(
        global_parameters={"master_subdir": str(tmp_path)},
        filename="test_gpt.in",
        initial_momentum=0e6,
        sigma_x=1e-4,
        sigma_px=1e3,
        sigma_y=1e-4,
        sigma_py=1e3,
        sigma_z=1e-3,
        sigma_pz=1e3,
        gaussian_cutoff_x=3,
        gaussian_cutoff_y=3,
        gaussian_cutoff_z=3,
        gaussian_cutoff_px=3,
        gaussian_cutoff_py=3,
        gaussian_cutoff_pz=3,
        charge=100e-12,
        number_of_particles=1000,
        species="electron",
        cathode=True
    )

def test_gpt_generator_initialization(gpt_generator):
    assert gpt_generator.code == "gpt"
    assert gpt_generator.filename == "test_gpt.in"
    assert gpt_generator.initial_momentum == 0e6

def test_gpt_generator_write(gpt_generator, tmp_path):
    gpt_generator.write()
    content = (tmp_path / f"{gpt_generator.objectname}.in").read_text()
    assert "E0" in content
    gpt_generator.cathode = False
    with pytest.raises(NotImplementedError):
        gpt_generator.write()

@pytest.mark.parametrize("species, sign", [("proton", 1), ("positrons", 1), ("electron", -1)])
def test_species_sets_instance_mass_and_sign(species, sign):
    from simba.Modules import constants
    gen = frameworkGenerator(species=species)
    assert gen.charge_sign == sign
    assert gen.particle_mass == (constants.m_p if species == "proton" else constants.m_e)
    assert frameworkGenerator().charge_sign == -1


def test_sample_gaussian_cutoff_follows_offset():
    from simba.Codes.Generators.Generators import sample_gaussian
    z = sample_gaussian(10.0, 1.0, 3, 2000)
    assert np.all(np.abs(z - 10.0) <= 3) and z.min() < 9


def test_gpt_gaussian_tlen_from_sigma_z():
    from simba.Modules.constants import speed_of_light
    gen = GPTGenerator(distribution_type_z="g", sigma_z=3e-4, sigma_t=0.0)
    assert f"tlen = {1e12 * 3e-4 / speed_of_light}e-12" in gen.generate_longitudinal_distribution()
    gen.distribution_type_z = "f"
