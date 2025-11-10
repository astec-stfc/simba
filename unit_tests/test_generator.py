from simba.Codes.Generators import (
    frameworkGenerator,
    ASTRAGenerator,
    GPTGenerator,
)
import numpy as np
import pytest
import os

@pytest.fixture
def simple_generator():
    gen = frameworkGenerator(
        global_parameters={
            "master_subdir": f"{os.path.dirname(os.path.abspath(__file__))}"
        },
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
    return gen

def test_generator_write(simple_generator):
    simple_generator.write()
    assert os.path.isfile(f"{os.path.dirname(os.path.abspath(__file__))}/generator.openpmd.hdf5")
    os.remove(f"{os.path.dirname(os.path.abspath(__file__))}/generator.openpmd.hdf5")

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

def test_load_defaults_dict(simple_generator):
    gen = simple_generator
    defaults = {"sigma_x": 2e-4, "sigma_y": 2e-4}
    gen.load_defaults(defaults)
    assert gen.sigma_x == 2e-4
    assert gen.sigma_y == 2e-4

@pytest.fixture
def astra_generator():
    return ASTRAGenerator(
        global_parameters={
            "master_subdir": f"{os.path.dirname(os.path.abspath(__file__))}"
        },
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

def test_astra_generator_write(monkeypatch, astra_generator):
    def mock_save_file(path, content):
        assert path.endswith("test_beam.in")
        assert "&INPUT" in content
        assert "FName = 'test_beam.txt'" in content or "FName = 'test_beam.txt'" in content
    import builtins
    import types
    mock_module = types.SimpleNamespace(saveFile=mock_save_file)
    builtins.simba = types.SimpleNamespace(FrameworkHelperFunctions=mock_module)
    astra_generator.write()

@pytest.fixture
def gpt_generator():
    return GPTGenerator(
        global_parameters={
            "master_subdir": f"{os.path.dirname(os.path.abspath(__file__))}"
        },
        filename="test_gpt.in",
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
        number_of_particles=1000,
        species="electron"
    )

def test_gpt_generator_initialization(gpt_generator):
    assert gpt_generator.code == "gpt"
    assert gpt_generator.filename == "test_gpt.in"
    assert gpt_generator.initial_momentum == 5e6

def test_gpt_generator_write(monkeypatch, gpt_generator):
    def mock_save_file(content):
        assert "beam" in content or "E0" in content
    import builtins
    import types
    mock_module = types.SimpleNamespace(saveFile=mock_save_file)
    builtins.simba = types.SimpleNamespace(FrameworkHelperFunctions=mock_module)
    gpt_generator.write()
    os.remove(f"{os.path.dirname(os.path.abspath(__file__))}/generator.in")