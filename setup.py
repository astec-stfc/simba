from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="simba-accelerator",  # must match pyproject.toml
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "deepdiff>=5",
        "h5py>=2.10",
        "munch>=2.5",
        "numpy>=2",
        "tqdm>=4",
        "PyQt5>=5.1",
        "PyYAML>=5.3",
        "mpl-axes-aligner>=1.1",
        "lox>=0.11",
        "fastKDE>=2.1.5",
        "pydantic>=2.5.3",
        "attrs>=23.2.0",
        "ocelot-desy==25.06.0",
        "scipy>=1.5",
        "soliday.sdds",
        "easygdf>=2.1.1",
        "xsuite>=0.39.0",
        "paramiko",
        "deap",
        "pyqtgraph",
        "numba",
        "pyfftw",
        "numexpr",
        "cheetah-accelerator>=0.7.5",
        "openpmd-beamphysics>=0.10",
        "xopt",
    ]
    package_data={
        "simba": [
            "Codes/*.yaml",
            "Codes/Elegant/*.yaml",
            "Codes/Cheetah/*.yaml",
            "Codes/Generators/*.yaml",
            "Codes/CSRTrack/*.yaml",
            "Codes/Ocelot/*.yaml",
            "Codes/Genesis/*.yaml",
            "Codes/OPAL/*.yaml",
            "*.yaml",
        ]
    },
        python_requires=">=3.10",
    long_description=readme,
    long_description_content_type="text/markdown",
)
