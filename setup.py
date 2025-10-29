from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "deepdiff>=5",
    "h5py>=2.10",
    "munch>=2.5",
    "numpy>=1.19,<2",
    "tqdm>=4",
    "PyQt5>=5.1",
    "PyYAML>=5.3",
    "mpl-axes-aligner>=1.1",
    "lox>=0.11",
    "fastKDE<2",
    "pydantic>=2.5.3",
    "attrs>=23.2.0",
    "easygdf>=2.1.1",
]

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_data={
        "SimulationFramework": [
            "Codes/*.yaml",
            "Codes/Elegant/*.yaml",
            "Codes/CSRTrack/*.yaml",
            "Codes/Ocelot/*.yaml",
            "Elements/*.yaml",
            "*.yaml",
        ]
    },
)
