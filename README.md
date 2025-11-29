# SIMBA: Simulation Interface for Machine and Beam Applications or
*Improving accelerators through consistent, complete, and easy to use simulations.*

## Mission
`SIMBA` is a framework for performing simulations of particle accelerators that aims to be simple to use, and complete.

Our mission statement is *"To create a start to end framework for consistent, transparent simulations of particle 
accelerators and FELs that anybody can use, and that everybody trusts."*

By leveraging a [standard accelerator lattice format](https://github.com/astec-stfc/nala.git), `SIMBA` 
is able to generate and run input files for a range of accelerator simulation codes, enabling seamless transfer 
of input and output distributions. 

The codes currently supported by ``SIMBA`` are:

* [ASTRA](https://www.desy.de/~mpyflo/)
* [GPT](https://pulsar.nl/)
* [ELEGANT](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software)
* [Ocelot](https://github.com/ocelot-collab/ocelot)
* [CSRTrack](https://www.desy.de/xfel-beam/csrtrack/)
* [Cheetah](https://www.github.com/desy-ml/cheetah/)
* [Wake-T](https://www.github.com/AngelFP/Wake-T/)
* [Xsuite](https://www.github.com/xsuite/)

A range of other codes are also currently under active development. 

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/simba-documentation-blue.svg)](https://simba-accelerator.readthedocs.io/)  |

## Installation

Cloning from Github
-------------------

Clone `SIMBA` from Github:

```bash
git clone https://github.com/astec-stfc/simba.git
```

The package and its dependencies can be installed using the following command in the ``SIMBA`` directory:

```bash
pip install .
```

Install from PyPI
-----------------

Alternatively, `SIMBA` can be installed directly from PyPI using:

```bash
pip install simba-accelerator
```
    
Optional Dependencies
---------------------

In order to access the accelerator code executables, they must also inform ``SIMBA`` of their
locations. See the entry on [SimCodes](https://simba-accelerator.readthedocs.io/en/latest/SimCodes.html).

Participation
-------------

We welcome contributions and suggestions from the community! ``SIMBA`` is 
currently under active development, and as such certain features may be missing 
or not working as expected. If you find any issues, please raise it 
[here](https://github.com/astec-stfc/simba/issues) or contact 
[Alex Brynes](mailto:alexander.brynes@stfc.ac.uk).

We are also happy to help with installation and setting up your accelerator lattice. 

## Example Lattice and Simulation

Getting started with SIMBA
-----------------------------

Accelerator lattices in `SIMBA` are derived from the [NALA](https://github.com/astec-stfc/nala/)
standard lattice format. This is a schema for providing generic descriptions of accelerator elements and
layouts; see [NALA documentation](https://nala-accelerator.readthedocs.io/en/latest/).

Given that this format is designed to capture all relevant information about accelerator elements,
and that it includes a built-in translator module for exporting lattice files to various simulation codes,
it can be used within :mod:`SIMBA` for loading, modifying, writing and exporting input and lattice
files for simulation codes.

Defining the Lattice Simulation
-------------------------------

The simulation of the lattice is defined in a separate ``YAML`` file, for example ``CLA-Injector.def``:

```yaml
generator:
    default: clara_400_3ps
files:
  injector400:
    code: astra
    charge:
      cathode: True
      space_charge_mode: 2D
      mirror_charge: True
    input:
      particle_definition: 'initial_distribution'
    output:
      zstart: 0
      end_element: CLA-S02-SIM-APER-01
  Linac:
    code: elegant
    output:
      start_element: CLA-S02-SIM-APER-01
      end_element: CLA-FEA-SIM-START-01
  FEBE:
    code: ocelot
    charge:
      cathode: False
      space_charge_mode: 3D
    input: {}
    output:
      start_element: CLA-FEA-SIM-START-01
      end_element: CLA-FED-SIM-DUMP-01-START
groups:
  bunch_compressor:
    type: chicane
    elements: [CLA-VBC-MAG-DIP-01, CLA-VBC-MAG-DIP-02, CLA-VBC-MAG-DIP-03, CLA-VBC-MAG-DIP-04]
layout: /path/to/nala-lattices/CLARA/layouts.yaml
section: /path/to/nala-lattices/CLARA/sections.yaml
element_list: /path/to/nala-lattices/CLARA/YAML/

```

This lattice definition would produce several output files (called ``injector400.in``, ``Linac.lte``,
and ``FEBE.py``) for running in the **ASTRA**, **Elegant** and **Ocelot** beam tracking codes.

The elements are loaded from the directory ``/path/to/nala-lattices/CLARA/YAML/`` defined above.

As this simulation starts from the cathode, the ``input`` definition is required for the first
`injector400` ``file`` block. An alternative method for starting is to specify ``input/particle_definition`` to
point to an existing 
[beam file](https://simba-accelerator.readthedocs.io/en/latest/examples/notebooks/beams_example.html).

For `follow-on` lattice runs, it is sufficient to define the ``output: start_element``, which should match the 
``output: end_element`` definition from the previous ``file`` block.


Running SIMBA
----------------

```python
import simba.Framework as fw


# Define a new framework instance, in directory 'example'.
#       "clean" will empty (delete everything!) in the directory if true
#       "verbose" will print a progressbar if true
simcodes_location = "/path/to/simcodes/directory"
framework = fw.Framework(
    master_lattice="/path/to/nala-lattices/CLARA",
    directory="./example",
    generator_defaults="/path/to/nala-lattices/CLARA/Generators/clara.yaml",
    simcodes_location=simcodes_location,
    clean=True,
    verbose=True,
    )
# Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
framework.loadSettings("Lattices/CLARA.def")
# Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector cannot be done in Elegant)
framework.change_Lattice_Code("All", "ASTRA", exclude=["Linac"])
# Again, but put the VBC in Elegant for CSR
framework.change_Lattice_Code("FEBE", "Elegant")
# This is the code that generates the laser distribution (ASTRA or GPT)
framework.change_generator("ASTRA")
# Load a starting laser distribution setting
framework.generator.load_defaults("clara_400_2ps_Gaussian")
# Set the thermal emittance for the generator
framework.generator.thermal_emittance = 0.0005
# This is a scaling parameter
# This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
framework.generator.number_of_particles = 512
# Track the lattice
framework.track()
```

Example Notebooks
-----------------

Some further examples on ``SIMBA`` usage can be found in the following notebooks:

* [getting_started.ipynb](./examples/notebooks/getting_started.ipynb)
* [beams_example.ipynb](./examples/notebooks/beams_example.ipynb)
* [utility_functions.ipynb](./examples/notebooks/utility_functions.ipynb)

Authors
-------

* [James Jones](mailto:james.jones@stfc.ac.uk)
* [Alex Brynes](mailto:alexander.brynes@stfc.ac.uk)
* [Mark Johnson](mailto:mark.johnson@stfc.ac.uk)
