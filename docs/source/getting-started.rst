.. _getting-started:

Getting started with SIMBA
==========================

.. _creating-the-lattice-elements:

Lattice Definition
------------------

Accelerator lattices in :mod:`SIMBA` are derived from the `NALA <https://github.com/astec-stfc/nala/>`_
standard lattice format. This is a schema for providing generic descriptions of accelerator elements and
layouts **TODO add hyperlinks to NALA doc page once it exists**.

Given that this format is designed to capture all relevant information about accelerator elements,
and that it includes a built-in translator module for exporting lattice files to various simulation codes,
it can be used within :mod:`SIMBA` for loading, modifying, writing and exporting input and lattice
files for simulation codes.

Defining the Lattice Simulation
-------------------------------

Given a :mod:`NALA` ``MachineModel``, which contains:

* All of the elements in an accelerator lattice;
* The various sections that compose that lattice;
* A list of layouts composed of lattice sections;

:mod:`SIMBA` can be used to interact with this structure.

The simulation of the lattice is defined in a separate ``YAML`` file, for example ``CLARA.def``
for the CLARA :cite:`PhysRevAccelBeams.23.044801` :cite:`PhysRevAccelBeams.27.041602` accelerator:

.. code-block:: yaml

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

This lattice definition would produce several output files (called ``injector400.in``, ``Linac.lte``,
and ``FEBE.py``) for running in the **ASTRA**, **Elegant** and **Ocelot** beam tracking codes.

The elements are loaded from the directory ``/path/to/nala-lattices/CLARA/YAML/`` defined above.

As this simulation starts from the cathode, the ``input`` definition is required for the first
`injector400` ``file`` block. An alternative method for starting is to specify ``input/particle_definition`` to
point to an existing beam file **#TODO add reference to beams page**.

For `follow-on` lattice runs, it is sufficient to define the ``output: start_element``, which should match the ``output: end_element`` definition 
from the previous ``file`` block.


Running SIMBA
-------------

The following example assumes that `NALA <https://github.com/astec-stfc/nala/>`_ has already been installed
(see :ref:`Installation <installation>`) and that the :ref:`SimCodes <simcodes>` directory has
been prepared.

.. code-block:: python

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
