.. _loading-a-lattice:

Loading in a Lattice File
=========================

:ref:`Getting started <getting-started>` demonstrated how to create a
`LAURA <https://github.com/astec-stfc/laura/>`_ lattice in :mod:`python`.
This page will describe how to generate a :mod:`SIMBA` instance based on pre-existing
`LAURA <https://github.com/astec-stfc/laura/>`_ element and lattice definitions.

.. _setup-from-file:

Setting up a Simulation From Files
----------------------------------

Given a :mod:`LAURA` ``MachineModel``, which contains:

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
    layout: /path/to/laura-lattices/CLARA/layouts.yaml
    section: /path/to/laura-lattices/CLARA/sections.yaml
    element_list: /path/to/laura-lattices/CLARA/YAML/

This lattice definition would produce several output files (called ``injector400.in``, ``Linac.lte``,
and ``FEBE.py``) for running in the **ASTRA**, **Elegant** and **Ocelot** beam tracking codes.

The elements are loaded from the directory ``/path/to/laura-lattices/CLARA/YAML/`` defined above.

As this simulation starts from the cathode, the ``input`` definition is required for the first
`injector400` ``file`` block. An alternative method for starting is to specify ``input/particle_definition`` to
point to an existing beam file.

For `follow-on` lattice runs, it is sufficient to define the ``output: start_element``, which should match the ``output: end_element`` definition 
from the previous ``file`` block.


Running SIMBA
-------------

The following example assumes that `LAURA <https://github.com/astec-stfc/laura/>`_ has already been installed
(see :ref:`Installation <installation>`) and that the :ref:`SimCodes <simcodes>` directory has
been prepared.

.. code-block:: python

    import simba.Framework as fw


    # Define a new framework instance, in directory 'example'.
    #       "clean" will empty (delete everything!) in the directory if true
    #       "verbose" will print a progressbar if true
    simcodes_location = "/path/to/simcodes/directory"
    framework = fw.Framework(
        master_lattice="/path/to/laura-lattices/CLARA",
        directory="./example",
        generator_defaults="clara.yaml",
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
