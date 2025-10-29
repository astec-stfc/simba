.. _getting-started:

Getting started with SimFrame
=============================

.. _creating-the-lattice-elements:

Creating the Lattice Elements
-----------------------------

The first step in starting a new ``SimFrame`` simulation is to define the lattice, using the :ref:`MasterLattice <masterlattice>`.

The following is part of the ``CLARA`` Injector, in a file  called ``YAML/CLA_Gun400.yaml`` for example:

.. code-block:: yaml

    elements:
        CLA-HRG1-SIM-APER-01:
            centre: [0, 0, 0]
            global_rotation: [0, 0, 0]
            horizontal_size: 0.017
            shape: rectangular
            type: aperture
            vertical_size: 0.017
        CLA-HRG1-GUN-CAV-01:
            Structure_Type: StandingWave
            centre: [0.0, 0.0, 0.16]
            crest: 145.789
            field_amplitude: 120000000.0
            field_definition: $master_lattice_location$Data_Files/HRRG_1D_RF.hdf5
            frequency: 2998500000.0
            global_rotation: [0, 0, 0]
            length: 0.32
            n_cells: 1.5
            phase: 9
            type: cavity
            sub_elements:
                EBT-HRG1-MAG-SOL-01:
                    centre: [0.0, 0.0, 0.16241]
                    field_amplitude: 0.345
                    field_definition: $master_lattice_location$Data_Files/HRRG_combined_sols_100mm_onaxis.hdf5
                    global_rotation: [0, 0, 0]
                    length: 0.32
                    type: solenoid
        CLA-S01-SIM-APER-01:
            centre: [0.0, 0.0, 0.32]
            global_rotation: [0, 0, 0]
            horizontal_size: 0.017
            shape: rectangular
            type: aperture

Note the use of ``sub_elements`` to define elements that overlap an existing element (in this case, a solenoid placed around the gun). 
We make extensive use of `substitutions` to define the locations of field definition files.

Element specific options (such as RF parameters) are also specified.

Defining the Lattice Simulation
-------------------------------

The simulation of the lattice is defined in a separate ``YAML`` file, for example ``CLA-Injector.def``:

.. code-block:: yaml

    generator:
        default: clara_400_3ps
    files:
        injector400:
            code: ASTRA
            charge:
                cathode: True
                space_charge_mode: 2D
                mirror_charge: True
            input:
                particle_definition: 'initial_distribution'
            output:
                zstart: 0
                end_element: CLA-L01-SIM-APER-01
        Linac:
            code: elegant
            starting_rotation: 0
            charge:
                cathode: False
                space_charge_mode: 3D
            input:
                {}
            output:
                start_element: CLA-L01-SIM-APER-01
                end_element: CLA-S02-SIM-APER-01
    elements:
        filename: YAML/CLA_Gun400.yaml

This lattice definition would produce several output files (called ``injector400.in`` and ``Linac.lte``) for running in the **ASTRA** and **Elegant** beam tracking codes.
The elements are loaded from the file ``YAML/CLA_Gun400.yaml`` defined above. Element definitions can also be defined directly in the ``.def`` file.

As this simulation starts from the cathode, the ``input`` definiton is required for the first `injector400` ``file`` block. 

For `follow-on` lattice runs, it is sufficient to define the ``output: start_element``, which should match the ``output: end_element`` definition 
from the previous ``file`` block.


Running SimFrame
----------------

The following example assumes that :ref:`MasterLattice <masterlattice>` has already been installed
(see :ref:`Installation <installation>`) and that the :ref:`SimCodes <simcodes>` directory has
been prepared.

.. code-block:: python

    import SimulationFramework.Framework as fw


    # Define a new framework instance, in directory 'example'.
    #       "clean" will empty (delete everything!) in the directory if true
    #       "verbose" will print a progressbar if true
    simcodes_location = "/path/to/simcodes/directory"
    framework = fw.Framework(
        directory="./example",
        simcodes_location=simcodes_location,
        clean=True,
        verbose=True,
        )
    # Load a lattice definition file. These can be found in Masterlattice/Lattices by default.
    framework.loadSettings("Lattices/clara400_v13.def")
    # Change all lattice codes to ASTRA/Elegant/GPT with exclusions (injector can not be done in Elegant)
    framework.change_Lattice_Code("All", "ASTRA", exclude=["VBC"])
    # Again, but put the VBC in Elegant for CSR
    framework.change_Lattice_Code("VBC", "Elegant")
    # This is the code that generates the laser distribution (ASTRA or GPT)
    framework.change_generator("ASTRA")
    # Load a starting laser distribution setting
    framework.generator.load_defaults("clara_400_2ps_Gaussian")
    # Set the thermal emittance for the generator
    framework.generator.thermal_emittance = 0.0005
    # This is a scaling parameter
    # This defines the number of particles to create at the gun (this is "ASTRA generator" which creates distributions)
    framework.generator.number_of_particles = 2 ** (3 * scaling)
    # Track the lattice
    framework.track()
