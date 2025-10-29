"""
Simframe Modules

Modules to handle particle beams, electromagnetic fields, matrices, plotting, optimisation and Twiss parameters,
along with various utility functions.

Classes:
    - :class:`~SimulationFramework.Modules.Beams.beam`: Handles particle distributions, including
    various analysis functions and the loading and writing of files to and from various formats.

    - :class:`~SimulationFramework.Modules.Fields.field`: Handles electromagnetic field distributions,
    including the loading and writing of files to and from various formats.

    - :class:`~SimulationFramework.Modules.Matrices.matrices`: Handles particle tracking matrices of
    various orders.

    - :class:`~SimulationFramework.Modules.Twiss.twiss`: Handles beam twiss parameters produced by
    simulations and joins them together.

    - :class:`~SimulationFramework.Modules.optimisation.optimiser.optimiser`: Generic optimiser class.

    - :class:`~SimulationFramework.Modules.units.UnitValue`: Class for storing arrays, floats and integers
    with units attached; used in many of these modules.

Other classes are defined in this submodule, but most of them are for expert use only.
"""