"""
Simframe Modules

Modules to handle particle beams, electromagnetic fields, matrices, plotting, optimisation and Twiss parameters,
along with various utility functions.

Classes:
    - :class:`~simba.Modules.Beams.Beam`: Handles particle distributions, including
    various analysis functions and the loading and writing of files to and from various formats.

    - :class:`~simba.Modules.Fields.FieldMap`: Handles electromagnetic field distributions,
    including the loading and writing of files to and from various formats.

    - :class:`~simba.Modules.Matrices.Matrices`: Handles particle tracking matrices of
    various orders.

    - :class:`~simba.Modules.Twiss.Twiss`: Handles beam twiss parameters produced by
    simulations and joins them together.

    - :class:`~simba.Modules.optimisation.optimiser.Optimiser`: Generic optimiser class.

    - :class:`~simba.Modules.units.UnitValue`: Class for storing arrays, floats and integers
    with units attached; used in many of these modules.

Other classes are defined in this submodule, but most of them are for expert use only.
"""