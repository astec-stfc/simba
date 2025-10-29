"""
Simframe Codes Module

This module converts the :class:`~SimulationFramework.Framework_objects.frameworkLattice` class
and its :class:`~SimulationFramework.Framework_objects.frameworkElement` objects into a format suitable
for the code defined.

The following codes are supported in `SimFrame`:
    - ASTRA

    - GPT

    - ELEGANT

    - Ocelot

A specific :class:`~SimulationFramework.Codes.Generators.Generators.frameworkGenerator` class
is provided for generating particle distributions for a subset of the supported codes.

Supported codes:
    - :class:`~SimulationFramework.Codes.ASTRA.ASTRA.astraLattice`

    - :class:`~SimulationFramework.Codes.GPT.GPT.gptLattice`

    - :class:`~SimulationFramework.Codes.Elegant.Elegant.elegantLattice`

    - :class:`~SimulationFramework.Codes.Ocelot.Ocelot.ocelotLattice`
"""