"""
Simframe Codes Module

This module converts the :class:`~simba.Framework_objects.frameworkLattice` class
and its :class:`~simba.Framework_objects.frameworkElement` objects into a format suitable
for the code defined.

The following codes are supported in `SimFrame`:
    - ASTRA

    - GPT

    - ELEGANT

    - Ocelot

A specific :class:`~simba.Codes.Generators.Generators.frameworkGenerator` class
is provided for generating particle distributions for a subset of the supported codes.

Supported codes:
    - :class:`~simba.Codes.ASTRA.ASTRA.astraLattice`

    - :class:`~simba.Codes.GPT.GPT.gptLattice`

    - :class:`~simba.Codes.Elegant.Elegant.elegantLattice`

    - :class:`~simba.Codes.Ocelot.Ocelot.ocelotLattice`
"""