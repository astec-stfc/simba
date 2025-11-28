.. SIMBA documentation master file, created by
   sphinx-quickstart on Tue Sep 24 10:00:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SIMBA - Simulations for Integrated Modeling of Beams in Accelerators
====================================================================

**SIMBA** is a ``python`` package for performing start-to-end (S2E) simulations of linear particle accelerators.

It provides a wrapper for several well-known particle tracking codes:

* `ASTRA <https://www.desy.de/~mpyflo/>`_ :cite:`ASTRA`
* `GPT <https://www.pulsar.nl/gpt/>`_ :cite:`GPT`
* `Elegant <https://www.aps.anl.gov/Accelerator-Operations-Physics/Software#elegant>`_ :cite:`Elegant`
* `CSRTrack <https://www.desy.de/xfel-beam/csrtrack/>`_ :cite:`CSRTrack`
* `Ocelot <https://github.com/ocelot-collab/ocelot>`_ :cite:`OCELOT`
* `Cheetah <https://github.com/desy-ml/cheetah>`_ :cite:`Cheetah`
* `Xsuite <https://github.com/xsuite>`_ :cite:`Xsuite`
* `Wake-T <https://github.com/AngelFP/Wake-T>`_ :cite:`WakeT`

Setup
-----
.. warning::
   | This site is currently **under construction**.
   | Some pages may have missing or incomplete reference documentation.

.. toctree::
   :maxdepth: 2
   
   installation
   getting-started
   loading-a-lattice
   SimCodes
   

.. Examples
   --------

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/notebooks/getting_started
   examples/notebooks/beams_example
   examples/notebooks/utility_functions
   
Participation
-------------

We welcome contributions and suggestions from the community! :mod:`SIMBA` is currently under active development,
and as such certain features may be missing or not working as expected. If you find any issues, please
raise it `here <https://github.com/astec-stfc/simba/issues>`_.

We are also happy to help with installation and setting up your accelerator lattice. 
   
.. API
   ---

.. toctree::
   :maxdepth: 2
   :caption: API
   
   Framework_objects
   Framework_elements
   simba.Codes
   simba.Modules
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

.. bibliography::
   :style: unsrt
