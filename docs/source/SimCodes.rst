.. _simcodes:

SimCodes
=============

.. note::
    | In case of any issues arising during installation or running :mod:`SIMBA`, contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_.
    | :mod:`SimFrame` has been tested with the most recent versions of the code below [2025], and may not be compatible with earlier versions.

:mod:`SimCodes` is a container for the particle accelerator tracking codes used by :mod:`SimFrame`.

While most of the accelerator code executables are open-source, we prefer not to provide these as an installable
package. The user can install the following codes from the links below:

* `ASTRA <https://www.desy.de/~mpyflo/>`_ :cite:`ASTRA`
* `GPT <https://www.pulsar.nl/gpt/>`_ :cite:`GPT`
* `Elegant <https://www.aps.anl.gov/Accelerator-Operations-Physics/Software#elegant>`_ :cite:`Elegant`
* `CSRTrack <https://www.desy.de/xfel-beam/csrtrack/>`_ :cite:`CSRTrack`

Note that the following python-based simulation packages are included in the dependencies of ``SimFrame``:

* `Ocelot <https://github.com/ocelot-collab/ocelot>`_ :cite:`OCELOT`
* `Xsuite <https://github.com/xsuite>`_ :cite:`Xsuite`
* `Cheetah <https://github.com/desy-ml/cheetah>`_ :cite:`Cheetah`
* `Wake-T <https://github.com/AngelFP/Wake-T>`_ :cite:`WakeT`

:mod:`SIMBA` does, however, require these codes to be accessible. This functionality is provided in various ways.

Creating a SimCodes Directory
-----------------------------

One can create a top-level directory containing sub-folders for each tracking code, and instantiate :mod:`SimFrame`
with a ``simcodes_location`` argument:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(
        directory=directory,
        simcodes_location=simcodes_location,
    )

Alternatively, one can set up :mod:`SimFrame` without this argument and set up the ``SimCodes`` location afterwards:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(directory=directory)

    fw.setSimCodesLocation(simcodes_location)

These executables are then accessible to the ``run()`` function of the ``frameworkLattice`` object.

In ``simba/Executables.yaml`` the required structure is provided for this
schema to work for different hardware architectures, either by the OS type or the computer name.

Editing the Executables.yaml file
---------------------------------

If the user already has these executables installed, they can point directly to them in
``simba/Executables.yaml``

Pointing to a specific location
-------------------------------

An instance of :mod:`SIMBA` has access to these executables via the ``executables`` attribute, and these
can be modified once :mod:`SIMBA` is instantiated.

For example, in order to point to a local install of the ``ELEGANT`` code, the user can run the following code:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    elegant_location = "/path/to/elegant/binary"

    fw = Framework(directory=directory)

    fw.executables.define_elegant_command(location=elegant_location)

This will then allow :mod:`SIMBA` to call the correct version of ``ELEGANT``.

Citing the codes used
---------------------

Please consider citing the code(s) used if any work performed with :mod:`SIMBA` leads to a publication:

.. bibliography::
   :style: unsrt
