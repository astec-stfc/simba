.. _simcodes:

SimCodes
=============

.. note::
    | In case of any issues arising during installation or running :mod:`SIMBA`, contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_.
    | :mod:`SIMBA` has been tested with the most recent versions of the codes mentioned below [2025], and may not be compatible with earlier versions.

:mod:`SimCodes` is a container for the particle accelerator tracking codes used by :mod:`SIMBA`.

While most of the accelerator code executables are open-source, we prefer not to provide these as an installable
package. The user can install the following codes from the links below:

* `ASTRA <https://www.desy.de/~mpyflo/>`_ :cite:`ASTRA`
* `GPT <https://www.pulsar.nl/gpt/>`_ :cite:`GPT`
* `Elegant <https://www.aps.anl.gov/Accelerator-Operations-Physics/Software#elegant>`_ :cite:`Elegant`
* `CSRTrack <https://www.desy.de/xfel-beam/csrtrack/>`_ :cite:`CSRTrack`
* `OPAL <https://amas.web.psi.ch/opal/Documentation/master/OPAL_Manual.html>`_
* `Genesis <https://github.com/svenreiche/Genesis-1.3-Version4>`_

Note that the following python-based simulation packages are **optional** dependencies of ``SIMBA``,
installable via the ``simcodes`` extra (``pip install simba-accelerator[simcodes]``). They run in-process
and are unaffected by ``container_runtime`` (see below), so they must be installed locally regardless of
whether a container runtime is used for the other codes:

* `Ocelot <https://github.com/ocelot-collab/ocelot>`_ :cite:`OCELOT`
* `Xsuite <https://github.com/xsuite>`_ :cite:`Xsuite`
* `Cheetah <https://github.com/desy-ml/cheetah>`_ :cite:`Cheetah`
* `Wake-T <https://github.com/AngelFP/Wake-T>`_ :cite:`WakeT`

:mod:`SIMBA` does, however, require these codes to be accessible. This functionality is provided in various ways.

Creating a SimCodes Directory
-----------------------------

One can create a top-level directory containing sub-folders for each tracking code, and instantiate :mod:`SIMBA`
with a ``simcodes`` argument:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(
        directory=directory,
        simcodes=simcodes_location,
    )

Alternatively, one can set up :mod:`SIMBA` without this argument and set up the ``SimCodes`` location afterwards:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(directory=directory)

    fw.setSimCodesLocation(simcodes_location)

These executables are then accessible to the ``run()`` function of the ``frameworkLattice`` object.

In ``simba/Executables.yaml`` the required structure is provided for this
schema to work for different hardware architectures, either by the OS type or the computer name.

Using a Container Runtime
--------------------------

.. note::
   | Running **SIMBA** via Apptainer or Docker is only possible with an OS that supports it.
   | For Windows, **SIMBA** with the container runtime option can only be run with WSL.

Rather than installing the tracking codes locally, :mod:`SIMBA` can run them from a prebuilt container image,
using either `Docker <https://www.docker.com/>`_ or `Apptainer <https://apptainer.org/>`_. This is enabled
by passing the ``container_runtime`` argument to :mod:`SIMBA` on instantiation:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"

    fw = Framework(
        directory=directory,
        container_runtime="apptainer",  # or "docker"
    )

When ``container_runtime="apptainer"`` is used, :mod:`SIMBA` looks for a ``.sif`` image file at
``<simcodes>/Apptainer/simcodes-apptainer_master.sif``, where ``<simcodes>`` is the directory
passed via the ``simcodes`` argument (see :ref:`above <simcodes>`):

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    fw = Framework(
        directory=directory,
        simcodes=simcodes_location,
        container_runtime="apptainer",
    )

If ``simcodes`` is not provided, the ``.sif`` file defaults to a per-OS cache location (e.g.
``~/.local/share/apptainer/`` on Linux and ``~/Library/Application Support/apptainer/`` on macOS).

If the ``.sif`` file is not already present at that location, :mod:`SIMBA` creates the containing
directory (if necessary) and pulls the image from the registry defined in ``simba/Executables.yaml``
(``ghcr.io/astec-stfc/simcodes-apptainer:master`` by default) the first time :mod:`SIMBA` is
instantiated with ``container_runtime="apptainer"``. This image is several GB, so the initial pull
can take some time; subsequent instantiations detect the existing ``.sif`` file and skip the pull.

``container_runtime="docker"`` works analogously, pulling the ``ghcr.io/astec-stfc/simcodes-docker:master``
image via the local Docker daemon instead of writing a ``.sif`` file to the ``simcodes`` directory.

.. note::
   | ``container_runtime`` covers ASTRA, Elegant, CSRTrack, OPAL, and Genesis. **GPT is not included**,
     as it is proprietary and requires a license; it must be installed locally (see
     :func:`~simba.Codes.Executables.Executables.define_gpt_command`) and run with ``GPTLICENSE`` set.
   | The python-based codes (Ocelot, Xsuite, Cheetah, Wake-T) run in-process and are unaffected by
     ``container_runtime`` - install them locally via the ``simcodes`` extra regardless.

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
