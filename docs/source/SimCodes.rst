cd .. _simcodes:

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
* MAD-X, through `cpymad <https://github.com/hibtc/cpymad>`_
* `ImpactX <https://impactx.readthedocs.io>`_, which is not on PyPI and so not in the ``simcodes`` extra: install
  it from conda-forge (``conda install -c conda-forge impactx``)
* `FBPIC <https://github.com/fbpic/fbpic>`_, through the separate ``fbpic`` extra
  (``pip install simba-accelerator[fbpic]``; on a GPU it also needs ``cupy`` for the local CUDA version)
* `Slipstream <https://github.com/astec-stfc/slipstream>`_, which is not in the ``simcodes`` extra: install it
  from its repository (it needs ``torch``). It tracks a line once, through ``LinacTracker``: quadrupoles, sextupoles,
  dipoles (with CSR and LSC if the lattice enables them) and cavities (with their wakefields). Apertures, screens and
  markers are drifts to it, and solenoids are drifts too. It has no space charge in this mode and writes no Twiss file.
  ``SLIPSTREAMsettings`` under ``global`` passes arguments to ``LinacTracker``; the defaults are in
  ``simba/Codes/Slipstream/slipstream_defaults.yaml``. Its CSR and LSC kernels are quadratic in the particle
  count unless ``fast_collective_effects: true`` is set there.

`RF-Track <https://gitlab.cern.ch/rf-track>`_ also runs in-process, but is not in the ``simcodes`` extra as it
is distributed under a CERN proprietary licence (free use, no redistribution). Install it from PyPI, where CERN
publishes it, with ``pip install simba-accelerator[rftrack]``, ``pip install RF-Track`` or ``simba-codes fetch`` (see
:ref:`below <fetch-codes>`). It tracks
in ``s`` through an ``RF_Track.Lattice``, writing the beam at screens and at the end of the line; cathode
(time-domain ``Volume``) tracking is not supported yet.

:mod:`SIMBA` does, however, require these codes to be accessible. This functionality is provided in various ways.

Creating a SimCodes Directory
-----------------------------

One can create a top-level directory containing sub-folders for each tracking code, and instantiate :mod:`SIMBA`
with a ``simcodes`` argument:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    framework = fw.Framework(
        directory=directory,
        simcodes=simcodes_location,
    )

Alternatively, one can set up :mod:`SIMBA` without this argument and set up the ``SimCodes`` location afterwards:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    framework = fw.Framework(directory=directory)

    framework.setSimCodesLocation(simcodes_location)

If ``simcodes`` is not given, :mod:`SIMBA` uses the directory in the ``SIMCODES`` environment variable, if it is set.

These executables are then accessible to the ``run()`` function of the ``frameworkLattice`` object.

In ``simba/Executables.yaml`` the required structure is provided for this
schema to work for different hardware architectures, either by the OS type or the computer name.

.. _finding-the-codes:

How :mod:`SIMBA` finds each code
--------------------------------

For each code, :mod:`SIMBA` takes the first of:

1. the entry for this computer's name in ``simba/Executables.yaml``;
2. the entry in its ``posix`` (Linux and macOS) or ``nt`` (Windows) section, most of which are paths under the
   ``simcodes`` directory;
3. a default path under the ``simcodes`` directory.

If the program that entry names does not exist, :mod:`SIMBA` looks for it on ``PATH``, by its file name (``genesis4``,
``Astra``, ``opal``, ...) and then by the code's name. So codes installed by a package manager (see
:ref:`below <conda-forge>`) need no configuration; nor does a code whose binary is in a directory on ``PATH``. An
explicit ``location`` (see :ref:`below <specific-location>`) is used as it is.

To change an entry without editing the installed ``simba/Executables.yaml``, write it to
``~/.config/simba/executables.yaml`` (or the file the ``SIMBA_EXECUTABLES`` environment variable names). Its entries
replace those of the same section and code; everything else is kept:

.. code-block:: yaml

    posix:
      astra: [/opt/ASTRA/Astra]
      csrtrack: [/opt/CSRTrack/csrtrack_1.204_Linux_x86_64_serial]

To see how each code would be run on this machine, and whether it is found:

.. code-block:: bash

    simba-codes check                             # natively
    simba-codes check --container-runtime auto

.. code-block:: text

    astragenerator   docker     /simcodes/Astra-serial/generator
    astra            docker     /simcodes/Astra-serial/Astra
    elegant          native     /opt/conda/envs/simba/bin/elegant
    gpt              not found  /path/to/simcodes/GPT/bin/gpt
    ...

``--simcodes`` gives the ``simcodes`` directory (default ``$SIMCODES``). With ``--container-runtime docker`` or
``apptainer`` the image is pulled if it is missing, as :mod:`SIMBA` itself does. (``simba-codes`` is the same as
``python -m simba.Codes.Executables``.)

.. _fetch-codes:

Downloading ASTRA, CSRTrack and RF-Track
----------------------------------------

ASTRA (with its generator) and the serial CSRTrack are single Linux x86-64 binaries, and RF-Track is on PyPI, so
:mod:`SIMBA` can fetch all three itself:

.. code-block:: bash

    simba-codes fetch

RF-Track is pip-installed into the Python running ``simba-codes`` (on any platform), unless it is already installed;
``--no-rftrack`` skips it. ASTRA and CSRTrack go to ``$SIMCODES``, or ``~/.local/share/simba/simcodes`` (``--simcodes`` gives another directory), in the
layout :mod:`SIMBA` expects, and :mod:`SIMBA` looks in ``~/.local/share/simba/simcodes`` when no other SimCodes
directory is given or found. Each file is taken from DESY if DESY still serves the version :mod:`SIMBA` knows, and
otherwise (DESY's pages are down at times) from the copy kept in
`simcodes-docker <https://github.com/astec-stfc/simcodes-docker/tree/main/vendor/desy>`__; either way it is kept only if
its sha256 matches. A file that is already there is never replaced. On macOS and Windows use a container runtime
for ASTRA and CSRTrack instead.

.. _conda-forge:

Codes from conda-forge
----------------------

Elegant (with Pelegant), Genesis, Bmad (with ``pytao``) and ImpactX are packaged on
`conda-forge <https://conda-forge.org/>`__. The ``environment.yml`` at the top of the :mod:`SIMBA` repository installs
them with the released :mod:`SIMBA` and its ``simcodes`` and ``rftrack`` extras (the file can be used on its own,
without the rest of the repository):

.. code-block:: bash

    conda env create -f environment.yml
    conda activate simba
    simba-codes fetch

`pixi <https://pixi.sh>`__ users get the same codes from ``pixi.toml``, with :mod:`SIMBA` installed from the
checkout (editable) and every version pinned in ``pixi.lock``, as tested:

.. code-block:: bash

    pixi run fetch      # installs the environment first
    pixi run test       # the unit tests, with the native codes
    pixi shell          # a shell in the environment

On Linux both take the CPU-only PyTorch (for Cheetah) instead of PyPI's, which brings about 4 GB of CUDA libraries,
and ``pixi.lock`` is for x86-64 CPUs with AVX2 (since about 2015), which the current ImpactX builds need.
:mod:`SIMBA` then finds the codes on ``PATH``, and ``libtao`` in the environment. ASTRA, CSRTrack, OPAL, GPT and BDSIM are not on conda-forge: ASTRA and
CSRTrack come from ``simba-codes fetch`` (see :ref:`above <fetch-codes>`); for the others install them into a
``simcodes`` directory, or let ``container_runtime="auto"`` run them in the container (see
:ref:`below <container-auto>`).

Using a Container Runtime
--------------------------

.. note::
   | Running **SIMBA** via Apptainer or Docker is only possible with an OS that supports it.
   | Apptainer has no native Windows build at all. On Windows, use WSL (see :ref:`below <windows-wsl>`).

Rather than installing the tracking codes locally, :mod:`SIMBA` can run them from a prebuilt container image,
using either `Docker <https://www.docker.com/>`_ or `Apptainer <https://apptainer.org/>`_. This is enabled
by passing the ``container_runtime`` argument to :mod:`SIMBA` on instantiation:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"

    framework = fw.Framework(
        directory=directory,
        container_runtime="apptainer",  # or "docker"
    )

When ``container_runtime="apptainer"`` is used, :mod:`SIMBA` looks for a ``.sif`` image file at
``<simcodes>/Apptainer/simcodes-apptainer_v0.3.sif``, where ``<simcodes>`` is the directory
passed via the ``simcodes`` argument (see :ref:`above <simcodes>`):

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    simcodes_location = "/path/to/simcodes/folder"

    framework = fw.Framework(
        directory=directory,
        simcodes=simcodes_location,
        container_runtime="apptainer",
    )

If ``simcodes`` is not provided, the ``.sif`` file defaults to a per-OS cache location (e.g.
``~/.local/share/apptainer/`` on Linux and ``~/Library/Application Support/apptainer/`` on macOS).

If the ``.sif`` file is not already present at that location, :mod:`SIMBA` creates the containing
directory (if necessary) and pulls the image from the registry defined in ``simba/Executables.yaml``
(``ghcr.io/astec-stfc/simcodes-apptainer:v0.3`` by default) the first time :mod:`SIMBA` is
instantiated with ``container_runtime="apptainer"``. This image is about 2 GB, so the initial pull
can take some time; subsequent instantiations detect the existing ``.sif`` file and skip the pull.

``container_runtime="docker"`` works analogously, pulling the ``ghcr.io/astec-stfc/simcodes-docker:v0.3``
image via the local Docker daemon instead of writing a ``.sif`` file to the ``simcodes`` directory.

The images are pinned to a release of `simcodes-docker <https://github.com/astec-stfc/simcodes-docker>`__.
A :mod:`SIMBA` version that moves to a newer release pulls the new image the first time it runs;
the old image (or ``.sif`` file) is left in place and can be deleted.

.. _container-auto:

Only the codes that are missing: ``container_runtime="auto"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``container_runtime="auto"``, each code runs natively if :mod:`SIMBA` finds it (see
:ref:`above <finding-the-codes>`), and otherwise in the container, with Docker if its daemon answers and else with
Apptainer. The image is pulled the first time a code needs it, not when :mod:`SIMBA` starts, so a machine with every
code it uses installed never pulls it. With the :ref:`conda-forge environment <conda-forge>`, for example, Elegant
and Genesis run natively and ASTRA, CSRTrack and OPAL in the container. Codes that are not in the image (GPT,
BDSIM) are always looked for natively.

.. _simcodes-image-contents:

What is in the image
^^^^^^^^^^^^^^^^^^^^

``container_runtime`` only runs the codes that :mod:`SIMBA` starts as a separate program. The python-based
codes run inside :mod:`SIMBA`'s own Python process, so with ``container_runtime`` they come from the host's
Python, whatever the image holds; the image's copies are only used when :mod:`SIMBA` itself runs inside the
image (see :ref:`below <simba-inside-image>`).

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Code
     - In the image
     - Notes
   * - ASTRA, Elegant, CSRTrack, OPAL, Genesis
     - yes
     - run through ``container_runtime``. ASTRA and CSRTrack are the serial builds; Pelegant runs on MPICH's
       ``mpiexec``.
   * - Ptarmigan, HiPACE++
     - since v0.3
     - run through ``container_runtime``. Ptarmigan is serial (with
       HDF5 output); HiPACE++ is the CPU (OpenMP) build. :mod:`SIMBA` has no HiPACE++ lattice yet.
   * - Bmad / Tao
     - yes
     - Tao runs in-process through ``pytao``, so not through ``container_runtime``: install Bmad and ``pytao``
       on the host (conda-forge). The image's ``long_term_tracking`` is not used by :mod:`SIMBA`.
   * - Ocelot, Xsuite, Cheetah, Wake-T, MAD-X (cpymad)
     - yes
     - in-process: install them on the host with the ``simcodes`` extra. The image has the versions its
       :mod:`SIMBA` release asks for, with the CPU build of ``torch``.
   * - ImpactX, FBPIC, Slipstream
     - since v0.3
     - in-process: install them on the host. In the image, ImpactX is the conda-forge build without MPI and
       FBPIC runs on the CPU.
   * - GPT
     - no
     - proprietary and licensed: install it locally (see
       :func:`~simba.Codes.Executables.Executables.define_gpt_command`) and run with ``GPTLICENSE`` set.
   * - RF-Track
     - no
     - its CERN licence allows free use but not redistribution: ``pip install RF-Track`` (or the ``rftrack``
       extra, or ``simba-codes fetch``), on the host or in the image.
   * - BDSIM
     - no
     - it needs Geant4 and ROOT, which would make the image several GB larger: install it locally
       (`installation <https://bdsim-collaboration.github.io/bdsim/sphinx/installation.html>`_).

The conda-forge builds of ImpactX and HiPACE++ in the image need an x86-64-v3 (AVX2) CPU.

.. _simba-inside-image:

Running :mod:`SIMBA` inside the image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The image also has :mod:`SIMBA` (its latest release when the image was built) and all the python-based codes
above, so a whole study can run in it with nothing installed on the host but Docker or Apptainer. Inside the
image, ``SIMCODES=/simba-codes`` holds the codes in the layout of the ``posix`` section of
``simba/Executables.yaml``; ``elegant``, ``Pelegant``, ``ptarmigan`` and ``hipace`` are on
``PATH``; and ``TAO_DIR`` is set for Bmad. Give :mod:`SIMBA` that directory and no ``container_runtime`` (a
:mod:`SIMBA` that reads ``SIMCODES`` itself needs neither):

.. code-block:: python

    import os
    import simba.Framework as fw

    framework = fw.Framework(directory="/path/to/working_directory", simcodes=os.environ["SIMCODES"])

.. _windows-wsl:

Running on Windows
------------------

The container images hold Linux binaries, so on Windows the supported route for running in containers is
`WSL2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_:

.. code-block:: powershell

   wsl --install -d Ubuntu

Then, inside the WSL shell, install Docker (or Apptainer) and :mod:`SIMBA` as on any Linux
machine, and work from a path under the WSL filesystem (``/home/...``) rather than under
``/mnt/c``; bind-mounting a ``/mnt/c`` working directory into a container is slow and the
tracking codes write a lot of intermediate files.

Native Windows supports the codes that ship a Windows build - ASTRA, Elegant, CSRTrack and
GPT - via the ``nt`` section of ``simba/Executables.yaml`` or the ``location`` argument
(see :ref:`below <specific-location>`).

**OPAL and Genesis are Linux-only.** There is no Windows binary to point ``Executables.yaml``
at, so requesting either from a Windows Python process raises a ``RuntimeError`` telling you to
use WSL or a container runtime, rather than failing later with a missing output file.

``container_runtime`` on native Windows is untested.

Editing the Executables.yaml file
---------------------------------

If the user already has these executables installed, they can point directly to them in
``simba/Executables.yaml``

.. _specific-location:

Pointing to a specific location
-------------------------------

An instance of :mod:`SIMBA` has access to these executables via the ``executables`` attribute, and these
can be modified once :mod:`SIMBA` is instantiated.

For example, in order to point to a local install of the ``ELEGANT`` code, the user can run the following code:

.. code-block:: python

    import simba.Framework as fw
    directory = "/path/to/working_directory"
    elegant_location = "/path/to/elegant/binary"

    framework = fw.Framework(directory=directory)

    framework.executables.define_elegant_command(location=elegant_location)

This will then allow :mod:`SIMBA` to call the correct version of ``ELEGANT``.

Citing the codes used
---------------------

Please consider citing the code(s) used if any work performed with :mod:`SIMBA` leads to a
publication. The citations above resolve against the :ref:`reference list <references>`.
