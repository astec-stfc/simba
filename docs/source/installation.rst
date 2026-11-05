.. _installation:

Installation
============

.. note::
   | **SIMBA** is compatible only with python `3.11` and above.
   | Contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_ in case of any issues during installation / testing / etc.

Cloning from Github
-------------------

Clone :mod:`SIMBA` from Github:

.. code-block:: bash

    git clone https://github.com/astec-stfc/simba.git

Install via pip
---------------

(It is recommended to activate a ``python3.11`` (or higher) virtual environment to run :mod:`SIMBA`.)

The package and its dependencies can be installed using the following command in the :mod:`SIMBA` directory:

.. code-block:: bash

    pip install .

In order to enable :mod:`SIMBA` to access the simulation codes, refer to the instructions
:ref:`here <simcodes>` -- this step is necessary to perform the tests.

To check that the install was completed successfully, run this command from the top level:

.. code-block:: bash

    pytest --cov



Conda environment
-----------------

``environment.yml`` in the :mod:`SIMBA` directory builds a conda environment with the compiled codes from
conda-forge (Bmad/PyTao, ELEGANT and Pelegant on Open MPI, Genesis 1.3 v4, ImpactX, FBPIC, Wake-T, PyTorch,
mpi4py), the
remaining Python codes and :mod:`LAURA` from PyPI, and :mod:`SIMBA` itself as an editable install:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate simba
    pytest unit_tests -n 4

To develop against a local :mod:`LAURA` checkout, install it over the PyPI release (and keep its branch
matching :mod:`SIMBA`'s):

.. code-block:: bash

    pip install -e ../laura

Activating the environment blanks ``PYTHONPATH`` and ``LD_LIBRARY_PATH``, so separately built copies of
ROOT, Bmad or MPI cannot shadow the conda ones. If ``SimCodes`` has no Bmad or Genesis build, :mod:`SIMBA`
uses the environment's own ``libtao`` and ``genesis4``.

ASTRA, GPT, OPAL, CSRTrack and BDSIM are not provided; set them up through
:ref:`SimCodes <simcodes>` as usual. ELEGANT needs ``RPN_DEFNS`` pointing at a ``defns.rpn`` file, which
the conda package does not ship; :mod:`SIMBA` sets it from ``SimCodes`` when one is there.

Install from pypi / conda-forge
-------------------------------

:mod:`SIMBA` is available from pypi:

.. code-block:: bash

    pip install simba-accelerator

Required Dependencies
---------------------

Check out the ``pyproject.toml`` file for a full list of dependencies for ``SIMBA``.

Optional Dependencies
---------------------

In order to set up the :mod:`SimCodes` required for running simulations, refer to
:ref:`SimCodes <simcodes>`
