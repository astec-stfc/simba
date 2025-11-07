.. _installation:

Installation
============

.. note::
   | **SIMBA** is compatible only with python `3.10` and above.
   | Contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_ in case of any issues during installation / testing / etc.

Cloning from Github
-------------------

Clone :mod:`SIMBA` from Github:

.. code-block:: bash

    git clone https://github.com/astec-stfc/simba.git

Install via pip
---------------

(It is recommended to activate a ``python3.10`` (or higher) virtual environment to run :mod:`SIMBA`.)

The package and its dependencies can be installed using the following command in the :mod:`SIMBA` directory:

.. code-block:: bash

    pip install .

In order to enable :mod:`SIMBA` to access the simulation codes, refer to the instructions
:ref:`here <simcodes>` -- this step is necessary to perform the tests.

To check that the install was completed successfully, run this command from the top level:

.. code-block:: bash

    pytest --cov



Install from pypi / conda-forge
-------------------------------

WIP.....

Required Dependencies
---------------------

Check out the ``pyproject.toml`` file for a full list of dependencies for :mod:`SimFrame`.

Optional Dependencies
---------------------

In order to set up the :mod:`SimCodes` required for running simulations, refer to
:ref:`SimCodes <simcodes>`
