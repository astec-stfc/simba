.. _installation:

Installation
============

.. note::
   | **SimFrame** is compatible only with python `3.12` and above.
   | Contact `Alex Brynes <mailto:alexander.brynes@stfc.ac.uk>`_ in case of any issues during installation / testing / etc.

Cloning from Github
-------------------

Clone ``SimFrame`` from Github:

.. code-block:: bash

    git clone https://github.com/astec-stfc/simframe.git

Install via pip
-------------------

(It is recommended to activate a ``python3.12`` virtual environment to run ``SimFrame``.)

The package and its dependencies can be installed using the following command in the ``SimFrame`` directory:

.. code-block:: bash

    pip install .


To install the ``MasterLattice`` package along with ``SimFrame``, use this command from
the ``SimFrame`` directory:

.. code-block:: bash

    pip install .[test]

In order to enable ``SimFrame`` to access the simulation codes, refer to the instructions
:ref:`here <simcodes>` -- this step is necessary to perform the tests.

To check that the install was completed successfully, run this command from the top level:

.. code-block:: bash

    pytest --cov



Install from pypi / conda-force
-------------------

WIP.....

Required Dependencies
---------------------

Check out the ``pyproject.toml`` file for a full list of dependencies for ``SimFrame``.

Optional Dependencies
---------------------

The following dependency is optional, but is generally required for running ``CLARA`` simulations:

* `MasterLattice <https://github.com/astec-stfc/masterlattice.git>`__

This can be installed via `pip`:

.. code-block:: bash

    pip install MasterLattice

Finally, in order to set up the ``SimCodes`` required for running simulations, refer to
:ref:`SimCodes <simcodes>`
