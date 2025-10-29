.. _masterlattice:

MasterLattice
=============

The :mod:`MasterLattice` is a container for elements belonging to a specific accelerator lattice. It is used to enable :mod:`SimFrame` to:

* Read in lattice elements and create the appropriate :ref:`frameworkElement <framework-elements>` objects.
* Define lattice lines based on these elements. Each line becomes an instance of a :py:class:`frameworkLattice <simba.Framework_objects.frameworkLattice>` depending on its `code` attribute.
* Add a :py:class:`frameworkGenerator <simba.Codes.Generators.Generators.frameworkGenerator>` for defining an initial particle distribution.
* Link to data files containing electromagnetic field distributions for given elements that are loaded into :py:class:`field <simba.Modules.Fields.field>` objects.

:mod:`MasterLattice` files for the CLARA accelerator can be cloned from the `Github repository <https://github.com/astec-stfc/masterlattice/>`_ or via pip:

.. code-block:: bash

    pip install MasterLattice

An example snippet of the definition of the `CLARA <https://www.astec.stfc.ac.uk/Pages/CLARA.aspx>`_ photoinjector is provided :ref:`here <creating-the-lattice-elements>`.

The keys in the YAML file define an individual accelerator element (which can contain overlapping `sub_elements` such as a solenoid surrounding an accelerating cavity). 

While :py:class:`drift <simba.Framework_objects.csrdrift>` elements exist in :mod:`SimFrame`, they do not have to be provided explicitly in the lattice definition. Rather, these elements can be created based on the distance between the `end` and `start` of two adjacent elements; see :py:func:`createDrifts <simba.Framework_objects.frameworkLattice.createDrifts>`.

Required and default parameters
-------------------------------

The definition of each `frameworkElement` defines the following fields (default values are specified below):

* :mod:`objectname`: `str` -- Name of the element

* :mod:`objecttype`: `str` -- Type of element; must be a subclass of :ref:`frameworkElement <framework-elements>`

* :mod:`length`: `float = 0` -- Length of the element in the simulation [m]

* :mod:`centre`: `Tuple[float, float, float] = (0.0, 0.0, 0.0)` -- Centre of the element [x,y,z] in Cartesian coordinates

* :mod:`position_errors:` Tuple[float, float, float] = (0.0, 0.0, 0.0)` -- Position errors of the element [x,y,z]

* :mod:`rotation_errors`: `Tuple[float, float, float] = (0.0, 0.0, 0.0)` -- Rotation errors of the element in the simulation [x,y,z]

* :mod:`global_rotation`: `Tuple[float, float, float] = (0.0, 0.0, 0.0)` -- Global rotation of the element in the simulation [x,y,z]

* :mod:`rotation`: `Tuple[float, float, float] = (0.0, 0.0, 0.0)` -- Local rotation of the element in the simulation [x,y,z]

* :mod:`subelement`: `bool = False` -- Flag indicating whether the element is a sub-element of a larger structure

* :mod:`field_definition`: `[field | str | None] = None` -- Field definition for the element, can be a field object or a string representing a file

* :mod:`wakefield_definition`: `[field | str | None] = None` -- Wakefield definition for the element, can be a field object or a string representing a file

Certain elements have additional requirements: magnets such as :py:class:`dipole <simba.Elements.dipole.dipole>` and :py:class:`quadrupole <simba.Elements.quadrupole.quadrupole>` must have non-zero length, an :py:class:`aperture <simba.Elements.aperture.aperture>` must define a shape. Additional properties can also provided to given elements; see the element-specific documentation :ref:`here <framework-elements>` to see those which can be interpreted by :mod:`SimFrame`, although note that other arbitrary attributes can be specified.

Note also that while it is not necessary to include :py:class:`marker <simba.Elements.marker.marker>` or :py:class:`screen <simba.Elements.screen.screen>` type elements at the beginning and end of a line, it is generally good practice. This enables :mod:`SimFrame` to have reliable reference points for the lattice.

Building a lattice: A simple FODO example
-----------------------------------------

The example below shows a very simple beamline consisting of two quadrupole magnets and a beginning and end marker. Elements do not have to be placed sequentially in longitudinal order in this file. When they are loaded into the :py:class:`frameworkLattice <simba.Framework_objects.frameworkLattice>` object, the order is not important. During pre-processing of the lattice before tracking, :mod:`SimFrame` arranges the elements in sequential order before writing the code-specific input files. 

.. code-block:: yaml

    elements:
        BEGIN:
            centre: [0, 0, 0]
            type: marker
        QUAD1:
            centre: [0.0, 0.0, 0.1]
            type: quadrupole
            length: 0.1
            k1l: 1.0
        QUAD2:
            centre: [0.0, 0.0, 0.3]
            type: quadrupole
            length: 0.1
            k1l: -1.0
        END:
            centre: [0, 0, 0.4]
            type: marker
