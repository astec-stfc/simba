"""
Simframe Fields Module

Thin re-export of :mod:`laura.translator.utils.fields` -- this package used to
be an independently-maintained duplicate of that module (same ``field``
class, same ``astra``/``gdf``/``hdf5``/``opal``/``sdds``/``rftrack``
sub-modules, hand-copied), which had already silently drifted apart in
``opal.py`` (different, incompatible unit-conversion factors between the two
copies) and meant every ``isinstance(x, field)`` check in SIMBA (``Ocelot.py``,
``FrameworkHelperFunctions.py``, ``Framework_objects.py``) was comparing
against the *wrong* class whenever ``x`` was actually a
``laura.translator.utils.fields.field`` -- which it always is in practice,
since every ``field_definition``/``wakefield_definition`` is resolved by
``laura.translator.converters.base.BaseElementTranslator.
update_field_definition()``. Re-exporting laura's own class fixes that
silently.

``laura`` is already an unconditional, module-level dependency of the rest of
SIMBA (``Framework.py``, every ``Codes/*.py``), so this adds no new import
requirement.

Classes:
    - :class:`~laura.translator.utils.fields.field`: Generic field definition.
    - :class:`~laura.translator.utils.fields.FieldParameter.FieldParameter`: Field parameter with a
      name and a :class:`~laura.translator.utils.units.UnitValue` associated with it.
"""
from laura.translator.utils.fields import (  # noqa: F401
    field,
    FieldParameter,
    allowed_fields,
    fieldtype,
    cavitytype,
    astra,
    gdf,
    hdf5,
    sdds,
    opal,
    rftrack,
)
