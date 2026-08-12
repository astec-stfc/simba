"""
Backwards-compatibility guards for the PEP 8 naming migration.

Every class/function/global renamed for PEP 8 compliance keeps working under
its old spelling: module-level names via ``simba._compat.deprecated_aliases``
(a module ``__getattr__``), instance methods and pydantic fields via the
``DeprecatedMethodAliases`` mixin, and renamed module/package paths via the
meta-path finder in ``simba._legacy``. These tests pin that surface so a
future edit can't quietly drop an alias.

Warnings are FutureWarning rather than DeprecationWarning: Python only
displays a DeprecationWarning raised from ``__main__``, so a downstream
script calling a legacy name from its own top-level code would otherwise see
nothing.
"""

import importlib
import warnings

import pytest

from simba._compat import SIMBA_RENAMES
from simba._legacy import LEGACY_MODULES

ALIAS_REGISTERING_MODULES = [
    "simba.codes.astra.astra",
    "simba.codes.cheetah.cheetah",
    "simba.codes.csrtrack.csrtrack",
    "simba.codes.elegant.elegant",
    "simba.codes.executables",
    "simba.codes.generators.generators",
    "simba.codes.genesis.genesis",
    "simba.codes.gpt.gpt",
    "simba.codes.ocelot.ocelot",
    "simba.codes.opal.opal",
    "simba.codes.wake_t.wake_t",
    "simba.codes.xsuite.xsuite",
    "simba.framework",
    "simba.framework_helper_functions",
    "simba.framework_objects",
    "simba.modules.beams",
    "simba.modules.beams.cheetah",
    "simba.modules.beams.hdf5",
    "simba.modules.beams.particles.centroids",
    "simba.modules.beams.particles.emittance",
    "simba.modules.beams.particles.kde",
    "simba.modules.beams.particles.minimum_volume_ellipse",
    "simba.modules.beams.particles.sigmas",
    "simba.modules.beams.particles.slice",
    "simba.modules.beams.particles.twiss",
    "simba.modules.beams.plot",
    "simba.modules.beams.sdds",
    "simba.modules.fields",
    "simba.modules.fields.hdf5",
    "simba.modules.fields.sdds",
    "simba.modules.gdf_beam",
    "simba.modules.gdf_emit",
    "simba.modules.id_number",
    "simba.modules.id_number_server",
    "simba.modules.matrices",
    "simba.modules.matrices.hdf5",
    "simba.modules.optimisation.constraints",
    "simba.modules.optimisation.optimiser",
    "simba.modules.plotting.lattice_draw",
    "simba.modules.plotting.multi_axis_plot",
    "simba.modules.plotting.multi_plot",
    "simba.modules.plotting.plotting",
    "simba.modules.pmd_units",
    "simba.modules.sdds_file",
    "simba.modules.twiss",
    "simba.modules.twiss.hdf5",
]


def _import_all():
    for mod in ALIAS_REGISTERING_MODULES:
        importlib.import_module(mod)


class TestModuleLevelAliases:
    """Renamed classes, functions and module globals -- served by __getattr__."""

    def test_every_registered_alias_resolves(self):
        _import_all()
        assert SIMBA_RENAMES, "no modules registered aliases"

        checked = 0
        for module_name, aliases in SIMBA_RENAMES.items():
            module = importlib.import_module(module_name)
            for legacy, current in aliases.items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    obj = getattr(module, legacy)
                assert obj is getattr(module, current), (
                    f"{module_name}.{legacy} does not resolve to {current}"
                )
                assert any(
                    issubclass(c.category, FutureWarning) for c in caught
                ), f"{module_name}.{legacy} resolved without a FutureWarning"
                checked += 1
        assert checked >= 140, f"expected the full alias surface, checked {checked}"

    def test_unknown_attribute_still_raises(self):
        from simba.codes.astra import astra

        with pytest.raises(AttributeError):
            astra.definitely_not_a_real_name


class TestLegacyModulePaths:
    """
    Module and package paths to lower_snake_case
    (``simba.Codes.ASTRA`` -> ``simba.codes.astra``, ...), using a
    metapath finder instead of shim files -- see simba/_legacy.py.
    """

    _UNIMPORTABLE = {
        "simba.modules.symmlinks",
        "simba.support_files.check_yaml_with_center_datums",
        "simba.support_files.convert_yaml_to_db",
        "simba.support_files.convert_txt_to_hdf5_field",
        "simba.support_files.elegant_to_yaml",
        "simba.support_files.gdf_to_hdf5",
        "simba.support_files.gdf_to_sdds",
        "simba.support_files.sdds_to_gdf",
        "simba.support_files.sdds_to_hdf5",
        "simba.support_files.update_yaml_to_center_datums",
    }

    def test_every_legacy_path_resolves_to_the_same_object(self):
        assert LEGACY_MODULES, "no legacy module paths registered"
        for old, new in LEGACY_MODULES.items():
            if new in self._UNIMPORTABLE:
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                old_module = importlib.import_module(old)
            new_module = importlib.import_module(new)
            assert old_module is new_module, f"{old} does not resolve to {new}"
            assert any(
                issubclass(c.category, FutureWarning) for c in caught
            ), f"import {old} did not warn"

    def test_unknown_legacy_path_is_not_registered(self):
        assert "simba.Codes.NotARealCode" not in LEGACY_MODULES


class TestInstanceMethodAndFieldAliases:
    """
    Renamed instance methods and pydantic fields, served by the
    DeprecatedMethodAliases mixin (attribute *reads* only, matching laura's
    own limitation for the same mechanism).
    """

    def test_renamed_method_still_reachable(self):
        from simba.framework_objects import RunSetup

        rs = RunSetup()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rs.setNRuns(5)
        assert rs.nruns == 5
        assert any(issubclass(c.category, FutureWarning) for c in caught)

    def test_renamed_field_still_readable(self):
        from simba.framework_objects import FrameworkLattice

        aliases = FrameworkLattice._DEPRECATED_METHOD_ALIASES
        assert aliases["elementObjects"] == "element_objects"
        assert aliases["groupObjects"] == "group_objects"

    def test_subclass_overriding_a_legacy_name_is_warned_about(self):
        """
        An alias can't save a downstream override: simba calls the new name,
        so a subclass still defining the old one is silently skipped. The
        base class warns instead.
        """
        from simba.framework_objects import FrameworkLattice

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class LegacyOverride(FrameworkLattice):
                # Deliberately the OLD name -- this is the trap being detected.
                def preProcess(self, *args, **kwargs):
                    return "never called"

        messages = [
            str(w.message) for w in caught if issubclass(w.category, FutureWarning)
        ]
        assert any("preProcess" in m for m in messages), messages

    def test_compliant_subclass_is_not_warned_about(self):
        from simba.framework_objects import FrameworkLattice

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class ModernOverride(FrameworkLattice):
                def pre_process(self, *args, **kwargs):
                    return "called fine"

        assert not [
            w for w in caught
            if issubclass(w.category, FutureWarning) and "renamed" in str(w.message)
        ]


class TestWireValuesWereNotRenamed:
    """
    A handful of strings are wire values -- written into lattice YAML, or
    matched against simulation-code file formats -- not Python identifiers,
    and renaming them would silently break existing data even though nothing
    would raise. Pin them here.
    """

    def test_lattice_code_lookup_keys_are_unchanged(self):
        """
        The ``code`` field of a lattice YAML block (e.g. ``code: ocelot``)
        selects the backend via this dict; those keys are user-facing wire
        values, independent of the (now CapWords) class names they map to.
        """
        from simba import framework_lattices

        assert set(framework_lattices.LATTICE_CLASSES) == {
            "astra", "cheetah", "csrtrack", "elegant", "genesis",
            "gpt", "ocelot", "opal", "waket", "xsuite",
        }

    def test_group_type_lookup_keys_are_unchanged(self):
        """Same idea for a group's ``type:`` value in lattice YAML."""
        from simba import framework_elements

        assert set(framework_elements.GROUP_CLASSES) == {
            "chicane", "s_chicane", "r56_group", "element_group",
        }

    def test_field_component_names_were_not_renamed(self):
        """
        Ex/Ey/Ez/Bx/By/Bz/... on FieldMap mirror column/key names written by
        ASTRA/GPT/OPAL/SDDS/GDF/HDF5 field-map readers and writers throughout
        simba/modules/fields/*.py; they are not simba's naming choice to
        change, and ruff never flagged them as violations here (only the
        identically-spelled, unrelated properties on Particles were).
        """
        from simba.modules.fields import FieldMap

        for name in ("Ex", "Ey", "Ez", "Er", "Bx", "By", "Bz", "Br",
                     "Wx", "Wy", "Wz", "Wr", "G"):
            assert name in FieldMap.model_fields, (
                f"FieldMap.{name} is missing or was renamed; this is a wire "
                f"value read from/written to simulation-code field files"
            )
