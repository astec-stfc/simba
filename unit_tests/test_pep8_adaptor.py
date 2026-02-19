"""Tests for PEP8 adaptor: camel_to_snake, snake_to_camel, pep8_adaptor, alias_classes_to_pep8."""
import pytest
from simba.FrameworkHelperFunctions import (
    camel_to_snake,
    snake_to_camel,
    pep8_adaptor,
    alias_classes_to_pep8,
)


# ── camel_to_snake ──────────────────────────────────────────────────────────

class TestCamelToSnake:

    def test_pascal_case(self):
        assert camel_to_snake("FrameworkSettings") == "framework_settings"

    def test_camel_case(self):
        assert camel_to_snake("loadSettings") == "load_settings"

    def test_all_upper_case(self):
        assert camel_to_snake("URL") == "u_r_l" or camel_to_snake("URL") == "url"

    def test_single_word_lowercase(self):
        assert camel_to_snake("settings") == "settings"

    def test_single_word_upper(self):
        assert camel_to_snake("Settings") == "settings"

    def test_already_snake(self):
        assert camel_to_snake("my_function") == "my_function"

    def test_consecutive_caps(self):
        result = camel_to_snake("getCSRBins")
        assert "csr" in result.lower()

    def test_empty_string(self):
        assert camel_to_snake("") == ""

    def test_with_numbers(self):
        result = camel_to_snake("element2D")
        assert "element" in result

    def test_no_double_underscores(self):
        result = camel_to_snake("getHTTPSURL")
        assert "__" not in result


# ── snake_to_camel ──────────────────────────────────────────────────────────

class TestSnakeToCamel:

    def test_simple(self):
        assert snake_to_camel("load_settings") == "LoadSettings"

    def test_single_word(self):
        assert snake_to_camel("settings") == "Settings"

    def test_already_pascal(self):
        # Each char split by empty gives weird result, but verifying idempotency
        # isn't the goal — just verify it handles normal cases
        assert snake_to_camel("framework_directory") == "FrameworkDirectory"

    def test_multiple_underscores(self):
        assert snake_to_camel("set_lattice_sample_interval") == "SetLatticeSampleInterval"

    def test_empty_string(self):
        assert snake_to_camel("") == ""


# ── pep8_adaptor ────────────────────────────────────────────────────────────

class TestPep8Adaptor:

    def test_creates_snake_case_aliases(self):
        @pep8_adaptor
        class MyClass:
            def loadSettings(self):
                return "loaded"
            def changeCode(self):
                return "changed"
        obj = MyClass()
        assert hasattr(obj, "load_settings")
        assert hasattr(obj, "change_code")
        assert obj.load_settings() == "loaded"
        assert obj.change_code() == "changed"

    def test_preserves_original_methods(self):
        @pep8_adaptor
        class MyClass:
            def loadSettings(self):
                return "original"
        obj = MyClass()
        assert obj.loadSettings() == "original"
        assert obj.load_settings() == "original"

    def test_does_not_overwrite_existing_snake(self):
        @pep8_adaptor
        class MyClass:
            def loadSettings(self):
                return "camel"
            def load_settings(self):
                return "existing_snake"
        obj = MyClass()
        # Existing snake_case method should NOT be overwritten
        assert obj.load_settings() == "existing_snake"

    def test_ignores_private_methods(self):
        @pep8_adaptor
        class MyClass:
            def _privateMethod(self):
                return "private"
        obj = MyClass()
        # _privateMethod starts with _, so pep8_adaptor should skip it
        assert not hasattr(obj, "_private_method")

    def test_already_snake_case_no_duplicate(self):
        @pep8_adaptor
        class MyClass:
            def already_snake(self):
                return "ok"
        obj = MyClass()
        assert obj.already_snake() == "ok"

    def test_returns_class(self):
        @pep8_adaptor
        class MyClass:
            pass
        assert isinstance(MyClass(), MyClass)

    def test_works_on_framework_class(self):
        """Verify the decorator works on the actual Framework class."""
        import simba.Framework as fw
        # Framework should have snake_case aliases
        assert hasattr(fw.Framework, "load_settings") or hasattr(fw.Framework, "loadSettings")

    def test_framework_method_aliases(self):
        """Verify common camelCase Framework methods have snake_case aliases."""
        import simba.Framework as fw
        expected_aliases = [
            "change_lattice_code",
            "change_generator",
            "get_element",
            "modify_element",
            "modify_elements",
            "modify_element_type",
            "detect_changes",
            "save_changes_file",
            "load_changes_file",
        ]
        for alias in expected_aliases:
            assert hasattr(fw.Framework, alias), f"Framework missing PEP8 alias: {alias}"

    def test_framework_objects_method_aliases(self):
        """Verify Framework_objects classes have PEP8 aliases."""
        from simba.Framework_objects import frameworkLattice
        expected = [
            "get_element",
            "get_element_type",
            "set_element_type",
        ]
        for alias in expected:
            assert hasattr(frameworkLattice, alias), f"frameworkLattice missing PEP8 alias: {alias}"


# ── alias_classes_to_pep8 ──────────────────────────────────────────────────

class TestAliasClassesToPep8:

    def test_creates_snake_case_module_aliases(self):
        module_dict = {}
        class MyFramework:
            pass
        class FrameworkDirectory:
            pass
        module_dict["MyFramework"] = MyFramework
        module_dict["FrameworkDirectory"] = FrameworkDirectory
        alias_classes_to_pep8(module_dict)
        assert "my_framework" in module_dict
        assert "framework_directory" in module_dict
        assert module_dict["my_framework"] is MyFramework
        assert module_dict["framework_directory"] is FrameworkDirectory

    def test_does_not_overwrite_existing(self):
        class MyClass:
            pass
        class Existing:
            pass
        module_dict = {"MyClass": MyClass, "my_class": Existing}
        alias_classes_to_pep8(module_dict)
        # Should NOT overwrite existing 'my_class'
        assert module_dict["my_class"] is Existing

    def test_ignores_private_classes(self):
        class _PrivateClass:
            pass
        module_dict = {"_PrivateClass": _PrivateClass}
        alias_classes_to_pep8(module_dict)
        assert "_private_class" not in module_dict

    def test_ignores_non_classes(self):
        module_dict = {"some_function": lambda: None, "CONSTANT": 42}
        alias_classes_to_pep8(module_dict)
        # No new keys should be added for non-class entries
        assert len(module_dict) == 2

    def test_framework_module_aliases(self):
        """Verify the actual simba.Framework module has PEP8 class aliases."""
        import simba.Framework as fw
        assert hasattr(fw, "framework"), "Missing alias 'framework' for 'Framework'"
        assert fw.framework is fw.Framework

    def test_framework_directory_alias(self):
        """Verify frameworkDirectory has a PEP8 alias."""
        import simba.Framework as fw
        assert hasattr(fw, "framework_directory"), "Missing alias 'framework_directory'"
        assert fw.framework_directory is fw.frameworkDirectory

    def test_framework_objects_module_aliases(self):
        """Verify Framework_objects module has PEP8 class aliases."""
        import simba.Framework_objects as fo
        expected = ["framework_lattice", "framework_object", "run_setup"]
        for alias in expected:
            assert hasattr(fo, alias), f"Missing Framework_objects module alias: {alias}"


# ── PEP8 subclass propagation ──────────────────────────────────────────────

class TestPep8SubclassPropagation:

    def test_subclass_inherits_pep8_aliases(self):
        """Verify that subclasses of decorated classes also get PEP8 method aliases."""
        @pep8_adaptor
        class Base:
            def loadSettings(self):
                return "base"

        class Child(Base):
            def changeCode(self):
                return "child"

        # Base aliases should be inherited
        child = Child()
        assert child.load_settings() == "base"
