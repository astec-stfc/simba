"""Tests for _FrozenSnapshot and Framework_Settings."""
import os
import pytest
import warnings
from pydantic import BaseModel

from simba.Framework import _FrozenSnapshot
from simba.Framework_Settings import FrameworkSettings


# ── _FrozenSnapshot ─────────────────────────────────────────────────────────

class SimplePydanticModel(BaseModel):
    name: str = "test_element"
    length: float = 1.5
    position: float = 0.0
    active: bool = True


class TestFrozenSnapshot:

    @pytest.fixture
    def model(self):
        return SimplePydanticModel(name="Q1", length=2.0, position=0.5, active=True)

    @pytest.fixture
    def snapshot(self, model):
        return _FrozenSnapshot(model)

    def test_getattr_returns_values(self, snapshot):
        assert snapshot.name == "Q1"
        assert snapshot.length == 2.0
        assert snapshot.position == 0.5
        assert snapshot.active is True

    def test_getattr_missing_raises(self, snapshot):
        with pytest.raises(AttributeError):
            _ = snapshot.nonexistent

    def test_contains(self, snapshot):
        assert "name" in snapshot
        assert "length" in snapshot
        assert "nonexistent" not in snapshot

    def test_model_dump_returns_copy(self, snapshot):
        d = snapshot.model_dump()
        assert isinstance(d, dict)
        assert d["name"] == "Q1"
        assert d["length"] == 2.0
        # Modifying the returned dict should not affect the snapshot
        d["name"] = "CHANGED"
        assert snapshot.name == "Q1"

    def test_iter(self, snapshot):
        items = list(snapshot)
        keys = [k for k, v in items]
        assert "name" in keys
        assert "length" in keys

    def test_name_attribute(self, snapshot):
        assert snapshot.name == "Q1"

    def test_snapshot_is_independent(self, model, snapshot):
        """Modifying the original model doesn't affect the snapshot."""
        model.length = 99.0
        assert snapshot.length == 2.0

    def test_snapshot_from_model_without_name(self):
        class NoNameModel(BaseModel):
            x: float = 1.0
        snap = _FrozenSnapshot(NoNameModel())
        assert snap.name is None
        assert snap.x == 1.0


# ── FrameworkSettings ───────────────────────────────────────────────────────

class TestFrameworkSettings:

    def test_init_creates_required_keys(self):
        settings = FrameworkSettings()
        for key in ["global", "generator", "files", "groups", "elements", "layout", "section", "element_list"]:
            assert key in settings

    def test_init_with_nonexistent_file(self):
        settings = FrameworkSettings(filename="nonexistent_file.def")
        # Should not raise, but file won't load
        assert settings.settingsFilename == "nonexistent_file.def"

    def test_load_and_save_settings(self, tmp_path):
        import yaml
        filepath = str(tmp_path / "test.def")
        data = {
            "global": {"key": "value"},
            "files": {"FODO": {"code": "elegant"}},
            "generator": {},
            "groups": {},
            "elements": {},
            "layout": {},
            "section": {},
            "element_list": {},
        }
        with open(filepath, "w") as f:
            yaml.dump(data, f)
        settings = FrameworkSettings(filename=filepath)
        assert settings["global"]["key"] == "value"
        assert settings["files"]["FODO"]["code"] == "elegant"

    def test_copy_returns_dict(self):
        settings = FrameworkSettings()
        settings["global"]["test_key"] = "test_value"
        copied = settings.copy()
        assert isinstance(copied, dict)
        assert copied["global"]["test_key"] == "test_value"

    def test_add_file(self):
        settings = FrameworkSettings()
        settings.add_file("FODO", "elegant", "M1", "M2", input={"twiss": {}}, charge={"mode": "off"})
        assert "FODO" in settings["files"]
        assert settings["files"]["FODO"]["code"] == "elegant"
        assert settings["files"]["FODO"]["output"]["start_element"] == "M1"
        assert settings["files"]["FODO"]["output"]["end_element"] == "M2"

    def test_add_file_with_zstart(self):
        settings = FrameworkSettings()
        settings.add_file("sect", "astra", 0.0, "END", input={}, charge={})
        assert settings["files"]["sect"]["output"]["zstart"] == 0.0

    def test_add_group(self):
        settings = FrameworkSettings()
        settings.add_group("mygroup", "chicane", ["E1", "E2", "E3"])
        assert settings["groups"]["mygroup"]["type"] == "chicane"
        assert settings["groups"]["mygroup"]["elements"] == ["E1", "E2", "E3"]

    def test_add_element(self):
        settings = FrameworkSettings()
        settings.add_element("Q1", "quadrupole", 0.5, [0, 0, 0], [0, 0, 0.5], k1l=1.0)
        assert settings["elements"]["Q1"]["type"] == "quadrupole"
        assert settings["elements"]["Q1"]["length"] == 0.5
        assert settings["elements"]["Q1"]["k1l"] == 1.0
