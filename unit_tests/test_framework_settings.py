from simba.Framework_Settings import FrameworkSettings


def test_init_creates_empty_sections():
    fs = FrameworkSettings()
    for key in ("global", "generator", "files", "groups", "elements", "layout", "section", "element_list"):
        assert fs[key] == {}
    assert fs.settingsFilename is None


def test_add_file_with_numeric_start():
    fs = FrameworkSettings()
    fs.add_file("sec1", "elegant", 0.0, "END", input={"a": 1}, charge={"b": 2})
    assert fs["files"]["sec1"] == {
        "code": "elegant",
        "output": {"zstart": 0.0, "end_element": "END"},
        "input": {"a": 1},
        "charge": {"b": 2},
    }


def test_add_file_with_named_start():
    fs = FrameworkSettings()
    fs.add_file("sec1", "elegant", "START", "END")
    assert fs["files"]["sec1"]["output"] == {"start_element": "START", "end_element": "END"}


def test_add_group():
    fs = FrameworkSettings()
    fs.add_group("g1", "quad", ["Q1", "Q2"])
    assert fs["groups"]["g1"] == {"type": "quad", "elements": ["Q1", "Q2"]}


def test_add_element():
    fs = FrameworkSettings()
    fs.add_element("Q1", "quadrupole", 1.0, 0.0, 1.0, k1l=-1.0)
    assert fs["elements"]["Q1"] == {
        "type": "quadrupole",
        "length": 1.0,
        "position_start": 0.0,
        "position_end": 1.0,
        "k1l": -1.0,
    }


def test_add_element_file():
    fs = FrameworkSettings()
    fs.add_element_file("elements1.yaml")
    fs.add_element_file("elements2.yaml")
    assert fs["elements"]["filename"] == ["elements1.yaml", "elements2.yaml"]


def test_load_settings_roundtrip(tmp_path):
    fs = FrameworkSettings()
    fs.add_group("g1", "quad", ["Q1"])
    path = tmp_path / "settings.yaml"

    import yaml

    with open(path, "w") as f:
        yaml.dump(fs.copy(), f)

    loaded = FrameworkSettings(filename=str(path))
    assert loaded["groups"]["g1"] == {"type": "quad", "elements": ["Q1"]}
    assert loaded.settingsFilename == str(path)


def test_load_settings_new_flag_skips_loading(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("groups:\n  g1: {type: quad, elements: [Q1]}\n")
    fs = FrameworkSettings(filename=str(path), new=True)
    assert fs["groups"] == {}


def test_load_settings_method_alias(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("groups:\n  g1: {type: quad, elements: [Q1]}\n")
    fs = FrameworkSettings()
    fs.load_settings(str(path))
    assert fs["groups"]["g1"] == {"type": "quad", "elements": ["Q1"]}


def test_copy_returns_plain_dict():
    fs = FrameworkSettings()
    fs.add_group("g1", "quad", ["Q1"])
    plain = fs.copy()
    assert isinstance(plain, dict)
    assert plain["groups"]["g1"] == {"type": "quad", "elements": ["Q1"]}
