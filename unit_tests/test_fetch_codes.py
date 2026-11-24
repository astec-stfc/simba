import hashlib
import os
import pytest
from simba.Codes import fetch_codes as fc

GOOD = b"known-good binary"
FILES = {"ASTRA/Astra": ("https://desy.example/Astra", "astra-serial/Astra", hashlib.sha256(GOOD).hexdigest())}


@pytest.fixture
def served(monkeypatch):
    """What each URL serves (None: unreachable), and the URLs asked for."""
    urls, asked = {}, []
    monkeypatch.setattr(fc, "FILES", FILES)
    monkeypatch.setattr(fc, "_download", lambda url: asked.append(url) or urls.get(url))
    monkeypatch.setattr(fc, "fetch_rftrack", lambda: False)
    monkeypatch.setattr(fc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fc.platform, "machine", lambda: "x86_64")
    return urls, asked


def test_from_desy_while_it_serves_the_known_good_file(tmp_path, served):
    urls, asked = served
    urls["https://desy.example/Astra"] = GOOD
    assert fc.fetch_codes(str(tmp_path)) == [str(tmp_path / "ASTRA" / "Astra")]
    assert (tmp_path / "ASTRA" / "Astra").read_bytes() == GOOD
    assert os.access(tmp_path / "ASTRA" / "Astra", os.X_OK)
    assert asked == ["https://desy.example/Astra"]


@pytest.mark.parametrize("desy", [None, b"a new version"])
def test_from_the_mirror_when_desy_is_down_or_changed(tmp_path, served, desy):
    urls, asked = served
    urls["https://desy.example/Astra"] = desy
    urls[fc.MIRROR + "astra-serial/Astra"] = GOOD
    fc.fetch_codes(str(tmp_path))
    assert (tmp_path / "ASTRA" / "Astra").read_bytes() == GOOD


def test_nothing_is_written_unless_it_is_the_known_good_file(tmp_path, served):
    urls, _ = served
    urls[fc.MIRROR + "astra-serial/Astra"] = b"tampered"
    with pytest.raises(RuntimeError, match="sha256"):
        fc.fetch_codes(str(tmp_path), desy=False)
    assert not (tmp_path / "ASTRA").exists()


def test_a_file_already_there_is_kept(tmp_path, served):
    urls, asked = served
    (tmp_path / "ASTRA").mkdir()
    (tmp_path / "ASTRA" / "Astra").write_bytes(b"my own build")
    assert fc.fetch_codes(str(tmp_path)) == []
    assert (tmp_path / "ASTRA" / "Astra").read_bytes() == b"my own build"
    assert asked == []


def test_only_linux_x86_64(tmp_path, served, monkeypatch):
    monkeypatch.setattr(fc.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="container_runtime"):
        fc.fetch_codes(str(tmp_path))


def test_rftrack_is_installed_before_the_linux_only_codes(tmp_path, served, monkeypatch):
    installed = []
    monkeypatch.setattr(fc, "fetch_rftrack", lambda: installed.append(True) or True)
    monkeypatch.setattr(fc.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="container_runtime"):
        fc.fetch_codes(str(tmp_path))
    assert installed
    installed.clear()
    served[0]["https://desy.example/Astra"] = GOOD
    monkeypatch.setattr(fc.platform, "system", lambda: "Linux")
    assert fc.fetch_codes(str(tmp_path), rftrack=False) == [str(tmp_path / "ASTRA" / "Astra")]
    assert not installed


@pytest.fixture
def pip(monkeypatch):
    """The commands run, each returning ``pip.returncode``."""
    class Pip(list):
        returncode = 0
    runs = Pip()
    monkeypatch.setattr(fc.subprocess, "run", lambda cmd: runs.append(cmd) or type("R", (), {"returncode": runs.returncode}))
    return runs


def test_rftrack_already_installed_is_not_reinstalled(pip, monkeypatch):
    monkeypatch.setattr(fc.importlib.util, "find_spec", lambda name: object())
    assert fc.fetch_rftrack() is False and pip == []


def test_rftrack_is_pip_installed_into_this_python(pip, monkeypatch):
    monkeypatch.setattr(fc.importlib.util, "find_spec", lambda name: None)
    assert fc.fetch_rftrack() is True
    assert pip == [[fc.sys.executable, "-m", "pip", "install", "RF-Track"]]
    pip.returncode = 1
    with pytest.raises(RuntimeError, match="pip install RF-Track"):
        fc.fetch_rftrack()
