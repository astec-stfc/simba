import os
import sys
import pytest
from unittest.mock import patch
from simba.Codes.Executables import Executables


@patch("simba.Codes.Executables.ensure_image", lambda **kwargs: None)
def test_docker_runtime_resolves_full_command(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path), "container_runtime": "docker"})
    assert ex.astra[0] == "docker"
    assert "/simcodes/Astra-serial/Astra" in ex.astra
    assert "/simcodes/elegant/bin/Linux-x86_64/elegant" in ex.elegant
    assert "/simcodes/CSRTrack-serial/csrtrack_1.204_Linux_x86_64_serial" in ex.csrtrack
    assert "/simcodes/OPAL-install/bin/opal" in ex.opalExecutable.executable
    assert "/simcodes/Genesis/genesis4" in ex.genesisExecutable.executable
    assert ex.astra[-1] == ex.settings["docker"]["image"]
    # runs as the host user, not root, so output files stay writable on the host -
    # but POSIX uid:gid has no Windows equivalent, so `--user` is dropped there
    if hasattr(os, "getuid"):
        expected_user = f"{os.getuid()}:{os.getgid()}"
        assert expected_user in ex.astra
        assert expected_user in ex.genesisExecutable.executable
    else:
        assert "--user" not in ex.astra
        assert "--user" not in ex.genesisExecutable.executable


@patch("simba.Codes.Executables.ensure_image", lambda **kwargs: None)
def test_apptainer_runtime_resolves_full_command(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path), "container_runtime": "apptainer"})
    assert ex.astra[0] == "apptainer"
    assert ex.astra[-1] == "/simcodes/Astra-serial/Astra"
    assert ex.opalExecutable.executable[-1] == "/simcodes/OPAL-install/bin/opal"
    assert ex.genesisExecutable.executable[-1] == "/simcodes/Genesis/genesis4"


@patch("simba.Codes.Executables.ensure_image", lambda **kwargs: None)
@patch("simba.Codes.Executables.os.name", "nt")
def test_linux_only_codes_raise_on_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")  # or this machine's elegant would stand in for the missing elegant.exe
    ex = Executables({"simcodes_location": str(tmp_path)})
    elegant_path = ex["elegant"][0]
    os.makedirs(os.path.dirname(elegant_path), exist_ok=True)
    open(elegant_path, "w").close()
    assert os.path.isfile(elegant_path)
    for code in ("opal", "genesis"):
        with pytest.raises(RuntimeError, match="WSL"):
            ex[code]
    # ...but a container runtime is a valid way out
    exd = Executables({"simcodes_location": str(tmp_path), "container_runtime": "docker"})
    assert exd["opal"][0] == "docker"
    assert exd["genesis"][0] == "docker"


@patch("simba.Codes.Executables.ensure_image", lambda **kwargs: None)
@patch("simba.Codes.Executables.socket.gethostname", lambda: "apclara2")
@patch("simba.Codes.Executables.os.name", "posix")
def test_code_missing_from_host_section_falls_back_to_default(tmp_path, monkeypatch):
    # apclara2 defines no genesis entry; this used to raise KeyError
    monkeypatch.setenv("PATH", "")  # or a genesis4 on this machine's PATH would be used instead
    ex = Executables({"simcodes_location": str(tmp_path)})
    assert ex["genesis"] == [f"{tmp_path}/Genesis/genesis4"]
    assert ex["opal"] == ["/opt/OPAL/opal"]


def test_literal_path_override_still_works(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path)})
    ex.define_elegant_command(location="/custom/path/to/elegant")
    assert ex.elegant == ["/custom/path/to/elegant"]


def _fake_codes(tmp_path, monkeypatch, *names):
    """A directory of executables `names`, alone on PATH, and no user executables file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in names:
        (bin_dir / name).write_text("#!/bin/sh\n")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SIMBA_EXECUTABLES", str(tmp_path / "none.yaml"))
    return bin_dir


@pytest.mark.skipif(os.name == "nt", reason="the posix section and executable bits")
def test_code_not_in_simcodes_is_found_on_path(tmp_path, monkeypatch):
    bin_dir = _fake_codes(tmp_path, monkeypatch, "genesis4", "opal", "Astra", "tao")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))  # with no lib/libtao.so, as a pip venv
    ex = Executables({"simcodes_location": str(tmp_path / "simcodes")})
    assert ex.genesis == [str(bin_dir / "genesis4")]  # by the file name the posix section gives
    assert ex.astra == [str(bin_dir / "Astra")]
    assert ex.opal == [str(bin_dir / "opal")]  # without sourcing the opal.sh of the install that is not there
    assert ex.csrtrack == [str(tmp_path / "simcodes" / "CSRTrack" / "csrtrack_1.204_Linux_x86_64_serial")]  # left as it is
    assert ex.tao == [str(tmp_path / "simcodes" / "Bmad" / "lib" / "libtao.so")]  # pytao loads it: not the tao program
    routes = {code: route for code, route, _ in ex.routes()}
    assert routes["genesis"] == "native" and routes["csrtrack"] == "not found"
    # a remote host's command is not checked against this machine
    ex.define_opal_command(override_location="apclara2")
    assert ex.opal == ["/opt/OPAL/opal"]
    # a conda environment's own libtao (conda-forge bmad), as pytao would find it
    (tmp_path / "env" / "lib").mkdir(parents=True)
    (tmp_path / "env" / "lib" / "libtao.so").write_bytes(b"")
    ex.define_tao_command()
    assert ex.tao == [str(tmp_path / "env" / "lib" / "libtao.so")]


@pytest.mark.skipif(os.name == "nt", reason="the posix section")
def test_user_executables_file_replaces_entries(tmp_path, monkeypatch):
    _fake_codes(tmp_path, monkeypatch)
    user = tmp_path / "executables.yaml"
    user.write_text("posix:\n  astra: [/my/astra, -v]\ndocker:\n  image: example.org/simcodes:mine\n")
    monkeypatch.setenv("SIMBA_EXECUTABLES", str(user))
    ex = Executables({"simcodes_location": str(tmp_path)})
    assert ex.astra == ["/my/astra", "-v"]
    assert ex.settings["docker"]["image"] == "example.org/simcodes:mine"
    assert ex.settings["docker"]["astra"][0] == "docker"  # the rest of the section stays


@pytest.mark.skipif(os.name == "nt", reason="the posix section and executable bits")
def test_auto_runtime_runs_only_missing_codes_in_the_container(tmp_path, monkeypatch):
    bin_dir = _fake_codes(tmp_path, monkeypatch, "elegant")
    pulled = []
    monkeypatch.setattr("simba.Codes.Executables.available_runtime", lambda: "docker")
    monkeypatch.setattr("simba.Codes.Executables.ensure_image", lambda **kwargs: pulled.append(kwargs["runtime"]))
    ex = Executables({"simcodes_location": str(tmp_path), "container_runtime": "auto"})
    assert ex.elegant == ["elegant"] and ex.elegantExecutable.found
    assert ex.astra[0] == "docker"
    assert ex.gpt[0] != "docker"  # not in the image
    assert pulled == []  # not until a code needs it
    cmd = ex.build_command(ex.astra, str(tmp_path))
    assert f"{tmp_path}:/workdir" in cmd and pulled == ["docker"]
    ex.build_command(ex.genesis, str(tmp_path))
    assert pulled == ["docker"]
    ex.define_astra_command(override_location="auto")  # as Framework.prepare_executables redefines it
    assert ex.astra[0] == "docker"
    assert bin_dir.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="the posix section")
def test_auto_runtime_without_a_container_runtime_stays_native(tmp_path, monkeypatch):
    _fake_codes(tmp_path, monkeypatch)
    monkeypatch.setattr("simba.Codes.Executables.available_runtime", lambda: None)
    ex = Executables({"simcodes_location": str(tmp_path), "container_runtime": "auto"})
    assert ex.astra == [f"{tmp_path}/ASTRA/Astra"]
    assert ex.build_command(ex.astra, str(tmp_path)) == ex.astra


class _StubExecutables(dict):
    """Minimal stand-in for `Executables`: dict lookup plus a no-op `build_command`
    (no container runtime is in play here, so the real thing would be a passthrough too)."""

    def build_command(self, cmd, workdir):
        return cmd


def _run_elegant_on_windows(tmp_path, simcodes_location):
    """Drive elegantLattice.run's `nt` branch against a stub, capturing the command."""
    from types import SimpleNamespace
    from simba import Framework_objects
    from simba.Codes.Elegant import Elegant

    captured = []
    stub = SimpleNamespace(
        remote_setup=False,
        code="elegant",
        objectname="test",
        executables=_StubExecutables(elegant=["mpiexec.exe", "-np", "4", "C:/sc/Elegant/Pelegant.exe"]),
        global_parameters={
            "simcodes_location": simcodes_location,
            "master_subdir": str(tmp_path),
        },
    )
    stub.run_command = lambda *args, **kwargs: Framework_objects.frameworkLattice.run_command(
        stub, *args, **kwargs
    )

    def fake_call(cmd, **kwargs):
        captured.append(cmd)
        return 0

    with patch.object(Elegant.os, "name", "nt"), patch.object(
        Framework_objects.subprocess, "call", fake_call
    ):
        Elegant.elegantLattice.run(stub)
    return captured[0]


def test_pelegant_windows_without_simcodes_location(tmp_path):
    # used to raise TypeError from os.path.abspath(None)
    command = _run_elegant_on_windows(tmp_path, None)
    assert "-env" not in command
    assert "RPN_DEFNS" not in command
    assert command[-1] == "test.ele"


def test_pelegant_windows_with_simcodes_location(tmp_path):
    command = _run_elegant_on_windows(tmp_path, str(tmp_path))
    assert command[1:3] == ["-env", "RPN_DEFNS"]
    assert command[3].endswith("\\Elegant\\defns.rpn")
    assert command[-1] == "test.ele"


def test_a_code_that_fails_says_so(tmp_path):
    """A code that gives up used to look like a successful run, and the failure only
    surfaced much later as a parse error on output that was never written."""
    from types import SimpleNamespace
    from simba import Framework_objects

    stub = SimpleNamespace(code="elegant", objectname="Linac")
    logfile = str(tmp_path / "Linac.log")

    def fake_call(cmd, stdout=None, **kwargs):
        stdout.write("error: No charge defined for LSC.\n")
        return 1

    with (
        patch.object(Framework_objects.subprocess, "call", fake_call),
        pytest.raises(RuntimeError, match="No charge defined for LSC"),
    ):
        Framework_objects.frameworkLattice.run_command(stub, ["elegant"], logfile)
