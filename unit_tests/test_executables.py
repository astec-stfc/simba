import os
import pytest
from pathlib import Path
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
def test_linux_only_codes_raise_on_windows(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path)})
    # codes with a real Windows build still resolve, to a file that actually exists -
    # not hardcoded to whichever subpath `Executables.yaml`'s `nt:` section uses today
    elegant_path = Path(ex["elegant"][0])
    elegant_path.parent.mkdir(parents=True, exist_ok=True)
    elegant_path.touch()
    assert elegant_path.is_file()
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
def test_code_missing_from_host_section_falls_back_to_default(tmp_path):
    # apclara2 defines no genesis entry; this used to raise KeyError
    ex = Executables({"simcodes_location": str(tmp_path)})
    assert ex["genesis"] == [f"{tmp_path}/Genesis/genesis4"]
    assert ex["opal"] == ["/opt/OPAL/opal"]


def test_literal_path_override_still_works(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path)})
    ex.define_elegant_command(location="/custom/path/to/elegant")
    assert ex.elegant == ["/custom/path/to/elegant"]


class _StubExecutables(dict):
    """Minimal stand-in for `Executables`: dict lookup plus a no-op `build_command`
    (no container runtime is in play here, so the real thing would be a passthrough too)."""

    def build_command(self, cmd, workdir):
        return cmd


def _run_elegant_on_windows(tmp_path, simcodes_location):
    """Drive elegantLattice.run's `nt` branch against a stub, capturing the command."""
    from types import SimpleNamespace
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
    with patch.object(Elegant.os, "name", "nt"), patch.object(
        Elegant.subprocess, "call", lambda cmd, **kw: captured.append(cmd)
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
