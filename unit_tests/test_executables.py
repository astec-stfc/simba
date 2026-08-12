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


@patch("simba.Codes.Executables.ensure_image", lambda **kwargs: None)
def test_apptainer_runtime_resolves_full_command(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path), "container_runtime": "apptainer"})
    assert ex.astra[0] == "apptainer"
    assert ex.astra[-1] == "/simcodes/Astra-serial/Astra"
    assert ex.opalExecutable.executable[-1] == "/simcodes/OPAL-install/bin/opal"
    assert ex.genesisExecutable.executable[-1] == "/simcodes/Genesis/genesis4"


def test_literal_path_override_still_works(tmp_path):
    ex = Executables({"simcodes_location": str(tmp_path)})
    ex.define_elegant_command(location="/custom/path/to/elegant")
    assert ex.elegant == ["/custom/path/to/elegant"]
