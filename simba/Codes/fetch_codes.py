"""
Download ASTRA (with its generator) and the serial CSRTrack into a SimCodes directory, and install RF-Track:
``simba-codes fetch``.

ASTRA and CSRTrack are distributed only from DESY's user pages, which are down at times. simcodes-docker keeps
known-good copies (its ``vendor/desy``), so each file comes from DESY while DESY still serves the known-good file, and
otherwise from there. Either way it is kept only if its sha256 is the one below.

RF-Track is pip-installed from PyPI, where CERN publishes it under its own licence (free use, no redistribution).
"""
import hashlib
import http.client
import importlib.util
import os
import platform
import subprocess
import sys
import urllib.request

FETCHED_SIMCODES = os.path.join(os.path.expanduser("~"), ".local", "share", "simba", "simcodes")
"""Where ``simba-codes fetch`` puts the codes by default, and the last place :mod:`SIMBA` looks for a SimCodes directory."""

MIRROR = "https://raw.githubusercontent.com/astec-stfc/simcodes-docker/v0.3/vendor/desy/"

# path under the SimCodes directory (as the posix section of Executables.yaml has it): (DESY URL, mirror path, sha256)
FILES = {
    "ASTRA/Astra": (
        "https://www.desy.de/~mpyflo/Astra_for_64_Bit_Linux/Astra",
        "astra-serial/Astra",
        "6a2fea8b1f654467f74dcdf01ba1c9cdc1e8dff3e0e8e86ed658c4d2a8713741",
    ),
    "ASTRA/generator": (
        "https://www.desy.de/~mpyflo/Astra_for_64_Bit_Linux/generator",
        "astra-serial/generator",
        "b3c4aa0456be107e6af656705b6d2fa959ed5772e85368e3b056fa75c8b7279e",
    ),
    "CSRTrack/csrtrack_1.204_Linux_x86_64_serial": (
        "https://www.desy.de/xfel-beam/csrtrack/files/csrtrack_1.204_Linux_x86_64_serial",
        "csrtrack-serial/csrtrack_1.204_Linux_x86_64_serial",
        "30c79fa664e6c97256052782895aa5d1109dbb64efeebdda18c4602ac8d5d0f6",
    ),
}


def _download(url: str) -> bytes | None:
    """The file at ``url``, or None if it cannot be had."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except (OSError, http.client.HTTPException):  # URLError, HTTPError, timeouts; a connection dropped half-way
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_rftrack() -> bool:
    """
    pip-install RF-Track into the running Python unless it is already importable.

    Returns
    -------
    bool
        True if it was installed.
    """
    if importlib.util.find_spec("RF_Track") is not None:
        print("ok      RF-Track (already installed)")
        return False
    print("installing RF-Track from PyPI; CERN's licence allows free use but not redistribution "
          "(https://gitlab.cern.ch/rf-track)", flush=True)
    if subprocess.run([sys.executable, "-m", "pip", "install", "RF-Track"]).returncode != 0:
        raise RuntimeError(f"Could not pip-install RF-Track into {sys.executable}; install it with `pip install RF-Track`")
    return True


def fetch_codes(simcodes: str = FETCHED_SIMCODES, desy: bool = True, rftrack: bool = True) -> list[str]:
    """
    Install RF-Track, and put ASTRA, its generator and the serial CSRTrack under ``simcodes``, where :mod:`SIMBA`
    looks for them.

    A file that is already there is left alone: kept if it is the known-good file, and kept with a message if it is
    not (it may be a version of your own).

    Parameters
    ----------
    simcodes : str
        The SimCodes directory.
    desy : bool
        Try DESY first; if False, take the simcodes-docker copies straight away.
    rftrack : bool
        Install RF-Track (on any platform; ASTRA and CSRTrack are Linux x86-64 only).

    Returns
    -------
    list[str]
        The files written, and "RF-Track" if it was installed.
    """
    written = ["RF-Track"] if rftrack and fetch_rftrack() else []
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError(
            "ASTRA and CSRTrack can only be fetched as Linux x86-64 binaries; on this machine use "
            "container_runtime='docker' or 'apptainer' (see the SimCodes page of the documentation)"
        )
    for rel, (url, mirror, sha) in FILES.items():
        path = os.path.join(simcodes, rel)
        if os.path.exists(path):
            with open(path, "rb") as f:
                same = _sha256(f.read()) == sha
            print(f"{'ok     ' if same else 'kept   '} {path}" + ("" if same else " (not the known-good file; remove it to fetch)"))
            continue
        data = _download(url) if desy else None
        source = "DESY"
        if data is None or _sha256(data) != sha:
            why = "not tried" if not desy else "not reachable" if data is None else "sends another file (a new version?)"
            source = f"the simcodes-docker copy (DESY {why})"
            data = _download(MIRROR + mirror)
        if data is None or _sha256(data) != sha:
            raise RuntimeError(f"Could not download {rel} with sha256 {sha} from DESY ({url}) or {MIRROR + mirror}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".download", "wb") as f:
            f.write(data)
        os.chmod(path + ".download", 0o755)
        os.replace(path + ".download", path)
        print(f"fetched {path} from {source}")
        written.append(path)
    return written
