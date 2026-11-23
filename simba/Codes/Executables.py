import socket
import os
import shutil
import yaml
import subprocess
import sys
from typing import Literal
import platform

def which(program):
    return shutil.which(program)

USER_EXECUTABLES = os.path.join(os.path.expanduser("~"), ".config", "simba", "executables.yaml")
"""Entries here (or in the file ``SIMBA_EXECUTABLES`` names) replace those of the same section and code in ``Executables.yaml``."""

CONTAINER_RUNTIMES = ("docker", "apptainer")
LAUNCHERS = ("mpiexec", "mpirun", "srun", "salloc")
LIBRARIES = (".so", ".dylib", ".dll")  # libtao, which pytao loads: a file, never looked for on PATH


def _program_index(cmd: list) -> int:
    """Where the program is in a command: the first word after the last shell ``;``, or, after an MPI or Slurm launcher,
    the last word before any ``>`` redirection."""
    start = max((i + 1 for i, s in enumerate(cmd) if s == ";"), default=0)
    if os.path.basename(str(cmd[start])).removesuffix(".exe") not in LAUNCHERS:
        return start
    end = next((i for i in range(start, len(cmd)) if cmd[i] == ">"), len(cmd))
    return end - 1


def is_found(cmd) -> bool:
    """Whether the program of a command exists, as a path or on ``PATH``."""
    if not isinstance(cmd, list) or not cmd:
        return False
    program = cmd[_program_index(cmd)]
    if isinstance(program, str) and program.endswith(LIBRARIES):
        return os.path.isfile(program)
    return isinstance(program, str) and which(program) is not None


def resolve_on_path(cmd, name: str):
    """
    ``cmd`` if its program exists; otherwise ``cmd`` with the program looked up on ``PATH``, by its file name and then
    by ``name``, without the ``source ... ;`` set-up of the install that is not there. ``cmd`` unchanged if neither
    is on ``PATH``.
    """
    if not isinstance(cmd, list) or not cmd or is_found(cmd):
        return cmd
    i = _program_index(cmd)
    if not isinstance(cmd[i], str):
        return cmd
    if cmd[i].endswith(LIBRARIES):
        stem = os.path.splitext(os.path.basename(cmd[i]))[0]
        for found in (os.path.join(sys.prefix, "lib", stem + ext) for ext in LIBRARIES):
            if os.path.isfile(found):
                return cmd[:i] + [found] + cmd[i + 1:]
        return cmd
    start = max((j + 1 for j, s in enumerate(cmd) if s == ";"), default=0)
    for candidate in dict.fromkeys((os.path.basename(cmd[i]), name)):
        found = which(candidate)
        if found:
            return cmd[start:i] + [found] + cmd[i + 1:]
    return cmd


def available_runtime() -> str | None:
    """The container runtime ``container_runtime="auto"`` uses: Docker if its daemon answers, else Apptainer."""
    if which("docker"):
        try:
            if subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0:
                return "docker"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "apptainer" if which("apptainer") else None

def default_sif_path(filename: str = "simcodes-apptainer_v0.3.sif") -> str:  # tag as in Executables.yaml
    system = platform.system()
    if system == "Linux":
        return os.path.join(os.path.expanduser("~"), ".local", "share", "apptainer", filename)
    elif system == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "apptainer", filename)
    else:
        return os.path.join(os.path.expanduser("~"), ".local", "share", "apptainer", filename)

SIMCODES_SIF = default_sif_path()

WINDOWS_UNSUPPORTED_CODES = ("opal", "genesis")

def container_user() -> str:
    """
    Get the ``uid:gid`` to run a container as, so files it writes to a mount are not owned by root.

    Returns
    -------
    str
        ``uid:gid`` on POSIX, otherwise (Windows) an empty string.
    """
    if hasattr(os, "getuid"):
        return f"{os.getuid()}:{os.getgid()}"
    return ""

def ensure_image(
    runtime: Literal["docker", "apptainer"],
    image: str,
    build_context: str | None = None,
    sif: str | None = None,
    simcodes_location: str | None = None,
) -> None:
    """
    Pull (or, for Docker, build) the container image if it is not available locally.

    Parameters
    ----------
    runtime : str
        'docker' or 'apptainer'.
    image : str
        Image name; the pull source for both runtimes.
    build_context : str, optional
        Directory with a Dockerfile; if given, a missing image is built rather than pulled. Docker only.
    sif : str, optional
        Apptainer .sif path; defaults to ``SIMCODES_SIF``. Apptainer only.
    simcodes_location : str, optional
        Unused; callers substitute ``$simcodes$`` in ``sif`` themselves.
    """
    if runtime == "docker":
        result = subprocess.run(
            ['docker', 'image', 'inspect', image],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"Docker image '{image}' found locally.")
            return
        if build_context is not None:
            print(f"Image '{image}' not found. Building from '{build_context}'...")
            subprocess.run(
                ['docker', 'build', '-t', image, build_context],
                check=True
            )
        else:
            print(f"Image '{image}' not found locally. Pulling from registry...")
            subprocess.run(['docker', 'pull', image], check=True)

    elif runtime == "apptainer":
        if isinstance(sif, str) and os.path.isfile(sif):
            print(f"Apptainer image found at '{sif}'.")
            return
        log = f"Apptainer .sif not found at '{sif}'." if isinstance(sif, str) else "Apptainer .sif path not provided."
        if not isinstance(sif, str):
            sif = SIMCODES_SIF
        os.makedirs(os.path.dirname(sif), exist_ok=True)
        print(f"{log} Pulling from '{image}' to {sif}...")
        subprocess.run(
            ['apptainer', 'pull', sif, f'oras://{image}'],
            check=True
        )

    else:
        raise ValueError(f"Unknown container runtime '{runtime}'. Use 'docker' or 'apptainer'.")

class executable:

    def __init__(
            self,
            name: str,
            settings: dict={},
            location: str | None = None,
            ncpu: int = 1,
            default: str | list = "",
            override_location: str = None,
    ):
        self.name = name
        self.settings = settings
        self.location = location
        self.ncpu = ncpu
        if location is not None:
            if isinstance(location, str):
                if location in self.settings and name in self.settings[location]:
                    self.executable = self._substitute_variables(
                        self.settings[location][name]
                    )
                else:
                    self.executable = self._substitute_variables([location])
            elif isinstance(location, list):
                self.executable = self._substitute_variables(location)
        else:
            hostname = socket.gethostname()
            for section in (hostname, hostname.split(".")[0], override_location, os.name):
                if section in self.settings and name in self.settings[section]:
                    self.executable = self._substitute_variables(
                        self.settings[section][name]
                    )
                    break
            else:
                section = None
            if section is None:
                self.executable = self._substitute_variables(default)
            if override_location is None or section != override_location:  # a remote host's paths are not ours to check
                self.executable = resolve_on_path(self.executable, name)
        self.found = is_found(self.executable)

    def _substitute_sif(self, param):
        if isinstance(param, list):
            return [self._substitute_sif(s) for s in param]
        else:
            return param.replace("$sif$", self.settings.get("apptainer", {}).get("sif", ""))

    def _substitute_user(self, param):
        """Substitute ``$user$`` with ``uid:gid``, or drop ``--user $user$`` where there is none (Windows)."""
        user = container_user()
        if user:
            return [s.replace("$user$", user) if isinstance(s, str) else s for s in param]
        return [
            s
            for i, s in enumerate(param)
            if s != "$user$" and not (s == "--user" and param[i + 1: i + 2] == ["$user$"])
        ]

    def _substitute_variables(self, param):
        if isinstance(param, list):
            return [self._substitute_variables(s) for s in self._substitute_user(param)]
        else:
            return (
                self._substitute_ncpu(
                    self._substitute_simcodes(
                        self._substitute_sif(
                            self._substitute_image(param)
                        )
                    )
                )
            )

    def _substitute_simcodes(self, param):
        if isinstance(param, list):
            return [self._substitute_simcodes(s) for s in param]
        else:
            return param.replace("$simcodes$", self.settings["sim_codes_location"])

    def _substitute_ncpu(self, param):
        if isinstance(param, list):
            return [self._substitute_ncpu(s) for s in param]
        else:
            return param.replace("$ncpu$", str(self.ncpu))

    def _substitute_image(self, param):
        if isinstance(param, list):
            return [self._substitute_image(s) for s in param]
        else:
            return param.replace("$image$", self.settings.get("docker", {}).get("image", ""))


class Executables:
    """
    The code executables in :download:`Executables <../../simba/Executables.yaml>` for this machine.

    Entries exist for Windows, POSIX and some Daresbury clusters; users can add others.
    Each ``define_<code>_command`` sets the attribute named after the code and takes:

    location: str, optional
        Location of the executable; overrides the default under ``SimCodes``.
    ncpu: int
        Number of CPUs to run.
    scaling: int, optional
        See :meth:`getNCPU`.
    override_location: str, optional
        Name of a remote server, defined in ``Executables.yaml``, to run on.
    """

    def __init__(self, global_parameters):
        super().__init__()
        self.global_parameters = global_parameters
        sim_codes = self.global_parameters.get("simcodes_location")
        if sim_codes is None:
            self.sim_codes_location = (
                os.path.relpath(
                    os.path.dirname(os.path.abspath(__file__)) + "/../SimCodes/SimCodes"
                )
                + "/"
            ).replace("\\", "/")
        else:
            self.sim_codes_location = sim_codes
        with open(
            os.path.join(os.path.dirname(__file__), "../Executables.yaml")
        ) as file:
            self.settings = yaml.load(file, Loader=yaml.Loader)
        user_file = os.environ.get("SIMBA_EXECUTABLES", USER_EXECUTABLES)
        if os.path.isfile(user_file):
            with open(user_file) as file:
                for section, entries in (yaml.safe_load(file) or {}).items():
                    self.settings.setdefault(section, {}).update(entries or {})
        runtime = global_parameters.get("container_runtime", None)  # 'docker', 'apptainer', 'auto' or None
        self.auto = runtime == "auto"
        self.runtime = available_runtime() if self.auto else runtime
        self._image_ready = False
        if runtime in CONTAINER_RUNTIMES:
            self._ensure_image(runtime)
        self.ASTRAgenerator = None
        self.astra = None
        self.elegant = None
        self.gpt = None
        self.csrtrack = None
        self.genesis = None
        self.opal = None
        self.tao = None
        self.madx = None
        self.bdsim = None
        self.ptarmigan = None
        self.settings["sim_codes_location"] = self.sim_codes_location
        section = None if self.auto else self.runtime
        for define in (
            self.define_ASTRAgenerator_command, self.define_astra_command, self.define_elegant_command,
            self.define_csrtrack_command, self.define_gpt_command, self.define_opal_command,
            self.define_genesis_command, self.define_tao_command, self.define_madx_command,
            self.define_bdsim_command, self.define_ptarmigan_command,
        ):
            define(override_location=section)

    def _ensure_image(self, runtime: str) -> None:
        """Pull the image of ``runtime`` (or build it, see :func:`ensure_image`) unless it is here already."""
        if runtime == "docker":
            ensure_image(
                runtime="docker",
                image=self.settings.get("docker", {}).get("image", ""),
                build_context=self.global_parameters.get("docker_build_context", None),
                simcodes_location=self.sim_codes_location,
            )
        else:
            apptainer_cfg = self.settings.get("apptainer", {})
            ensure_image(
                runtime="apptainer",
                image=apptainer_cfg.get("registry", ""),
                sif=apptainer_cfg.get("sif", "").replace("$simcodes$", self.sim_codes_location),
                simcodes_location=self.sim_codes_location,
            )
        self._image_ready = True

    def routes(self) -> list[tuple[str, str, str]]:
        """``(code, route, program)`` for every code: the route is ``docker``, ``apptainer``, ``native`` or ``not found``."""
        rows = []
        for attr, exe in vars(self).items():
            if not attr.endswith("Executable"):
                continue
            cmd = exe.executable
            if isinstance(cmd, list) and cmd and cmd[0] in CONTAINER_RUNTIMES:
                route = cmd[0]
                image_last = cmd[-1] == self.settings.get("docker", {}).get("image") and "--entrypoint" in cmd
                program = cmd[cmd.index("--entrypoint") + 1] if image_last else cmd[-1]
            else:
                route = "native" if exe.found else "not found"
                program = cmd[_program_index(cmd)] if isinstance(cmd, list) and cmd else cmd
                program = (exe.found and which(program)) or program
            rows.append((exe.name, route, str(program)))
        return rows

    def __getitem__(self, item):
        if os.name == "nt" and self.runtime is None and item in WINDOWS_UNSUPPORTED_CODES:
            raise RuntimeError(
                f"'{item}' has no native Windows build, so it cannot be run by a Python "
                "process on Windows itself. Either run SIMBA from within WSL, or "
                "instantiate it with container_runtime='docker'. "
                "See docs/source/SimCodes.rst for both routes."
            )
        return getattr(self, item)

    def build_command(self, cmd: list, workdir: str) -> list:
        """
        Substitute ``$workdir$`` in a container command, first pulling the image if this is the first command that needs it.

        Parameters
        ----------
        cmd : list
            Command and arguments.
        workdir : str
            Working directory to mount in the container.

        Returns
        -------
        list
            The command with ``$workdir$`` substituted.
        """
        if self.runtime is None:
            return cmd
        if cmd and cmd[0] == self.runtime and not self._image_ready:
            self._ensure_image(self.runtime)
        return [
            s.replace('$workdir$', workdir) if isinstance(s, str) else s
            for s in cmd
        ]

    def getNCPU(
            self,
            ncpu: int,
            scaling: int,
    ) -> int:
        """
        Get the number of CPUs for tracking.

        Parameters
        ----------
        ncpu : int
            Requested CPUs.
        scaling : int
            Particle-number scaling; if given and ``ncpu`` is 1, ``3 * scaling`` CPUs are used.

        Returns
        -------
        int
            Number of CPUs to run.
        """
        if scaling is not None and ncpu == 1:
            return 3 * scaling
        else:
            return ncpu

    def _define(
            self,
            attr: str,
            name: str,
            default: list,
            location: str | None,
            ncpu: int,
            override_location: str | None,
    ) -> None:
        """Build :class:`executable` ``name``; keep it as ``<attr>Executable`` and its command as ``attr``."""
        exe = executable(
            name,
            settings=self.settings,
            location=location,
            ncpu=ncpu,
            default=default,
            override_location=override_location,
        )
        if self.auto and self.runtime and location is None and not exe.found and name in self.settings.get(self.runtime, {}):
            exe = executable(name, settings=self.settings, location=self.runtime, ncpu=ncpu, default=default)
        setattr(self, f"{attr}Executable", exe)
        setattr(self, attr, exe.executable)

    def define_ASTRAgenerator_command(self, location=None, override_location=None) -> None:
        """Define the ASTRA generator executable and set :attr:`~ASTRAgenerator`."""
        default = [self.sim_codes_location + "ASTRA/generator"]
        self._define("ASTRAgenerator", "astragenerator", default, location, 1, override_location)

    def define_astra_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the ASTRA executable and set :attr:`~astra`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "ASTRA/astra"]
        self._define("astra", "astra", default, location, ncpu, override_location)

    def define_elegant_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the ELEGANT executable, Pelegant if `ncpu` > 1, and set :attr:`~elegant`."""
        ncpu = self.getNCPU(ncpu, scaling)
        if ncpu > 1:
            np_ = str(min([2, int(ncpu / 3)]))
            default = [which("mpiexec.exe"), "-np", np_, which("Pelegant.exe")]
            self._define("elegant", "Pelegant", default, location, ncpu, override_location)
        else:
            default = [self.sim_codes_location + "Elegant/elegant"]
            self._define("elegant", "elegant", default, location, ncpu, override_location)

    def define_csrtrack_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the CSRTrack executable and set :attr:`~csrtrack`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "CSRTrack/csrtrack"]
        self._define("csrtrack", "csrtrack", default, location, ncpu, override_location)

    def define_gpt_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the GPT executable and set :attr:`~gpt`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "GPT/gpt.exe", "-j", str(ncpu)]
        self._define("gpt", "gpt", default, location, ncpu, override_location)

    def define_opal_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the OPAL executable and set :attr:`~opal`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "OPAL/bin/opal"]
        self._define("opal", "opal", default, location, ncpu, override_location)

    def define_genesis_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the Genesis executable and set :attr:`~genesis`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "Genesis/genesis4"]
        self._define("genesis", "genesis", default, location, ncpu, override_location)

    def define_madx_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the MAD-X executable and set :attr:`~madx`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "MADX/madx"]
        self._define("madx", "madx", default, location, ncpu, override_location)

    def define_bdsim_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the BDSIM executable and set :attr:`~bdsim`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "BDSIM/bdsim-build/bin/bdsim"]
        self._define("bdsim", "bdsim", default, location, ncpu, override_location)

    def define_ptarmigan_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the Ptarmigan executable and set :attr:`~ptarmigan`: ``ptarmigan`` on the
        ``PATH`` unless ``Executables.yaml`` says otherwise for this host."""
        ncpu = self.getNCPU(ncpu, scaling)
        self._define("ptarmigan", "ptarmigan", ["ptarmigan"], location, ncpu, override_location)

    def define_tao_command(self, location=None, ncpu=1, scaling=None, override_location=None) -> None:
        """Define the Tao library and set :attr:`~tao`."""
        ncpu = self.getNCPU(ncpu, scaling)
        default = [self.sim_codes_location + "Bmad/lib/libtao.so"]
        self._define("tao", "tao", default, location, ncpu, override_location)


def main(argv=None) -> None:
    """``simba-codes check``: how SIMBA would run each external code here; ``simba-codes fetch``: install RF-Track,
    and download ASTRA and CSRTrack."""
    import argparse
    from .fetch_codes import FETCHED_SIMCODES, fetch_codes

    parser = argparse.ArgumentParser(prog="simba-codes", description="The external codes SIMBA runs.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="how SIMBA would run each code here, and whether it is found")
    check.add_argument(
        "--simcodes", default=os.environ.get("SIMCODES"),
        help=f"SimCodes directory (default: $SIMCODES, else {FETCHED_SIMCODES} if it exists)",
    )
    check.add_argument(
        "--container-runtime", choices=CONTAINER_RUNTIMES + ("auto",),
        help="as Framework's container_runtime; docker and apptainer pull the image if it is missing",
    )
    fetch = commands.add_parser(
        "fetch", help="install RF-Track, and download ASTRA, its generator and the serial CSRTrack (Linux x86-64)"
    )
    fetch.add_argument(
        "--simcodes", default=os.environ.get("SIMCODES") or FETCHED_SIMCODES,
        help=f"where to put ASTRA and CSRTrack (default: $SIMCODES, else {FETCHED_SIMCODES})",
    )
    fetch.add_argument("--no-desy", action="store_true", help="take the simcodes-docker copies without trying DESY")
    fetch.add_argument("--no-rftrack", action="store_true", help="do not pip-install RF-Track")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        fetch_codes(args.simcodes, desy=not args.no_desy, rftrack=not args.no_rftrack)
        return
    simcodes = args.simcodes or (FETCHED_SIMCODES if os.path.isdir(FETCHED_SIMCODES) else None)
    simcodes = os.path.join(os.path.abspath(simcodes), "") if simcodes else None
    ex = Executables({"simcodes_location": simcodes, "container_runtime": args.container_runtime})
    for code, route, program in ex.routes():
        print(f"{code:16} {route:10} {program}")


if __name__ == "__main__":
    main()