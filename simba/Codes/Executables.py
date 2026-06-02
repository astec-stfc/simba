import socket
import os
import yaml
import logging
import subprocess

SIMCODES_IMAGE = "ghcr.io/astec-stfc/simcodes:latest"

def which(program):
    def is_exe(filepath):
        return os.path.isfile(filepath) and os.access(filepath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def ensure_image(image: str = SIMCODES_IMAGE, build_context: str | None = None) -> None:
    """
    Check if the Docker image exists locally. If not, either pull it from
    the registry or build it from a local Dockerfile, depending on whether
    build_context is provided.
    """
    result = subprocess.run(
        ['docker', 'image', 'inspect', image],
        capture_output=True
    )
    if result.returncode == 0:
        logging.info(f"Docker image '{image}' found locally.")
        return

    if build_context is not None:
        logging.info(f"Image '{image}' not found. Building from '{build_context}'...")
        subprocess.run(
            ['docker', 'build', '-t', image, build_context],
            check=True
        )
    else:
        logging.info(f"Image '{image}' not found locally. Pulling from registry...")
        subprocess.run(
            ['docker', 'pull', image],
            check=True
        )

class executable:

    def __init__(
            self,
            name: str,
            settings: dict={},
            location: str | None = None,
            ncpu: int = 1,
            workdir: str | None = None,
            default: str | list = "",
            override_location: str = None,
    ):
        self.name = name
        self.settings = settings
        self.location = location
        self.ncpu = ncpu
        self.workdir = workdir
        if location is not None:
            if isinstance(location, str):
                self.executable = self._substitute_variables([location])
            elif isinstance(location, list):
                self.executable = self._substitute_variables(location)
        elif socket.gethostname() in self.settings:
            self.executable = self._substitute_variables(
                self.settings[socket.gethostname()][name]
            )
        elif socket.gethostname().split(".")[0] in self.settings:
            self.executable = self._substitute_variables(
                self.settings[socket.gethostname().split(".")[0]][name]
            )
        elif override_location in self.settings:
            self.executable = self._substitute_variables(
                self.settings[override_location][name]
            )
        elif os.name in self.settings:
            self.executable = self._substitute_variables(self.settings[os.name][name])
        else:
            self.executable = self._substitute_variables(default)

    def _substitute_variables(self, param):
        if isinstance(param, list):
            return [self._substitute_variables(s) for s in param]
        else:
            return self._substitute_ncpu(self._substitute_simcodes(param))

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

    def _substitute_workdir(self, param):
        if isinstance(param, list):
            return [self._substitute_workdir(s) for s in param]
        else:
            return param.replace("$workdir$", str(self.workdir))

    def _substitute_image(self, param):
        if isinstance(param, list):
            return [self._substitute_image(s) for s in param]
        else:
            return param.replace("$image$", self.settings.get("docker", {}).get("image", ""))


class Executables(object):
    """
    Class for interpreting the accelerator code executables defined in
    :download:`Executables <../Executables.yaml>` for a given computer architecture and linking
    to the `SimCodes` directory. This enables the simulation code with the lattice input file
    to be called from within the `Framework` instance.

    Executables for Windows and POSIX architectures are defined, as are executables for
    specific clusters based at Daresbury Laboratory. Others can be added by the user.
    """

    def __init__(self, global_parameters):
        super(Executables, self).__init__()
        self.global_parameters = global_parameters
        sim_codes = (
            self.global_parameters["simcodes_location"]
            if "simcodes_location" in self.global_parameters
            else None
        )
        if sim_codes is None:
            self.sim_codes_location = (
                os.path.relpath(
                    os.path.dirname(os.path.abspath(__file__)) + "/../SimCodes/SimCodes"
                )
                + "/"
            ).replace("\\", "/")
            # print('Using SimCodes at ', os.path.abspath(self.sim_codes_location))
        else:
            self.sim_codes_location = sim_codes
        # try:
        with open(
            os.path.join(os.path.dirname(__file__), "../Executables.yaml"), "r"
        ) as file:
            self.settings = yaml.load(file, Loader=yaml.Loader)
        # except:
        #     self.settings = {}
        self.use_docker = global_parameters.get("use_docker", False)
        if self.use_docker:
            self.settings["_active_location"] = "docker"
            docker_image = global_parameters.get("docker_image", SIMCODES_IMAGE)
            build_context = global_parameters.get("docker_build_context", None)
            ensure_image(docker_image, build_context)
        self.ASTRAgenerator = None
        self.astra = None
        self.elegant = None
        self.gpt = None
        self.csrtrack = None
        self.genesis = None
        self.settings["sim_codes_location"] = self.sim_codes_location
        self.define_ASTRAgenerator_command()
        self.define_astra_command()
        self.define_elegant_command()
        self.define_csrtrack_command()
        self.define_gpt_command()
        self.define_opal_command()
        self.define_genesis_command()

    def __getitem__(self, item):
        return getattr(self, item)

    def build_command(self, cmd: list, workdir: str) -> list:
        """
        If using Docker, inject the volume mount for workdir into the command.
        Otherwise return the command unchanged.

        Parameters
        ----------
        cmd: list
            List of commands as strings
        workdir: str
            Working directory to mount in Docker

        Returns
        -------
        int:
            Number of CPUs to run
        """
        if not self.use_docker:
            return cmd
        idx = cmd.index('--rm') + 1
        return cmd[:idx] + ['-v', f'{workdir}:/workdir'] + cmd[idx:]

    def getNCPU(
            self,
            ncpu: int,
            scaling: int,
    ) -> int:
        """
        Get the number of CPUs for tracking.

        Parameters
        ----------
        ncpu: int
            Number of CPUs for multi-threaded runs
        scaling: int
            Scaling factor for the number of particles

        Returns
        -------
        int:
            Number of CPUs to run
        """
        if scaling is not None and ncpu == 1:
            return 3 * scaling
        else:
            return ncpu

    def define_ASTRAgenerator_command(
            self,
            location: str | None = None
    ) -> None:
        """
        Define the ASTRA generator :class:`~executable` object and sets :attr:`~ASTRAgenerator`

        Parameters
        ----------
        location: str, optional
            Location of ASTRA generator executable; overrides `default`.
        """
        ASTRAgeneratorExecutable = executable(
            "astragenerator",
            settings=self.settings,
            location=location,
            default=[self.sim_codes_location + "ASTRA/generator"],
        )
        self.ASTRAgenerator = ASTRAgeneratorExecutable.executable

    def define_astra_command(
            self,
            location: str | None = None,
            ncpu: int = 1,
            scaling: int | None = None,
            override_location: str | None = None,
    ) -> None:
        """
        Define the ASTRA :class:`~executable` object and sets :attr:`~astra`

        Parameters
        ----------
        location: str
            Location of ASTRA executable; overrides `default`.
        ncpu: int
            Number of CPUs to run
        scaling: int, optional
            Scaling parameter for number of CPUs.
        override_location: str, optional
            Name of remote server on which to run the executable;
            must be defined in `Executables.yaml`
        """
        ncpu = self.getNCPU(ncpu, scaling)
        astraExecutable = executable(
            "astra",
            settings=self.settings,
            location=location,
            ncpu=ncpu,
            default=[self.sim_codes_location + "ASTRA/astra"],
            override_location=override_location,
        )
        self.astra = astraExecutable.executable

    def define_elegant_command(
            self,
            location: str | None = None,
            ncpu: int = 1,
            scaling: int | None = None,
            override_location: str | None = None,
    ) -> None:
        """
        Define the ELEGANT :class:`~executable` object and sets :attr:`~elegant`

        Parameters
        ----------
        location: str
            Location of ELEGANT executable; overrides `default`.
        ncpu: int
            Number of CPUs to run
        scaling: int, optional
            Scaling parameter for number of CPUs.
        override_location: str, optional
            Name of remote server on which to run the executable;
            must be defined in `Executables.yaml`
        """
        ncpu = self.getNCPU(ncpu, scaling)
        if ncpu > 1:
            elegantExecutable = executable(
                "Pelegant",
                settings=self.settings,
                location=location,
                ncpu=ncpu,
                default=[
                    which("mpiexec.exe"),
                    "-np",
                    str(min([2, int(ncpu / 3)])),
                    which("Pelegant.exe"),
                ],
                override_location=override_location,
            )
        else:
            elegantExecutable = executable(
                "elegant",
                settings=self.settings,
                location=location,
                ncpu=ncpu,
                default=[self.sim_codes_location + "Elegant/elegant"],
                override_location=override_location,
            )
        self.elegant = elegantExecutable.executable

    def define_csrtrack_command(
            self,
            location: str | None = None,
            ncpu: int = 1,
            scaling: int | None = None,
            override_location: str | None = None,
    ) -> None:
        """
        Define the CSRTrack :class:`~executable` object and sets :attr:`~csrtrack`

        Parameters
        ----------
        location: str
            Location of CSRTrack executable; overrides `default`.
        ncpu: int
            Number of CPUs to run
        scaling: int, optional
            Scaling parameter for number of CPUs.
        override_location: str, optional
            Name of remote server on which to run the executable;
            must be defined in `Executables.yaml`
        """
        ncpu = self.getNCPU(ncpu, scaling)
        csrtrackExecutable = executable(
            "csrtrack",
            settings=self.settings,
            location=location,
            ncpu=ncpu,
            default=[self.sim_codes_location + "CSRTrack/csrtrack"],
            override_location=override_location,
        )
        self.csrtrack = csrtrackExecutable.executable

    def define_gpt_command(
            self,
            location: str | None = None,
            ncpu: int = 1,
            scaling: int | None = None,
            override_location: str | None = None,
    ) -> None:
        """
        Define the GPT :class:`~executable` object and sets :attr:`~gpt`

        Parameters
        ----------
        location: str
            Location of GPT executable; overrides `default`.
        ncpu: int
            Number of CPUs to run
        scaling: int, optional
            Scaling parameter for number of CPUs.
        override_location: str, optional
            Name of remote server on which to run the executable;
            must be defined in `Executables.yaml`
        """
        ncpu = self.getNCPU(ncpu, scaling)
        gptExecutable = executable(
            "gpt",
            settings=self.settings,
            location=location,
            ncpu=ncpu,
            default=[self.sim_codes_location + "GPT/gpt.exe", "-j", str(ncpu)],
            override_location=override_location,
        )
        self.gpt = gptExecutable.executable

    def define_opal_command(
            self,
            location: str | None = None,
            ncpu: int = 1,
            scaling: int | None = None,
            override_location: str | None = None,
    ) -> None:
        """
        Define the OPAL :class:`~executable` object and sets :attr:`~opal`

        Parameters
        ----------
        location: str
            Location of OPAL executable; overrides `default`.
        ncpu: int
            Number of CPUs to run
        scaling: int, optional
            Scaling parameter for number of CPUs.
        override_location: str, optional
            Name of remote server on which to run the executable;
            must be defined in `Executables.yaml`
        """
        ncpu = self.getNCPU(ncpu, scaling)
        self.opalExecutable = executable(
            "opal",
            settings=self.settings,
            location=location,
            ncpu=ncpu,
            default=[self.sim_codes_location + "OPAL/bin/opal"],
            override_location=override_location,
        )
        self.opal = self.opalExecutable.executable

    def define_genesis_command(
            self,
            location: str | None = None,
            ncpu: int = 1,
            scaling: int | None = None,
            override_location: str | None = None,
    ) -> None:
        """
        Define the Genesis :class:`~executable` object and sets :attr:`~genesis`

        Parameters
        ----------
        location: str
            Location of Genesis executable; overrides `default`.
        ncpu: int
            Number of CPUs to run
        scaling: int, optional
            Scaling parameter for number of CPUs.
        override_location: str, optional
            Name of remote server on which to run the executable;
            must be defined in `Executables.yaml`
        """
        ncpu = self.getNCPU(ncpu, scaling)
        self.genesisExecutable = executable(
            "genesis",
            settings=self.settings,
            location=location,
            ncpu=ncpu,
            default=[self.sim_codes_location + "Genesis/genesis4"],
            override_location=override_location,
        )
        self.genesis = self.genesisExecutable.executable