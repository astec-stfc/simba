from simba import Framework as fw
from simba.Support_Files.tempdir import TemporaryDirectory
import re
from warnings import warn

beam_evaluate = (
    "sigma_x",
    "sigma_y",
    "sigma_t",
    "sigma_z",
    "sigma_cp",
    "linear_chirp_z",
    "beta_x",
    "beta_y",
    "alpha_x",
    "alpha_y",
    "peak_current",
    "enx",
    "eny",
    "mean_energy",
    "momentum_spread",
)

def xopt_optimisation(
        settings: dict,
        directory: str,
        settings_file: str,
        start_lattice: str | None = None,
        end_lattice: str | None = None,
        prefix: list | None = None,
        params: list=beam_evaluate,
        sample_interval: int = 1,
        **kwargs,
    ):
    """
    Optimisation function for use with xopt.

    Parameters:
    -----------
    settings : dict
        Variables from the Xopt `VOCS`, i.e. parameters to be changed.
        The keys in this dictionary are formatted as `elem:param` with `{elem}` the name of the element
        and `param` the attribute to be changed. The values in the dictionary are the upper and lower bounds of `param`.
    directory : str
        The root framework run directory. Each iteration of the optimisation will produce a subdirectory.
    settings_file : str
        The .def file in MasterLattice/Lattices/
    start_lattice : Optional[str]
        The starting lattice line
    end_lattice : Optional[str]
        The ending lattice line
    prefix: Optional[list]
        Used for `framework.set_lattice_prefix(prefix[0], prefix[1])`, with [0] the starting line and [1] the location
        of an existing beam file at the start of that section. `prefix[0]` will overwrite `start_lattice` if
        they are different.
    params: list = beam_evaluate
        List of attributes of the `beam` objects in the line. These are possible variables to be optimised at
        every point along the beamline where a `beam` is dumped. Can be customised to be any `float` attribute
        available to `beam`.

    Returns:
    -----------
    dict
        A dictionary of `elem:param : val` with `elem` the `beam` file names, and `param` in `params`
    """
    if directory is None:
        raise ValueError("directory must be provided.")
    else:
        tmpdir = TemporaryDirectory(directory)
        directory = tmpdir.__enter__()
        framework = fw.Framework(
            directory=directory,
            **kwargs,
        )
    if settings_file is None:
        raise ValueError("settings_file must be provided.")
    else:
        framework.loadSettings(settings_file)
    startfile = start_lattice if start_lattice is not None else framework.lines[0]
    endfile = end_lattice if end_lattice is not None else framework.lines[-1]
    if "code" in settings:
        if isinstance(settings["code"], str):
            framework.change_Lattice_Code("All", settings["code"])
        elif isinstance(settings["code"], dict):
            for line, code in settings["code"].items():
                if line == "generator":
                    framework.change_generator(code)
                else:
                    framework.change_Lattice_Code(line, code)
        else:
            raise ValueError("settings['code'] must be a string or a dictionary.")
    if isinstance(prefix, list):
        if len(prefix) == 2 and isinstance(prefix[0], str) and isinstance(prefix[1], str):
            if prefix[0] != start_lattice:
                warn(f"start_lattice is different to prefix[0], using {prefix[0]} as start_lattice.")
            framework.set_lattice_prefix(prefix[0], prefix[1])
            startfile = prefix[0]
    if isinstance(sample_interval, int) and isinstance(start_lattice, str):
        framework.set_lattice_sample_interval(start_lattice, sample_interval)
    for elem, val in settings.items():
        name, param = elem.split(':')
        if name == "generator":
            setattr(framework.generator, name, val)
        elif (name in framework.elements) or (name in framework.groups):
            framework.modifyElement(name, param, val)
    framework.track(startfile=startfile, endfile=endfile)
    fwdir = fw.load_directory(directory, beams=True)
    data = {}
    for index in range(len(fwdir.beams)):
        beam = fwdir.beams[index]
        scr = re.split(r' |/|\\', beam.filename)[-1].split('.')[0]
        for param in params:
            data.update({f'{scr}:{param}': float(getattr(beam, param))})
    return data
