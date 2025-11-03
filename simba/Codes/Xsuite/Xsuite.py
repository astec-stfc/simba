"""
SIMBA Xsuite Module

Various objects and functions to handle Xsuite lattices and commands. See `Xsuite github`_ for more details.

    .. Xsuite github: https://github.com/xsuite

Classes:
    - :class:`~simba.Codes.Xsuite.Xsuite.xsuiteLattice`: The Xsuite lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an Xsuite lattice object,
    and for tracking through it.

"""
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

from ...Framework_objects import frameworkLattice, getGrids
from ...Modules import Beams as rbf
from copy import deepcopy
import numpy as np
import json

from typing import Dict, List, Any, Literal


class xsuiteLattice(frameworkLattice):
    """
    Class for defining the Xsuite lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an Xsuite lattice object,
    and for tracking through it.
    """

    code: str = "xsuite"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam.
    If False, run in single-particle mode; the beam is not tracked and the output beam distributions
    will be generated from Gaussians."""

    names: List = None
    """Names of elements in the lattice"""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    final_screen: Any = None
    """Final screen object"""

    env: Any = None
    """Xsuite environment object"""

    line: Any = None
    """Xsuite line object"""

    pin: Any | None = None
    """Xsuite input particle distribution"""

    pout: Any | None = None
    """Xsuite output particle distribution"""

    tws: Any | None = None
    """Xsuite Twiss Table output"""

    matrices: Dict | None = None
    """Dictionary of R-matrices produced by tracking"""

    context: Any | None = None
    """Xsuite context object"""

    beam_data: Dict = {}
    """Data containing beam statistics at each element"""

    grids: getGrids = None
    """Class for calculating the required number of space charge grids"""

    pic_solver: Literal['FFTSolver2p5DAveraged'] = 'FFTSolver2p5DAveraged'
    """PIC solver to use for space charge calculations"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        import xobjects as xo
        if (
            "input" in self.file_block
            and "particle_definition" in self.file_block["input"]
        ):
            if (
                self.file_block["input"]["particle_definition"]
                == "initial_distribution"
            ):
                self.particle_definition = "laser"
            else:
                self.particle_definition = self.file_block["input"][
                    "particle_definition"
                ]
        else:
            self.particle_definition = self.start
        if has_cupy:
            self.context = xo.ContextCupy()
        else:
            self.context = xo.ContextCpu()
        self.grids = getGrids()


    def writeElements(self) -> None:
        """
        Create Xsuite objects for all the elements in the lattice and set the
        :attr:`~simba.Codes.Xsuite.Xsuite.xsuiteLattice.lat_obj` and
        :attr:`~simba.Codes.Xsuite.Xsuite.xsuiteLattice.names`.
        """
        import xtrack as xt
        self.env = xt.Environment()
        particle_ref = xt.Particles(
            p0c=[self.global_parameters["beam"].centroids.mean_cp.val],
            mass0=[self.global_parameters["beam"].particle_rest_energy_eV.val],
            q0=-1,
            zeta=0.0,
        )
        beam_length = len(self.global_parameters["beam"].x.val)
        self.line = self.section.to_xsuite(beam_length, env=self.env, particle_ref=particle_ref, save=True)
        self.names = self.line.element_names

    def setup_collective_effects(self) -> None:
        import xfields as xf
        if "charge" in list(self.file_block.keys()):
            if (
                "space_charge_mode" in list(self.file_block["charge"].keys())
                and self.file_block["charge"]["space_charge_mode"].lower() == "3d"
            ):
                be = self.global_parameters["beam"]
                gridsize = self.grids.getGridSizes(
                    (len(be.x) / self.sample_interval)
                )
                sigma_z = be.sigmas.sigma_z.val
                lprofile = xf.LongitudinalProfileQGaussian(
                    number_of_particles=len(be.x),
                    sigma_z=sigma_z,
                )
                xf.install_spacecharge_frozen(
                    line=self.line,
                    longitudinal_profile=lprofile,
                    nemitt_x=be.emittance.normalized_horizontal_emittance.val,
                    nemitt_y=be.emittance.normalized_horizontal_emittance.val,
                    sigma_z=sigma_z,
                    num_spacecharge_interactions=len(self.getNames())
                )
                pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
                    line=self.line,
                    n_sigmas_range_pic_x=8,
                    n_sigmas_range_pic_y=8,
                    nx_grid=gridsize,
                    ny_grid=gridsize,
                    nz_grid=gridsize,
                    n_lims_x=7,
                    n_lims_y=3,
                    z_range=(-3 * sigma_z, 3 * sigma_z),
                    solver=self.pic_solver)

    def write(self) -> None:
        """
        Create the lattice object via :func:`~simba.Codes.Xsuite.Xsuite.xsuiteLattice.writeElements`
        and save it as a python file to `master_subdir`.
        """
        self.writeElements()

    def preProcess(self) -> None:
        """
        Get the initial particle distribution defined in `file_block['input']['prefix']` if it exists.
        """
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.hdf5_to_json(prefix)

    def hdf5_to_json(self, prefix: str = "", write: bool = True) -> None:
        """
        Convert the initial HDF5 particle distribution to Xsuite format and set
        :attr:`~simba.Codes.Xsuite.Xsuite.xsuiteLattice.pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        self.read_input_file(prefix, self.particle_definition)
        xsuitebeamfilename = self.global_parameters["master_subdir"] + "/" + self.particle_definition + ".xsuite.json"
        self.pin = rbf.beam.write_xsuite_beam_file(
            self.global_parameters["beam"],
            xsuitebeamfilename,
            write=write,
            s_start=self.startObject.physical.start.z,
        )

    def run(self) -> None:
        """
        Run the code, and set :attr:`~tws` and :attr:`~pout`
        """
        import xtrack as xt
        from xtrack import Cavity
        self.line.build_tracker(_context=self.context)
        self.line.freeze_longitudinal(state=False)
        self.line.freeze_energy(state=False, force=True)
        pin = deepcopy(self.pin)

        for el, name in zip(self.line.elements, self.line.element_names):
            if isinstance(el, Cavity):
                xt.ReferenceEnergyIncrease(
                    Delta_p0c=el.voltage * np.sin(el.lag * np.pi / 180)
                ).track(pin)
            pin.zeta -= np.mean(pin.zeta)  # Center zeta
            el.track(pin, increment_at_element=True)  # Track in-place
            stats = {
                'mean_x': np.mean(pin.x),
                'mean_y': np.mean(pin.y),
                'sigma_x': np.std(pin.x),
                'sigma_px': np.std(pin.px),
                'sigma_y': np.std(pin.y),
                'sigma_py': np.std(pin.py),
                'sigma_zeta': np.std(pin.zeta),
                'sigma_delta': np.std(pin.delta) * np.mean(pin.energy),
                'momentum': np.mean(pin.energy) - pin.mass0,
                'emit_xn': np.mean(self.compute_norm_emit(pin.x, pin.px, pin)),
                'emit_yn': np.mean(self.compute_norm_emit(pin.y, pin.py, pin)),
            }
            self.beam_data.update({name: stats})
        self.beam_data.update({"_end_point": stats})
        self.pout = pin
        self.tws = self.line.twiss(
            betx=self.global_parameters["beam"].twiss.beta_x.val,
            alfx=self.global_parameters["beam"].twiss.alpha_x.val,
            bety=self.global_parameters["beam"].twiss.beta_y.val,
            alfy=self.global_parameters["beam"].twiss.alpha_y.val,
            compute_R_element_by_element=False,
            method="6d",
            freeze_energy=False,
        )

    def compute_norm_emit(self, coord, mom, particles):
        cov = np.cov(coord, mom)
        emit = np.sqrt(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2)
        gamma = particles.energy / particles.mass0
        beta = particles.beta0 * (1 + particles.delta.mean()) / (1 + particles.delta.mean() * particles.beta0 ** 2)
        return gamma * beta * emit

    def postProcess(self) -> None:
        """
        Convert the outputs from Ocelot to HDF5 format and save them to `master_subdir`.
        """
        import xobjects as xo
        super().postProcess()
        bfname = f'{self.global_parameters["master_subdir"]}/{self.end}.xsuite.json'
        with open(bfname, 'w') as fid:
            json.dump(self.pout.to_dict(), fid, cls=xo.JEncoder)
        beam = deepcopy(self.global_parameters["beam"])
        beam.read_xsuite_beam_file(bfname)#, s_start=self.endObject.physical.start.z)
        rbf.openpmd.write_openpmd_beam_file(beam, f'{self.global_parameters["master_subdir"]}/{self.end}.openpmd.hdf5')
        for elem in self.screens_and_bpms:
            fname = f'{self.global_parameters["master_subdir"]}/{elem.name}.xsuite.json'
            with open(fname, 'w') as fid:
                json.dump(self.line[elem.name].data.to_dict(), fid, cls=xo.JEncoder)
            beam = deepcopy(self.global_parameters["beam"])
            beam.read_xsuite_beam_file(fname)#, s_start=elem.physical.start.z)
            rbf.openpmd.write_openpmd_beam_file(beam, f'{self.global_parameters["master_subdir"]}/{elem.name}.openpmd.hdf5')
        df = self.tws.to_pandas()
        df["s"] += self.startObject.physical.start.z
        for k in self.beam_data[list(self.beam_data.keys())[0]].keys():
            df[k] = [x[k] for x in self.beam_data.values()]
        df.to_csv(f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.csv')
