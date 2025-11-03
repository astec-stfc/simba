"""
Simframe Ocelot Module

Various objects and functions to handle OCELOT lattices and commands. See `Ocelot github`_ for more details.

    .. _Ocelot github: https://github.com/ocelot-collab/ocelot

Classes:
    - :class:`~simba.Codes.Ocelot.Ocelot.ocelotLattice`: The Ocelot lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an Ocelot lattice object,
    and for tracking through it.

"""

from ...Framework_objects import frameworkLattice, getGrids
from ...Modules import Beams as rbf
from ...Modules.Fields import field
from copy import deepcopy
from numpy import array, savez_compressed, linspace, save
import os
from yaml import safe_load

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/ocelot_defaults.yaml",
    "r",
) as infile:
    oceglobal = safe_load(infile)
from typing import Dict, List, Any


class ocelotLattice(frameworkLattice):
    """
    Class for defining the OCELOT lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an Ocelot lattice object,
    and for tracking through it.
    """

    code: str = "ocelot"
    """String indicating the lattice object type"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    lat_obj: Any = None
    """Lattice object as an Ocelot `MagneticLattice`_
    
    .. _MagneticLattice: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/magnetic_lattice.py
    """

    pin: Any = None
    """Initial particle distribution as an Ocelot `ParticleArray`_
    
    .. _ParticleArray: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/beam.py"""

    pout: Any = None
    """Final particle distribution as an Ocelot `ParticleArray`_"""

    tws: List = None
    """List containing Ocelot `Twiss`_ objects
    
    .. _Twiss: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/beam.py
    """

    names: List = None
    """Names of elements in the lattice"""

    grids: getGrids = None
    """Class for calculating the required number of space charge grids"""

    oceglobal: Dict = {}
    """Global settings for Ocelot, read in from `ocelotLattice.settings["global"]["OCELOTsettings"]` and
    `ocelot_defaults.yaml`"""

    unit_step: float = 0.01
    """Step for Ocelot `PhysProc`_ objects
    
    .. _PhysProc: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/physics_proc.py
    """

    smooth_param: float = 0.01
    """Smoothing parameter"""

    lsc: bool = True
    """Flag to enable LSC calculations"""

    random_mesh: bool = True
    """Random meshing for space charge calculations"""

    nbin_csr: int = 10
    """Number of longitudinal bins for CSR calculations"""

    mbin_csr: int = 5
    """Number of macroparticle bins for CSR calculations"""

    wake_factor: float = 1.0
    """Multiplication factor for wakefields"""

    sigmamin_csr: float = 1e-5
    """Minimum size for CSR calculations"""

    wake_sampling: int = 1000
    """Number of samples for wake calculations"""

    wake_filter: int = 10
    """Filter parameter for wake calculations"""

    particle_definition: str = None
    """Initial particle distribution as a string"""

    final_screen: Any = None
    """Final screen object"""

    mbi_navi: Any | None = None
    """Physics process for calculating microbunching gain"""

    mbi: Dict = {}
    """Dictionary containing settings for microbunching gain calculation"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.oceglobal = (
            self.settings["global"]["OCELOTsettings"]
            if "OCELOTsettings" in list(self.settings["global"].keys())
            else oceglobal
        )
        cls = self.__class__
        for f in cls.model_fields:
            if f in list(self.oceglobal.keys()):
                setattr(self, f, self.oceglobal[f])
            elif f in self.file_block:
                setattr(self, f, self.file_block[f])

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
        self.grids = getGrids()

    def writeElements(self) -> None:
        """
        Create Ocelot objects for all the elements in the lattice and set the
        :attr:`~simba.Codes.Ocelot.Ocelot.ocelotLattice.lat_obj` and
        :attr:`~simba.Codes.Ocelot.Ocelot.ocelotLattice.names`.
        """
        self.lat_obj = self.section.to_ocelot(save=True)
        self.names = [str(x) for x in array([lat.id for lat in self.lat_obj.sequence])]

    def write(self) -> None:
        """
        Create the lattice object via :func:`~simba.Codes.Ocelot.Ocelot.ocelotLattice.writeElements`
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
        self.hdf5_to_npz(prefix)

    def hdf5_to_npz(self, prefix: str="", write: bool=True) -> None:
        """
        Convert the initial HDF5 particle distribution to Ocelot format and set
        :attr:`~simba.Codes.Ocelot.Ocelot.ocelotLattice.pin` accordingly.

        Parameters
        ----------
        prefix: str
            Prefix for particle file
        write: bool
            Flag to indicate whether to save the file
        """
        from ...Modules.Beams import ocelot as rbf_ocelot
        self.read_input_file(prefix, self.particle_definition)
        self.pin = rbf_ocelot.particle_group_to_parray(
            self.global_parameters["beam"],
            s_start=self.startObject.physical.start.z
        )


    def run(self) -> None:
        """
        Run the code, and set :attr:`~tws` and :attr:`~pout`
        """
        from ocelot.cpbd.track import track
        navi = self.navi_setup()
        pin = deepcopy(self.pin)
        if self.sample_interval > 1:
            pin = pin.thin_out(nth=self.sample_interval)
        self.tws, self.pout = track(
            self.lat_obj,
            pin,
            navi=navi,
            calc_tws=True,
            twiss_disp_correction=True,
        )

    def postProcess(self) -> None:
        """
        Convert the outputs from Ocelot to HDF5 format and save them to `master_subdir`.
        """
        from ocelot.cpbd.io import load_particle_array, save_particle_array
        super().postProcess()
        bfname = f'{self.global_parameters["master_subdir"]}/{self.end}.ocelot.npz'
        save_particle_array(bfname, self.pout)
        from ...Modules.Beams import ocelot as rbf_ocelot
        rbf_ocelot.particle_array_to_beam(
            self.global_parameters["beam"],
            self.pout,
            zstart=self.startObject.physical.start.z
        )
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            bfname.replace(".ocelot.npz", ".openpmd.hdf5"),
        )
        for w in self.screens_and_bpms:
            beamname = f'{self.global_parameters["master_subdir"]}/{w.name + ".ocelot.npz"}'
            if os.path.isfile(beamname):
                beam = load_particle_array(beamname)
                rbf.ocelot.particle_array_to_beam(
                    self.global_parameters["beam"],
                    beam,
                    zstart=w.physical.start.z
                )
                rbf.openpmd.write_openpmd_beam_file(
                    self.global_parameters["beam"],
                    beamname.replace(".ocelot.npz", ".openpmd.hdf5"),
                )
        twsdat = {e: [] for e in self.tws[0].__dict__.keys()}
        for t in self.tws:
            for k, v in t.__dict__.items():
                # Offset the s values to the start of the lattice
                if k == "s":
                    v += self.startObject.physical.start.z
                twsdat[k].append(v)
        savez_compressed(
            f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.npz',
            **twsdat,
        )
        if self.mbi_navi is not None:
            save(
                f'{self.global_parameters["master_subdir"]}/{self.objectname}_mbi.dat',
                self.mbi_navi.bf,
            )

    def navi_setup(self) -> "Navigator":
        """
        Set up the physics processes for Ocelot (i.e. space charge, CSR, wakes etc).

        .. _Navigator: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/navi.py

        Returns
        -------
        Navigator
            An Ocelot `Navigator`_ object
        """
        from ocelot.cpbd.navi import Navigator
        from ocelot import Twiss
        from ocelot.cpbd.physics_proc import SaveBeam
        from .mbi import MBI
        navi_processes = []
        navi_locations_start = []
        navi_locations_end = []
        # settings = self.settings
        navi = Navigator(self.lat_obj, unit_step=self.unit_step)
        if self.lsc:
            lsc = self.physproc_lsc()
            navi_processes += [lsc]
            navi_locations_start += [self.lat_obj.sequence[0]]
            navi_locations_end += [self.lat_obj.sequence[-1]]
        space_charge_set = False
        csr_set = False
        if "charge" in list(self.file_block.keys()):
            if (
                "space_charge_mode" in list(self.file_block["charge"].keys())
                and self.file_block["charge"]["space_charge_mode"].lower() == "3d"
            ):
                gridsize = self.grids.getGridSizes(
                    (len(self.global_parameters["beam"].x) / self.sample_interval)
                )
                g1 = self.sc_grid if hasattr(self, "sc_grid") else gridsize
                grids = [g1 for _ in range(3)]
                sc = self.physproc_sc(grids)
                navi_processes += [sc]
                navi_locations_start += [self.lat_obj.sequence[0]]
                navi_locations_end += [self.lat_obj.sequence[-1]]
                space_charge_set = True
        if "csr" in list(self.file_block.keys()):
            csr, start, end = self.physproc_csr()
            for i in range(len(csr)):
                navi_processes += [csr[i]]
                navi_locations_start += [start[i]]
                navi_locations_end += [end[i]]
        if self.mbi["set_mbi"]:
            self.mbi_navi = MBI(
                lattice=self.lat_obj,
                lamb_range=list(
                    linspace(
                        float(self.mbi["min"]),
                        float(self.mbi["max"]),
                        int(self.mbi["nstep"]),
                    )
                ),
                lsc=space_charge_set,
                csr=csr_set,
                slices=self.mbi["slices"],
            )
            # mbi1.step = self.unit_step
            self.mbi_navi.navi = deepcopy(navi)
            self.mbi_navi.lattice = deepcopy(self.lat_obj)
            self.mbi_navi.lsc = True
            navi.add_physics_proc(
                self.mbi_navi, self.lat_obj.sequence[0], self.lat_obj.sequence[-1]
            )
        for name, obj in self.elements.items():
            fieldstr = None
            if "cavity" in obj.hardware_type.lower():
                fieldstr = "wakefield_definition"
            elif "wake" in obj.hardware_type.lower():
                fieldstr = "field_definition"
            if fieldstr is not None:
                if getattr(obj.simulation, fieldstr) is not None:
                    wake, w_ind = self.physproc_wake(
                        name, getattr(obj.simulation, fieldstr), obj.cavity.n_cells
                    )
                    navi_processes += [wake]
                    navi_locations_start += [self.lat_obj.sequence[w_ind]]
                    navi_locations_end += [self.lat_obj.sequence[w_ind + 1]]
            if obj.hardware_type.lower() == "twissmatch":
                twsobj = Twiss(
                    beta_x=obj.simulation.beta_x,
                    beta_y=obj.simulation.beta_y,
                    alpha_x=obj.simulation.alpha_x,
                    alpha_y=obj.simulation.alpha_y,
                    Dx=obj.simulation.eta_x,
                    Dy=obj.simulation.eta_y,
                    Dxp=obj.simulation.eta_xp,
                    Dyp=obj.simulation.eta_yp,
                    )
                navi_processes += [self.physproc_beamtransform(tws=twsobj)]
                navi_locations_start += [self.lat_obj.sequence[self.names.index(name)]]
                navi_locations_end += [self.lat_obj.sequence[self.names.index(name)]]
        for w in self.screens_and_bpms:
            loc = self.lat_obj.sequence[self.names.index(w.name)]
            subdir = self.global_parameters["master_subdir"]
            navi_processes += [SaveBeam(filename=f"{subdir}/{w.name}.ocelot.npz")]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        loc = self.lat_obj.sequence[-1]
        subdir = self.global_parameters["master_subdir"]
        navi_processes += [SaveBeam(filename=f"{subdir}/{self.objectname}.ocelot.npz")]
        navi_locations_start += [loc]
        navi_locations_end += [loc]
        navi.add_physics_processes(
            navi_processes, navi_locations_start, navi_locations_end
        )
        return navi

    def physproc_lsc(self) -> "LSC":
        """
        Get an Ocelot `LSC`_ physics process

        .. LSC: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/sc.py

        Returns
        -------
        LSC
            The Ocelot LSC PhysProc
        """
        from ocelot.cpbd.sc import LSC
        lsc = LSC()
        lsc.smooth_param = self.smooth_param
        return lsc

    def physproc_sc(self, grids: List[int]) -> "SpaceCharge":
        """
        Get an Ocelot `SpaceCharge`_ physics process

        .. _SpaceCharge: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/sc.py

        Parameters
        ----------
        grids: List[int]
            The space charge grid number in x,y,z

        Returns
        -------
        SpaceCharge
            The Ocelot SpaceCharge PhysProc
        """
        from ocelot.cpbd.sc import SpaceCharge
        sc = SpaceCharge(step=1)
        sc.nmesh_xyz = grids
        sc.random_mesh = self.random_mesh
        return sc

    def physproc_csr(self) -> tuple:
        """
        Get Ocelot `CSR`_ physics processes based on the start and end positions provided in `file_block`.
        If these are not provided, just include CSR for the entire lattice.

        .. _CSR: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/csr.py

        Returns
        -------
        tuple
            A list of CSR PhysProcs, and their start and end positions
        """
        csrlist = []
        stlist = []
        enlist = []
        from ocelot.cpbd.csr import CSR
        if ("start" in list(self.file_block["csr"].keys())) and (
            "end" in list(self.file_block["csr"].keys())
        ):
            start = self.file_block["csr"]["start"]
            st = [start] if isinstance(start, str) else start
            end = self.file_block["csr"]["end"]
            en = [end] if isinstance(end, str) else end
            for i in range(len(st)):
                stelem = self.lat_obj.sequence[self.names.index(st[i])]
                enelem = self.lat_obj.sequence[self.names.index(en[i])]
                csr = CSR()
                csr.n_bin = self.nbin_csr
                csr.m_bin = self.mbin_csr
                csr.sigma_min = self.sigmamin_csr
                csrlist.append(csr)
                stlist.append(stelem)
                enlist.append(enelem)
        else:
            csr = CSR()
            csr.n_bin = self.nbin_csr
            csr.m_bin = self.mbin_csr
            csr.sigma_min = self.sigmamin_csr
            stlist = [self.lat_obj.sequence[0]]
            enlist = [self.lat_obj.sequence[-1]]
        return csrlist, stlist, enlist

    def physproc_wake(
            self,
            name: str,
            loc: field | str,
            ncell: int,
    ) -> tuple:
        """
        Get an Ocelot `Wake`_ physics process based on the wakefield provided.

        .. _Wake: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/wake.py

        Parameters
        ----------
        name: str
            Name of lattice object associated with the wake
        loc: :class:`~simba.Modules.Fields.field` or str
            If `field`, then write the field file to ASTRA format
        ncell: int
            Number of cells, which provides a multiplication factor for the wake

        Returns
        -------
        tuple
            A Wake PhysProc, and its index in the lattice
        """
        from ocelot.cpbd.wake3D import Wake, WakeTable
        if isinstance(loc, field):
            loc = loc.write_field_file(code="astra")
        subdir = self.global_parameters["master_subdir"]
        fname = subdir + '/' + os.path.basename(loc).replace('.hdf5', '.astra')
        wake = Wake(
            step=100,
            w_sampling=self.wake_sampling,
            filter_order=self.wake_filter,
        )
        wake.factor = ncell * self.wake_factor
        wake.wake_table = WakeTable(fname)
        w_ind = self.names.index(name)
        return wake, w_ind

    def physproc_beamtransform(
            self,
            tws: "Twiss",
    ) -> "BeamTransform":
        """
        Get an Ocelot `BeamTransform`_ physics process based on the wakefield provided.

        .. _Wake: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/physproc.py

        Parameters
        ----------
        tws: Ocelot `Twiss` object
            Object containing Twiss parameters

        Returns
        -------
        tuple
            A BeamTransform PhysProc
        """
        from ocelot.cpbd.physics_proc import BeamTransform
        return BeamTransform(tws=tws)
