"""
SIMBA Ocelot Module

Various objects and functions to handle OCELOT lattices and commands. See `Ocelot github`_ for more details.

    .. _Ocelot github: https://github.com/ocelot-collab/ocelot

Classes:
    - :class:`~simba.Codes.Ocelot.Ocelot.ocelotLattice`: The Ocelot lattice object, used for
      converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
      :class:`~simba.Framework_objects.frameworkLattice` into an Ocelot lattice object,
      and for tracking through it.

"""

from ...Framework_objects import frameworkLattice, getGrids
from ...Modules.Fields import field
from ...Modules.Twiss.ocelot import save_ocelot_twiss_hdf
from copy import deepcopy
from numpy import array, linspace, save, interp, searchsorted, clip, mean, pi
import os
from yaml import safe_load

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/ocelot_defaults.yaml",
    "r",
) as infile:
    oceglobal = safe_load(infile)
from lox.worker.thread import ScatterGatherDescriptor
from typing import Dict, List, Any, ClassVar, TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from ocelot.cpbd.beam import Twiss
    from ocelot.cpbd.navi import Navigator
    from ocelot.cpbd.physics_proc import BeamTransform
    from ocelot.cpbd.sc import LSC, SpaceCharge


class ocelotLattice(frameworkLattice):
    """
    Class for defining the OCELOT lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into an Ocelot lattice object,
    and for tracking through it.
    """

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Function for converting all screen outputs from ELEGANT into the SIMBA generic 
    :class:`~simba.Modules.Beams.beam` object and writing files"""

    code: str = "ocelot"
    """String indicating the lattice object type"""

    supports_turns: ClassVar[bool] = True
    """By looping ``cpbd.track.track`` and feeding the bunch back in.

    **Not** ``track_nturns``, despite the name: we need to track a ``ParticleArray``
    via a ``Navigator``. ``track_nturns`` takes a list of single particles and is
    the right tool for dynamic aperture, which is a different job.
    """

    supports_frequency_map: ClassVar[bool] = True
    """``cpbd.track.freq_analysis`` over the same tracked grid."""

    supports_dynamic_aperture: ClassVar[bool] = True
    """Using ``cpbd.track.track_nturns``."""

    supports_nsuperperiods: ClassVar[bool] = True
    """``track_nturns`` takes ``nsuperperiods`` directly, and the bunch loop
    in :meth:`run` repeats the sector itself."""

    supports_radiation: ClassVar[bool] = True
    """``cpbd.physics_proc.SpontanRadEffects`` with ``type="dipole"``, one per
    bend; see :meth:`physproc_radiation`."""

    supports_periodic: ClassVar[bool] = True
    """``cpbd.optics.twiss`` with ``tws0=None``, which defers to
    ``lattice.periodic_twiss``."""

    supports_programs: ClassVar[bool] = True
    """By setting the attribute on the sequence element between turns."""

    otm_convention: ClassVar[str] = "x, xp, y, yp, tau, p"
    """One-turn map convention"""

    otm_longitudinal_sign: ClassVar[int] = -1
    """``tau`` runs the other way"""

    otm_longitudinal_scale: ClassVar[int | None] = 2
    """Magnitude follows MAD-X (scales with beta**2)."""

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

    ref_s: float = None
    """Reference s position"""

    ref_idx: int = None
    """Reference particle index"""

    _s_values: Dict | None = None
    """Cached :meth:`section.get_s_values`, both ends, for
    :attr:`_s_values_section`. See :meth:`section_s_values`."""

    _s_values_section: Any = None
    """The section :attr:`_s_values` was computed for."""

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

    def section_s_values(self, at_entrance: bool) -> Dict:
        """
        ``section.get_s_values``, computed once per section.
        :meth:`navi_setup` runs once a turn per superperiod and asks for
        both ends every time.

        Keyed on the section object:
        :attr:`section` rebuilds itself when ``start``/``end`` change, and
        a new object misses the cache on its own.

        Parameters
        ----------
        at_entrance: bool
            Whether to measure each element at its entrance or its exit.

        Returns
        -------
        Dict
            Element name to s position.
        """
        section = self.section
        if self._s_values_section is not section:
            self._s_values_section = section
            self._s_values = {
                end: section.get_s_values(as_dict=True, at_entrance=end)
                for end in (True, False)
            }
        return self._s_values[at_entrance]

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
        self.read_input_file(prefix, self.particle_definition)
        if self.initial_twiss["horizontal"]["beta"]:
            self.global_parameters["beam"].beam.rematchXPlane(
                **self.initial_twiss["horizontal"]
            )
        if self.initial_twiss["vertical"]["beta"]:
            self.global_parameters["beam"].beam.rematchYPlane(
                **self.initial_twiss["vertical"]
            )
        self.ref_s = self.global_parameters["beam"].s
        self.ref_idx = self.global_parameters["beam"].reference_particle_index
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
        self.pin = rbf_ocelot.particle_group_to_parray(
            self.global_parameters["beam"],
            s_start=self.ref_s
        )

    def apply_programs(self, turn: int) -> None:
        """
        Set each programmed element's attribute for `turn`; see
        :attr:`supports_programs`.
        ``angle`` is the default attribute but is can be overridden
        by ``program.parameter`` if it is in :attr:`programs`.

        Parameters
        ----------
        turn: int
            Turn number, 1-based
        """
        for program in self.programs:
            matches = [e for e in self.lat_obj.sequence if str(e.id) == program.element]
            if not matches:
                warn(
                    f"Line '{self.objectname}' programs '{program.element}', "
                    "which is not in the Ocelot lattice. Nothing is varied."
                )
                continue
            value = program.value_at(turn)
            for element in matches:
                attribute = program.parameter or "angle"
                if not hasattr(element, attribute):
                    warn(
                        f"Line '{self.objectname}' programs '{program.element}', "
                        f"an Ocelot {type(element).__name__}, which has no "
                        f"'{attribute}' to set. Name the attribute with "
                        "'parameter:' in the program."
                    )
                    continue
                setattr(element, attribute, value)

    def run(self) -> None:
        """
        Run the code, and set :attr:`~tws` and :attr:`~pout`
        """
        from ocelot.cpbd.track import track
        pin = deepcopy(self.pin)
        if self.sample_interval > 1:
            pin = pin.thin_out(nth=self.sample_interval)
        wanted = dict(self.output_turns())
        for turn in range(1, self.turns + 1):
            key = turn if self.turns > 1 else None
            self.apply_programs(turn)
            for sector in range(1, self.nsuperperiods + 1):
                navi = self.navi_setup(
                    turn=wanted.get(key),
                    write_beams=sector == self.nsuperperiods and key in wanted,
                    beam_turn=turn,
                )
                navi.go_to_start()
                self.tws, self.pout = track(
                    self.lat_obj,
                    pin,
                    navi=navi,
                    calc_tws=True,
                    twiss_disp_correction=False,
                )
                pin = self.pout
        if self.periodic:
            self.tws = self._periodic_twiss()

    def _periodic_twiss(self) -> List:
        """
        The lattice's closed solution, replacing the tracked beam's Twiss;
        based on ``optics.twiss`` with ``tws0=None``.

        Returns
        -------
        List
            The periodic Twiss, or the tracked Twiss if no periodic solution
            exists.

        Raises
        ------
        warning
            If no periodic solution exists.
        """
        from ocelot.cpbd.optics import twiss
        periodic = twiss(self.lat_obj, tws0=None)
        if periodic is None:
            warn(
                f"Line '{self.objectname}' asks for the periodic solution, but "
                "Ocelot found none: the one-turn map is unstable, so the ring "
                "has no matched optics. Falling back to the tracked beam's "
                "Twiss, which is not the ring's."
            )
            return self.tws
        return periodic

    def read_closed_orbit(self):
        """
        Ocelot's periodic Twiss carries the orbit on its first element.
        """
        from ocelot.cpbd.optics import twiss as ocelot_twiss

        periodic = ocelot_twiss(self.lat_obj, tws0=None)
        if not periodic:
            return None
        first = periodic[0]
        return array(
            [first.x, first.xp, first.y, first.yp, 0.0, 0.0], dtype=float
        )

    def read_optics_summary(self) -> dict:
        """
        Ocelot keeps the phase advance on the Twiss objects and computes
        chromaticity separately, from the periodic solution.
        """
        from ocelot.cpbd.chromaticity import chromaticity
        from ocelot.cpbd.optics import twiss as ocelot_twiss

        periodic = ocelot_twiss(self.lat_obj, tws0=None)
        if not periodic:
            return {}
        summary = {
            "tune_x_total": float(periodic[-1].mux) / (2 * pi),
            "tune_y_total": float(periodic[-1].muy) / (2 * pi),
        }
        try:
            chrom_x, chrom_y = chromaticity(self.lat_obj, periodic[0])[:2]
            summary["chromaticity_x"] = float(chrom_x)
            summary["chromaticity_y"] = float(chrom_y)
        except Exception as error:
            warn(f"Ocelot chromaticity unavailable for {self.objectname}: {error}")
        return summary

    def read_one_turn_map(self):
        """
        ``cpbd.optics.lattice_transfer_map``, which composes the element
        maps analytically rather than tracking anything.

        Returns
        -------
        numpy.ndarray
            The 6x6 map.
        """
        from ocelot.cpbd.optics import lattice_transfer_map

        energy_gev = float(mean(self.global_parameters["beam"].energy.val)) / 1e9
        return array(lattice_transfer_map(self.lat_obj, energy_gev), dtype=float)

    def postProcess(self) -> None:
        """
        Convert the outputs from Ocelot to HDF5 format and save them to `master_subdir`.
        """
        from ocelot.cpbd.io import save_particle_array
        super().postProcess()
        twsdat = {e: [] for e in self.tws[0].__dict__.keys()}
        for t in self.tws:
            for k, v in t.__dict__.items():
                # Offset the s values to the start of the lattice
                if k == "s":
                    v += self.entrance_s
                twsdat[k].append(v)
        svals = array(self.getSValues(at_entrance=False)) + twsdat["s"][0]
        zvals = [a[-1] for a in self.getZValues()]
        twsdat['z'] = interp(twsdat["s"], svals, zvals)
        elem_names = array(
            [e.name for e in self.createDrifts().values()], dtype="U"
        )
        if len(elem_names):
            idx = clip(
                searchsorted(svals, twsdat["s"], side="left"),
                0,
                len(elem_names) - 1,
            )
            twsdat['id'] = [str(n).encode("utf-8") for n in elem_names[idx]]
        save_ocelot_twiss_hdf(
            self,
            filename=f'{self.global_parameters["master_subdir"]}/{self.objectname}_twiss.oh5',
            twiss=twsdat,
        )
        if self.mbi_navi is not None:
            save(
                f'{self.global_parameters["master_subdir"]}/{self.objectname}_mbi.dat',
                self.mbi_navi.bf,
            )

    def run_frequency_map(self) -> list:
        """
        Tune footprint over the aperture grid.

        Needs ``save_track=True``, unlike :meth:`run_dynamic_aperture`.

        The tunes come from :func:`~simba.Modules.Matrices.tune_from_trajectory`
        rather than Ocelot's ``freq_analysis``.

        Returns
        -------
        list
            ``(x, y, tune_x, tune_y)`` per surviving grid point.
        """
        from ocelot.cpbd.track import create_track_list, track_nturns

        import math

        from ...Modules.Matrices import tune_diffusion

        xs, ys = self.da_grid()
        energy_gev = float(mean(self.global_parameters["beam"].energy.val)) / 1e9
        track_list = create_track_list(xs, ys, [0.0], energy=energy_gev)
        track_list = track_nturns(
            self.lat_obj,
            self.turns,
            track_list,
            nsuperperiods=self.nsuperperiods,
            save_track=True,
            print_progress=False,
        )
        twiss = self.normalisation_twiss()
        footprint = []
        for particle in track_list:
            if particle.turn < self.turns - 1:
                continue
            tune_x, tune_y, diffusion = tune_diffusion(
                [p[0] for p in particle.p_list],
                [p[1] for p in particle.p_list],
                [p[2] for p in particle.p_list],
                [p[3] for p in particle.p_list],
                twiss=twiss,
            )
            if math.isnan(tune_x):
                continue
            footprint.append(
                (float(particle.x), float(particle.y), tune_x, tune_y, diffusion)
            )
        if not footprint:
            warn(
                f"Line '{self.objectname}': no particle survived the "
                "frequency-map scan, so there is no footprint."
            )
        self.frequency_map = footprint
        return footprint

    def run_dynamic_aperture(self) -> list:
        """
        Dynamic aperture via ``track_nturns`` over a grid of single particles.
        With no ``Aperture`` elements the aperture limit is
        Ocelot's default of +/- 1 m.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per grid point.
        """
        from ocelot.cpbd.track import create_track_list, track_nturns

        xs, ys = self.da_grid()
        energy_gev = float(mean(self.global_parameters["beam"].energy.val)) / 1e9
        track_list = create_track_list(xs, ys, [0.0], energy=energy_gev)
        track_list = track_nturns(
            self.lat_obj,
            self.turns,
            track_list,
            nsuperperiods=self.nsuperperiods,
            save_track=False,
            print_progress=False,
        )
        self.dynamic_aperture = [
            (float(p.x), float(p.y), int(p.turn)) for p in track_list
        ]
        return self.dynamic_aperture

    def track_reference_particle(self) -> dict:
        """
        One particle, recorded every turn, via ``track_nturns``.
        Launched slightly off the closed
        orbit, since a particle sitting on it has no oscillation to show.
        """
        from ocelot.cpbd.track import create_track_list, track_nturns

        orbit = self.read_closed_orbit()
        if orbit is None:
            orbit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        nudge = float((self.da_settings or {}).get("x_max", 1e-3)) / 100.0
        energy_gev = float(mean(self.global_parameters["beam"].energy.val)) / 1e9
        track_list = create_track_list(
            [orbit[0] + nudge], [orbit[2] + nudge], [0.0], energy=energy_gev
        )
        track_list = track_nturns(
            self.lat_obj,
            self.turns,
            track_list,
            nsuperperiods=self.nsuperperiods,
            save_track=True,
            print_progress=False,
        )
        if not len(track_list) or len(track_list[0].p_list) < 2:
            warn(
                f"Line '{self.objectname}': the reference particle did not "
                "survive, so there is no trajectory."
            )
            return {}
        record = track_list[0].p_list[1:]
        return {
            name: array([step[index] for step in record])
            for index, name in enumerate(("x", "px", "y", "py"))
        }

    def physproc_radiation(self) -> list:
        """
        One ``SpontanRadEffects`` per dipole, for :meth:`navi_setup`.

        ``quant_diff`` separates the two models. ``mean`` gives the
        energy loss and hence damping; only ``quantum`` adds the excitation,
        and an equilibrium emittance needs both.

        Returns
        -------
        list
            ``(process, element, radius)`` triples, empty when
            :meth:`radiation` is off.
        """
        model = self.radiation
        if model in (None, "off"):
            return []
        from ocelot.cpbd.physics_proc import SpontanRadEffects

        out = []
        for name in self.names:
            element = self.elementObjects.get(name)
            magnetic = getattr(element, "magnetic", None)
            if magnetic is None:
                continue
            try:
                angle = float(magnetic.KnL(0))
                length = float(element.physical.length or 0.0)
            except (TypeError, ValueError, AttributeError, KeyError):
                continue
            if not angle or not length:
                continue
            out.append(
                (
                    SpontanRadEffects(
                        type="dipole",
                        radius=abs(length / angle),
                        energy_loss=True,
                        quant_diff=(model == "quantum"),
                    ),
                    self.lat_obj.sequence[self.names.index(name)],
                    abs(length / angle),
                )
            )
        return out

    def navi_setup(
        self,
        turn: int | None = None,
        write_beams: bool = True,
        beam_turn: int | None = None,
    ) -> "Navigator":
        """
        Set up the physics processes for Ocelot (i.e. space charge, CSR, wakes etc).

        .. _Navigator: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/navi.py

        Parameters
        ----------
        turn: int, optional
            Number of turns; passed to :meth:`output_basename` for saving beams
        write_beams: bool, optional
            Whether to write beam files at each turn
        beam_turn: int, optional
            The turn the beam is on

        Returns
        -------
        Navigator
            An Ocelot `Navigator`_ object
        """
        from ocelot.cpbd.navi import Navigator
        from ocelot import Twiss
        from .savebeamopenpmd import SaveBeamOpenPMD
        from .mbi import MBI
        navi_processes = []
        navi_locations_start = []
        navi_locations_end = []
        # settings = self.settings
        navi = Navigator(self.lat_obj, unit_step=self.unit_step)
        if self.lsc and self.lsc_enable:
            lsc = self.physproc_lsc()
            navi_processes += [lsc]
            navi_locations_start += [self.lat_obj.sequence[0]]
            navi_locations_end += [self.lat_obj.sequence[-1]]
        space_charge_set = False
        csr_set = False
        if "charge" in list(self.file_block.keys()):
            if (
                "space_charge_mode" in list(self.file_block["charge"].keys())
                and str(self.file_block["charge"]["space_charge_mode"]).lower() == "3d"
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
        if "csr" in list(self.file_block.keys()) and self.csr_enable:
            csr, start, end = self.physproc_csr()
            for i in range(len(csr)):
                navi_processes += [csr[i]]
                navi_locations_start += [start[i]]
                navi_locations_end += [end[i]]
            csr_set = len(csr) > 0
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
            if fieldstr is not None and self.wakefield_enable:
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
        sval_in = self.section_s_values(at_entrance=True)
        sval_out = self.section_s_values(at_entrance=False)
        for bend, loc, radius in self.physproc_radiation():
            navi_processes += [bend]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        for w in (self.screens_and_bpms + self.apertures) if write_beams else []:
            if w.name == self.start:
                continue
            loc = self.lat_obj.sequence[self.names.index(w.name)]
            subdir = self.global_parameters["master_subdir"]
            navi_processes += [
                SaveBeamOpenPMD(
                    filename=(
                        f"{subdir}/"
                        f"{self.output_basename(w.name, turn=turn)}.openpmd.hdf5"
                    ),
                    global_parameters=self.global_parameters,
                    zstart=w.physical.start.z,
                    sstart=self.entrance_s + sval_in[w.name],
                    ref_idx=self.ref_idx,
                    beam_turn=beam_turn,
                )
            ]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        if write_beams:
            loc = self.lat_obj.sequence[-1]
            subdir = self.global_parameters["master_subdir"]
            navi_processes += [
                SaveBeamOpenPMD(
                    filename=(
                        f"{subdir}/"
                        f"{self.output_basename(self.names[-1], turn=turn)}"
                        ".openpmd.hdf5"
                    ),
                    global_parameters=self.global_parameters,
                    zstart=self.endObject.physical.end.z,
                    sstart=self.entrance_s + sval_out[self.end],
                    ref_idx=self.ref_idx,
                    beam_turn=beam_turn,
                )
            ]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        navi.add_physics_processes(
            navi_processes, navi_locations_start, navi_locations_end
        )
        return navi

    def physproc_lsc(self) -> "LSC":
        """
        Get an Ocelot `LSC`_ physics process

        .. _LSC: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/sc.py

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
        block = self.file_block["csr"] if "csr" in self.file_block else {}
        if ("start" in list(block.keys())) and ("end" in list(block.keys())):
            start = block["start"]
            st = [start] if isinstance(start, str) else start
            end = block["end"]
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
            csrlist = [csr]
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

        .. _BeamTransform: https://github.com/ocelot-collab/ocelot/blob/master/ocelot/cpbd/physproc.py

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
