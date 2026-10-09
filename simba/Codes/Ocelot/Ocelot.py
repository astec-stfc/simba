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
from laura.translator.utils.fields import FieldMap
from ...Modules.Twiss.ocelot import save_ocelot_twiss_hdf
from ...Modules.constants import speed_of_light
from .cavitymaps import stable_cavity_maps
from .latticefile import fast_lattice_files
from copy import deepcopy
from inspect import signature
from numpy import array, linspace, save, interp, searchsorted, clip, pi, sqrt
import os
from yaml import safe_load

with open(
    os.path.dirname(os.path.abspath(__file__)) + "/ocelot_defaults.yaml",
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

    electrons_only: ClassVar[bool] = True
    """Ocelot's physics processes, and simba's beam conversion, assume electrons."""

    supports_turns: ClassVar[bool] = True
    """By looping ``cpbd.track.track`` and feeding the bunch back in.

    **Not** ``track_nturns``, despite the name: we need to track a ``ParticleArray``
    via a ``Navigator``. ``track_nturns`` takes a list of single particles and is
    the right tool for dynamic aperture, which is a different job.
    """

    native_time: ClassVar[tuple[str, str]] = ("tau", "m")
    """``tau = c t``, measured from the reference: positive behind it."""

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

    supports_ramp: ClassVar[bool] = True
    """By re-referencing the particle array between turns; see
    :meth:`apply_ramp`."""

    native_rf: ClassVar[str] = "synchronous"
    """The cavity map's phase is ``phi - k tau``, with ``tau`` measured from
    the reference, so the reference sees ``phi`` every pass. Moved by
    :meth:`apply_rf_phases`."""

    rf_phase_sign: ClassVar[float] = -1.0
    """``phi`` (degrees) moved by this times a phase the reference sees."""

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

    _s_values: Dict | None = None
    """Cached :meth:`section.get_s_values`, both ends, for
    :attr:`_s_values_section`. See :meth:`section_s_values`."""

    _s_values_section: Any = None
    """The section :attr:`_s_values` was computed for."""

    _periodic: Any = None
    """The run's periodic Twiss (``[]`` if there is none), solved once by
    :meth:`_periodic_twiss` for the closed orbit and optics summary too."""

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
            elif f in oceglobal:
                setattr(self, f, oceglobal[f])
        self.particle_definition = self.input_particle_definition
        self.grids = getGrids()

    @classmethod
    def native_time_scale(cls, beta0: float) -> float:
        """``tau = c (t - reference_time)``."""
        return speed_of_light

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
        stable_cavity_maps()
        fast_lattice_files()
        self.lat_obj = self.section.to_ocelot(save=True)
        self._periodic = None
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
        self.load_input_beam(prefix, self.particle_definition)
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
            s_start=self.entrance_s,
            energy=self.reference_energy,
            t0=self.reference_t0,
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
                self.program_attribute(program, None)
                continue
            value = program.value_at(turn)
            for element in matches:
                attribute = program.parameter or "angle"
                if attribute not in ("dx", "dy", "tilt") and (
                    attribute not in signature(type(element).__init__).parameters
                ):
                    warn(
                        f"Line '{self.objectname}' programs '{program.element}', "
                        f"an Ocelot {type(element).__name__}, which has no "
                        f"'{attribute}' to set. Name the attribute with "
                        "'parameter:' in the program."
                    )
                    continue
                setattr(element, attribute, value)

    def _cavities(self, name: str) -> list:
        return [e for e in self.lat_obj.sequence if str(e.id) == name]

    def cavity_phase(self, name: str) -> float | None:
        """A cavity's ``phi``, in degrees; see
        :meth:`~simba.Framework_objects.frameworkLattice.apply_rf_phases`."""
        cavities = self._cavities(name)
        return float(cavities[0].phi) if cavities else None

    def set_cavity_phase(self, name: str, phase: float) -> None:
        """
        Set a cavity's ``phi``, in degrees. Setting it clears the cavity's
        cached transfer map, and the ``Navigator`` is built afresh each pass,
        so the next pass sees it.
        """
        for element in self._cavities(name):
            element.phi = phase

    def apply_ramp(self, particles: Any, turn: int) -> Any:
        """
        Re-reference `particles` to the ramp's momentum for `turn`, keeping
        every particle as it was; the model in :mod:`simba.Modules.EnergyRamp`.
        :func:`~simba.Codes.Ocelot.fixedreference.rereference` does this.

        Parameters
        ----------
        particles: ParticleArray
            The beam entering `turn`; changed in place
        turn: int
            Turn number, 1-based

        Returns
        -------
        ParticleArray
            `particles`
        """
        from .fixedreference import rereference
        p0c_new = self.ramp_p0c(turn)
        if p0c_new is None:
            return particles
        return rereference(particles, p0c_new, self.rest_energy)

    def run(self) -> None:
        """
        Run the code, and set :attr:`~tws` and :attr:`~pout`
        """
        from ocelot.cpbd.track import track
        from .navigator import lattice_pass
        self.pout = deepcopy(self.pin)

        def start_turn(turn):
            self.pout = self.apply_ramp(self.pout, turn)

        def track_pass(turn, pass_index, name_turn, record):
            navi = self.navi_setup(
                turn=name_turn,
                write_beams=record,
                beam_turn=turn,
                reference_energy=self.pout.E if self.fixed_reference else None,
                pass_index=pass_index if self.uses_reference_clock else None,
            )
            navi.go_to_start()
            with lattice_pass(self.lat_obj):
                self.tws, self.pout = track(
                    self.lat_obj,
                    self.pout,
                    navi=navi,
                    calc_tws=True,
                    twiss_disp_correction=False,
                    print_progress=False,
                )

        self._periodic = None
        self.run_turns(track_pass, start_turn)
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
        periodic = self._ocelot_periodic()
        if not periodic:
            warn(
                f"Line '{self.objectname}' asks for the periodic solution, but "
                "Ocelot found none: the one-turn map is unstable, so the ring "
                "has no matched optics. Falling back to the tracked beam's "
                "Twiss, which is not the ring's."
            )
            return self.tws
        return periodic

    def _ocelot_periodic(self) -> List:
        """
        ``optics.twiss`` for the closed solution, seeded with the reference energy.
        ``tws0=None`` is how Ocelot is asked for the periodic solution, but with a
        cavity in the lattice it refuses outright.

        Solved once a run: the run puts the lattice back as it found it, so
        every later ask is the same question.

        Returns
        -------
        List
            The periodic Twiss, empty if there is no periodic solution
        """
        if self._periodic is not None:
            return self._periodic
        stable_cavity_maps()
        from ocelot.cpbd.beam import Twiss
        from ocelot.cpbd.optics import twiss as ocelot_twiss

        tws0 = Twiss()
        tws0.E = self.reference_energy / 1e9
        self._periodic = ocelot_twiss(self.lat_obj, tws0=tws0) or []
        return self._periodic

    def _track_nturns(self, track_list, save_track: bool):
        """
        Ocelot's ``track_nturns``, with its aperture limits taken from
        :meth:`_ocelot_periodic`, inside
        :func:`~simba.Codes.Ocelot.navigator.lattice_pass`.
        """
        import ocelot.cpbd.track as octrack

        stable_cavity_maps()

        def aperture_limit(lat, xlim=1, ylim=1):
            tws = self._ocelot_periodic()
            bxmax = max(t.beta_x for t in tws)
            bymax = max(t.beta_y for t in tws)
            bx0, by0 = tws[0].beta_x, tws[0].beta_y
            return (
                float(xlim) * sqrt(bx0 / bxmax),
                float(ylim) * sqrt(by0 / bymax),
                float(xlim) / sqrt(bxmax * bx0),
                float(ylim) / sqrt(bymax * by0),
            )

        from .navigator import lattice_pass

        original = octrack.aperture_limit
        octrack.aperture_limit = aperture_limit
        try:
            with lattice_pass(self.lat_obj):
                return octrack.track_nturns(
                    self.lat_obj,
                    self.turns,
                    track_list,
                    nsuperperiods=self.nsuperperiods,
                    save_track=save_track,
                    print_progress=False,
                )
        finally:
            octrack.aperture_limit = original

    def read_closed_orbit(self):
        """
        Ocelot's periodic Twiss carries the orbit on its first element.
        """
        periodic = self._ocelot_periodic()
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

        periodic = self._ocelot_periodic()
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

        energy_gev = self.reference_energy / 1e9
        return array(lattice_transfer_map(self.lat_obj, energy_gev), dtype=float)

    def postProcess(self) -> None:
        """
        Convert the outputs from Ocelot to HDF5 format and save them to `master_subdir`.
        """
        super().postProcess()
        twsdat = {e: [] for e in self.tws[0].__dict__}
        for t in self.tws:
            for k, v in t.__dict__.items():
                # Offset the s values to the start of the lattice
                if k == "s":
                    v += self.entrance_s
                twsdat[k].append(v)
        svals = array(self.getSValues(at_entrance=False)) + twsdat["s"][0]
        elems = self.createDrifts().values()
        zvals = [e.physical.end.z for e in elems]
        twsdat['z'] = interp(twsdat["s"], svals, zvals)
        elem_names = array([e.name for e in elems], dtype="U")
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
        from ocelot.cpbd.track import create_track_list

        xs, ys = self.da_grid()
        energy_gev = self.reference_energy / 1e9
        track_list = create_track_list(xs, ys, [0.0], energy=energy_gev)
        track_list = self._track_nturns(track_list, save_track=True)
        return self._footprint(
            (particle.x, particle.y, *([p[k] for p in particle.p_list] for k in range(4)))
            for particle in track_list
            if particle.turn >= self.turns - 1
        )

    def run_dynamic_aperture(self) -> list:
        """
        Dynamic aperture via ``track_nturns``, one particle per point of
        :meth:`da_rays`. With no ``Aperture`` elements the aperture limit is
        Ocelot's default of +/- 1 m.

        Returns
        -------
        list
            ``(x, y, turns_survived)`` per start.
        """
        from ocelot.cpbd.beam import Particle
        from ocelot.cpbd.track import Track_info

        energy_gev = self.reference_energy / 1e9
        track_list = [
            Track_info(Particle(x=x, y=y, p=0.0, E=energy_gev), x, y)
            for x, y in self.da_rays()
        ]
        track_list = self._track_nturns(track_list, save_track=False)
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
        from ocelot.cpbd.track import create_track_list

        orbit = self.read_closed_orbit()
        if orbit is None:
            orbit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        nudge = float((self.da_settings or {}).get("x_max", 1e-3)) / 100.0
        energy_gev = self.reference_energy / 1e9
        track_list = create_track_list(
            [orbit[0] + nudge], [orbit[2] + nudge], [0.0], energy=energy_gev
        )
        track_list = self._track_nturns(track_list, save_track=True)
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
        reference_energy: float | None = None,
        pass_index: int | None = None,
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
        reference_energy: float, optional
            The line's reference energy for this pass [GeV], to be restored
            after every cavity; see
            :class:`~simba.Codes.Ocelot.fixedreference.FixedReference`.
        pass_index: int, optional
            The 0-based pass the beams are written on, for their ``t``; see
            :meth:`reference_time`. None leaves Ocelot's own clock, as in a linac.

        Returns
        -------
        Navigator
            An Ocelot `Navigator`_ object
        """
        from ocelot import Twiss
        from .navigator import PassNavigator
        from .savebeamopenpmd import SaveBeamOpenPMD
        from .mbi import MBI
        navi_processes = []
        navi_locations_start = []
        navi_locations_end = []
        navi = PassNavigator(self.lat_obj, unit_step=self.unit_step)
        if reference_energy is not None:
            # first, so anything else at a cavity's exit sees the line's reference
            for proc, loc in self.physproc_fixed_reference(reference_energy):
                navi_processes += [proc]
                navi_locations_start += [loc]
                navi_locations_end += [loc]
        if self.lsc and self.lsc_enable:
            lsc = self.physproc_lsc()
            navi_processes += [lsc]
            navi_locations_start += [self.lat_obj.sequence[0]]
            navi_locations_end += [self.lat_obj.sequence[-1]]
        space_charge_set = False
        csr_set = False
        if (
            "charge" in self.file_block
            and "space_charge_mode" in self.file_block["charge"]
            and str(self.file_block["charge"]["space_charge_mode"]).lower() == "3d"
        ):
            gridsize = self.grids.getGridSizes(len(self.global_parameters["beam"].x))
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
            if (
                fieldstr is not None
                and self.wakefield_enable
                and getattr(obj.simulation, fieldstr) is not None
            ):
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
        for bend, loc, _ in self.physproc_radiation():
            navi_processes += [bend]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        recorded = self.screens_and_markers_and_bpms + self.apertures

        def t_reference(s):
            if pass_index is None:
                return None
            return self.reference_time(s, pass_index)

        file_turn = turn if self.bundles_turns else None
        for w in recorded if write_beams else []:
            if w.name == self.names[-1] or not self.writes_output(w.name):
                continue
            loc = self.lat_obj.sequence[self.names.index(w.name)]
            navi_processes += [
                SaveBeamOpenPMD(
                    filename=self.output_beam_file(w.name),
                    global_parameters=self.global_parameters,
                    zstart=w.physical.start.z,
                    sstart=self.entrance_s + sval_in[w.name],
                    ref_idx=self.ref_idx,
                    beam_turn=beam_turn,
                    t_reference=t_reference(sval_in[w.name]),
                    file_turn=file_turn,
                )
            ]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        if write_beams:
            loc = self.lat_obj.sequence[-1]
            navi_processes += [
                SaveBeamOpenPMD(
                    filename=self.output_beam_file(self.names[-1]),
                    global_parameters=self.global_parameters,
                    zstart=self.endObject.physical.end.z,
                    sstart=self.entrance_s + sval_out[self.end],
                    ref_idx=self.ref_idx,
                    beam_turn=beam_turn,
                    t_reference=t_reference(sval_out[self.end]),
                    file_turn=file_turn,
                )
            ]
            navi_locations_start += [loc]
            navi_locations_end += [loc]
        navi.add_physics_processes(
            navi_processes, navi_locations_start, navi_locations_end
        )
        return navi

    def physproc_fixed_reference(self, reference_energy: float) -> List:
        """
        A :class:`~simba.Codes.Ocelot.fixedreference.FixedReference` at the exit
        of every cavity that changes the energy.

        Parameters
        ----------
        reference_energy: float
            The reference total energy to restore [GeV]

        Returns
        -------
        List
            ``(process, element)`` pairs

        Raises
        ------
        warning
            If a cavity is the last element, so has no exit to put it on.
        """
        from ocelot.cpbd.elements import Cavity, TWCavity
        from .fixedreference import FixedReference
        sequence = self.lat_obj.sequence
        processes = []
        for index, element in enumerate(sequence):
            if not isinstance(element, (Cavity, TWCavity)):
                continue
            if index + 1 == len(sequence):
                warn(
                    f"Line '{self.objectname}' ends on cavity '{element.id}', so "
                    "Ocelot's move of the reference energy there cannot be undone "
                    "until the next pass. End the line on a marker."
                )
                continue
            processes.append(
                (FixedReference(reference_energy, self.rest_energy), sequence[index + 1])
            )
        return processes

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
        block = self.file_block.get("csr", {})
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
            loc: field | FieldMap | str,
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
        if isinstance(loc, (field, FieldMap)):
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
