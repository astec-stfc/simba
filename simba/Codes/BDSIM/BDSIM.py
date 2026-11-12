"""
SIMBA BDSIM Module

Various objects and functions to handle BDSIM lattices and commands. See `BDSIM github`_ for more details.

    .. _BDSIM github: https://github.com/bdsim-collaboration/bdsim

Classes:
    - :class:`~simba.Codes.BDSIM.BDSIM.bdsimLattice`: The BDSIM lattice object, used for
      converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
      :class:`~simba.Framework_objects.frameworkLattice` into a BDSIM lattice object,
      and for tracking through it.

"""

from ... import exceptions
from ...Framework_objects import frameworkLattice
from ...Modules import Beams as rbf

import math
import os
import numpy as np
from typing import Any, ClassVar
from warnings import warn

BDSIM_PARTICLES = {1: "e-", 2: "e+", 3: "proton"}
"""BDSIM particle names, keyed by the SIMBA particle index returned by
:meth:`~simba.Modules.Beams.Particles.Particles.get_particle_index`.

Index 4 is H-, which BDSIM has no built-in name for (it would need a PDG ion
id), so it is deliberately absent rather than mapped to ``antiproton``."""

MATERIAL_CATEGORIES = frozenset(
    {"ecol", "rcol", "jcol", "target", "degrader", "crystalcol", "wirescanner", "dump"}
)
"""BDSIM element types that put material in the beam, which does nothing without a
physics list. (An aperture or collimator with no shape is built as a drift.)"""

TURN_PAD = 1e-6
"""Length [m] of the drift that keeps the samplers off BDSIM's terminator; see
:meth:`bdsimLattice.close_ring`."""

BDSIM_INTEGRATOR_SETS = frozenset(
    {"bdsimmatrix", "bdsimmatrixfringescaling", "bdsimone", "bdsimtwo", "geant4", "geant4dp"}
)
"""BDSIM's ``integratorSet`` names. LAURA negates quadrupole ``k1`` for a negative
beam in the ``bdsim*`` sets only, whose quadrupole scales by ``|brho|``."""

BDSIM_MATRIX_INTEGRATOR_SETS = frozenset({"bdsimmatrix", "bdsimmatrixfringescaling"})
"""Sets whose bends leave path length out of the arrival time (measured with BDSIM
1.7.7), so z does not change through dispersion; see
:class:`~simba.exceptions.BunchLengthWarning`."""


def use_before_samplers(main: str) -> None:
    """
    Move the ``use`` line of a pybdsim main file to just after the sequence include.

    pybdsim 3.9 writes ``use`` last, after the samplers, but BDSIM looks each sampler
    up in the beamline already in use ("current beamline doesn't contain element").
    Earlier pybdsim put ``use`` in the sequence file, so there is nothing to move.

    Parameters
    ----------
    main: str
        Main ``.gmad`` file written by ``pybdsim.Builder.Machine.Write``
    """
    with open(main) as fh:
        lines = fh.readlines()
    use = [line for line in lines if line.startswith("use,")]
    rest = [line for line in lines if not line.startswith("use,")]
    sequence = [i for i, line in enumerate(rest) if line.startswith("include") and "_sequence.gmad" in line]
    if len(use) != 1 or not sequence:
        return
    rest.insert(sequence[0] + 1, use[0])
    with open(main, "w") as fh:
        fh.writelines(rest)


def match_samplers(wanted: list, names: list, svals: dict, available) -> dict:
    """
    The BDSIM sampler to read each of ``wanted``'s beams from: its own, else one at the same s.

    BDSIM samples a marker at the end of the element before it, so of two markers side by
    side only one keeps a sampler; the other's beam is the same, so it shares it. At s = 0
    there is no element before, but the ``Primary`` sampler holds the same beam.

    Parameters
    ----------
    wanted: list
        Names to write beams at.
    names: list
        Every name that might have a sampler, ``wanted`` among them.
    svals: dict
        s of each name.
    available:
        The samplers BDSIM wrote.

    Returns
    -------
    dict
        ``{name: sampler}``; a sampler not in ``available`` if there is none.
    """
    own = {n: n.replace("-", "_") for n in names}
    matched = {}
    for name in wanted:
        sampler, s = own[name], svals.get(name)
        if sampler not in available and s is not None:
            twins = [own[n] for n in names if own[n] in available and abs(svals.get(n, s + 1) - s) < 1e-9]
            if twins:
                sampler = twins[0]
            elif "Primary" in available and abs(s) < 1e-9:
                sampler = "Primary"
        matched[name] = sampler
    return matched


class bdsimLattice(frameworkLattice):
    """
    Class for defining the BDSIM lattice object, used for
    converting the :class:`~simba.Framework_objects.frameworkObject`s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a BDSIM lattice object,
    and for tracking through it.
    """

    code: str = "bdsim"
    """String indicating the lattice object type"""

    supports_turns: ClassVar[bool] = True
    """Via BDSIM's circular mode (``option, circular=1, nturns=N``), whose samplers
    record every turn with its ``turnNumber``."""

    merges_carried_species: ClassVar[bool] = True

    trackBeam: bool = True
    """Flag to indicate whether to track the beam"""

    lattice: Any | None = None
    """
    Lattice elements arranged into a BDSIM `Machine`_

    .. _Machine: https://github.com/bdsim-collaboration/pybdsim/blob/develop/src/pybdsim/Builder.py
    """

    particle_definition: str = None
    """Initial particle distribution as a string"""

    def model_post_init(self, __context):
        super().model_post_init(__context)
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

    @property
    def gmadname(self) -> str:
        """
        Name of the gmad/ROOT files for this lattice (sanitized).
        """
        return self.section.name.replace("-", "_")

    @property
    def particle_type(self) -> str:
        """
        BDSIM name of the beam particle, from the mass and charge of the beam.
        """
        beam = self.global_parameters["beam"]
        index = beam.beam.get_particle_index(
            float(np.asarray(beam.particle_mass)[0]), np.asarray(beam.charge)[0]
        )
        if index not in BDSIM_PARTICLES:
            raise ValueError(
                f"Particle index {index} has no BDSIM particle name; "
                "BDSIM cannot track this species by name"
            )
        return BDSIM_PARTICLES[index]

    @property
    def charge_sign(self) -> int:
        """
        Sign of the charge of the beam particle; needed for setting magnet polarities.
        """
        return int(self.global_parameters["beam"].chargesign[0])

    def write(self) -> None:
        """
        Create the lattice object and save it as a JSON file to `master_subdir`.
        """
        from pybdsim.Beam import Beam  # not at the top: pybdsim is only needed to run BDSIM

        beam = Beam()
        beam.SetParticleType(self.particle_type)
        beam.SetDistributionType("userfile")
        beam.SetDistrFile(self.bdsim_beam_file)
        beam.SetDistrFileFormat(rbf.bdsim.BDSIM_DISTRIBUTION_FORMAT)
        beam.SetEnergy(self.global_parameters["beam"].centroids.mean_energy.val, unitsstring="eV")
        self.lattice = self.section.to_bdsim(
            save=True,
            beam=beam,
            charge_sign=self.charge_sign,
            integrator_set=self.integrator_set,
            Brho=self.rigidity(self.reference_p0c),
        )
        # LAURA has already set integratorSet and buildPoleFaceGeometry
        self.lattice.options.update(self.bdsim_options)
        sampled = {s.name for s in self.lattice.samplers}
        for name in self.samplers:
            if name.replace("-", "_") not in sampled:
                self.lattice.AddSampler(name.replace("-", "_"))
        material = list(dict.fromkeys(
            name for name in self.lattice.sequence
            if self.lattice.elements[name].category in MATERIAL_CATEGORIES
        ))
        if material and not self.physics_list:
            warn(exceptions.MaterialWithoutPhysicsWarning(self.objectname, material))
        crystals = [name for name in material if self.lattice.elements[name].category == "crystalcol"]
        if crystals and self.physics_list and "channel" not in self.physics_list.lower():
            warn(exceptions.CrystalWithoutChannellingWarning(self.objectname, crystals, self.physics_list))
        if self.circular:
            self.close_ring()
            self.lattice.options.update(circular=1, nturns=self.turns)
        main = os.path.join(self.global_parameters["master_subdir"], self.gmadname + ".gmad")
        self.lattice.Write(main)
        use_before_samplers(main)
        if self.integrator_set in BDSIM_MATRIX_INTEGRATOR_SETS and any(
            getattr(e, "hardware_type", None) == "Dipole" for e in self.elements.values()
        ):
            warn(exceptions.BunchLengthWarning(self.objectname, self.integrator_set))

    @property
    def integrator_set(self) -> str:
        """
        BDSIM's ``integratorSet`` (see :data:`BDSIM_INTEGRATOR_SETS`); ``bdsimmatrix``,
        BDSIM's default, unless the ``tracking`` block says::

            files:
              ARC:
                code: bdsim
                tracking: {integrator_set: bdsimtwo}
        """
        tracking = self.file_block.get("tracking") or {}
        name = str(tracking.get("integrator_set", "bdsimmatrix")).lower()
        if name not in BDSIM_INTEGRATOR_SETS:
            raise ValueError(
                f"Unknown BDSIM integrator_set {name!r}; one of {sorted(BDSIM_INTEGRATOR_SETS)}"
            )
        return name

    @property
    def physics_list(self) -> str:
        """
        BDSIM's ``physicsList``. Empty, BDSIM's default, unless the ``tracking`` block
        says otherwise. With no physics list, BDSIM only transports::

            files:
              TARGET:
                code: bdsim
                tracking: {physics_list: em}
        """
        tracking = self.file_block.get("tracking") or {}
        return str(tracking.get("physics_list") or "")

    @property
    def samplers(self) -> list:
        """
        Elements to record the beam at the exit of, as well as the screens, markers and
        BPMs, which always are; each gets ``<element>.openpmd.hdf5``::

            tracking:
              physics_list: em
              samplers: [CLA-FEC1-TARGET-01]
        """
        tracking = self.file_block.get("tracking") or {}
        names = tracking.get("samplers") or []
        names = [names] if isinstance(names, str) else list(names)
        unknown = [name for name in names if name not in self.elements]
        if unknown:
            raise ValueError(f"Line '{self.objectname}' has no elements {unknown} to sample at")
        return names

    @property
    def secondaries(self):
        """
        Species of the secondaries BDSIM makes that are kept and passed on: none unless
        the ``tracking`` block lists them, or says ``all`` (every species SIMBA has)::

            tracking:
              physics_list: em
              secondaries: [photon, positron]

        A secondary of the injected species (an electron knocked on by an electron
        beam, say) joins the injected beam's group in the output.
        """
        tracking = self.file_block.get("tracking") or {}
        wanted = tracking.get("secondaries") or ()
        if wanted == "all":
            return "all"
        wanted = (wanted,) if isinstance(wanted, str) else tuple(wanted)
        unknown = [w for w in wanted if rbf.SPECIES_ALIASES.get(w, w) not in rbf.SPECIES]
        if unknown:
            raise ValueError(f"Unknown secondaries {unknown}; one of {sorted(rbf.SPECIES)} or 'all'")
        return tuple(rbf.SPECIES_ALIASES.get(w, w) for w in wanted)

    @property
    def bdsim_options(self) -> dict:
        """
        The BDSIM options this line adds: :attr:`physics_list`, and anything in the
        ``tracking`` block's ``options``, by BDSIM's own names::

            tracking:
              physics_list: em
              options: {defaultRangeCut: 1e-3, minimumKineticEnergy: 1e-3}

        Strings are quoted for gmad. ``integratorSet`` comes from :attr:`integrator_set`,
        and ``circular`` and ``nturns`` from the turn count, after these.

        With no :attr:`secondaries` asked for, ``stopSecondaries=1`` too, unless the
        ``options`` say otherwise: they would be dropped anyway, and tracking them is
        what costs. The primaries still lose what they gave them.

        ``samplersSplitLevel=1`` too, unless the ``options`` say otherwise: a split
        sampler is read in one go (:func:`~simba.Modules.Beams.bdsim.read_bdsim_sampler_arrays`)
        rather than event by event.
        """
        tracking = self.file_block.get("tracking") or {}
        options = dict(tracking.get("options") or {})
        if self.physics_list:
            options["physicsList"] = self.physics_list
        if not self.secondaries:
            options.setdefault("stopSecondaries", 1)
        options.setdefault("samplersSplitLevel", 1)
        return {
            key: f'"{value}"' if isinstance(value, str) else value
            for key, value in options.items()
        }

    @property
    def circular(self) -> bool:
        """
        Whether BDSIM runs in circular mode: for more than one turn, or for a line that
        bends through a whole turn, which would otherwise loop for ever.
        """
        return self.turns > 1 or abs(self.net_bend_angle) >= 2 * math.pi - 1e-6

    def close_ring(self) -> None:
        """
        Set up BDSIM's circular mode (``option, circular=1, nturns=N``).

        BDSIM puts a terminator, which counts the turns, and a teleporter, which bridges
        the gap back to the start, after the last element. The teleporter needs a gap of
        more than 0.03 um, and less than 1 m, along the beam, so the line must close
        geometrically.

        A sampler next to the terminator makes it count each turn twice, so a
        :data:`TURN_PAD` drift goes after the last element. The options themselves
        are added in :meth:`write`.
        """
        drifts = [
            name for name in self.lattice.sequence
            if self.lattice.elements[name].category == "drift"
        ]
        if drifts:
            last = self.lattice.elements[drifts[-1]]
            last["l"] = float(last["l"]) - 2 * TURN_PAD
        self.lattice.AddDrift("BDSIM_TURN_PAD", TURN_PAD)

    def preProcess(self) -> None:
        """
        Load the input beam (see :meth:`load_input_beam`) and write it for BDSIM.
        """
        super().preProcess()
        prefix = self.get_prefix()
        prefix = prefix if self.trackBeam else prefix + self.particle_definition
        self.load_input_beam(prefix, self.particle_definition)
        self.hdf5_to_bdsim()

    def hdf5_to_bdsim(self) -> None:
        """
        Write the input beam as :attr:`bdsim_beam_file`.
        """
        rbf.bdsim.write_bdsim_beam_file(
            beam=self.global_parameters["beam"],
            filename=self.bdsim_beam_file,
        )

    @property
    def bdsim_beam_file(self) -> str:
        """
        The input beam in BDSIM's ``userfile`` format, ``<line>.bdsim`` in ``master_subdir``:
        named after the line, as elegant's is, since ``particle_definition`` may be a path
        (``$master_lattice$Data_Files/...``).
        """
        return os.path.join(self.global_parameters["master_subdir"], self.objectname + ".bdsim")

    def _generate_optics_config(self):
        with open(f"{self.global_parameters['master_subdir']}/optics_config.txt", "w") as f:
            f.write("CalculateOpticalFunctions 1 \nMergeHistograms 0\n")

    def run(self) -> None:
        """
        Run the code with input 'filename'
        This method constructs the command to run the simulation using the specified executable
        and the name of the lattice. It redirects the output to a log file in the master subdirectory.

        If  :attr:`~remote_setup` is set, then :func:`~run_remote` will be called instead.

        Raises
        ------
        FileNotFoundError
            If the executable for the specified code is not found in the executables dictionary.
        """
        if self.remote_setup:
            self.run_remote()
        else:
            from pybdsim.Run import Bdsim, RebdsimOptics

            Bdsim(
                f"{self.global_parameters['master_subdir']}/{self.gmadname}.gmad",
                f"{self.global_parameters['master_subdir']}/{self.gmadname}",
                ngenerate=len(self.global_parameters["beam"].x),
                bdsimExecutable=self.executables[self.code][-1],
            )
            RebdsimOptics(
                f"{self.global_parameters['master_subdir']}/{self.gmadname}.root",
                f"{self.global_parameters['master_subdir']}/{self.gmadname}.optics.root",
            )

    def screen_function(
        self, name: str, arrays: dict, data_turn: int | None, turn: int | None, ninjected: int,
        injected=None,
    ) -> None:
        """
        Convert one turn of one BDSIM sampler into a SIMBA beam and write it
        with :meth:`write_beam_file`.

        Parameters
        ----------
        name: str
            Name of the element the sampler belongs to.
        arrays: dict
            Raw sampler arrays, see
            :func:`~simba.Modules.Beams.bdsim.read_bdsim_sampler_arrays`.
        data_turn, turn: int or None
            An :meth:`output_turns` pair; ``data_turn`` picks the hits, see
            :func:`~simba.Modules.Beams.bdsim.select_turn`.
        ninjected: int
            Number of particles tracked, for scaling the charge by the losses.
        injected: :class:`~simba.Modules.Beams.beam`, optional
            The beam tracked, for its charge and species; by default the line's beam.
        """
        injected = injected if injected is not None else self.global_parameters["beam"]
        if self.circular:
            # in circular mode only the primaries are kept, one per turn
            arrays = rbf.bdsim.select_turn(arrays, data_turn or 1)
        beam = rbf.beam()
        # channelling physics records garbage weights (1e-41, 0: BDSIM develop @ 17ddb954,
        # in hand-written gmad too), so take each event's from what was written
        input_weights = None
        if "channel" in self.physics_list.lower():
            input_weights = np.asarray(rbf.openpmd._weight(injected), dtype=float)
            input_weights = input_weights / np.mean(input_weights)
        rbf.bdsim.interpret_bdsim_sampler(
            beam,
            arrays,
            charge=injected.total_charge,
            zstart=self.startObject.physical.start.z,
            ref_index=self.ref_idx,
            ninjected=ninjected,
            species=injected.species,
            secondaries=self.secondaries,
            input_weights=input_weights,
            sstart=self.entrance_s,
        )
        beam.turn = self.beam_turn(turn)
        self.write_beam_file(beam, name, turn)
        if name == self.end and beam.turn == self.turns:
            self.global_parameters["beam"] = beam

    def postProcess(self) -> None:
        """
        Convert the BDSIM sampler outputs to HDF5 format and save them to
        `master_subdir`.

        BDSIM writes one sampler per marker into the ``Event`` tree of its raw ROOT
        output.
        """
        super().postProcess()
        if not self.trackBeam:
            return
        rootfile = os.path.join(
            self.global_parameters["master_subdir"], self.gmadname + ".root"
        )
        if not os.path.isfile(rootfile):
            hint = (
                "; BDSIM tracks turns only round a line that closes geometrically"
                if self.circular else ""
            )
            raise FileNotFoundError(
                f"BDSIM output file {rootfile} not found; the tracking run failed{hint}"
            )
        # every sampler in one read: a second pybdsim DataLoader on the same file in one
        # process corrupted the heap (a segfault reading the samplers, ~half the runs)
        arrays = rbf.bdsim.read_bdsim_sampler_arrays(rootfile)
        names = [scr.name for scr in self.screens_and_markers_and_bpms] + self.samplers
        wanted = match_samplers(
            [n for n in names if self.writes_output(n)], names, self.getSValues(as_dict=True), arrays
        )
        missing = [n for n, s in wanted.items() if s not in arrays]
        for name in missing:
            warn(f"No BDSIM sampler found for {name}; no beam file will be written")
        wanted = {n: s for n, s in wanted.items() if n not in missing}
        if not wanted:
            warn(f"No BDSIM samplers found in {rootfile}")
            return
        # the end screen replaces the line's beam, so keep the one that was tracked
        injected = self.global_parameters["beam"]
        ninjected = len(injected.x)

        def convert(name):
            for data_turn, turn in self.output_turns():
                self.screen_function(name, arrays[wanted[name]], data_turn, turn, ninjected, injected)

        self.convert_outputs(convert, [name for name in wanted if name != self.end])
        if self.end in wanted:
            convert(self.end)
