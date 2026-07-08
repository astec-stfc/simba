"""
SIMBA MAD-X Module

Objects and functions to handle MAD-X lattices and commands. See `MAD-X manual`_ for
more details.

    .. _MAD-X manual: https://mad.web.cern.ch/mad/

Classes:
    - :class:`~simba.Codes.MADX.madx.madxLattice`: The MAD-X lattice object, used for
      converting the :class:`~simba.Framework_objects.frameworkObject` s defined in the
      :class:`~simba.Framework_objects.frameworkLattice` into a string representation of
      the lattice suitable for a MAD-X input file (via the ``LAURA`` translator's
      ``section.to_madx`` converter).

MAD-X input files are run either through the `cpymad`_ python bindings, if available,
or by calling the ``madx`` executable defined in :download:`Executables <../Executables.yaml>`.

    .. _cpymad: https://hibtc.github.io/cpymad/
"""

import os
import re
import subprocess
from warnings import warn
import numpy as np
import lox
from lox.worker.thread import ScatterGatherDescriptor
from typing import Dict, Tuple, Any, ClassVar

from ...Framework_objects import frameworkLattice
from ...FrameworkHelperFunctions import saveFile
from ...Modules import Beams as rbf
from laura.translator.utils.functions import sanitize_string


class madxLattice(frameworkLattice):
    """
    Class for defining the MAD-X lattice object, used for converting the
    :class:`~simba.Framework_objects.frameworkObject` s defined in the
    :class:`~simba.Framework_objects.frameworkLattice` into a string
    representation of the lattice suitable for a MAD-X input file.
    """

    code: str = "madx"
    """String indicating the lattice object type"""

    particle_definition: str | None = None
    """String representation of the initial particle distribution"""

    trackBeam: bool = True
    """Flag to indicate whether to track the beam through the lattice"""

    use_tracking: bool | None = True
    """How to propagate the beam. ``False`` uses the (fast, constant-energy) TWISS
    sectormap transfer map; ``True`` runs a full MAD-X ``TRACK`` (needed when the
    reference energy changes, e.g. through accelerating cavities). ``None`` (default)
    auto-selects: ``TRACK`` if the lattice contains cavities, otherwise the sectormap."""

    single_particle: bool = False
    """Whether to propagate a single (reference) particle rather than the full
    distribution: the beams at each screen are then reconstructed by applying the
    (energy-staged) TWISS transfer matrices to the input distribution instead of
    tracking every macroparticle -- much cheaper. Settable from the ``.def`` lattice
    block (``single_particle: true``)."""

    generate_beams: bool = True
    """Whether to write a beam at every screen / marker / BPM (``True``, default) or
    only the final beam at the end of the lattice (:func:`endObject`, ``False``). This
    applies in every mode -- full ``TRACK``, sectormap, and the :attr:`single_particle`
    transfer-map reconstruction -- and only affects which beams are *written* (the beam
    is still propagated through the whole line either way). Settable from the ``.def``
    lattice block (``generate_beams: false``)."""

    madx: Any | None = None
    """Instance of `Madx` from `cpymad`"""

    staged: bool = False
    """Set by :func:`write` when the line accelerates enough that tracking must be
    energy-staged (MAD-X ``TRACK`` uses a constant reference momentum, so a large
    ``PT`` corrupts downstream dispersive elements -- see :func:`run_staged`).
    Requires `cpymad`; the constant-momentum single-pass ``TRACK`` is used otherwise."""

    staged_results: Dict = {}
    """Per-diagnostic staged-tracking output, keyed by sanitised element name:
    ``(Xf (6,N), s_position, original_indices, (p0c, E0, Eref))``. Populated by
    :func:`run_staged`, consumed by :func:`postProcess_staged`."""

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Threaded function for propagating the beam to each diagnostic and writing
    its output (see :func:`screen_threaded_function`)"""


    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.elementObjects[self.start].name
        # honour single_particle / generate_beams from the .def lattice block. These
        # live under the "input" sub-block (like particle_definition), with a top-level
        # key accepted as a fallback.
        block = self.file_block.get("input") or {}
        for flag in ("single_particle", "generate_beams"):
            if flag in block:
                setattr(self, flag, block[flag])
            elif flag in self.file_block:
                setattr(self, flag, self.file_block[flag])

    def _output_diagnostics(self) -> list:
        """Screens / markers / BPMs at which a beam should be written. Empty when
        :attr:`generate_beams` is ``False`` (only the final :func:`endObject` beam is
        produced); the beam is still propagated through the whole line regardless."""
        return list(self.screens_and_markers_and_bpms) if self.generate_beams else []

    def tracking_mode(self) -> bool:
        """
        Whether to propagate the beam by full MAD-X ``TRACK`` (as opposed to the
        constant-energy TWISS sectormap map). Honours :attr:`use_tracking`, or
        auto-selects tracking when the lattice contains (deflecting) cavities, whose
        acceleration a constant-reference sectormap cannot represent.
        """
        if self.use_tracking is not None:
            return self.use_tracking
        # self.elements are LAURA elements, whose hardware_type is "RFCavity" /
        # "RFDeflectingCavity" (not the SIMBA type key "cavity"), so match by
        # substring rather than the exact-name getElementType("cavity").
        return any(
            "cavity" in getattr(el, "hardware_type", "").lower()
            for el in self.elements.values()
        )

    def accelerating_energy_gain(self, madx_text: str) -> float:
        """
        Peak absolute cumulative change [eV] in the synchronous-particle energy
        along the line due to accelerating RF cavities, read from the generated
        MAD-X ``rfcavity`` lines (``ΔE = VOLT[MV]·1e6·sin(2π·LAG)`` -- the same gain
        MAD-X applies in ``TRACK``).

        This is ~0 for non-accelerating RF (e.g. a storage-ring / bunching cavity,
        whose synchronous gain is negligible), so it distinguishes those from a true
        accelerating linac. Using the cumulative peak (not just the net) also flags
        lines that ramp up then down (e.g. energy recovery). Deflecting
        ``crabcavity`` elements carry no ``VOLT`` acceleration and are ignored.
        """
        cum = 0.0
        peak = 0.0
        for line in madx_text.splitlines():
            low = line.lower()
            if ": rfcavity" not in low:
                continue
            vm = re.search(r"volt\s*=\s*([-\d.eE+]+)", low)
            lm = re.search(r"lag\s*=\s*([-\d.eE+]+)", low)
            if not (vm and lm):
                continue
            cum += float(vm.group(1)) * 1e6 * np.sin(2 * np.pi * float(lm.group(1)))
            peak = max(peak, abs(cum))
        return peak

    def _warn_if_accelerating(self, madx_text: str) -> None:
        """
        Warn that the (constant reference energy) MAD-X ``TWISS`` optics are
        unreliable when the line accelerates the beam. MAD-X ``TWISS`` does not ramp
        the reference momentum through RF cavities, so it omits the adiabatic
        damping / RF focusing that keeps the beta functions bounded, and the optics
        diverge. No warning is issued for non-accelerating RF (storage-ring /
        bunching cavities), whose synchronous energy gain is negligible.
        """
        _, _, Eref = self.reference_from_beam(self.global_parameters["beam"])
        gain = self.accelerating_energy_gain(madx_text)
        # threshold at 1% of the injection energy: below this the constant-energy
        # optics error is small; above it (a real accelerating linac) TWISS diverges.
        if gain > 0.01 * Eref:
            warn(
                f"MAD-X TWISS uses a constant reference energy, but this line "
                f"accelerates the beam by ~{gain / 1e6:.1f} MeV "
                f"(injection {Eref / 1e6:.1f} MeV). The optics in "
                f"{sanitize_string(self.name)}-twiss.tfs omit adiabatic damping and "
                f"will diverge -- treat the twiss summary as unreliable and use the "
                f"tracked beam for optics. (Non-accelerating RF does not trigger this.)"
            )

    def _should_stage(self, madx_text: str) -> bool:
        """
        Whether tracking should be driven interactively by :func:`run_staged` (which
        needs `cpymad`). This is the case when either

        * :attr:`single_particle` -- the beams are reconstructed from the segment
          transfer maps rather than tracked, so run_staged builds them (a single
          segment when the line does not accelerate), or
        * the line accelerates by more than 1% of the injection energy, so the
          constant-momentum single-pass ``TRACK`` would corrupt downstream dispersive
          elements and the constant-energy ``TWISS`` would blow the beta functions up.
        """
        try:
            import cpymad.madx  # noqa: F401
        except ImportError:
            return False
        if self.single_particle:
            return True
        _, _, Eref = self.reference_from_beam(self.global_parameters["beam"])
        return self.accelerating_energy_gain(madx_text) > 0.01 * Eref

    def write(self) -> None:
        """
        Write the MAD-X input file to `master_subdir` by calling the ``LAURA``
        translator's :func:`section.to_madx`, passing in the SIMBA generic beam
        object (:attr:`global_parameters["beam"]`) as the ``beam`` argument, then
        appending a ``TWISS``/``SECTORMAP`` block (:func:`madx_twiss_block`) and/or a
        ``TRACK`` block (:func:`madx_track_block`).

        In sectormap mode only the ``TWISS`` block is written. In tracking mode the
        ``TWISS`` block is emitted first (on the still-thick lattice, before
        ``TRACK``'s ``MAKETHIN``, so the twiss summary / ``-twiss.tfs`` is accurate and
        produced alongside the track output), then the ``TRACK`` block. TWISS is a
        lattice-optics computation independent of the number of tracked particles, so
        it is written regardless of :attr:`single_particle`.
        """
        tracking = self.tracking_mode()
        # TRACK needs a thin (MAKETHIN) lattice, which in turn needs refer=centre.
        refer = "centre" if tracking else "entry"
        output = self.section.to_madx(beam=self.hdf5_to_madx(), refer=refer)
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".madx"
        )
        if tracking:
            # When the line accelerates significantly, both a single constant-momentum
            # TRACK (huge PT corrupts dispersive elements) and a constant-energy TWISS
            # (RF defocusing at the wrong energy blows the beta functions up) are wrong.
            # Stage both instead, interactively in run_staged (needs cpymad); the file
            # then holds only the sequence. Otherwise write the single-pass TWISS+TRACK.
            self.staged = self._should_stage(output)
            if not self.staged:
                self._warn_if_accelerating(output)
                output += self.madx_twiss_block()
                output += self.madx_track_block()
        else:
            self.staged = False
            output += self.madx_twiss_block()
        saveFile(command_file, output, "w")
        self.files.append(command_file)

    def madx_twiss_block(self) -> str:
        """
        Build the ``TWISS`` command (with ``SECTORMAP``/``SECTORACC``) that writes the
        accumulated transfer maps used by the sectormap propagation.
        """
        beam = self.global_parameters["beam"]
        name = sanitize_string(self.name)
        twsstr = f"\nTWISS, SEQUENCE={name},\n"
        twsstr += "\tRMATRIX,\n"
        twsstr += "\tCHROM,\n"
        twsstr += "\tSECTORMAP,\n"
        twsstr += "\tSECTORACC,\n"
        twsstr += f"\tSECTORFILE=\"{name}-sectormap.tfs\",\n"
        twsstr += f"\tFILE=\"{name}-twiss.tfs\",\n"
        twsstr += f"\tBETX={beam.twiss.beta_x.val},\n"
        twsstr += f"\tBETY={beam.twiss.beta_y.val},\n"
        twsstr += f"\tALFX={beam.twiss.alpha_x.val},\n"
        twsstr += f"\tALFY={beam.twiss.alpha_y.val},\n"
        twsstr += f"\tDX={beam.twiss.eta_x.val},\n"
        twsstr += f"\tDY={beam.twiss.eta_y.val},\n"
        twsstr += f"\tDPX={beam.twiss.eta_xp.val},\n"
        twsstr += f"\tDPY={beam.twiss.eta_yp.val};\n"
        return twsstr

    def madx_track_block(self) -> str:
        """
        Build a MAD-X ``TRACK`` block that tracks the beam through the lattice,
        observing at every screen / BPM / marker and writing the result to
        ``<name>-trackone`` (via ``ONETABLE``).

        Unlike the sectormap map, ``TRACK`` applies the real RF kick, so the beam
        energy updates through accelerating cavities (as a ``PT`` deviation about
        the still-constant MAD-X reference momentum).

        The sequence is first sliced with ``MAKETHIN`` because MAD-X ``TRACK``
        cannot thick-track special elements such as crab cavities. ``THICK=true``
        keeps the magnets (quadrupoles, dipoles) as exact thick slices; only the
        elements that must be thin (RF / crab cavities, collimators) are thinned.
        This relies on the sequence being written with ``refer=centre`` (see
        :func:`write`), as ``MAKETHIN`` silently ignores ``refer=entry`` sequences.

        With :attr:`single_particle`, a single ``START`` is emitted from the beam
        centroid; otherwise (the default) one ``START`` per macroparticle.
        """
        beam = self.global_parameters["beam"]
        name = sanitize_string(self.name)
        p0c, E0, Eref = self.reference_from_beam(beam)
        X, _ = self.simba_to_canonical(beam, p0c, E0, Eref)
        if self.single_particle:
            X = X.mean(axis=1, keepdims=True)
        lines = [
            f"\nUSE, SEQUENCE={name};",
            'SELECT, FLAG=makethin, PATTERN=".*", THICK=true;',
            f"MAKETHIN, SEQUENCE={name};",
            f"USE, SEQUENCE={name};",
            f'TRACK, ONEPASS, DUMP, ONETABLE, FILE="{name}-track";',
        ]
        for diag in self._output_diagnostics():
            lines.append(f"OBSERVE, PLACE={sanitize_string(diag.name)};")
        for i in range(X.shape[1]):
            lines.append(
                "START, X={:.15g}, PX={:.15g}, Y={:.15g}, PY={:.15g}, "
                "T={:.15g}, PT={:.15g};".format(*X[:, i])
            )
        lines.append("RUN, TURNS=1;")
        lines.append("ENDTRACK;\n")
        return "\n".join(lines)

    def hdf5_to_madx(self) -> Dict:
        """
        Convert the HDF5 beam input file into a dict describing the MAD-X input distribution
        (a dict containing the relevant information) for tracking.
        """
        beam = self.global_parameters["beam"]
        madx_input_dict = {
            "PARTICLE": beam.species.upper(),
            "GAMMA": beam.centroids.mean_gamma,
            "EXN": beam.emittance.normalized_horizontal_emittance.val,
            "EYN": beam.emittance.normalized_vertical_emittance.val,
            "SIGE": float(beam.sigmas.sigma_cp.val / beam.centroids.mean_cp.val),
            "SIGT": float(beam.sigmas.sigma_z.val),
        }
        return madx_input_dict

    def preProcess(self) -> None:
        """
        Prepare the input distribution for MAD-X based on the `prefix` in the settings
        file for this lattice section, and write the MAD-X input file.
        """
        super().preProcess()
        prefix = self.get_prefix()
        self.read_input_file(prefix, self.particle_definition)
        self.write()

    def postProcess(self) -> None:
        """
        Post-process the simulation results and propagate the initial distribution to
        every screen / BPM / marker and to the end of the lattice.

        Dispatches to :func:`postProcess_tracking` when a full MAD-X ``TRACK`` was run
        (see :func:`tracking_mode`), otherwise to :func:`postProcess_sectormap`, which
        applies the accumulated TWISS sectormap transfer map. Each propagated beam is
        written (with the reconstructed energy) to ``<name>.openpmd.hdf5`` in parallel
        via :func:`screen_threaded_function`.
        """
        super().postProcess()
        if not self.trackBeam:
            return
        if self.staged:
            self.postProcess_staged()
        elif self.tracking_mode():
            self.postProcess_tracking()
        else:
            self.postProcess_sectormap()

    def postProcess_sectormap(self) -> None:
        """
        Propagate the beam using the TWISS ``SECTORMAP`` (constant-energy) transfer
        map. With ``SECTORACC`` each sectormap row is the accumulated map from the
        start of the line to that element; the last row is the full-lattice map.
        """
        import tfs

        subdir = self.global_parameters["master_subdir"]
        name = sanitize_string(self.objectname)
        twiss_file = f"{subdir}/{name}-twiss.tfs"
        sectormap_file = f"{subdir}/{name}-sectormap.tfs"
        if not (os.path.isfile(twiss_file) and os.path.isfile(sectormap_file)):
            warn(
                f"MAD-X output not found ({twiss_file}, {sectormap_file}); "
                "skipping beam propagation in postProcess"
            )
            return
        twiss = tfs.read(twiss_file)
        sectormap = tfs.read(sectormap_file)
        headers = dict(twiss.headers)
        beam = self.global_parameters["beam"]
        p0c, E0, Eref = self.reference_from_headers(headers)

        # The map (R, T, K) and the PX/PT normalisation are all defined about MAD-X's
        # reference momentum p0 (set by the BEAM command via to_madx). If that does not
        # match the actual beam energy, every PT is offset and the propagated energies
        # (and, through longitudinal coupling, times) are wrong.
        beam_mean_energy = float(np.mean(np.asarray(beam.energy.val, dtype=float)))
        if abs(beam_mean_energy - Eref) > 1e-3 * Eref:
            warn(
                f"MAD-X reference energy ({Eref / 1e9:.4g} GeV) does not match the beam "
                f"mean energy ({beam_mean_energy / 1e9:.4g} GeV). The transfer map was "
                "computed about a different momentum, so the propagated energies and "
                "times will be wrong. Check the beam/GAMMA passed to section.to_madx "
                "(see hdf5_to_madx). If the lattice accelerates the beam, use tracking "
                "(use_tracking=True) instead, as a sectormap cannot represent it."
            )

        X, t_ref = self.simba_to_canonical(beam, p0c, E0, Eref)
        ref = (p0c, E0, Eref)

        def final_coords(row):
            K, R, T = self.extract_madx_map(row)
            return K[:, None] + R @ X + np.einsum("ijk,jn,kn->in", T, X, X)

        rows_by_name: Dict[str, list] = {}
        for _, row in sectormap.iterrows():
            rows_by_name.setdefault(str(row["NAME"]).upper(), []).append(row)
        # Collect one task per output file, keyed by element name (the end element is
        # added first). Deduplicating avoids two threads writing the same
        # <name>.openpmd.hdf5 when the end element is itself a diagnostic.
        last = sectormap.iloc[-1]
        tasks: Dict = {
            self.endObject.name: (self.endObject, final_coords(last), float(last["POS"]))
        }
        for diag in self._output_diagnostics():
            if diag.name in tasks:
                continue
            candidates = rows_by_name.get(sanitize_string(diag.name).upper(), [])
            if not candidates:
                warn(f"No MAD-X sectormap row found for diagnostic {diag.name}")
                continue
            # disambiguate repeated names by the diagnostic's s-position (MAD-X frame)
            madx_s = diag.physical.end.z - self.startObject.physical.start.z
            row = min(candidates, key=lambda r: abs(r["POS"] - madx_s))
            tasks[diag.name] = (diag, final_coords(row), float(row["POS"]))
        for element, Xf, s_pos in tasks.values():
            self.screen_threaded_function.scatter(element, Xf, ref, s_pos, t_ref, None)
        self.screen_threaded_function.gather()

    def postProcess_tracking(self) -> None:
        """
        Propagate the beam by reading back the MAD-X ``TRACK`` (``ONETABLE``) output,
        which contains the fully-tracked canonical coordinates of every particle at
        each observation point (screens / BPMs / markers and the line end). Unlike the
        sectormap, this captures the energy change through accelerating cavities.
        """
        subdir = self.global_parameters["master_subdir"]
        name = sanitize_string(self.objectname)
        trackfile = f"{subdir}/{name}-trackone"
        if not os.path.isfile(trackfile):
            warn(
                f"MAD-X track output not found ({trackfile}); "
                "skipping beam propagation in postProcess"
            )
            return
        segments = self.read_trackone(trackfile)
        beam = self.global_parameters["beam"]
        p0c, E0, Eref = self.reference_from_beam(beam)
        _, t_ref = self.simba_to_canonical(beam, p0c, E0, Eref)
        ref = (p0c, E0, Eref)

        # Collect one task per output file, keyed by element name (the end element is
        # added first), so overlapping names don't have two threads write the same file.
        tasks: Dict = {}
        end_key = f"{sanitize_string(self.name)}$END".upper()
        if end_key in segments:
            Xf, s_end, numbers = segments[end_key]
            tasks[self.endObject.name] = (self.endObject, Xf, s_end, numbers - 1)
        else:
            warn(f"No end-of-line segment ({end_key}) found in {trackfile}")
        for diag in self._output_diagnostics():
            if diag.name in tasks:
                continue
            key = sanitize_string(diag.name).upper()
            if key not in segments:
                warn(f"No MAD-X track segment found for diagnostic {diag.name}")
                continue
            Xf, s_pos, numbers = segments[key]
            tasks[diag.name] = (diag, Xf, s_pos, numbers - 1)
        for element, Xf, s_pos, idx in tasks.values():
            self.screen_threaded_function.scatter(element, Xf, ref, s_pos, t_ref, idx)
        self.screen_threaded_function.gather()

    def postProcess_staged(self) -> None:
        """
        Reconstruct the beams from :func:`run_staged`'s energy-staged tracking output.

        Each diagnostic in :attr:`staged_results` carries the reference triplet
        ``(p0c, E0, Eref)`` of the segment it was observed in, so :func:`canonical_to_beam`
        reconstructs its energy about the correct (ramped) reference rather than the
        constant injection momentum. The stored indices are the original particle
        positions (lost particles pruned), so per-particle charge/mass carry over.
        """
        beam = self.global_parameters["beam"]
        p0c, E0, Eref = self.reference_from_beam(beam)
        _, t_ref = self.simba_to_canonical(beam, p0c, E0, Eref)
        if not self.staged_results:
            warn("MAD-X staged tracking produced no results; skipping propagation")
            return
        diagmap = {
            sanitize_string(d.name).lower(): d
            for d in self._output_diagnostics() + [self.endObject]
        }
        # one task per output file, keyed by element name (avoids two threads writing
        # the same <name>.openpmd.hdf5 when a diagnostic name repeats)
        tasks: Dict = {}
        for key, (Xf, s_pos, idx, ref) in self.staged_results.items():
            diag = diagmap.get(key)
            if diag is None or diag.name in tasks:
                continue
            tasks[diag.name] = (diag, Xf, ref, s_pos, idx)
        for diag, Xf, ref, s_pos, idx in tasks.values():
            self.screen_threaded_function.scatter(diag, Xf, ref, s_pos, t_ref, idx)
        self.screen_threaded_function.gather()

    @staticmethod
    def read_trackone(filename: str) -> Dict:
        """
        Parse a MAD-X ``TRACK`` ``ONETABLE`` output file (``<name>-trackone``).

        The file holds one ``#segment ... <place>`` block per observation point, each
        followed by rows ``NUMBER TURN X PX Y PY T PT S E`` (one per surviving
        particle).

        Returns
        -------
        dict
            Maps the upper-cased observation-point name to a tuple
            ``(X, s, numbers)`` where ``X`` is the ``(6, N)`` array of canonical
            coordinates (ordered by particle number), ``s`` the s-position [m], and
            ``numbers`` the 1-based MAD-X particle numbers that survived.
        """
        seg_rows: Dict[str, list] = {}
        place = None
        with open(filename, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "#segment":
                    place = parts[-1]
                    seg_rows.setdefault(place, [])
                elif parts[0] in ("@", "*", "$"):
                    continue
                elif place is not None:
                    seg_rows[place].append(parts)
        result: Dict = {}
        for place, rows in seg_rows.items():
            if not rows:
                continue
            arr = np.array(rows, dtype=float)
            arr = arr[np.argsort(arr[:, 0])]  # order by particle NUMBER
            # a place can be written twice in one segment (an explicit OBSERVE plus
            # MAD-X's automatic end-of-range dump); keep one row per particle
            _, uniq = np.unique(arr[:, 0], return_index=True)
            arr = arr[uniq]
            result[place.upper()] = (
                arr[:, 2:8].T,          # X, PX, Y, PY, T, PT
                float(arr[0, 8]),       # S
                arr[:, 0].astype(int),  # NUMBER
            )
        return result

    @lox.thread(60)
    def screen_threaded_function(
        self, diag, Xf: np.ndarray, ref: Tuple, s_position: float, t_ref: float, indices
    ) -> None:
        """
        Build the propagated beam at one observation point from its final canonical
        coordinates and write its openPMD file (run in parallel across diagnostics).

        Parameters
        ----------
        diag:
            The diagnostic (or end) element the beam is written for.
        Xf:
            The ``(6, N)`` final canonical coordinates ``(x, px/p0, y, py/p0, -c*dt,
            pt)`` at ``diag``.
        ref:
            The ``(p0c, E0, Eref)`` reference triplet [eV].
        s_position:
            The arc length from the start of the line to ``diag`` [m].
        t_ref:
            The initial reference time [s].
        indices:
            Original (0-based) indices of the surviving particles, or ``None`` for all.
        """
        beam = self.global_parameters["beam"]
        out = self.canonical_to_beam(
            Xf, beam, ref[0], ref[1], ref[2], float(s_position), t_ref, indices
        )
        outname = f"{self.global_parameters['master_subdir']}/{diag.name}.openpmd.hdf5"
        rbf.openpmd.write_openpmd_beam_file(out, outname)
        self.files.append(outname)

    @staticmethod
    def extract_madx_map(row) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the MAD-X transfer map from a single row of the ``SECTORMAP`` table.

        Parameters
        ----------
        row:
            A row of the MAD-X sectormap TFS table (e.g. ``sectormap.iloc[-1]``),
            holding the ``K1..K6``, ``R11..R66`` and ``T111..T666`` columns.

        Returns
        -------
        tuple
            ``K`` (shape ``(6,)``), ``R`` (shape ``(6, 6)``) and ``T`` (shape
            ``(6, 6, 6)``), such that the final canonical coordinates are
            ``x_i = K_i + R_ij x_j + T_ijk x_j x_k``.
        """
        K = np.array([row[f"K{i}"] for i in range(1, 7)])
        R = np.array([[row[f"R{i}{j}"] for j in range(1, 7)] for i in range(1, 7)])
        T = np.array(
            [
                [[row[f"T{i}{j}{k}"] for k in range(1, 7)] for j in range(1, 7)]
                for i in range(1, 7)
            ]
        )
        return K, R, T

    @staticmethod
    def reference_from_headers(headers: Dict) -> Tuple[float, float, float]:
        """Return ``(p0c, E0, Eref)`` [eV] from MAD-X TWISS headers (``PC``, ``MASS``,
        ``ENERGY``, all in GeV)."""
        return (
            float(headers["PC"]) * 1e9,
            float(headers["MASS"]) * 1e9,
            float(headers["ENERGY"]) * 1e9,
        )

    @staticmethod
    def reference_from_beam(beam) -> Tuple[float, float, float]:
        """Return ``(p0c, E0, Eref)`` [eV] from the beam itself, matching the MAD-X
        reference set by the ``BEAM`` command (``GAMMA`` from the beam mean)."""
        rest = beam.particle_rest_energy_eV
        E0 = float(np.mean(np.asarray(rest.val if hasattr(rest, "val") else rest, dtype=float)))
        cp = np.asarray(beam.cp.val, dtype=float)
        Eref = float(np.mean(np.sqrt(cp**2 + E0**2)))
        p0c = float(np.sqrt(max(Eref**2 - E0**2, 0.0)))
        return p0c, E0, Eref

    @staticmethod
    def simba_to_canonical(beam, p0c: float, E0: float, Eref: float) -> Tuple[np.ndarray, float]:
        """
        Convert a SIMBA beam into MAD-X canonical coordinates
        ``(x, px/p0, y, py/p0, -c*dt, (E - Eref)/(p0*c))``.

        Returns the ``(6, N)`` coordinate array and the reference time ``t_ref`` [s].
        The coordinates are absolute deviations from the on-axis design orbit (we
        assume no closed-orbit distortion).
        """
        c = rbf.constants.speed_of_light
        t = np.asarray(beam.t.val, dtype=float)
        idx = beam.reference_particle_index
        t_ref = float(t[idx]) if idx is not None else float(np.mean(t))
        cpx = np.asarray(beam.cpx.val, dtype=float)   # [eV/c] (numerically p*c in eV)
        cpy = np.asarray(beam.cpy.val, dtype=float)
        cp = np.asarray(beam.cp.val, dtype=float)     # total momentum [eV/c]
        E = np.sqrt(cp**2 + E0**2)
        X = np.vstack(
            [
                np.asarray(beam.x.val, dtype=float),
                cpx / p0c,
                np.asarray(beam.y.val, dtype=float),
                cpy / p0c,
                -c * (t - t_ref),
                (E - Eref) / p0c,
            ]
        )
        return X, t_ref

    def canonical_to_beam(
        self,
        Xf: np.ndarray,
        beam,
        p0c: float,
        E0: float,
        Eref: float,
        s_position: float,
        t_ref: float,
        indices=None,
    ):
        """
        Convert final MAD-X canonical coordinates back into a SIMBA beam.

        The reference momentum ``p0c`` and rest/reference energies (``E0``, ``Eref``)
        define the conversion, so the reconstructed per-particle energy
        ``E = Eref + PT*p0c`` matches MAD-X exactly. ``indices`` gives the original
        (0-based) positions of the surviving particles, used to carry over the input
        beam's per-particle charge/status/mass (MAD-X may drop lost particles).

        Parameters
        ----------
        Xf:
            The ``(6, N)`` final canonical coordinates.
        beam:
            The initial SIMBA beam (source of species/charge/mass).
        p0c, E0, Eref:
            Reference momentum*c, rest energy and reference total energy [eV].
        s_position:
            Arc length to this location [m], used for the nominal time of flight and
            the output ``s``.
        t_ref:
            Initial reference time [s].
        indices:
            Original indices of the surviving particles, or ``None`` for all.

        Returns
        -------
        :class:`~simba.Modules.Beams.beam`
            The propagated beam.
        """
        UnitValue = rbf.UnitValue
        c = rbf.constants.speed_of_light
        xf, PXf, yf, PYf, Tf, PTf = Xf
        n_in = len(np.asarray(beam.x.val))
        if indices is None:
            indices = np.arange(xf.shape[0])

        def sel(value, units):
            if value is None:
                return None
            a = np.asarray(value.val if hasattr(value, "val") else value)
            if a.ndim > 0 and a.shape[0] == n_in:
                a = a[indices]
            return UnitValue(a, units)

        cpxf = PXf * p0c
        cpyf = PYf * p0c
        Ef = Eref + PTf * p0c
        cpf = np.sqrt(np.clip(Ef**2 - E0**2, 0.0, None))
        cpzf = np.sqrt(np.clip(cpf**2 - cpxf**2 - cpyf**2, 0.0, None))
        q_over_c = beam.q_over_c
        beta0 = p0c / Eref
        # the synchronous particle arrives after the nominal time of flight
        t_ref_out = t_ref + s_position / (beta0 * c)
        tf = t_ref_out - Tf / c

        out = rbf.beam()
        out.species = beam.species
        out._beam.particle_mass = sel(beam._beam.particle_mass, "kg")
        out._beam.particle_charge = sel(beam._beam.particle_charge, "C")
        out._beam.particle_rest_energy = sel(beam._beam.particle_rest_energy, "J")
        out._beam.particle_rest_energy_eV = beam._beam.particle_rest_energy_eV
        out._beam.charge = sel(beam._beam.charge, "C")
        out._beam.total_charge = beam._beam.total_charge
        out._beam.nmacro = sel(beam._beam.nmacro, "")
        out._beam.status = sel(beam._beam.status, "")
        out._beam.x = UnitValue(xf, "m")
        out._beam.y = UnitValue(yf, "m")
        out._beam.px = UnitValue(cpxf * q_over_c, "kg*m/s")
        out._beam.py = UnitValue(cpyf * q_over_c, "kg*m/s")
        out._beam.pz = UnitValue(cpzf * q_over_c, "kg*m/s")
        out._beam.t = UnitValue(tf, "s")
        # z is the absolute longitudinal position: the plane's s-position plus each
        # particle's deviation from the synchronous particle (matching the ocelot /
        # xsuite beam readers, which add ``zstart``). Without the offset the bunch
        # would sit at z=0 instead of its actual location along the line.
        z_plane = self.startObject.physical.start.z + s_position
        out._beam.z = UnitValue(z_plane + (-1 * out.Bz * c) * (tf - t_ref_out), "m")
        out._beam.s = UnitValue(z_plane, "m")
        idx = beam.reference_particle_index
        if idx is not None:
            pos = np.where(np.asarray(indices) == idx)[0]
            if len(pos):
                out.reference_particle_index = int(pos[0])
                out.reference_particle = [
                    np.asarray(getattr(out, coord))[int(pos[0])]
                    for coord in out.reference_particle_coords
                ]
        return out

    def run(self) -> None:
        """
        Run MAD-X on the generated input file.

        Uses the `cpymad` python bindings if available; otherwise falls back to
        calling the ``madx`` executable defined in
        :download:`Executables <../Executables.yaml>`, feeding the input file via stdin.
        """
        if self.remote_setup:
            self.run_remote()
            return
        if self.staged:
            self.run_staged()
            return
        subdir = self.global_parameters["master_subdir"]
        command_file = self.objectname + ".madx"
        logfile = os.path.abspath(subdir + "/" + self.objectname + ".log")
        try:
            from cpymad.madx import Madx
        except ImportError:
            Madx = None
        if Madx is not None:
            # chdir=True runs MAD-X from the input file's directory, so relative
            # output paths (e.g. the TWISS .tfs file) resolve into master_subdir.
            with open(logfile, "w") as log:
                self.madx = Madx(stdout=log)
                try:
                    self.madx.call(
                        file=os.path.abspath(os.path.join(subdir, command_file)),
                        chdir=True,
                    )
                except Exception as e:
                    print(e)
                    self.madx.quit()
        else:
            command = self.executables[self.code]
            with open(subdir + "/" + command_file, "r") as inp, open(
                logfile, "w"
            ) as f:
                subprocess.call(command, stdin=inp, stdout=f, cwd=subdir)

    def run_staged(self) -> None:
        """
        Run MAD-X with **energy-staged** tracking (see :func:`_should_stage`).

        MAD-X ``TRACK`` uses a constant reference momentum ``p0`` fixed by ``BEAM``,
        so after acceleration each particle carries a huge ``PT`` and the dispersive
        elements (chicane dipoles) act on that full offset -- corrupting the beam.
        The fix is to split the line at each accelerating cavity, run each segment
        with its own ``BEAM`` reference (set to that segment's energy), and rescale
        the canonical coordinates between segments so the next one sees ``PT ~ 0``.
        This is what ELEGANT does internally (its ``RFCA`` updates the reference).

        Drives `cpymad` interactively: loads the sequence + runs the (constant-energy)
        TWISS summary from the ``.madx`` file, installs a boundary marker just after
        each accelerating cavity via ``SEQEDIT``, ``MAKETHIN`` s once, then tracks
        segment by segment (:func:`_track_segments`).

        The per-segment commands (with the runtime ``START`` values from each
        re-referenced hand-off) are appended to the ``.madx`` file, so the staged
        tracking is inspectable and re-runnable even though it cannot be a single
        static file (the hand-off coordinates are computed in Python between runs).
        """
        from cpymad.madx import Madx

        subdir = self.global_parameters["master_subdir"]
        name = sanitize_string(self.name)
        command_file = os.path.abspath(os.path.join(subdir, self.objectname + ".madx"))
        logfile = os.path.abspath(subdir + "/" + self.objectname + ".log")
        beam = self.global_parameters["beam"]
        _, _, Eref0 = self.reference_from_beam(beam)
        cmds: list = []  # MAD-X commands issued after the .madx file, appended back to it
        with open(logfile, "w") as log:
            self.madx = Madx(stdout=log)
            madx = self.madx
            try:
                # defines BEAM + sequence (staged: no single-pass TWISS/TRACK in file)
                madx.call(file=command_file, chdir=True)
                # accelerating cavities (resolved VOLT/LAG) -> stage boundaries + ΔE
                cavs = []
                for el in madx.sequence[name].elements:
                    if el.base_type.name == "rfcavity":
                        dE = float(el.volt) * 1e6 * np.sin(2 * np.pi * float(el.lag))
                        if abs(dE) > 0.01 * Eref0:
                            cavs.append((el.name, float(el.l), dE))
                # install a zero-length boundary marker just after each such cavity
                self._emit(madx, f"\nSEQEDIT, SEQUENCE={name};", cmds)
                for cname, clen, _ in cavs:
                    self._emit(
                        madx,
                        f"INSTALL, ELEMENT={cname}_stg, CLASS=marker, "
                        f"AT={clen / 2}, FROM={cname};",
                        cmds,
                    )
                self._emit(madx, "FLATTEN;\nENDEDIT;", cmds)
                self._emit(madx, f"USE, SEQUENCE={name};", cmds)
                # staged TWISS on the thick lattice (bounded beta + ramped energy).
                # In single-particle mode this also populates staged_results by applying
                # the (energy-correct) transfer matrices to the input distribution.
                # Non-fatal: a twiss failure must not stop the (primary) staged track.
                try:
                    self._twiss_segments(madx, name, cavs, cmds)
                except Exception as e:
                    warn(f"MAD-X staged TWISS failed ({e}); -twiss.tfs may be stale")
                # single-particle: beams come from the transfer maps (above); otherwise
                # MAKETHIN and track the full distribution one segment at a time
                if not self.single_particle:
                    self._emit(madx, 'SELECT, FLAG=makethin, PATTERN=".*", THICK=true;', cmds)
                    self._emit(madx, f"MAKETHIN, SEQUENCE={name};", cmds)
                    self._emit(madx, f"USE, SEQUENCE={name};", cmds)
                    self.staged_results = self._track_segments(madx, name, cavs, cmds)
            except Exception as e:
                print(e)
                madx.quit()
        if cmds:
            with open(command_file, "a") as f:
                f.write("\n\n! ---- energy-staged tracking (run_staged) ----\n")
                f.write("\n".join(cmds) + "\n")

    @staticmethod
    def _emit(madx, cmd: str, log: list | None = None) -> None:
        """Send ``cmd`` to MAD-X and, if ``log`` is given, record it for writing back
        to the ``.madx`` file (so the staged run is inspectable / re-runnable)."""
        if log is not None:
            log.append(cmd)
        madx.input(cmd)

    def _track_segments(self, madx, name: str, cavs: list, log: list = None) -> Dict:
        """
        Track the beam through the line one constant-energy segment at a time,
        re-referencing the momentum at each cavity boundary.

        Each segment ``i`` spans ``[boundary_i, boundary_{i+1}]`` (boundaries are the
        line start, the ``*_stg`` markers just after each accelerating cavity, and the
        line end) and is tracked with ``BEAM`` set to the energy at its start, so every
        element sees ``PT ~ spread`` only. Diagnostics are observed within their
        segment and recorded with that segment's reference. Between segments the
        coordinates at the boundary are rescaled to the next reference
        (:func:`_rescale_reference`), and lost particles are pruned from the running
        map of original indices.

        Returns the :attr:`staged_results` dict.
        """
        beam = self.global_parameters["beam"]
        p0c, E0, Eref = self.reference_from_beam(beam)
        X, _ = self.simba_to_canonical(beam, p0c, E0, Eref)

        spos = {e.name: float(e.at) for e in madx.sequence[name].elements}
        names = [e.name for e in madx.sequence[name].elements]
        bounds = [names[0]] + [f"{c[0]}_stg" for c in cavs] + [names[-1]]
        sb = [spos[b] for b in bounds]
        # diagnostics (and the end element) to observe, keyed by MAD-X element name
        diags = {}
        for diag in self._output_diagnostics() + [self.endObject]:
            key = sanitize_string(diag.name).lower()
            if key in spos:
                diags[key] = diag

        results: Dict = {}
        orig = np.arange(X.shape[1])  # original 0-based indices still being tracked
        p = p0c
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            Eref_i = float(np.sqrt(p**2 + E0**2))
            ref_i = (p, E0, Eref_i)
            # diagnostics inside this segment (half-open, so a diag on a boundary
            # falls in the next segment), plus the boundary for the hand-off
            in_seg = [k for k in diags if sb[i] - 1e-6 <= spos[k] < sb[i + 1] - 1e-9]
            if log is not None:
                log.append(f"\n! segment {i}: {a} -> {b}  (ref {Eref_i / 1e6:.3f} MeV)")
            out, nums = self._run_one_segment(
                madx, name, a, b, Eref_i / E0, X, in_seg + [b], log
            )
            for k in in_seg:
                if k in out:
                    results[k] = (out[k], spos[k], orig[nums[k] - 1], ref_i)
            if b in diags and b in out:  # e.g. the line-end marker
                results[b] = (out[b], spos[b], orig[nums[b] - 1], ref_i)
            if b not in out or len(nums[b]) == 0:
                warn(f"MAD-X staged tracking: no particles reached boundary {b}")
                break
            # the final boundary is the MAD-X '<seq>$end' marker, which the endObject
            # (frameworkLattice.end) sits at but is not itself a track boundary, so it
            # never lands in `diags`. Attribute the end dump to it so the final beam is
            # always produced (setdefault: don't clobber an interior endObject capture).
            if i == len(bounds) - 2:
                ek = sanitize_string(self.endObject.name).lower()
                results.setdefault(
                    ek, (out[b], spos.get(ek, sb[-1]), orig[nums[b] - 1], ref_i)
                )
            # hand off to the next segment: rescale to the new mean momentum
            Xb = out[b]
            E = Eref_i + Xb[5] * p
            p_new = float(np.sqrt(np.clip(E**2 - E0**2, 0.0, None)).mean())
            X = self._rescale_reference(Xb, p, p_new, E0)
            orig = orig[nums[b] - 1]
            p = p_new
        return results

    def _run_one_segment(
        self, madx, name: str, a: str, b: str, gamma: float, X: np.ndarray,
        obs: list, log: list = None,
    ) -> Tuple[Dict, Dict]:
        """
        Track one segment ``[a, b]`` with ``BEAM`` at ``gamma`` and starting
        coordinates ``X`` (6, N), observing at the element names in ``obs``. If ``log``
        is given, the issued commands are recorded (see :func:`_emit`).

        Returns ``(coords, numbers)`` where ``coords[name]`` is the ``(6, N)`` canonical
        array at that observation point and ``numbers[name]`` the 1-based MAD-X particle
        numbers that reached it (order-matched to ``coords``).
        """
        subdir = self.global_parameters["master_subdir"]
        # absolute path: cpymad restores the working directory after the initial
        # call(chdir=True), so a bare FILE= would land in the process cwd, not subdir.
        trackfile = os.path.abspath(os.path.join(subdir, f"{name}-stg"))
        self._emit(madx, f"BEAM, PARTICLE={self.global_parameters['beam'].species.upper()}, "
                   f"GAMMA={gamma}, SEQUENCE={name};", log)
        self._emit(madx, f"USE, SEQUENCE={name}, RANGE={a}/{b};", log)
        self._emit(madx, f'TRACK, ONEPASS, DUMP, ONETABLE, FILE="{trackfile}";', log)
        for o in dict.fromkeys(obs):
            self._emit(madx, f"OBSERVE, PLACE={o};", log)
        for j in range(X.shape[1]):
            self._emit(
                madx,
                "START, X={:.15g}, PX={:.15g}, Y={:.15g}, PY={:.15g}, "
                "T={:.15g}, PT={:.15g};".format(*X[:, j]),
                log,
            )
        self._emit(madx, "RUN, TURNS=1;", log)
        self._emit(madx, "ENDTRACK;", log)
        segments = self.read_trackone(f"{trackfile}one")
        coords = {k.lower(): v[0] for k, v in segments.items()}
        numbers = {k.lower(): v[2] for k, v in segments.items()}
        return coords, numbers

    @staticmethod
    def _rescale_reference(
        X: np.ndarray, p_old: float, p_new: float, E0: float,
        synchronous: bool = False,
    ) -> np.ndarray:
        """
        Re-express canonical coordinates ``X`` (6, N) about a new reference momentum.

        The physical particle is unchanged; only the reference momentum changes from
        ``p_old`` to ``p_new`` [eV/c]. The transverse divergences ``px, py`` (which MAD-X
        normalises by the reference momentum) scale by ``p_old/p_new``; ``x, y, T`` are
        unchanged.

        ``pt`` handling depends on how the beam reached the boundary:

        * *tracking* (``synchronous=False``): the beam was really pushed through the
          cavity, so its energy is already correct and ``p_new`` is its actual mean
          momentum. ``pt`` is recomputed as ``(E - Eref_new)/p_new`` so the next segment
          sees only the residual energy spread.
        * *transfer map* (``synchronous=True``): a constant-reference-energy TWISS
          R-matrix carries the energy spread / chirp but **not** the synchronous energy
          gain, so it must be added here. The synchronous particle gains exactly the
          reference step (mean ``pt`` stays 0) and the spread renormalises by
          ``p_old/p_new`` (adiabatic damping): ``pt -> pt * p_old/p_new``.
        """
        Y = X.copy()
        Y[1] = X[1] * (p_old / p_new)
        Y[3] = X[3] * (p_old / p_new)
        if synchronous:
            Y[5] = X[5] * (p_old / p_new)
        else:
            Eref_old = float(np.sqrt(p_old**2 + E0**2))
            Eref_new = float(np.sqrt(p_new**2 + E0**2))
            E = Eref_old + X[5] * p_old
            Y[5] = (E - Eref_new) / p_new
        return Y

    def _twiss_segments(self, madx, name: str, cavs: list, log: list = None) -> None:
        """
        Energy-staged ``TWISS`` -> a bounded ``<name>-twiss.tfs``.

        A constant-reference-momentum ``TWISS`` blows the beta functions up on an
        accelerating line because MAD-X's ``rfcavity`` has a transverse effect that
        scales like 1/energy -- at a fixed low reference it defocuses far too hard.
        Twissing each segment at its own (ramped) reference energy keeps the optics
        bounded. The optics are threaded across cavity boundaries with the reference
        transform: ``beta`` and the dispersion ``dx`` scale by ``p_new/p_old`` (the
        divergence normalisation changes), ``alpha`` and ``dpx`` are unchanged; the
        phase advance ``mu`` accumulates. The per-row ``ENERGY`` is the segment energy,
        so the twiss reader also recovers the ramped energy profile.
        """
        import tfs

        beam = self.global_parameters["beam"]
        tw = beam.twiss
        p0c, E0, Eref = self.reference_from_beam(beam)
        names = [e.name for e in madx.sequence[name].elements]
        # global s-positions of every element (captured before the RANGE USEs below,
        # which make TWISS report S local to the range and would otherwise overlap)
        spos_global = {e.name: float(e.at) for e in madx.sequence[name].elements}
        bounds = [names[0]] + [f"{c[0]}_stg" for c in cavs] + [names[-1]]
        # initial optics (MAD-X DX ~ physical eta since beta_rel ~ 1 here)
        bx, ax = float(tw.beta_x.val), float(tw.alpha_x.val)
        by, ay = float(tw.beta_y.val), float(tw.alpha_y.val)
        dx, dpx = float(tw.eta_x.val), float(tw.eta_xp.val)
        dy, dpy = float(tw.eta_y.val), float(tw.eta_yp.val)

        keys = ["NAME", "KEYWORD", "S", "BETX", "ALFX", "MUX", "BETY", "ALFY",
                "MUY", "DX", "DPX", "DY", "DPY", "X", "Y", "ENERGY", "L", "SIGMA_T"]
        cols = {k: [] for k in keys}
        # longitudinal sigma matrix S_L = [[<T,T>, <T,pt>], [<T,pt>, <pt,pt>]] in
        # (T [m], pt) coordinates, propagated through the twiss R-matrix so the cavity
        # R65 builds the chirp and the chicane R56 compresses it (a linear-optics twiss
        # otherwise only carries a constant SIGT). Initial correlation assumed 0.
        c_light = rbf.constants.speed_of_light
        mx = self.hdf5_to_madx()
        sig_T0, sig_d0 = float(mx["SIGT"]), float(mx["SIGE"])
        SL = np.array([[sig_T0**2, 0.0], [0.0, sig_d0**2]])
        p, mux0, muy0 = p0c, 0.0, 0.0
        # single-particle mode: instead of tracking the distribution, apply the
        # (energy-staged) TWISS transfer matrix to the input distribution at each
        # screen. X_map is the distribution propagated segment-by-segment.
        map_beams = self.single_particle
        if map_beams:
            X_map, _ = self.simba_to_canonical(beam, p0c, E0, Eref)
            orig_map = np.arange(X_map.shape[1])
            map_results: Dict = {}
            diagmap = {
                sanitize_string(d.name).lower(): d
                for d in self._output_diagnostics() + [self.endObject]
            }
            end_key = sanitize_string(self.endObject.name).lower()
            end_beam = None  # last segment's end state -> guaranteed frameworkLattice.end
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            Eref_i = float(np.sqrt(p**2 + E0**2))
            ref_i = (p, E0, Eref_i)  # segment reference triplet for reconstruction
            self._emit(madx, f"BEAM, PARTICLE={beam.species.upper()}, "
                       f"GAMMA={Eref_i / E0}, SEQUENCE={name};", log)
            self._emit(madx, f"USE, SEQUENCE={name}, RANGE={a}/{b};", log)
            try:
                self._emit(
                    madx,
                    f"TWISS, SEQUENCE={name}, RANGE={a}/{b}, RMATRIX, BETX={bx}, ALFX={ax}, "
                    f"BETY={by}, ALFY={ay}, DX={dx}, DPX={dpx}, DY={dy}, DPY={dpy};",
                    log,
                )
            except Exception as e:
                # write whatever ramped/bounded twiss we have so far rather than
                # abandoning it (and leaving a stale constant-energy file)
                warn(f"MAD-X staged TWISS diverged at segment {i} ({a} -> {b}): {e}")
                break
            t = madx.table.twiss
            col = {k: np.array(getattr(t, k.lower())) for k in
                   ["s", "betx", "alfx", "mux", "bety", "alfy", "muy",
                    "dx", "dpx", "dy", "dpy", "x", "y", "l",
                    "re55", "re56", "re65", "re66"]}
            nm = np.array(t.name, dtype=str)
            kw = np.array(t.keyword, dtype=str)
            n = len(col["s"])
            rng = slice(1 if i > 0 else 0, n)  # drop the duplicated boundary row
            # bunch length at each row: S_L(s) = M(s) S_L_start M(s)^T with the
            # segment-accumulated longitudinal 2x2 map M = [[R55,R56],[R65,R66]]
            sigt_row = np.empty(n)
            for j in range(n):
                M = np.array([[col["re55"][j], col["re56"][j]],
                              [col["re65"][j], col["re66"][j]]])
                sigt_row[j] = np.sqrt(max((M @ SL @ M.T)[0, 0], 0.0)) / c_light
            # RANGE-restricted TWISS reports S local to the segment; shift it back to
            # the sequence-local s of the segment's start boundary (so segments don't
            # overlap). The twiss FILE additionally gets the section's absolute start z
            # (s_global) so its frame matches the beams; the beams themselves are stored
            # with the sequence-local s because canonical_to_beam re-adds z0.
            z0 = self.startObject.physical.start.z
            s_local = col["s"] + (spos_global[a] - float(col["s"][0]))
            s_global = s_local + z0
            # single-particle beam generation: apply the full 6x6 transfer matrix
            # (from this segment's start) to the propagated distribution at each screen
            if map_beams:
                RE = np.array([[np.array(getattr(t, f"re{aa}{bb}"))
                                for bb in range(1, 7)] for aa in range(1, 7)])  # (6,6,n)
                # MAD-X twiss names carry a ':<occurrence>' suffix (e.g. 'name:1');
                # strip it so the clean sanitised diagnostic names match
                row_of = {}
                for j in range(n):
                    row_of.setdefault(str(nm[j]).split(":")[0].lower(), j)
                # observe each requested diagnostic (diagmap already respects
                # generate_beams and always includes the final element) at its own row
                for key in diagmap:
                    if key in row_of and key not in map_results:
                        j = row_of[key]
                        map_results[key] = (
                            RE[:, :, j] @ X_map, s_local[j], orig_map, ref_i
                        )
                # advance to the segment end; remember it as a fallback end-of-lattice
                # beam in case the end element never matched a twiss row above
                X_map = RE[:, :, -1] @ X_map
                end_beam = (X_map, s_local[-1], orig_map, ref_i)
            cols["NAME"] += list(nm[rng])
            cols["KEYWORD"] += list(kw[rng])
            cols["S"] += list(s_global[rng])
            for k in ["BETX", "ALFX", "BETY", "ALFY", "DX", "DPX", "DY", "DPY", "X", "Y", "L"]:
                cols[k] += list(col[k.lower()][rng])
            cols["MUX"] += list(col["mux"][rng] + mux0)
            cols["MUY"] += list(col["muy"][rng] + muy0)
            cols["SIGMA_T"] += list(sigt_row[rng])
            cols["ENERGY"] += [Eref_i / 1e9] * (n - (1 if i > 0 else 0))
            mux0 += float(col["mux"][-1])
            muy0 += float(col["muy"][-1])
            # carry the longitudinal sigma to the segment end
            Mend = np.array([[col["re55"][-1], col["re56"][-1]],
                             [col["re65"][-1], col["re66"][-1]]])
            SL = Mend @ SL @ Mend.T
            if i < len(cavs):  # re-reference across the cavity at this boundary
                Enew = Eref_i + cavs[i][2]
                p_new = float(np.sqrt(max(Enew**2 - E0**2, 0.0)))
                r = p_new / p
                bx, by = float(col["betx"][-1]) * r, float(col["bety"][-1]) * r
                ax, ay = float(col["alfx"][-1]), float(col["alfy"][-1])
                dx, dy = float(col["dx"][-1]) * r, float(col["dy"][-1]) * r
                dpx, dpy = float(col["dpx"][-1]), float(col["dpy"][-1])
                # pt spread scales by p_old/p_new (delta = (E-Eref)/p); T unchanged
                D = np.array([[1.0, 0.0], [0.0, p / p_new]])
                SL = D @ SL @ D.T
                if map_beams:  # hand the propagated distribution to the next ref,
                    # adding the synchronous energy gain the TWISS map omits
                    X_map = self._rescale_reference(
                        X_map, p, p_new, E0, synchronous=True
                    )
                p = p_new

        if map_beams:
            # always emit the final element's beam (frameworkLattice.end), matching the
            # sectormap path, even when generate_beams skipped the intermediate screens
            if end_beam is not None and end_key not in map_results:
                map_results[end_key] = end_beam
            self.staged_results = map_results

        if not cols["S"]:
            warn("MAD-X staged TWISS produced no rows; -twiss.tfs not written")
            return
        gamma_inj, beta_inj = Eref / E0, p0c / Eref
        headers = {
            "NAME": "TWISS", "TYPE": "TWISS", "SEQUENCE": name.upper(),
            "PARTICLE": beam.species.upper(), "MASS": E0 / 1e9,
            "ENERGY": Eref / 1e9, "PC": p0c / 1e9, "GAMMA": gamma_inj,
            "EX": float(mx["EXN"]) / (beta_inj * gamma_inj),
            "EY": float(mx["EYN"]) / (beta_inj * gamma_inj),
            "SIGE": float(mx["SIGE"]), "SIGT": float(mx["SIGT"]),
        }
        subdir = self.global_parameters["master_subdir"]
        df = tfs.TfsDataFrame(cols)
        df.headers = headers
        # write to <name>-twiss.tfs (self.name), the same file the constant-energy
        # madx_twiss_block would produce and the Twiss reader globs, so the staged
        # (ramped-energy, bounded-beta) table overwrites it rather than sitting under
        # a different name while a stale constant-energy file is read.
        outfile = os.path.abspath(os.path.join(subdir, f"{name}-twiss.tfs"))
        tfs.write(outfile, df)
        self.files.append(outfile)
