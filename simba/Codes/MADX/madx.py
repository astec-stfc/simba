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
import shlex
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
    """Whether to ``TRACK`` a single particle built from the beam's average
    (centroid) coordinates, rather than one ``START`` per macroparticle. This
    gives the reference trajectory and energy profile (all we need through
    accelerating cavities) at a fraction of the cost."""

    madx: Any | None = None
    """Instance of `Madx` from `cpymad`"""

    screen_threaded_function: ClassVar[ScatterGatherDescriptor] = (
        ScatterGatherDescriptor
    )
    """Threaded function for propagating the beam to each diagnostic and writing
    its output (see :func:`screen_threaded_function`)"""


    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.particle_definition = self.elementObjects[self.start].name

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

    def write(self) -> None:
        """
        Write the MAD-X input file to `master_subdir` by calling the ``LAURA``
        translator's :func:`section.to_madx`, passing in the SIMBA generic beam
        object (:attr:`global_parameters["beam"]`) as the ``beam`` argument, then
        appending either a ``TRACK`` block (:func:`madx_track_block`) or a
        ``TWISS``/``SECTORMAP`` block (:func:`madx_twiss_block`).
        """
        tracking = self.tracking_mode()
        # TRACK needs a thin (MAKETHIN) lattice, which in turn needs refer=centre.
        refer = "centre" if tracking else "entry"
        output = self.section.to_madx(beam=self.hdf5_to_madx(), refer=refer)
        command_file = (
            self.global_parameters["master_subdir"] + "/" + self.objectname + ".madx"
        )
        if tracking:
            output += self.madx_track_block()
        else:
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

        With :attr:`single_particle` (the default), a single ``START`` is emitted
        from the beam centroid; otherwise one ``START`` per macroparticle.
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
        for diag in self.screens_and_markers_and_bpms:
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
        if self.tracking_mode():
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
        for diag in self.screens_and_markers_and_bpms:
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
        for diag in self.screens_and_markers_and_bpms:
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
        out._beam.z = UnitValue((-1 * out.Bz * c) * (tf - t_ref_out), "m")
        out._beam.s = UnitValue(self.startObject.physical.start.z + s_position, "m")
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

    def apply_madx_map(
        self,
        beam,
        K: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        headers: Dict,
        s_position: float,
    ):
        """
        Apply a MAD-X (constant-energy) sectormap transfer map to a SIMBA beam and
        return the propagated beam.

        The beam is converted to MAD-X canonical coordinates
        (:func:`simba_to_canonical`), the second-order map
        ``x_i = K_i + R_ij x_j + T_ijk x_j x_k`` is applied, and the result is
        converted back (:func:`canonical_to_beam`). The reference (``PC``, ``MASS``,
        ``ENERGY``) is taken from the TWISS headers. Note this cannot represent an
        energy change through cavities; use ``TRACK`` for that.
        """
        p0c, E0, Eref = self.reference_from_headers(headers)
        X, t_ref = self.simba_to_canonical(beam, p0c, E0, Eref)
        Xf = K[:, None] + R @ X + np.einsum("ijk,jn,kn->in", T, X, X)
        return self.canonical_to_beam(Xf, beam, p0c, E0, Eref, s_position, t_ref)

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
