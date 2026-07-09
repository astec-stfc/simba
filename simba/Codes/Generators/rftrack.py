"""
This file generates an RF-Track cathode/emission beam from the provided
parameters, as a drop-in replacement for :class:`~simba.Codes.Generators.astra.ASTRAGenerator`.

Unlike ASTRA/GPT (external executables driven through an input deck), RF-Track
is a pure-Python in-process library: its ``Bunch6dT_Generator`` (reference
manual §7) builds the emission bunch directly, so ``write``/``run``/``postProcess``
call the library in memory instead of writing a deck and shelling out. The rest
of the RF-Track integration follows the same in-process pattern — see
``laura/RFTrack/PLAN.md``.
"""
from typing import Any

from laura.translator.conversion_rules.codes import rftrack_conversion

from .Generators import frameworkGenerator
from ...Modules import Beams as rbf
from ...Modules.Beams import rftrack as rbf_rftrack

# Generic frameworkGenerator attribute -> (RF-Track Bunch6dT_Generator option,
# multiplier from SI to RF-Track units). RF-Track uses mm / ns / MeV-c / nC and
# option names that deliberately echo ASTRA's generator (§7.2 of the manual).
_ALIASES = {
    "charge": ("q_total", 1e9),                          # C -> nC
    "reference_time": ("ref_clock", 1e9),                # s -> ns
    "sigma_x": ("sig_x", 1e3),                           # m -> mm
    "sigma_y": ("sig_y", 1e3),
    "sigma_z": ("sig_z", 1e3),
    "sigma_t": ("sig_t", 1e9),                           # s -> ns
    "offset_x": ("x_off", 1e3),
    "offset_y": ("y_off", 1e3),
    "gaussian_cutoff_x": ("c_sig_x", 1),
    "gaussian_cutoff_y": ("c_sig_y", 1),
    "normalized_horizontal_emittance": ("nemit_x", 1e6),  # m rad -> mm mrad
    "normalized_vertical_emittance": ("nemit_y", 1e6),
    "noise_reduction": ("noise_reduc", 1),
    "work_function_ev": ("phi_eff", 1),                  # eV
}

# Generic distribution_type_* letter codes -> RF-Track distribution strings.
# Anything not listed (e.g. "fd_300" Fermi-Dirac photoemission) is passed
# through unchanged, so users can set native RF-Track distribution names.
_DIST = {
    "g": "gaussian", "gaussian": "gaussian", "2dgaussian": "gaussian",
    "r": "gaussian", "radial": "gaussian",
    "u": "plateau", "uniform": "plateau",
    "p": "plateau", "plateau": "plateau", "flattop": "plateau",
}


class RFTrackGenerator(frameworkGenerator):
    """
    Generate a photocathode/emission beam using RF-Track's ``Bunch6dT_Generator``
    (reference manual §7), as a drop-in replacement for
    :class:`~simba.Codes.Generators.astra.ASTRAGenerator`.
    """

    # Widen the base class's strict Literal distribution types to plain strings:
    # RF-Track accepts richer names than the ASTRA/GPT letter codes, notably
    # ``fd_300`` (Fermi-Dirac photoemission) for ``dist_pz`` (manual §7.4).
    distribution_type_x: str = "r"
    distribution_type_y: str = "r"
    distribution_type_z: str = "g"
    distribution_type_pz: str = "i"

    gen_obj: Any = None
    """The built ``RF_Track.Bunch6dT_Generator`` (set by :func:`write`)."""

    bunch: Any = None
    """The emitted ``RF_Track.Bunch6dT`` (set by :func:`run`)."""

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.code = "rftrack"

    def write(self) -> None:
        """Build the ``Bunch6dT_Generator`` and set its options from this class."""
        rft = rftrack_conversion.get_rftrack()
        G = rft.Bunch6dT_Generator()

        species = self.species if self.species.endswith("s") else self.species + "s"
        G.species = species
        G.cathode = bool(self.cathode)

        # Extract extra fields (user-supplied or from YAML defaults).
        extra = self.__pydantic_extra__ or {}

        for attr, (rft_attr, mult) in _ALIASES.items():
            val = getattr(self, attr, None)
            if val is not None:
                # Don't multiply booleans (SWIG enforces strict types).
                if isinstance(val, bool):
                    setattr(G, rft_attr, val)
                else:
                    setattr(G, rft_attr, val * mult)

        for attr, rft_attr in (
            ("distribution_type_x", "dist_x"),
            ("distribution_type_y", "dist_y"),
            ("distribution_type_z", "dist_z"),
            ("distribution_type_pz", "dist_pz"),
        ):
            v = getattr(self, attr, None)
            if v:
                setattr(G, rft_attr, _DIST.get(v.lower(), v))

        # Plateau distributions require Lt (length) or rt (rise time). If not
        # explicitly set (e.g. via load_defaults), use fallback defaults.
        # ponytail: user should ideally call load_defaults("clara_400_*") after
        # change_generator() to load the full preset; this fallback applies if
        # they don't.
        dist_z_str = str(getattr(self, "distribution_type_z", "")).lower()
        if dist_z_str in ("p", "plateau", "u", "uniform", "flattop"):
            if "Lt" not in extra or extra.get("Lt") is None:
                G.Lt = 1.0  # ns — sensible default
            else:
                G.Lt = extra["Lt"]
            if "rt" not in extra or extra.get("rt") is None:
                G.rt = 0.2  # ns — reasonable rise time
            else:
                G.rt = extra["rt"]

        # Longitudinal cutoff: RF-Track keeps separate c_sig_z (3D) and c_sig_t
        # (emission time); feed both from the single generic cutoff.
        G.c_sig_z = self.gaussian_cutoff_z
        if self.cathode:
            G.c_sig_t = self.gaussian_cutoff_z

        # Map CLARA-style plateau parameters to RF-Track names (s → ns).
        if "plateau_bunch_length" in extra:
            G.Lt = extra["plateau_bunch_length"] * 1e9  # s -> ns
        if "plateau_rise_time" in extra:
            G.rt = extra["plateau_rise_time"] * 1e9  # s -> ns

        # Forward any native RF-Track generator option the user set directly as
        # an extra field (model_config extra="allow"), e.g. e_photon, phi_eff,
        # rand_generator, shape_x/y/z, disp_x/y, cor_px/py, ref_ekin, ref_zpos.
        for name in _NATIVE_OPTIONS:
            if name in extra:
                setattr(G, name, extra[name])

        self.gen_obj = G

    def run(self) -> None:
        """Emit the bunch: ``B0 = Bunch6dT(G, Nparticles)`` (manual §7.4)."""
        rft = rftrack_conversion.get_rftrack()
        self.bunch = rft.Bunch6dT(self.gen_obj, self.particles)

    def postProcess(self) -> None:
        """Convert the emitted bunch to a SIMBA beam and write it to openPMD HDF5."""
        self.global_parameters["beam"] = rbf.beam()
        rbf_rftrack.bunch6dt_to_beam(self.global_parameters["beam"], self.bunch)
        filename = "laser.openpmd.hdf5" if self.cathode else self.filename
        rbf.openpmd.write_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + filename,
        )


# RF-Track Bunch6dT_Generator options with no generic frameworkGenerator
# equivalent — forwarded verbatim if the user supplies them (see §7.2).
_NATIVE_OPTIONS = (
    "e_photon", "phi_eff", "rand_generator", "p_ref", "ref_ekin", "ref_zpos",
    "shape_x", "shape_y", "shape_z", "disp_x", "disp_y",
    "cor_px", "cor_py", "cor_ekin", "emit_z", "emit_t",
    "lx", "ly", "lz", "rx", "ry", "rz",
    "Lt", "rt",  # temporal plateau parameters (for dist_z = "plateau" in cathode mode)
)
