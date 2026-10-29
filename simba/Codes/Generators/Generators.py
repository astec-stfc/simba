"""
Beam generators.

Each code's ``<code>.yaml`` lists the :class:`frameworkGenerator` attributes it
accepts, and ``aliases.yaml`` maps the generic names (with unit multipliers) to
the code's own.
"""

import os
import numpy as np
from pydantic import (
    BaseModel,
    model_validator,
    field_validator,
    confloat,
    ConfigDict,
    ValidationInfo,
)
from typing import Literal, Dict, Any, List
from ...Modules import constants
from ...Modules.units import UnitValue
from ...Modules import Beams as rbf
import yaml
from easygdf import load
import warnings

with open(
        os.path.dirname(os.path.abspath(__file__)) + "/astra.yaml",
) as infile:
    astra_generator_keywords = yaml.safe_load(infile)

with open(
        os.path.dirname(os.path.abspath(__file__)) + "/gpt.yaml",
) as infile:
    gpt_generator_keywords = yaml.safe_load(infile)

with open(
        os.path.dirname(os.path.abspath(__file__)) + "/elegant.yaml",
) as infile:
    elegant_generator_keywords = {"defaults": {}}
    elegant_generator_keywords.update(yaml.safe_load(infile))

with open(
        os.path.dirname(os.path.abspath(__file__)) + "/opal.yaml",
) as infile:
    opal_generator_keywords = yaml.safe_load(infile)

with open(
        os.path.dirname(os.path.abspath(__file__)) + "/aliases.yaml",
) as infile:
    aliases = yaml.safe_load(infile)

with open(
        os.path.dirname(os.path.abspath(__file__)) + "/species.yaml",
) as infile:
    code_species = yaml.safe_load(infile)

allowed_species = ["electron", "proton", "positron", "hydrogen"]

# OPAL emits from a photocathode natively (DISTRIBUTION, EMITTED = TRUE with an
# EMISSIONMODEL), which is how an OPAL section starting at the cathode generates
# and accelerates the bunch in a single run.
cathode_codes = ["ASTRA", "astra", "GPT", "gpt", "OPAL", "opal"]


class frameworkGenerator(BaseModel):
    """
    Code-agnostic beam-generator parameters: a cathode laser profile
    (``cathode=True``) or a 6D phase space.

    Named parameter sets can be kept in the YAML file named by
    :attr:`~simba.Framework.Framework.generator_defaults` (in
    ``<master_lattice>/Generators/``):

    .. code-block:: yaml

        defaults:
          species: electron
          cathode: true
          thermal_emittance: 0.0009
          distribution_type_x: radial
          sigma_x: 0.00025
        laser_3ps_gaussian:
          distribution_type_z: g
          sigma_t: 0.000000000003

    and chosen with ``generator: {default: laser_3ps_gaussian}`` in ``settings.def``
    or ``framework.generator.load_defaults("laser_3ps_gaussian")``.
    """
    name: str = "generator"
    """Name of this generator."""

    code: Literal[
        "ASTRA", "astra", "GPT", "gpt", "OPAL", "opal",
        "generic", "framework", "simba", "SIMBA",
    ] = "ASTRA"
    """Code generating the distribution. ``opal`` only generates and accelerates
    in a single OPAL run; see :attr:`~simba.Codes.OPAL.OPAL.opalLattice.all_in_one`."""

    sigma_x: float = 0.0
    """Horizontal beam sigma [m]"""

    sigma_y: float = 0.0
    """Vertical beam sigma [m]"""

    sigma_z: float = 0.0
    """Longitudinal beam size [m]"""

    sigma_px: float = 0.0
    """Horizontal beam momentum sigma [eV/c]"""

    sigma_py: float = 0.0
    """Vertical beam momentum sigma [eV/c]"""

    sigma_pz: float = 0.0
    """Longitudinal beam momentum sigma [eV/c]"""

    sigma_t: float = 0.0
    """Longitudinal beam size [s]"""

    number_of_particles: int = 512
    """Number of particles"""

    filename: str = "generator.hdf5"
    """Beam distribution filename to be generated"""

    probe_particle: bool = True
    """[ASTRA only] Generate 6 probe particles."""

    noise_reduction: bool = False
    """[ASTRA only] Quasi-random (Hammersley) coordinates."""

    high_resolution: bool = True
    """[ASTRA only] High-resolution cathode emission."""

    combine_distributions: bool = False
    """[ASTRA only] Add N_add distributions, each given in the input list."""

    cathode: bool = False
    """Emit the beam from a cathode."""

    cathode_radius: float = 0.0
    """[ASTRA only] Radius of a curved cathode."""

    charge: float = 0.0
    """Bunch charge [C]"""

    species: str = "electron"
    """Particle type"""

    thermal_emittance: float = 0.9e-3
    """Thermal emittance of beam [um-rad/m]"""

    initial_momentum: float = 0.0
    """Mean initial momentum [eV/c]"""

    distribution_type_z: Literal["p", "plateau", "flattop", "g", "gaussian", "i", "f", "file"] = "g"
    """Longitudinal distribution type."""

    distribution_type_x: Literal[
        "g", "gaussian", "2dgaussian", "u", "uniform", "r", "radial"
    ] = "r"
    """Horizontal distribution type."""

    distribution_type_y: Literal[
        "g", "gaussian", "2dgaussian", "u", "uniform", "r", "radial"
    ] = "r"
    """Vertical distribution type."""

    distribution_type_pz: Literal[
        "g", "gaussian", "2dgaussian", "u", "uniform", "r", "radial", "i",
    ] = "i"
    """Longitudinal momentum distribution type."""

    distribution_type_px: Literal[
        "g", "gaussian", "2dgaussian", "u", "uniform", "r", "radial"
    ] = "r"
    """Horizontal momentum distribution type."""

    distribution_type_py: Literal[
        "g", "gaussian", "2dgaussian", "u", "uniform", "r", "radial"
    ] = "r"
    """Vertical momentum distribution type."""

    gaussian_cutoff_x: float = 3
    """Cut-off for Gaussian distribution in horizontal direction [sigma]"""

    gaussian_cutoff_y: float = 3
    """Cut-off for Gaussian distribution in vertical direction [sigma]"""

    gaussian_cutoff_z: float = 3
    """Cut-off for Gaussian distribution in longitudinal direction [sigma]"""

    gaussian_cutoff_px: float = 3
    """Cut-off for Gaussian distribution in horizontal momentum plane [sigma]"""

    gaussian_cutoff_py: float = 3
    """Cut-off for Gaussian distribution in vertical momentum plane [sigma]"""

    gaussian_cutoff_pz: float = 3
    """Cut-off for Gaussian distribution in longitudinal momentum plane [sigma]"""

    plateau_bunch_length: float = 0.0
    """Flat-top bunch length [s]"""

    plateau_rise_time: float = 0.0
    """Rise-time for flat-top distribution [s]"""

    plateau_fall_time: float = 0.0
    """Fall-time for flat-top distribution [s]"""

    plateau_rise_distance: float = 0.0  # TODO deprecated?
    """[ASTRA only] Rise-distance for flat-top distribution [m]"""

    offset_x: float = 0
    """Horizontal offset from axis [m]"""

    offset_y: float = 0
    """Vertical offset from axis [m]"""

    offset_z: float = 0
    """Reference beam position [m]"""

    reference_time: float = 0
    """Reference beam time [s]"""

    normalized_horizontal_emittance: float = 0e-6
    """Normalised horizontal emittance [m-rad]"""

    normalized_vertical_emittance: float = 0e-6
    """Normalised vertical emittance [m-rad]"""

    image_filename: str = ""
    """Image file used to generate transverse beam distribution (GPT only)"""

    longitudinal_profile: str = ""
    """File used to generate longitudinal beam distribution (GPT only)"""

    longitudinal_fields: list = []
    """Fields defining longitudinal beam distribution (GPT only)"""

    correlation_px: float = 0  # TODO is this ever used?
    """Correlated horizontal divergence (ASTRA ``cor_px``, GPT ``addxdiv``)."""

    correlation_py: float = 0  # TODO is this ever used?
    """Correlated vertical divergence (ASTRA ``cor_py``)."""

    correlation_kinetic_energy: float = 0  # TODO is this ever used?
    """Correlated energy spread (ASTRA ``cor_Ekin``)."""

    sigma_kinetic_energy: float = 0  # TODO is this ever used?
    """Kinetic-energy spread [eV] (ASTRA ``sig_Ekin``)."""

    covariance_xxp: confloat(lt=1, gt=-1) = 0.0
    """Correlation coefficient of horizontal position and momentum."""

    covariance_yyp: confloat(lt=1, gt=-1) = 0.0
    """Correlation coefficient of vertical position and momentum."""

    chirp: float | list = 0.0
    """Energy chirp of the beam [eV/m] or list of higher-order chirps for each particle"""

    laser_energy: float = 0.0
    """[OPAL only] Photoinjector laser energy [eV]"""

    work_function_ev: float = 0.0
    """[OPAL only] Work function of photocathode [eV]"""

    fermi_energy_ev: float = 0.0
    """[OPAL only] Fermi energy of photocathode [eV]"""

    cathode_temperature: float = 0.0
    """[OPAL only] Photocathode temperature [K]"""

    rf_frequency: float = 2.9985e9
    """[OPAL only] Photoinjector RF frequency [Hz]"""

    emission_model: Literal["ASTRA", "NONEQUIL"] = "ASTRA"
    """[OPAL only] Photoemission model"""

    particle_mass: float = constants.m_e
    """Particle mass [kg]"""

    elementary_charge: float = constants.elementary_charge
    """Elementary charge [C]"""

    charge_sign: int = -1
    """Sign of charge (+1 for protons, -1 for electrons)"""

    speed_of_light: float = constants.speed_of_light
    """Speed of light [m/s]"""

    tstep: float = 1e-15
    """[OPAL only] Time step for tracking [s]"""

    emission_steps: int = 500
    """[OPAL only] Number of emission steps; capped by
    :meth:`~simba.Codes.Generators.opal.OPALGenerator.capped_emission_steps`."""

    n_bin: int = 10
    """[OPAL only] Number of energy bins"""

    max_steps: int = 1000000000
    """[OPAL only] Max steps for tracking"""

    global_parameters: Dict = {}
    """Global parameters from :class:`~simba.Framework.Framework` class"""

    objectdefaults: Dict = {}
    """Unused."""

    executables: Any = {}
    """Generator executables from :class:`~simba.Framework.Framework` class"""

    kwargs: Dict = {}
    """Additional arguments"""

    allowedKeyWords: List = []
    """Unused."""

    generator_keywords: Dict = {}
    """Generator keywords from :class:`~simba.Framework.Framework` class"""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True,
    )

    def apply_alias_and_multiplier(self, config: Dict, code: str) -> None:
        """
        Set each aliased field under its `code` name, times any multiplier (e.g. s to ns for ASTRA).

        :param config: contents of ``aliases.yaml``
        :param code: code whose aliases to apply
        """
        alias_config = config.get("aliases", {}).get(code, {})
        for k, v in alias_config.items():
            if hasattr(self, k):
                value = getattr(self, k) * v["multiplier"] if "multiplier" in list(v.keys()) else getattr(self, k)
                setattr(self, v["alias"], value)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    @model_validator(mode="after")
    def validate_generator(self):
        if self.cathode and self.code not in cathode_codes:
            raise ValueError(
                f"cathode can only be used with {cathode_codes}, not {self.code}"
            )
        return self

    @field_validator("species", mode="after")
    @classmethod
    def validate_particle_mass(cls, v: str) -> str:
        if v[-1] == "s":
            v = v[:-1]
        if v not in allowed_species:
            raise NotImplementedError(f"species must be in {allowed_species}")
        return v

    @field_validator("longitudinal_profile", mode="before")
    @classmethod
    def validate_longitudinal_profile(cls, v: str) -> str:
        if len(v) > 0 and ".gdf" not in v:
            raise NotImplementedError("Longitudinal profiles only defined for GPT; fields must be GDF format")
        return v

    @model_validator(mode="after")
    def apply_species_and_profile(self, info: ValidationInfo):
        # write __dict__ directly: setattr would re-trigger validate_assignment
        if info.field_name in (None, "species"):
            self.__dict__["particle_mass"] = constants.m_p if self.species in ("proton", "hydrogen") else constants.m_e
            self.__dict__["charge_sign"] = -1 if self.species == "electron" else 1
        if info.field_name in (None, "longitudinal_profile") and len(self.longitudinal_profile) > 0:
            self.__dict__["longitudinal_fields"] = [p["name"] for p in load(self.longitudinal_profile)["blocks"]]
        return self

    def update_species(self, name: str) -> None:
        if self.cathode and "electron" not in name:
            raise ValueError("cathode can only be used with electron")
        if name == "electron":
            self.particle_mass = constants.m_e
            self.charge_sign = -1
        elif name == "proton":
            self.particle_mass = constants.m_p
            self.charge_sign = 1
        elif name == "positron":
            self.particle_mass = constants.m_e
            self.charge_sign = 1
        elif name == "hydrogen":
            self.particle_mass = constants.m_p
            self.charge_sign = 1
        else:
            raise NotImplementedError(f"name must be in {allowed_species}")
        self.species = name

    def run(self):
        pass

    def load_defaults(self, defaults: str | Dict) -> None:
        """
        Set attributes from a named set in :attr:`generator_keywords`, or from a dict.

        :param defaults: name of the set, or the settings
        """
        if isinstance(defaults, str) and defaults in self.generator_keywords:
            for k, v in self.generator_keywords[defaults].items():
                setattr(self, k, v)
        elif isinstance(defaults, dict):
            for k, v in defaults.items():
                setattr(self, k, v)
        else:
            raise ValueError(f"Could not find {defaults} in {self.generator_keywords} or it is not a valid dictionary")

    @property
    def particles(self) -> int:
        """
        Number of particles (default 512).
        """
        return self.number_of_particles if self.number_of_particles is not None else 512

    @particles.setter
    def particles(self, npart):
        """"""
        self.number_of_particles = npart

    @property
    def thermal_kinetic_energy(self) -> float:
        """
        Thermal kinetic energy [eV] from :attr:`thermal_emittance`, Eq. (39) of `Dowell & Schmerge`_.

        .. _Dowell & Schmerge: https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.12.074201
        """
        return float(
            (
                    3
                    * self.thermal_emittance ** 2
                    * self.speed_of_light ** 2
                    * self.particle_mass
            )
            / 2
            / self.elementary_charge
        )

    @property
    def objectname(self):
        """Name of this object."""
        return self.name

    def generate(self):
        if self.initial_momentum <= 0:
            raise ValueError("initial_momentum must be set to a non-zero value")
        q_over_c = UnitValue(
            constants.elementary_charge / constants.speed_of_light, "C/c"
        )
        xxp = self.generate_transverse_distribution("x")
        x = xxp[:, 0]
        xp = xxp[:, 1]
        yyp = self.generate_transverse_distribution("y")
        y = yyp[:, 0]
        yp = yyp[:, 1]
        zpz = self.generate_longitudinal_distribution()
        z = zpz[:, 0]
        pz = zpz[:, 1] * q_over_c
        px = xp * self.initial_momentum * q_over_c
        py = yp * self.initial_momentum * q_over_c
        beam = rbf.beam()
        beam.Particles.x = UnitValue(x, units="m")
        beam.Particles.y = UnitValue(y, units="m")
        beam.Particles.z = UnitValue(z, units="m")
        beam.Particles.px = UnitValue(px, units="kg*m/s")
        beam.Particles.py = UnitValue(py, units="kg*m/s")
        beam.Particles.pz = UnitValue(pz, units="kg*m/s")
        beam.Particles.status = UnitValue(np.full(len(x), 5), units="")
        beam.Particles.set_total_charge(self.charge)
        beam.set_species(self.species)
        beam.Particles.t = UnitValue(
            -z / (beam.Particles.Bz.val * constants.speed_of_light), units="s"
        )
        return beam

    def write(self, beam=None):
        if beam is None:
            beam = self.generate()
        rbf.openpmd.write_openpmd_beam_file(
            beam,
            self.global_parameters["master_subdir"] + "/" + self.filename,
            toffset=self.reference_time,
        )

    def generate_transverse_distribution(self, name: str) -> np.ndarray:
        """
        Sample a transverse plane's position and divergence.

        :param name: the plane, ``"x"`` or ``"y"``
        :return: ``(N, 2)`` samples from its ``sigma_``, ``offset_``, ``covariance_``
            and ``gaussian_cutoff_`` attributes
        """
        dist_i = getattr(self, f"distribution_type_{name}")
        dist_pi = getattr(self, f"distribution_type_p{name}")
        offset_i = getattr(self, f"offset_{name}")
        sigma_i = getattr(self, f"sigma_{name}")
        sigma_pi = getattr(self, f"sigma_p{name}") / self.initial_momentum
        cutoff_i = getattr(self, f"gaussian_cutoff_{name}")
        cutoff_pi = getattr(self, f"gaussian_cutoff_p{name}")
        cov_ipi = getattr(self, f"covariance_{name}{name}p")
        if sigma_i <= 0 or sigma_pi <= 0:
            raise ValueError(
                f"Sigma for {name} and p{name} must be set to a non-zero value"
            )
        if dist_i.lower() not in ["g", "gaussian", "r", "radial"]:
            raise NotImplementedError(
                f"Distribution type {dist_i} not implemented for transverse distribution"
            )
        if dist_pi.lower() not in ["g", "gaussian", "r", "radial"]:
            raise NotImplementedError(
                f"Distribution type {dist_pi} not implemented for transverse distribution"
            )
        mu = np.array([offset_i, 0])
        cov = np.array([[sigma_i ** 2, cov_ipi * sigma_i * sigma_pi],
                        [cov_ipi * sigma_i * sigma_pi, sigma_pi ** 2]])
        return sample_2d_gaussian_with_axis_cutoffs(self.particles, mu, cov, (cutoff_i, cutoff_pi))

    def generate_longitudinal_distribution(self) -> np.ndarray:
        """
        Sample ``(z, pz)``, with optional polynomial chirp.
        """
        if self.sigma_t > 0 and self.sigma_z > 0:
            warnings.warn(
                "Both sigma_t and sigma_z are set, using sigma_z for longitudinal distribution"
            )
        elif self.sigma_t == self.sigma_z == 0:
            raise ValueError("Either sigma_t or sigma_z must be non-zero")
        elif self.sigma_t != 0 and self.sigma_z == 0:
            self.sigma_z = self.sigma_t * constants.speed_of_light
            warnings.warn("sigma_z set to sigma_t * speed_of_light")

        if self.sigma_pz <= 0:
            raise ValueError("sigma_pz must be positive")

        if self.distribution_type_z.lower() in ["g", "gaussian", "r", "radial"]:
            z = sample_gaussian(self.offset_z, self.sigma_z, self.gaussian_cutoff_z, self.particles)
        elif self.distribution_type_z.lower() in ["u", "uniform", "flat", "flattop", "i", "plateau", "p"]:
            z = sample_flat_top(self.offset_z, self.sigma_z, self.gaussian_cutoff_z, 0.1, self.particles)
        else:
            raise NotImplementedError(f"Unsupported z distribution: {self.distribution_type_z}")

        pz_base = sample_gaussian(0, self.sigma_pz, self.gaussian_cutoff_pz, self.particles)

        chirp_coeffs = [self.chirp] if isinstance(self.chirp, float) else list(self.chirp)
        chirped_curve = poly_curve(z - np.mean(z), chirp_coeffs)
        chirped_curve -= np.mean(chirped_curve)

        pz_chirped = self.initial_momentum + chirped_curve + pz_base

        if np.any(pz_chirped < 0):
            warnings.warn("Some pz values are negative — consider reducing sigma_pz or curvature")

        return np.transpose([z, pz_chirped])

    def postProcess(self):
        self.global_parameters["beam"] = rbf.beam()
        rbf.openpmd.read_openpmd_beam_file(
            self.global_parameters["beam"],
            self.global_parameters["master_subdir"] + "/" + self.filename
        )


def poly_curve(x, coeffs):
    return sum(c * x ** i for i, c in enumerate(coeffs, start=1))


def sample_2d_gaussian_with_axis_cutoffs(N, mean, cov, cutoffs):
    """
    `N` samples of a 2D Gaussian, cut per axis at `cutoffs` sigma in whitened space.
    """
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)

    samples = []
    batch_size = int(N * 1.5)

    while len(samples) < N:
        z = np.random.randn(batch_size, 2)
        x = z @ L.T + mean

        z_white = (x - mean) @ L_inv.T
        mask = (
                (np.abs(z_white[:, 0]) <= cutoffs[0]) &
                (np.abs(z_white[:, 1]) <= cutoffs[1])
        )

        accepted = x[mask]
        samples.extend(accepted)

    return np.array(samples[:N])


def sample_gaussian(offset, sigma, cutoff, size):
    while True:
        samples = np.random.normal(offset, sigma, size * 2)
        accepted = samples[np.abs(samples - offset) <= cutoff * sigma]
        if len(accepted) >= size:
            return accepted[:size]


def sample_flat_top(offset, sigma, cutoff, edge_width, size):
    """
    Flat-top samples about `offset`, half-width ``cutoff * sigma``, with cosine edges
    ``edge_width * sigma`` wide.
    """
    total_width = cutoff * sigma
    ramp = edge_width * sigma

    while True:
        samples = np.random.uniform(-total_width, total_width, size * 3)
        # weight: flat center, cosine edges
        weight = np.ones_like(samples)
        mask_ramp = np.abs(samples) > (total_width - ramp)
        weight[mask_ramp] = 0.5 * (1 + np.cos(np.pi * (np.abs(samples[mask_ramp]) - (total_width - ramp)) / ramp))
        keep = np.random.rand(len(samples)) < weight
        final = samples[keep] + offset
        if len(final) >= size:
            return final[:size]