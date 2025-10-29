from wake_t import (
    PlasmaStage,
    ActivePlasmaLens,
    Dipole,
    Quadrupole,
    Sextupole,
    GaussianPulse,
)

wake_t_conversion_rules = {
    "dipole": Dipole,
    "quadrupole": Quadrupole,
    "sextupole": Sextupole,
    "laser": GaussianPulse,
    "plasma": PlasmaStage,
    "plasma_lens": ActivePlasmaLens,
}
