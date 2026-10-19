"""Cross-code agreement for the JFEL line.

The ASTRA injector is tracked once, then the Linac is tracked through cheetah,
elegant, madx and ocelot from that identical distribution. Every screen the four
codes have in common is compared, and each pair of codes is held to a ceiling on
how far apart it is allowed to drift.

Ceilings in ``TOLERANCES`` are values with headroom of
roughly 1.4x. The measured numbers are in the comment beside each row so a future
reader can see how much slack is actually being given. Absolute values are pinned
separately in ``test_exit_values`` as a plain regression guard.

Requires the JFEL master lattice; set ``JFEL_MASTER_LATTICE`` to point at it.
"""
import os
import shutil

import numpy as np
import pytest

import simba.Framework as fw
import simba.Modules.Beams as rbf

pytestmark = pytest.mark.needs_container

CODES = ("cheetah", "elegant", "madx", "ocelot")
END_ELEMENT = "JFEL-FEL-SIM-MARK-01"
SUFFIX = ".openpmd.hdf5"
N_PARTICLES = 4096
BUNCH_COMPRESSOR_ANGLE = 0.1
SEED = 1234

MASTER_LATTICE = os.environ.get(
    "JFEL_MASTER_LATTICE",
    os.path.expanduser("~/Documents/laura-lattices-gh/JFEL/"),
)
_runtime = os.environ.get("JFEL_CONTAINER_RUNTIME", "docker")
CONTAINER_RUNTIME = None if _runtime.lower() in ("none", "") else _runtime

SIMCODES = os.environ.get("SIMCODES_LOCATION")

QUANTITIES = {
    "E": (lambda b: 1e-6 * np.mean(b.Particles.cp), True),
    "dE": (lambda b: 100 * np.std(b.Particles.cp) / np.mean(b.Particles.cp), False),
    "enx": (lambda b: 1e6 * b.normalized_horizontal_emittance, True),
    "eny": (lambda b: 1e6 * b.normalized_vertical_emittance, True),
    "sx": (lambda b: 1e6 * b.Sx, True),
    "sy": (lambda b: 1e6 * b.Sy, True),
    "sz": (lambda b: 1e6 * b.Sz, True),
}

#                                   E      dE    enx    eny     sx    sy     sz
TOLERANCES = {
    # measured:                  0.011  0.022   3.77  10.93   2.79  0.66   2.02
    ("madx", "ocelot"):         (0.05,  0.10,   8.0,  18.0,   8.0,  3.0,   6.0),
    # measured:                  0.019  0.008   7.32   7.81   8.62  2.44   1.51
    ("cheetah", "madx"):        (0.05,  0.05,  12.0,  14.0,  14.0,  6.0,   5.0),
    # measured:                  0.028  0.022   6.87  10.07   8.65  3.10   3.15
    ("cheetah", "ocelot"):      (0.06,  0.06,  12.0,  18.0,  14.0,  6.0,   7.0),
    # measured:                  0.106  0.081  26.45   4.37  21.46  1.10  16.07
    ("cheetah", "elegant"):     (0.20,  0.20,  35.0,  10.0,  30.0,  4.0,  24.0),
    # measured:                  0.125  0.082  33.28   3.90  28.29  2.06  14.79
    ("elegant", "madx"):        (0.25,  0.20,  42.0,  10.0,  38.0,  6.0,  22.0),
    # measured:                  0.134  0.066  31.45  14.43  27.84  2.72  12.94
    ("elegant", "ocelot"):      (0.25,  0.20,  42.0,  22.0,  38.0,  6.0,  22.0),
}

EXIT_VALUES = {
    "cheetah": {"E": 576.6, "dE": 0.9165, "enx": 0.8972, "eny": 0.8823, "sx": 99.38, "sy": 669.8, "sz": 140.0},
    "elegant": {"E": 576.0, "dE": 0.9402, "enx": 1.169, "eny": 0.8446, "sx": 121.5, "sy": 675.5, "sz": 164.5},
    "madx": {"E": 576.7, "dE": 0.9246, "enx": 0.8385, "eny": 0.8748, "sx": 100.5, "sy": 670.2, "sz": 142.2},
    "ocelot": {"E": 576.8, "dE": 0.9249, "enx": 0.8552, "eny": 0.9759, "sx": 103.2, "sy": 672.3, "sz": 144.5},
}
EXIT_BANDS = {"E": 0.005, "dE": 0.15, "enx": 0.15, "eny": 0.15, "sx": 0.15, "sy": 0.05, "sz": 0.15}

MIN_SCREENS = 60
"""Guard against the comparison quietly shrinking to a handful of screens."""


def _make_framework(directory):
    framework = fw.Framework(
        master_lattice=MASTER_LATTICE,
        **({"simcodes": SIMCODES} if SIMCODES else {}),
        generator_defaults="/jfel.yaml",
        directory=directory,
        clean=False,
        verbose=False,
        eager_mode=False,
        container_runtime=CONTAINER_RUNTIME,
    )
    framework.loadSettings("Lattices/jfel_combined.def")
    framework["generator"].number_of_particles = N_PARTICLES
    framework["bunch_compressor"].set_angle(BUNCH_COMPRESSOR_ANGLE)
    framework.setSeedValue(SEED)
    return framework


def _stats(beamfile):
    beam = rbf.load_file(beamfile)
    stats = {"s": float(np.atleast_1d(beam.s)[0])}
    stats.update({label: float(fn(beam)) for label, (fn, _) in QUANTITIES.items()})
    return stats


def _disagreement(a, b, relative):
    if not relative:
        return abs(a - b)
    mean = 0.5 * (a + b)
    return 100 * abs(a - b) / abs(mean) if abs(mean) > 0 else 0.0


@pytest.fixture(scope="module")
def tracked(tmp_path_factory):
    """Track the injector once, then the Linac through every code."""
    if not os.path.isdir(MASTER_LATTICE):
        pytest.skip(f"JFEL master lattice not available at {MASTER_LATTICE}")

    base = tmp_path_factory.mktemp("jfel")
    injector = str(base / "injector")
    _make_framework(injector).track(endfile="injector400", check_lattice=False)

    per_code = {}
    for code in CODES:
        direc = str(base / code)
        shutil.copytree(injector, direc)
        framework = _make_framework(direc)
        framework.change_Lattice_Code("Linac", code)
        framework.track(files=["Linac"], check_lattice=False)
        per_code[code] = {
            f[: -len(SUFFIX)]: os.path.join(direc, f)
            for f in os.listdir(direc)
            if f.endswith(SUFFIX)
        }

    common = sorted(set.intersection(*(set(v) for v in per_code.values())))
    return {
        code: {name: _stats(files[name]) for name in common} for code, files in per_code.items()
    }


def test_enough_common_screens(tracked):
    n = len(tracked[CODES[0]])
    assert n >= MIN_SCREENS, f"only {n} screens common to all four codes, expected >= {MIN_SCREENS}"


@pytest.mark.parametrize("pair", list(TOLERANCES))
def test_codes_agree_along_line(tracked, pair):
    """No pair of codes may drift further apart than it was measured to be."""
    a, b = pair
    ceilings = dict(zip(QUANTITIES, TOLERANCES[pair]))
    failures = []
    for label, (_, relative) in QUANTITIES.items():
        worst, at = 0.0, None
        for name, stats_a in tracked[a].items():
            d = _disagreement(stats_a[label], tracked[b][name][label], relative)
            if d > worst:
                worst, at = d, name
        if worst > ceilings[label]:
            unit = "%" if relative else "pp"
            failures.append(
                f"{label}: {worst:.3f}{unit} > {ceilings[label]}{unit} at {at} "
                f"(s = {tracked[a][at]['s']:.2f} m)"
            )
    assert not failures, f"{a} vs {b} disagree more than allowed:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("code", CODES)
def test_exit_values(tracked, code):
    """Each code still gives the answer it gave when these bounds were set."""
    stats = tracked[code][END_ELEMENT]
    failures = []
    for label, expected in EXIT_VALUES[code].items():
        band = EXIT_BANDS[label] * abs(expected)
        if abs(stats[label] - expected) > band:
            failures.append(f"{label}: {stats[label]:.4g} not within +/-{band:.4g} of {expected}")
    assert not failures, f"{code} exit values moved:\n  " + "\n  ".join(failures)
