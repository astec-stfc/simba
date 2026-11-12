"""Cross-code agreement for the JFEL line.

The ASTRA injector is tracked once, then the Linac is tracked through cheetah,
elegant, madx and ocelot from that identical distribution, with CSR off as
cheetah and madx have none. Every screen the four codes have in common is
compared, and each pair of codes is held to a ceiling on how far apart it is
allowed to drift. ELEGANT and Ocelot are then tracked again with CSR on (as the
lattice's elements ask) and compared in the same way.

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
CSR_CODES = ("elegant", "ocelot")
"""The codes here that model CSR, compared again with it on."""
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
# sy peaks at JFEL-S11-DIA-BPM-01, a ~25 um vertical waist where a few um is a big percentage
# CSR off in every code; floors of 0.02/0.01/1.0 where 1.4x would be tighter
TOLERANCES = {
    # measured:                  0.130  0.062   6.69   3.96   6.19  2.17  11.79
    ("madx", "ocelot"):         (0.19,  0.087,  9.4,   5.6,   8.7,  3.1,  17.0),
    # measured:                  0.006  0.001   0.14   1.41   0.55  0.37   0.16
    ("cheetah", "madx"):        (0.02,  0.01,   1.0,   2.0,   1.0,  1.0,   1.0),
    # measured:                  0.131  0.062   6.76   4.21   6.16  2.54  11.63
    ("cheetah", "ocelot"):      (0.19,  0.087,  9.5,   5.9,   8.7,  3.6,  17.0),
    # measured:                  0.113  0.073   7.65   7.11   7.40  6.35  14.23
    ("cheetah", "elegant"):     (0.16,  0.11,  11.0,  10.0,  11.0,  8.9,  20.0),
    # measured:                  0.110  0.073   7.58   6.79   7.42  6.49  14.38
    ("elegant", "madx"):        (0.16,  0.11,  11.0,   9.6,  11.0,  9.1,  21.0),
    # measured:                  0.152  0.018   1.97   4.77   2.78  5.55   2.64
    ("elegant", "ocelot"):      (0.22,  0.026,  2.8,   6.7,   3.9,  7.8,   3.7),
}

# At the undulator entrance, through S13 matched into the undulator. With the
# chicane at 0.1 rad the 0.93% energy spread is barely compressed, so the strong
# final focusing grows eps_ny chromatically (~2.5 um) in every code alike.
EXIT_VALUES = {
    "cheetah": {"E": 576.7, "dE": 0.9265, "enx": 0.8512, "eny": 2.462, "sx": 51.92, "sy": 113.9, "sz": 142.8},
    "elegant": {"E": 576.1, "dE": 0.9441, "enx": 0.8665, "eny": 2.622, "sx": 53.0, "sy": 111.7, "sz": 164.2},
    "madx": {"E": 576.7, "dE": 0.9256, "enx": 0.8512, "eny": 2.497, "sx": 52.03, "sy": 113.9, "sz": 142.5},
    "ocelot": {"E": 577.0, "dE": 0.9258, "enx": 0.8496, "eny": 2.562, "sx": 51.55, "sy": 113.5, "sz": 159.9},
}
EXIT_BANDS = {"E": 0.005, "dE": 0.15, "enx": 0.15, "eny": 0.15, "sx": 0.15, "sy": 0.05, "sz": 0.15}

# CSR on, as the lattice's elements ask; enx grows ~0.09 um through the chicane in both.
# Ocelot's CSR spans only the chicane (see ocelotLattice.csr_regions): over the whole
# line it slices the cavities and its sz dropped to MAD-X's (142.7 um).
TOLERANCES_CSR = {
    # measured:                  0.163  0.019   4.61   4.80   3.81  5.80   2.75
    ("elegant", "ocelot"):      (0.23,  0.027,  6.5,   6.8,   5.4,  8.2,   3.9),
}
EXIT_VALUES_CSR = {
    "elegant": {"E": 576.0, "dE": 0.9446, "enx": 0.9535, "eny": 2.621, "sx": 55.27, "sy": 111.7, "sz": 164.8},
    "ocelot": {"E": 576.9, "dE": 0.9252, "enx": 0.9985, "eny": 2.560, "sx": 55.15, "sy": 113.6, "sz": 160.3},
}

MIN_SCREENS = 57
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
def injector(tmp_path_factory):
    """Track the injector once, for every Linac run to start from."""
    if not os.path.isdir(MASTER_LATTICE):
        pytest.skip(f"JFEL master lattice not available at {MASTER_LATTICE}")
    base = tmp_path_factory.mktemp("jfel")
    directory = str(base / "injector")
    _make_framework(directory).track(endfile="injector400", check_lattice=False)
    return directory


@pytest.fixture(scope="module")
def tracked(injector):
    """The Linac through every code with CSR off, as cheetah and madx have none."""
    return _track_linac(injector, CODES, csr=False)


@pytest.fixture(scope="module")
def tracked_csr(injector):
    """The Linac through the codes that model CSR, with it on as the lattice's elements ask."""
    return _track_linac(injector, CSR_CODES, csr=True)


def _track_linac(injector, codes, csr):
    """Statistics at every screen the ``codes`` share, from the Linac tracked through each."""
    per_code = {}
    for code in codes:
        direc = os.path.join(os.path.dirname(injector), f"{code}-csr-{'on' if csr else 'off'}")
        shutil.copytree(injector, direc)
        framework = _make_framework(direc)
        framework.change_Lattice_Code("Linac", code)
        if not csr:
            framework["Linac"].csr_enable = False
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


def worst_disagreements(tracked, a, b):
    """``{label: (worst disagreement, screen)}`` between codes ``a`` and ``b`` along the line."""
    worst = {}
    for label, (_, relative) in QUANTITIES.items():
        worst[label] = max(
            (_disagreement(stats_a[label], tracked[b][name][label], relative), name)
            for name, stats_a in tracked[a].items()
        )
    return worst


def _check_pair(tracked, pair, tolerances):
    a, b = pair
    ceilings = dict(zip(QUANTITIES, tolerances[pair]))
    failures = []
    for label, (worst, at) in worst_disagreements(tracked, a, b).items():
        if worst > ceilings[label]:
            unit = "%" if QUANTITIES[label][1] else "pp"
            failures.append(
                f"{label}: {worst:.3f}{unit} > {ceilings[label]}{unit} at {at} "
                f"(s = {tracked[a][at]['s']:.2f} m)"
            )
    assert not failures, f"{a} vs {b} disagree more than allowed:\n  " + "\n  ".join(failures)


def _check_exit(tracked, code, exit_values):
    stats = tracked[code][END_ELEMENT]
    failures = []
    for label, expected in exit_values[code].items():
        band = EXIT_BANDS[label] * abs(expected)
        if abs(stats[label] - expected) > band:
            failures.append(f"{label}: {stats[label]:.4g} not within +/-{band:.4g} of {expected}")
    assert not failures, f"{code} exit values moved:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("pair", list(TOLERANCES))
def test_codes_agree_along_line(tracked, pair):
    """No pair of codes may drift further apart than it was measured to be."""
    _check_pair(tracked, pair, TOLERANCES)


@pytest.mark.parametrize("code", CODES)
def test_exit_values(tracked, code):
    """Each code still gives the answer it gave when these bounds were set."""
    _check_exit(tracked, code, EXIT_VALUES)


@pytest.mark.parametrize("pair", list(TOLERANCES_CSR))
def test_csr_codes_agree_along_line(tracked_csr, pair):
    """With CSR on, Ocelot's CSR stays as close to ELEGANT's as it was measured to be."""
    _check_pair(tracked_csr, pair, TOLERANCES_CSR)


@pytest.mark.parametrize("code", CSR_CODES)
def test_exit_values_with_csr(tracked_csr, code):
    _check_exit(tracked_csr, code, EXIT_VALUES_CSR)
