from warnings import warn

import numpy as np

from .. import constants
from ..units import UnitValue

BDSIM_DISTRIBUTION_FORMAT = "x[m]:xp[rad]:y[m]:yp[rad]:t[s]:E[eV]"
"""Time, not ``z``, so every particle starts on the first element's face: in a closed
ring a ``z`` behind it starts beyond the last element instead."""

SAMPLER_VARIABLES = (
    "x",
    "y",
    "xp",
    "yp",
    "zp",
    "energy",
    "T",
    "S",
    "partID",
    "parentID",
    "weight",
    "turnNumber",
)

PDGID_SPECIES = {
    11: "electron",
    -11: "positron",
    22: "photon",
    2212: "proton",
    -2212: "antiproton",
}
"""SIMBA species (:data:`~simba.Modules.Beams.SPECIES`) of the PDG IDs it reads from BDSIM."""


def write_bdsim_beam_file(beam, filename):
    """
    Write a BDSIM ``userfile`` ASCII bunch distribution written into gmad's
    ``distrFileFormat``.

    Parameters
    ----------
    beam: :class:`~simba.Modules.Beams.beam`
        The beam to write.
    filename: str
        Name of the file to write.
    """
    np.savetxt(
        filename,
        np.transpose(
            np.array(
                [
                    beam.x.val,
                    beam.xp.val,
                    beam.y.val,
                    beam.yp.val,
                    -(beam.z.val - np.mean(beam.z.val))
                    / (beam.Bz.val * constants.speed_of_light),
                    beam.energy.val,
                ]
            )
        ),
    )


def read_bdsim_beam_file(
        self,
        filename: str,
        charge: float | None=None,
        zstart: float | None=0.0,
        s: float | None=0.0,
        ref_index: int | None=None):
    """
    Read back a BDSIM ``userfile`` ASCII distribution written by
    :func:`write_bdsim_beam_file` and update the
    :attr:`~simba.Modules.Beams.beam.Particles` object.

    Parameters
    ----------
    filename: str
        Name of the ``.bdsim`` file.
    charge: float, optional
        Total bunch charge [C]; if not given the existing charge is kept.
    zstart: float, optional
        Longitudinal offset [m] added to the reconstructed `z`.
    s: float, optional
        Curvilinear position [m] of the distribution.
    ref_index: int, optional
        Reference particle index.
    """
    data = np.atleast_2d(np.loadtxt(filename))
    if data.shape[1] != 6:
        raise ValueError(
            f"{filename} has {data.shape[1]} columns; expected 6 "
            f"({BDSIM_DISTRIBUTION_FORMAT})"
        )
    x, xp, y, yp, t, energy = data.T
    self.filename = filename
    self.code = "BDSIM"
    _set_particles(
        self,
        x=x,
        y=y,
        xp=xp,
        yp=yp,
        energy=energy,
        mass=np.full(len(x), constants.m_e),
        charge=charge,
        s=s,
    )
    self._beam.t = UnitValue(t, units="s")
    self._beam.z = UnitValue(
        zstart - t * self._beam.Bz * constants.speed_of_light, units="m"
    )
    _set_reference_particle(self, ref_index)


def load_bdsim_output(filename):
    """
    Load a BDSIM raw ROOT output file.

    Parameters
    ----------
    filename: str
        Path to the BDSIM ``.root`` output file.

    Returns
    -------
    ROOT.DataLoader
        The loaded BDSIM data.

    Raises
    ------
    ImportError
        If ROOT is not importable. BDSIM sampler branches hold a custom ROOT
        class and cannot be read without it.
    """
    try:
        import pybdsim
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "pybdsim is required to read BDSIM output files"
        ) from exc
    try:
        import ROOT  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "The ROOT python bindings are required to read BDSIM sampler data. "
            "Source ROOT's thisroot.sh *and* BDSIM's bdsim.sh before running "
            "SIMBA."
        ) from exc
    return pybdsim.Data.Load(filename)


def get_bdsim_sampler_names(filename):
    """
    List the samplers present in a BDSIM output file.

    Parameters
    ----------
    filename: str
        Path to the BDSIM ``.root`` output file.

    Returns
    -------
    list
        Sampler names, without the trailing ``.`` that BDSIM appends.
    """
    data = load_bdsim_output(filename)
    return ["Primary"] + [str(n).rstrip(".") for n in data.GetSamplerNames()]


def read_bdsim_sampler_arrays(filename, samplers=None, variables=SAMPLER_VARIABLES):
    """
    Pull the raw sampler arrays out of a BDSIM ROOT output file.


    Parameters
    ----------
    filename: str
        Path to the BDSIM ``.root`` output file.
    samplers: list, optional
        Sampler names to read; all of them (including ``Primary``) by default.
    variables: tuple, optional
        Sampler branch variables to read.

    Returns
    -------
    dict
        ``{sampler_name: {variable: ndarray}}``.
    """
    data = load_bdsim_output(filename)
    tree = data.GetEventTree()
    event = data.GetEvent()

    names = ["Primary"] + [str(n).rstrip(".") for n in data.GetSamplerNames()]
    branches = [event.GetPrimaries()] + list(event.Samplers)

    if samplers is None:
        samplers = names
    missing = [s for s in samplers if s not in names]
    if missing:
        raise ValueError(
            f"Sampler(s) {missing} not found in {filename}; available: {names}"
        )
    wanted = {s: branches[names.index(s)] for s in samplers}

    result = {s: {v: [] for v in variables} for s in samplers}
    for s in samplers:
        result[s]["eventID"] = []
    for i in range(int(tree.GetEntries())):
        tree.GetEntry(i)
        for name, branch in wanted.items():
            nhits = int(branch.n)
            if nhits == 0:
                continue
            for var in variables:
                result[name][var].extend(_as_hit_list(getattr(branch, var), nhits))
            result[name]["eventID"].extend([i] * nhits)
    return {
        s: {v: np.array(vals) for v, vals in arrays.items()}
        for s, arrays in result.items()
    }


def _as_hit_list(value, nhits):
    """Normalise one sampler variable of one event to a list with one entry per hit."""
    try:
        return list(value)
    except TypeError:
        return [value] * nhits


def select_turn(arrays, turn):
    """
    One sampler's hits by the primaries on ``turn`` (BDSIM's ``turnNumber``), one each.

    A primary starting behind the first element of a closed ring crosses the last
    sampler at once, so only its last crossing of the turn is kept.

    Parameters
    ----------
    arrays: dict
        ``{variable: ndarray}`` for a single sampler, see :func:`read_bdsim_sampler_arrays`.
    turn: int
        The turn, from 1.
    """
    hit = np.flatnonzero((arrays["turnNumber"] == turn) & (arrays["parentID"] == 0))
    hit = hit[np.argsort(-arrays["T"][hit])]
    _, last = np.unique(arrays["eventID"][hit], return_index=True)
    return {k: v[np.sort(hit[last])] for k, v in arrays.items()}


def interpret_bdsim_sampler(
    self,
    arrays,
    charge=None,
    zstart=0,
    ref_index=None,
    secondaries=(),
    ninjected=None,
    species="electron",
):
    """
    Populate a :class:`~simba.Modules.Beams.beam` from one sampler's arrays as
    returned by :func:`read_bdsim_sampler_arrays`.

    Each particle keeps its own species (from its PDG ID, :data:`PDGID_SPECIES`), so a
    beam with secondaries is several species, which the openPMD writer splits into
    groups (see :meth:`~simba.Modules.Beams.beam.split_species`). Each hit stands for
    ``|charge| / (e ninjected)`` real particles, times its BDSIM ``weight``.

    Parameters
    ----------
    arrays: dict
        ``{variable: ndarray}`` for a single sampler.
    charge: float, optional
        Total charge of the bunch injected [C]; without it, each hit is one particle.
    zstart: float, optional
        Longitudinal offset [m] added to the reconstructed `z`.
    ref_index: int, optional
        Index of the reference particle in the injected bunch, so its event.
    secondaries: iterable of str, or "all"
        Species of the secondaries to keep, e.g. ``("photon", "positron")``; none by default.
    ninjected: int, optional
        Number of particles tracked; by default one more than the last event seen.
    species: str
        The species injected: this beam's :attr:`species`, whose primaries come first.
    """
    from . import SPECIES

    if ninjected is None:
        ninjected = int(np.max(arrays["eventID"])) + 1 if len(arrays["eventID"]) else 0
    partid = arrays["partID"].astype(int)
    names = np.array([PDGID_SPECIES.get(p) for p in partid], dtype=object)
    primary = arrays["parentID"] == 0
    keep_all = secondaries == "all"
    wanted_secondary = ~primary & (names != None)  # noqa: E711 (elementwise)
    if not keep_all:
        wanted_secondary &= np.isin(names, list(secondaries))
    if keep_all and np.any(~primary & (names == None)):  # noqa: E711
        unknown = sorted(set(partid[~primary & (names == None)]))  # noqa: E711
        warn(f"Secondaries with PDG IDs {unknown}, which SIMBA has no species for, are dropped")
    nprimaries = int(np.count_nonzero(primary))
    if nprimaries < ninjected:
        warn(
            f"{ninjected - nprimaries} of {ninjected} particles were lost before "
            "this sampler; scaling the bunch charge accordingly"
        )
    # primaries first, so the injected species is the first group
    keep = np.concatenate([np.flatnonzero(primary), np.flatnonzero(wanted_secondary)])
    if not len(keep):
        raise ValueError("No particles found at this sampler")
    dat = {k: v[keep] for k, v in arrays.items()}
    kept = names[keep]
    kept[: nprimaries] = species  # the primaries are what was injected

    per_macroparticle = abs(float(charge)) / (constants.elementary_charge * ninjected) if charge is not None else 1.0
    real = per_macroparticle * np.asarray(dat["weight"], dtype=float)
    mass = np.array([SPECIES[n][0] for n in kept])
    sign = np.array([SPECIES[n][1] for n in kept], dtype=float)

    self.code = "BDSIM"
    _set_particles(
        self,
        x=dat["x"],
        y=dat["y"],
        xp=dat["xp"],
        yp=dat["yp"],
        energy=dat["energy"] * 1e9,
        mass=mass,
        charge=None,
        s=float(np.mean(dat["S"])),
        zp=dat["zp"],
        particle_charge=sign * constants.elementary_charge,
    )
    self._beam.nmacro = UnitValue(real)
    self._beam.charge = UnitValue(sign * constants.elementary_charge * real, units="C")
    self._beam.total_charge = UnitValue(float(np.sum(self._beam.charge)), units="C")
    self.species = species
    self._beam.t = UnitValue(dat["T"] * 1e-9, units="s")
    # the reference particle is the primary from the reference's event, wherever losses
    # have left it in the arrays
    reference = None
    if ref_index is not None:
        found = np.flatnonzero(dat["eventID"][:nprimaries] == int(ref_index))
        if found.size:
            reference = int(found[0])
        else:
            warn("The reference particle was lost before this sampler; using the mean instead")
    _set_reference_particle(self, reference, zstart=zstart)


def _set_particles(
    self, x, y, xp, yp, energy, mass, charge, s, zp=None, particle_charge=-constants.elementary_charge
):
    """
    Fill the particle arrays shared by both BDSIM readers; ``particle_charge`` is
    each physical particle's charge [C] (electrons' by default).
    """
    self.reset_dicts()
    nparticles = len(x)
    self._beam.particle_mass = UnitValue(mass, units="kg")
    self._beam.particle_rest_energy = UnitValue(
        self._beam.particle_mass * constants.speed_of_light**2, units="J"
    )
    self._beam.particle_rest_energy_eV = UnitValue(
        self._beam.particle_rest_energy / constants.elementary_charge, units="eV/c"
    )
    self._beam.particle_charge = UnitValue(
        np.broadcast_to(np.asarray(particle_charge, dtype=float), (nparticles,)).copy(), units="C"
    )
    self._beam.x = UnitValue(x, units="m")
    self._beam.y = UnitValue(y, units="m")

    cp = np.sqrt(energy**2 - self._beam.particle_rest_energy_eV**2)
    if zp is None:
        cpz = cp / np.sqrt(xp**2 + yp**2 + 1)
        cpx, cpy = xp * cpz, yp * cpz
    else:
        cpx, cpy, cpz = cp * xp, cp * yp, cp * zp
    self._beam.px = UnitValue(cpx * self.q_over_c, units="kg*m/s")
    self._beam.py = UnitValue(cpy * self.q_over_c, units="kg*m/s")
    self._beam.pz = UnitValue(cpz * self.q_over_c, units="kg*m/s")

    self._beam.nmacro = UnitValue(np.full(nparticles, 1))
    self._beam.status = UnitValue(np.full(nparticles, 5))
    self._beam.s = UnitValue(s, units="m")
    if charge is not None:
        self._beam.set_total_charge(charge)


def _set_reference_particle(self, ref_index, zstart=None):
    """
    Set the reference particle and, when `zstart` is given, derive `z` from the
    arrival times relative to it (or to the mean, if there is no reference).
    """
    have_ref = ref_index is not None and int(ref_index) < len(self._beam.x)
    if ref_index is not None and not have_ref:
        warn(
            f"Reference particle index {ref_index} is beyond the {len(self._beam.x)} "
            "particles at this sampler (particles were lost); using the mean instead"
        )
    if zstart is not None:
        t0 = (
            self._beam.t[int(ref_index)] if have_ref else np.mean(self._beam.t)
        )
        self._beam.z = UnitValue(
            zstart
            + (-1 * self._beam.Bz * constants.speed_of_light) * (self._beam.t - t0),
            units="m",
        )
    if have_ref:
        self.reference_particle_index = int(ref_index)
        self.reference_particle = [
            getattr(self._beam, coord)[self.reference_particle_index]
            for coord in self.reference_particle_coords
        ]
    else:
        self.reference_particle = None
