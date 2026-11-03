import os
from warnings import warn

import numpy as np

try:
    from beamphysics import ParticleGroup, pmd_init, particle_paths
    from beamphysics.readers import load_species_data
except ImportError:
    from pmd_beamphysics import ParticleGroup, pmd_init, particle_paths
    from pmd_beamphysics.readers import load_species_data
from h5py import File
from .. import constants
from ..units import UnitValue


TURN_BASE_PATH = "/data/%T/"
"""openPMD ``basePath`` of a multi-turn file; each turn is an iteration, ``/data/<turn>/particles``."""

openpmd_coords = [
    "x",
    "y",
    "z",
    "px",
    "py",
    "pz",
    "t",
    "weight",
    "status",
]

PRIMARY_SPECIES_ATTR = "simbaPrimarySpecies"
"""Attribute on a ``particles`` group naming the species that is tracked when the
file holds more than one; the rest are :attr:`~simba.Modules.Beams.beam.others`."""


def read_particle_group(self, particles, s=None, reference_particle_index=None):
    """Populate a SIMBA beam from an openPMD ``ParticleGroup``."""
    self._beam.x = UnitValue(particles.x, units="m")
    self._beam.y = UnitValue(particles.y, units="m")
    self._beam.t = UnitValue(particles.t, units="s")
    self._beam.z = UnitValue(particles.z, units="m")
    self.set_momenta(particles.px, particles.py, particles.pz)
    self._beam.charge = UnitValue(particles.weight, units="C")
    self._beam.total_charge = UnitValue(particles.charge, units="C")
    self._beam.nmacro = UnitValue(particles.weight / constants.elementary_charge)
    self._beam.status = UnitValue(particles.status)
    self.set_species(particles.species)
    self._beam.s = UnitValue(s, units="m") if s is not None else None
    self.longitudinal_reference = "t"
    self.reference_particle = None
    self.reference_particle_index = None
    if reference_particle_index is not None:
        if 0 <= reference_particle_index < len(particles):
            self.reference_particle = [
                getattr(particles, coord)[reference_particle_index]
                for coord in openpmd_coords
            ]
            self.reference_particle_index = int(reference_particle_index)
        else:
            warn("Reference particle is not present in the tracked bunch")


def _turns(h5file: File) -> list:
    """The turns in an open multi-turn file, in order; empty for a single beam."""
    base = h5file.attrs.get("basePath", b"/")
    base = base.decode("utf-8") if isinstance(base, bytes) else str(base)
    if "%T" not in base:
        return []
    return sorted(int(turn) for turn in h5file[base.split("%T")[0]])


def openpmd_turns(filename) -> list:
    """The turns ``filename`` holds, in order; empty for a single beam.

    A multi-turn run stores each turn as one openPMD iteration of the same file.
    """
    with File(os.path.expandvars(filename), "r") as h5file:
        return _turns(h5file)


def is_openpmd_beam_file(h5file: File) -> bool:
    """Whether an open file holds a beam: a single one or a multi-turn file."""
    return "/particles" in h5file or bool(_turns(h5file))


def _particles_group(h5file: File, filename, turn=None):
    """The ``particles`` group of an open file: ``turn`` of a multi-turn file (default the last), else the only one."""
    turns = _turns(h5file)
    if not turns:
        return h5file[particle_paths(h5file)[0]]
    chosen = turns[-1] if turn is None else int(turn)
    if chosen not in turns:
        raise ValueError(
            f"{filename} holds turns {turns[0]} to {turns[-1]}, not {turn}"
        )
    return h5file[f"{TURN_BASE_PATH.split('%T')[0]}{chosen}/particles"]


def file_species(bunch_data) -> list:
    """The species groups in an openPMD ``particles`` group."""
    return [name for name in bunch_data if "position" in bunch_data[name]]


def _choose_species(bunch_data, filename, species=None) -> str:
    """``species`` if given, else the file's primary species, else its only one."""
    present = file_species(bunch_data)
    if species is None:
        species = bunch_data.attrs.get(PRIMARY_SPECIES_ATTR)
        species = species.decode("utf-8") if isinstance(species, bytes) else species
    if species is None:
        if len(present) != 1:
            raise ValueError(
                f"{filename} holds species {present}; say which one to track "
                "(``input: species`` in the section, or ``species=``)"
            )
        species = present[0]
    if species not in present:
        raise ValueError(f"{filename} holds species {present}, not {species}")
    return species


def _read_species_group(self, bunch_data, species):
    """Read one species group into beam ``self``."""
    read_particle_group(self, ParticleGroup(data=load_species_data(bunch_data[species])))
    bunch_species = bunch_data[species]
    self._beam.s = UnitValue(bunch_species["s"], units="m") if "s" in bunch_species else None
    self.turn = int(bunch_species["turn"][()]) if "turn" in bunch_species else None
    if "reference_particle" in bunch_species:
        ref_particle = bunch_species["reference_particle"]
        self.reference_particle = [ref_particle[coord][()] for coord in openpmd_coords]
        self.reference_particle_index = int(ref_particle["index"][()])


def read_openpmd_beam_file(self, filename, turn=None, species=None):
    """Read an openPMD beam file.

    ``turn`` picks one turn of a multi-turn file (see :func:`openpmd_turns`);
    the default is the last. It is ignored for a single-beam file.

    ``species`` picks which species group becomes this beam when the file holds
    several; the default is the file's primary species (:data:`PRIMARY_SPECIES_ATTR`),
    or its only one. The other groups go to :attr:`~simba.Modules.Beams.beam.others`.
    """
    self.filename = filename
    fname = os.path.expandvars(filename)
    with File(fname, "r") as h5file:
        bunch_data = _particles_group(h5file, filename, turn)
        chosen = _choose_species(bunch_data, filename, species)
        _read_species_group(self, bunch_data, chosen)
        self._others = {}
        for name in file_species(bunch_data):
            if name != chosen:
                other = type(self)()
                _read_species_group(other, bunch_data, name)
                self._others[name] = other


def write_openpmd_beam_file(
    self,
    filename,
    pos=[0, 0, 0],
    toffset=0,
    turn=None,
):
    """Write the beam to an openPMD file.

    ``turn`` writes it as that turn of a multi-turn file (:data:`TURN_BASE_PATH`);
    turn 1, or a file that is not yet multi-turn, starts the file afresh.
    """
    fname = os.path.expandvars(filename)
    if turn is None:
        with File(fname, "w") as h5file:
            pmd_init(h5file, basePath="/", particlesPath="particles")
            return _write_particles(self, h5file.create_group("particles"), pos, toffset)
    adding = int(turn) != 1 and os.path.isfile(fname)
    if adding:
        with File(fname, "r") as h5file:
            adding = bool(_turns(h5file))
    with File(fname, "a" if adding else "w") as h5file:
        if not adding:
            pmd_init(h5file, basePath=TURN_BASE_PATH, particlesPath="particles")
        group = h5file.require_group(f"{TURN_BASE_PATH.split('%T')[0]}{int(turn)}")
        if "particles" in group:
            del group["particles"]
        return _write_particles(self, group.create_group("particles"), pos, toffset)


def _write_particles(self, h5file_particles, pos, toffset):
    """
    Write the beam, and its :attr:`~simba.Modules.Beams.beam.others`, into an openPMD ``particles`` group.

    A beam tracked as several species at once (:meth:`~simba.Modules.Beams.beam.absorb`)
    is split back into one group per species. The primary one is the beam's own
    :attr:`species` if any are left, else the reference particle's, else the first.
    """
    groups = self.split_species() if hasattr(self, "split_species") else {self.species: self}
    if self.species in groups:
        primary = self.species
    else:
        primary = next(
            (name for name, b in groups.items() if b.reference_particle_index is not None),
            next(iter(groups)),
        )
    particles = _write_species(groups.pop(primary), h5file_particles, pos, toffset)
    others = list(groups.values()) + [
        b for name, b in getattr(self, "others", {}).items() if name not in groups and name != primary
    ]
    if others:
        h5file_particles.attrs[PRIMARY_SPECIES_ATTR] = primary
        for other in others:
            _write_species(other, h5file_particles, pos, toffset)
    return particles


def append_species(filename, beams) -> list:
    """
    Add ``beams`` to an existing openPMD beam file as extra species groups.

    The species already in the file stays the primary one. A species the file
    already holds is not overwritten.

    Parameters
    ----------
    filename: str
        openPMD file to add to; for a multi-turn file, its last turn.
    beams: iterable of :class:`~simba.Modules.Beams.beam`
        The beams to add.

    Returns
    -------
    list
        The species that were added.
    """
    added = []
    with File(os.path.expandvars(filename), "a") as h5file:
        bunch_data = _particles_group(h5file, filename)
        present = file_species(bunch_data)
        if PRIMARY_SPECIES_ATTR not in bunch_data.attrs and len(present) == 1:
            bunch_data.attrs[PRIMARY_SPECIES_ATTR] = present[0]
        for other in beams:
            if other.species in present:
                warn(f"{filename} already holds {other.species}; not adding another")
                continue
            _write_species(other, bunch_data, [0, 0, 0], 0)
            present.append(other.species)
            added.append(other.species)
    return added


def _weight(self):
    """
    openPMD ``weight`` [C]: the macroparticle's |charge|, or for a neutral species,
    the number of real particles it stands for times e, so that it means the same
    "macroparticle size" for every species.
    """
    if np.any(np.asarray(self.charge) != 0):
        return abs(self.charge)
    if self._beam.nmacro is None:
        raise ValueError(
            f"A {self.species} beam has no charge, so needs nmacro (real particles per "
            "macroparticle) to write its openPMD weight"
        )
    return UnitValue(
        np.broadcast_to(np.asarray(self._beam.nmacro, dtype=float), (len(self.x),))
        * constants.elementary_charge,
        units="C",
    )


def _write_species(self, h5file_particles, pos, toffset):
    """Write one species group into an openPMD ``particles`` group."""
    xoffset, yoffset, zoffset = pos
    data = {
        "x": self.x + UnitValue(xoffset, units="m"),
        "y": self.y + UnitValue(yoffset, units="m"),
        "z": self.z + UnitValue(zoffset, units="m"),
        "px": self.cpx,
        "py": self.cpy,
        "pz": self.cpz,
        "t": self.t + UnitValue(toffset, units="s"),
        "weight": _weight(self),
        "status": self._beam.status,
        "species": [self.species],
    }
    particles = ParticleGroup(data=data)
    particles.write(h5file_particles)
    h5file_species = h5file_particles[self.species]
    if self.s is not None:
        h5file_species["s"] = self.s
    if self.turn is not None:
        h5file_species["turn"] = int(self.turn)
    if hasattr(self, "reference_particle") and self.reference_particle is not None:
        write_openpmd_reference_particle(self, h5file_species)
    return particles


def write_openpmd_reference_particle(self, h5: File):
    ref_particle = self.reference_particle
    h5file_reference_particle = h5.create_group("reference_particle")
    for i, coord in enumerate(openpmd_coords):
        h5file_reference_particle[coord] = UnitValue(ref_particle[i])
    h5file_reference_particle['index'] = int(self.reference_particle_index)
