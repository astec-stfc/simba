import os
from warnings import warn

try:
    from beamphysics import ParticleGroup, pmd_init, particle_paths
except ImportError:
    from pmd_beamphysics import ParticleGroup, pmd_init, particle_paths
from h5py import File
from .. import constants
from ..units import UnitValue


TURN_BASE_PATH = "/data/%T/"
"""openPMD ``basePath`` of a multi-turn file: every turn is an iteration in the
one file (openPMD's group-based encoding), ``/data/<turn>/particles``."""

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
    """The turns ``filename`` holds, in order: empty for a single beam.
    A multi-turn run that keeps every turn writes each element's beams into
    one file, a turn to an iteration; :func:`read_openpmd_beam_file` reads
    one.
    """
    with File(os.path.expandvars(filename), "r") as h5file:
        return _turns(h5file)


def is_openpmd_beam_file(h5file: File) -> bool:
    """Whether an open file holds a beam: a single one or a multi-turn file."""
    return "/particles" in h5file or bool(_turns(h5file))


def read_openpmd_beam_file(self, filename, turn=None):
    """Read an openPMD beam file.
    ``turn`` picks one turn of a multi-turn file (see :func:`openpmd_turns`),
    which otherwise reads as its last. A single beam is read whatever ``turn`` says.
    """
    self.filename = filename
    fname = os.path.expandvars(filename)
    with File(fname, "r") as h5file:
        turns = _turns(h5file)
        if turns:
            chosen = turns[-1] if turn is None else int(turn)
            if chosen not in turns:
                raise ValueError(
                    f"{filename} holds turns {turns[0]} to {turns[-1]}, not {turn}"
                )
            bunch_data = h5file[f"{TURN_BASE_PATH.split('%T')[0]}{chosen}/particles"]
        else:
            bunch_data = h5file[particle_paths(h5file)[0]]
        particles = ParticleGroup(h5=bunch_data)
        read_particle_group(self, particles)
        bunch_species = bunch_data[particles.species]
        self._beam.s = UnitValue(bunch_species["s"], units="m") if "s" in bunch_species else None
        self.turn = int(bunch_species["turn"][()]) if "turn" in bunch_species else None
        if "reference_particle" in bunch_species:
            ref_particle = bunch_species["reference_particle"]
            self.reference_particle = [ref_particle[coord][()] for coord in openpmd_coords]
            self.reference_particle_index = int(ref_particle["index"][()])


def write_openpmd_beam_file(
    self,
    filename,
    pos=[0, 0, 0],
    toffset=0,
    turn=None,
):
    """Write the beam to an openPMD file.
    ``turn`` makes it that turn of a multi-turn file (:data:`TURN_BASE_PATH`):
    turn 1, or a file that is not yet multi-turn, starts it afresh.
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
    """The beam into an openPMD ``particles`` group."""
    xoffset, yoffset, zoffset = pos
    data = {
        "x": self.x + UnitValue(xoffset, units="m"),
        "y": self.y + UnitValue(yoffset, units="m"),
        "z": self.z + UnitValue(zoffset, units="m"),
        "px": self.cpx,
        "py": self.cpy,
        "pz": self.cpz,
        "t": self.t + UnitValue(toffset, units="s"),
        "weight": abs(self.charge),
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
