from .codes.astra.astra import AstraLattice  # noqa F401
from .codes.gpt.gpt import GptLattice  # noqa F401
from .codes.elegant.elegant import ElegantLattice  # noqa F401
from .codes.ocelot.ocelot import OcelotLattice  # noqa F401
from .codes.csrtrack.csrtrack import CsrTrackLattice  # noqa F401
from .codes.cheetah.cheetah import CheetahLattice    # noqa F401
from .codes.xsuite.xsuite import XsuiteLattice  # noqa F401
from .codes.wake_t.wake_t import WaketLattice           # noqa F401
from .codes.genesis.genesis import GenesisLattice           # noqa F401
from .codes.opal.opal import OpalLattice                # noqa F401
# from .MAD8.MAD8 import mad8Lattice

LATTICE_CLASSES = {
    "astra": AstraLattice,
    "cheetah": CheetahLattice,
    "csrtrack": CsrTrackLattice,
    "elegant": ElegantLattice,
    "genesis": GenesisLattice,
    "gpt": GptLattice,
    "ocelot": OcelotLattice,
    "opal": OpalLattice,
    "waket": WaketLattice,
    "xsuite": XsuiteLattice,
}
