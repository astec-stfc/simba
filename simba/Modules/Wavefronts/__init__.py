"""
Photon wavefronts from FELs or wigglers; work in progress.

Wavefronts are openpmd-beamphysics ``Wavefront`` objects; only GENESIS 4 field files are supported.
"""
import os
import glob
from pydantic import BaseModel
from typing import Dict
try:
    from beamphysics.wavefront.wavefront import Wavefront
except ImportError:
    from pmd_beamphysics.wavefront.wavefront import Wavefront


class wavefrontGroup(BaseModel):
    """A group of wavefronts, e.g. from :func:`~simba.Modules.Wavefronts.load_directory`."""

    sddsindex: int = 0
    """Index for SDDS files."""

    wavefronts: Dict = {}
    """``Wavefront`` objects keyed by filename."""

    def __repr__(self):
        return repr(list(self.wavefronts.keys()))

    def __len__(self):
        return len(list(self.wavefronts.keys()))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.wavefronts.values())[key]
        elif isinstance(key, slice):
            return wavefrontGroup(wavefronts=list(self.wavefronts.items())[key])
        else:
            return getattr(self, key)

    def __init__(self, filenames=[], wavefronts=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sddsindex = 0
        self.wavefronts = {}
        for k, v in wavefronts:
            self.wavefronts[k] = v
        if isinstance(filenames, str):
            filenames = [filenames]
        for f in filenames:
            self.add(f)

    def add(self, filename):
        if isinstance(filename, str):
            filename = [filename]
        for file in filename:
            if os.path.isdir(file):
                self.add(glob.glob(os.path.join(file, "*.fld.h5")))
            elif os.path.isfile(file):
                file = file.replace("\\", "/")
                try:
                    self.wavefronts[file] = Wavefront.from_genesis4(file)
                except Exception:
                    if file in self.wavefronts:
                        del self.wavefronts[file]

    def getWavefront(self, wavefront):
        for b in self.wavefronts:
            if wavefront == ".".join(os.path.splitext(os.path.basename(b))[0].split('.')[:-1]):
                return self.wavefronts[b]
        return None

    def getWavefronts(self):
        return {".".join(os.path.splitext(os.path.basename(b))[0].split('.')[:-1]): b for b in self.wavefronts}


def load_directory(directory=".", types={"Genesis": ".fld.h5"}, verbose=False) -> wavefrontGroup:
    """
    Load every wavefront file in a directory into a new :class:`wavefrontGroup`.

    Parameters
    ----------
    directory: str
        Directory to load
    types: Dict
        Code name to filename suffix
    verbose: bool
        Print progress

    Returns
    -------
    :class:`~simba.Modules.Wavefronts.wavefrontGroup`
    """
    wg = wavefrontGroup()
    if verbose:
        print("Directory:", directory)
    for code, string in types.items():
        wavefront_files = glob.glob(directory + "/*" + string)
        if verbose:
            print(code, [os.path.basename(t) for t in wavefront_files])
        wg.add(wavefront_files)
    return wg