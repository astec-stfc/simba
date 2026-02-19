import importlib

# Lazy import mapping: attribute name -> (module path, class name)
_LATTICE_MODULES = {
    'astraLattice': ('.Codes.ASTRA.ASTRA', 'astraLattice'),
    'gptLattice': ('.Codes.GPT.GPT', 'gptLattice'),
    'elegantLattice': ('.Codes.Elegant.Elegant', 'elegantLattice'),
    'ocelotLattice': ('.Codes.Ocelot.Ocelot', 'ocelotLattice'),
    'csrtrackLattice': ('.Codes.CSRTrack.CSRTrack', 'csrtrackLattice'),
    'cheetahLattice': ('.Codes.Cheetah.Cheetah', 'cheetahLattice'),
    'xsuiteLattice': ('.Codes.Xsuite.Xsuite', 'xsuiteLattice'),
    'waketLattice': ('.Codes.Wake_T.Wake_T', 'waketLattice'),
    'genesisLattice': ('.Codes.Genesis.Genesis', 'genesisLattice'),
    'opalLattice': ('.Codes.OPAL.OPAL', 'opalLattice'),
}

# List of supported codes (avoids eagerly importing all modules)
supported_codes = [name.replace('Lattice', '').replace('lattice', '') for name in _LATTICE_MODULES]


def __getattr__(name):
    if name in _LATTICE_MODULES:
        module_path, attr_name = _LATTICE_MODULES[name]
        module = importlib.import_module(module_path, package='simba')
        cls = getattr(module, attr_name)
        globals()[name] = cls  # Cache for subsequent access
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LATTICE_MODULES.keys()) + list(globals().keys())
