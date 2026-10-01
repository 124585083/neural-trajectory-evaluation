"""Load the two retained pilot modules without global search-path changes."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load(name):
    identity = '_mental_pong_publication_' + name
    if identity not in sys.modules:
        spec = spec_from_file_location(identity, Path(__file__).resolve().parent.parent/(name+'.py'))
        module = module_from_spec(spec)
        sys.modules[identity] = module
        spec.loader.exec_module(module)
    return sys.modules[identity]


verified_file = _load('data_pipeline').verified_file
SharedTimescaleGPFA = _load('gpfa_shared').SharedTimescaleGPFA
