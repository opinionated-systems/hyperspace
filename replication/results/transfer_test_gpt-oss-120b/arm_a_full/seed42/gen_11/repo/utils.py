'''Utility functions for the HyperAgents repo.'''

import os
from pathlib import Path

def list_python_files(root: str) -> list[str]:
    """Return a list of all .py files under the given root directory."""
    py_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.py'):
                py_files.append(os.path.join(dirpath, f))
    return py_files
