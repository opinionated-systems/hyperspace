"""Utility helpers for the HyperAgents repository.

The original implementation of :func:`list_python_files` used ``os.walk`` and
manually concatenated paths.  While functional, it had a few drawbacks:

* It returned **absolute** paths even when the caller passed a relative one,
  making the output harder to compare in tests.
* The order of files depended on the underlying filesystem traversal order,
  which is nondeterministic across platforms.
* It silently ignored symbolic‑link loops and did not expose type information
  for static analysis tools.

The new implementation leverages :class:`pathlib.Path` which provides a more
expressive and cross‑platform API.  It returns **sorted relative paths** (as
strings) from the supplied ``root`` directory, guaranteeing deterministic
output.  The function now also includes a small amount of defensive coding –
it raises a clear ``ValueError`` if ``root`` does not exist or is not a
directory, helping callers surface configuration errors early.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def list_python_files(root: str | Path) -> List[str]:
    """Return a **sorted** list of all ``.py`` files under ``root``.

    The paths are returned as strings **relative** to ``root``.  This makes the
    function suitable for both display purposes and deterministic unit tests.

    Parameters
    ----------
    root:
        The directory to search.  It can be a string or a :class:`Path`
        instance.

    Returns
    -------
    List[str]
        Sorted relative file paths (e.g. ``["agent/__init__.py", "utils.py"]``).
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f"{root!r} is not a valid directory")

    # ``rglob`` recursively yields matching files.  ``relative_to`` converts each
    # absolute path to a path relative to the root, which we then cast to ``str``.
    py_files = [p.relative_to(root_path).as_posix() for p in root_path.rglob("*.py")]

    # Sorting provides deterministic ordering across runs and platforms.
    py_files.sort()
    return py_files

