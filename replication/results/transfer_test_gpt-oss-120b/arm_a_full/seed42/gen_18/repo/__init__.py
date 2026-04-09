"""Top-level package for the HyperAgents repository.

Provides convenient imports for the main components:
- MetaAgent
- TaskAgent
- utility functions
"""

try:
    from .meta_agent import MetaAgent
except Exception:  # pragma: no cover
    class MetaAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("MetaAgent could not be imported due to missing dependencies.")

try:
    from .task_agent import TaskAgent
except Exception:  # pragma: no cover
    class TaskAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("TaskAgent could not be imported due to missing dependencies.")

from .utils import list_python_files

__all__ = [
    "MetaAgent",
    "TaskAgent",
    "list_python_files",
]
