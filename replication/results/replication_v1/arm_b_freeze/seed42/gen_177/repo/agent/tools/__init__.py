"""
Agent tools package.

Provides tools for the agentic loop:
- bash: Run shell commands with persistent state
- editor: View, create, and edit files
- file_stats: Get file statistics
- search: Search for patterns in files
- view_tree: Display directory structure
- time: Get current time and date information
"""

from __future__ import annotations

from agent.tools.bash_tool import tool_function as bash, tool_info as bash_info
from agent.tools.editor_tool import tool_function as editor, tool_info as editor_info
from agent.tools.file_stats_tool import tool_function as file_stats, tool_info as file_stats_info
from agent.tools.search_tool import tool_function as search, tool_info as search_info
from agent.tools.view_tree_tool import tool_function as view_tree, tool_info as view_tree_info
from agent.tools.time_tool import get_current_time as time, TIME_TOOL_SCHEMA as time_info

__all__ = [
    "bash",
    "bash_info",
    "editor",
    "editor_info",
    "file_stats",
    "file_stats_info",
    "search",
    "search_info",
    "view_tree",
    "view_tree_info",
    "time",
    "time_info",
]
