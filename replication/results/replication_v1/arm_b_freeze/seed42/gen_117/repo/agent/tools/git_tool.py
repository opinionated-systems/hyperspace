"""
Git tool: provides git operations for version control.

Allows the agent to check repository status, view commit history,
and track changes in the codebase.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for OpenAI function calling."""
    return {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Execute git commands to check repository status, view commit history, or track changes. Returns command output or error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["status", "log", "diff", "branch"],
                        "description": "The git command to execute",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional path to the git repository (defaults to current directory)",
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for the git command (e.g., '--oneline -5' for log)",
                    },
                },
                "required": ["command"],
            },
        },
    }


def tool_function(command: str, path: str = ".", args: str = "") -> str:
    """Execute a git command and return the output."""
    # Validate command for security
    allowed_commands = {"status", "log", "diff", "branch"}
    if command not in allowed_commands:
        return f"Error: Command '{command}' is not allowed. Allowed commands: {allowed_commands}"

    cmd_parts = ["git", "-C", path, command]
    if args:
        cmd_parts.extend(args.split())

    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout or "Command executed successfully (no output)"
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing git command: {e}"
