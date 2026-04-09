"""
Git tool: execute git commands for version control operations.

Provides safe git operations with validation to prevent destructive actions.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "name": "git",
        "description": "Execute git commands for version control. Supports status, log, diff, branch, and safe operations. Prevents destructive commands like reset --hard, clean -f, or push --force.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Git subcommand and arguments (e.g., 'status', 'log --oneline -5', 'diff HEAD~1'). Do not include 'git' prefix.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the git command. Defaults to current directory.",
                },
            },
            "required": ["command"],
        },
    }


# List of dangerous git commands that are blocked
_BLOCKED_PATTERNS = [
    "reset --hard",
    "clean -f",
    "push --force",
    "push -f",
    "filter-branch",
    "rm -rf",
    "rm -r",
    "rm .",
    "rm *",
    "checkout -f",
    "branch -D",
    "branch -d",
]


def _is_safe_command(command: str) -> bool:
    """Check if the git command is safe to execute."""
    cmd_lower = command.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return False
    return True


def tool_function(command: str, cwd: str | None = None) -> str:
    """Execute a git command safely."""
    if not _is_safe_command(command):
        return f"Error: Command '{command}' contains blocked patterns for safety."

    try:
        full_cmd = ["git"] + command.split()
        result = subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {result.stderr}"
        return result.stdout or "Command executed successfully (no output)."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing git command: {e}"
