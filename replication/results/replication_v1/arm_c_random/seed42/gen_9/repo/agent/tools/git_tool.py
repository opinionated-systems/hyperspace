"""
Git tool: execute git commands for version control operations.

Provides git status, diff, log, and basic commit operations.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool specification for LLM function calling."""
    return {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Execute git commands for version control operations. Supports status, diff, log, add, commit, and branch operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "add", "commit", "branch", "checkout"],
                        "description": "The git subcommand to execute",
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for the git command (e.g., file paths, commit message)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Working directory for the git command (default: current directory)",
                    },
                },
                "required": ["command"],
            },
        },
    }


def tool_function(command: str, args: str = "", path: str = ".") -> str:
    """Execute a git command and return the output.

    Args:
        command: The git subcommand (status, diff, log, add, commit, branch, checkout)
        args: Additional arguments for the command
        path: Working directory for git operations

    Returns:
        Command output or error message
    """
    # Build the full command
    full_cmd = f"git {command}"
    if args:
        full_cmd += f" {args}"

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            cwd=path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing git command: {e}"
