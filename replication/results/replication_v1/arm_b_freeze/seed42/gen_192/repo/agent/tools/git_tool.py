"""
Git tool for repository operations.

Provides git commands for the agent to interact with version control.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "git",
        "description": "Execute git commands for version control operations. Supports status, log, diff, branch, checkout, commit, and other git operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git subcommand to execute (e.g., 'status', 'log', 'diff', 'branch', 'checkout', 'commit', 'add', 'clone', 'pull', 'push'). Do not include 'git' prefix.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional arguments for the git command.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the git command. Defaults to current directory.",
                },
            },
            "required": ["command"],
        },
    }


def tool_function(command: str, args: list[str] | None = None, cwd: str | None = None) -> str:
    """Execute a git command and return the output."""
    if args is None:
        args = []
    
    # Build the full command
    full_cmd = ["git", command] + args
    
    try:
        result = subprocess.run(
            full_cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        
        if result.returncode != 0:
            return f"Error (exit code {result.returncode}): {output}"
        
        return output if output else "Command executed successfully (no output)."
    
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out after 60 seconds."
    except FileNotFoundError:
        return "Error: Git executable not found. Is git installed?"
    except Exception as e:
        return f"Error executing git command: {e}"
