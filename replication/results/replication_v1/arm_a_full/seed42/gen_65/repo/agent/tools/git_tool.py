"""
Git tool: execute git commands for version control operations.

Provides git status, diff, log, add, commit, and branch operations.
"""

from __future__ import annotations

import subprocess
from typing import Any


def tool_info() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "git",
            "description": (
                "Execute git commands for version control operations. "
                "Supports: status, diff, log, add, commit, branch, checkout. "
                "Returns command output or error message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The git subcommand to execute (e.g., 'status', 'diff', 'log', 'add', 'commit')",
                        "enum": ["status", "diff", "log", "add", "commit", "branch", "checkout"],
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments for the git command (e.g., file paths, commit message)",
                        "default": "",
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the git repository (defaults to current directory)",
                        "default": ".",
                    },
                },
                "required": ["command"],
            },
        },
    }


def tool_function(command: str, args: str = "", repo_path: str = ".") -> str:
    """Execute a git command and return the output."""
    allowed_commands = {"status", "diff", "log", "add", "commit", "branch", "checkout"}
    
    if command not in allowed_commands:
        return f"Error: Unknown git command '{command}'. Allowed: {', '.join(allowed_commands)}"
    
    # Build the full command
    full_cmd = f"git {command}"
    if args:
        full_cmd += f" {args}"
    
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {output}"
        
        return output if output else "(no output)"
        
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing git command: {e}"
