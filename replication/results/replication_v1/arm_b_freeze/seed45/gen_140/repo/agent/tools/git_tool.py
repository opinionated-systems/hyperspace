"""
Git tool: execute git commands for repository management.

Provides git operations like status, log, diff, and branch management
to help track changes and understand repository history.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Execute git commands for repository management. "
            "Provides operations like status, log, diff, and branch management "
            "to help track changes and understand repository history. "
            "Only safe read-only commands are allowed by default."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "log", "diff", "show", "branch", "remote"],
                    "description": "The git command to execute.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the git repository (must be within allowed root).",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional arguments for the git command.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: 100).",
                    "default": 100,
                },
            },
            "required": ["command", "path"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict git operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _check_path(path: str) -> tuple[bool, str]:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is not None:
        resolved = os.path.abspath(path)
        if not resolved.startswith(_ALLOWED_ROOT):
            return False, f"Error: access denied. Git operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _is_git_repo(path: str) -> bool:
    """Check if the path is a git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_git_command(
    path: str,
    command: str,
    args: list[str] | None = None,
    max_lines: int = 100,
) -> str:
    """Execute a git command and return the output."""
    cmd = ["git", "-C", path, command]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            error = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: git {command} failed: {error}"
        
        output = result.stdout
        lines = output.split("\n")
        
        if len(lines) > max_lines:
            truncated = lines[:max_lines // 2] + ["... (output truncated) ..."] + lines[-max_lines // 2:]
            output = "\n".join(truncated)
        
        return output if output else "(no output)"
        
    except subprocess.TimeoutExpired:
        return f"Error: git {command} timed out after 30 seconds"
    except FileNotFoundError:
        return "Error: git command not found. Is git installed?"
    except Exception as e:
        return f"Error executing git {command}: {type(e).__name__}: {e}"


def tool_function(
    command: str,
    path: str,
    args: list[str] | None = None,
    max_lines: int = 100,
) -> str:
    """Execute a git command."""
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        # Check if path exists
        if not p.exists():
            return f"Error: {p} does not exist."
        
        # Check if it's a git repository
        if not _is_git_repo(str(p)):
            return f"Error: {p} is not a git repository (no .git directory found)."
        
        # Execute the command
        return _run_git_command(str(p), command, args, max_lines)
        
    except Exception as e:
        return f"Error: {e}"
