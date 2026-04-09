"""
Git tool: run git commands to inspect repository state.

Provides safe git operations for the meta agent to understand
the repository structure, recent changes, and branch status.
"""

from __future__ import annotations

import os
import subprocess


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Run git commands to inspect repository state. "
            "Useful for checking status, recent commits, branches, and diffs. "
            "Only read-only operations are allowed (status, log, diff, show, branch)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git subcommand to run (e.g., 'status', 'log --oneline -5', 'diff').",
                }
            },
            "required": ["command"],
        },
    }


_ALLOWED_COMMANDS = {
    "status", "log", "diff", "show", "branch", "remote", "config",
    "ls-files", "ls-tree", "rev-parse", "describe", "tag",
}

_TIMEOUT = 30.0
_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set working directory for git commands."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_safe_command(command: str) -> bool:
    """Check if the git command is read-only and safe."""
    parts = command.strip().split()
    if not parts:
        return False
    
    # Check main command
    main_cmd = parts[0].lower()
    if main_cmd not in _ALLOWED_COMMANDS:
        return False
    
    # Block dangerous flags
    dangerous = {"--exec", "-c", "--git-dir", "--work-tree", ";", "|", "&", "`", "$"}
    for part in parts:
        if any(d in part for d in dangerous):
            return False
    
    return True


def tool_function(command: str) -> str:
    """Execute a safe git command. Returns output.
    
    Only read-only git commands are allowed for safety.
    Commands run within the allowed root directory.
    """
    if not _is_safe_command(command):
        return f"Error: Command '{command}' is not allowed. Only read-only operations (status, log, diff, show, branch, etc.) are permitted."
    
    cwd = _ALLOWED_ROOT or os.getcwd()
    
    # Verify cwd is within allowed root
    if _ALLOWED_ROOT:
        real_cwd = os.path.realpath(cwd)
        real_root = os.path.realpath(_ALLOWED_ROOT)
        if not real_cwd.startswith(real_root):
            return f"Error: Working directory {cwd} is outside allowed root {_ALLOWED_ROOT}"
    
    try:
        result = subprocess.run(
            ["git"] + command.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {output}"
        
        return output if output else "(no output)"
        
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {e}"
