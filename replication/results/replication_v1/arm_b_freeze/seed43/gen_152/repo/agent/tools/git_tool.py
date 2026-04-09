"""
Git tool: perform basic git operations for version control.

Provides git status, diff, and log operations to help track changes
during the self-improvement process.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Perform basic git operations. "
            "Supports status, diff, and log commands. "
            "Useful for tracking changes during code modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "show"],
                    "description": "The git command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the git repository (default: allowed root).",
                },
                "args": {
                    "type": "string",
                    "description": "Additional arguments for the git command.",
                },
            },
            "required": ["command"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict git operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _validate_path(path: str) -> tuple[bool, str]:
    """Validate that a path is within the allowed root."""
    if _ALLOWED_ROOT is None:
        return True, ""
    
    resolved = os.path.abspath(path)
    if not resolved.startswith(_ALLOWED_ROOT):
        return False, f"Error: access denied. Git operations restricted to {_ALLOWED_ROOT}"
    return True, ""


def _truncate_output(output: str, max_len: int = 10000) -> str:
    """Truncate output to prevent context overflow."""
    if len(output) <= max_len:
        return output
    half = max_len // 2
    return output[:half] + "\n... [output truncated] ...\n" + output[-half:]


def _run_git_command(
    command: str,
    repo_path: str,
    extra_args: str = ""
) -> str:
    """Execute a git command in the specified repository."""
    
    # Build the git command
    cmd = ["git", "-C", repo_path]
    
    if command == "status":
        cmd.extend(["status", "--short"])
        if extra_args:
            cmd.extend(extra_args.split())
    elif command == "diff":
        cmd.extend(["diff"])
        if extra_args:
            cmd.extend(extra_args.split())
    elif command == "log":
        cmd.extend(["log", "--oneline", "-10"])
        if extra_args:
            cmd.extend(extra_args.split())
    elif command == "show":
        if extra_args:
            cmd.extend(["show", extra_args])
        else:
            cmd.extend(["show", "HEAD"])
    else:
        return f"Error: Unknown git command '{command}'"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if not output:
                return f"Git {command}: (no output)"
            return _truncate_output(output)
        else:
            error = result.stderr.strip()
            return f"Git error: {error}"
            
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out (30s limit)"
    except FileNotFoundError:
        return "Error: Git not found. Is git installed?"
    except Exception as e:
        return f"Error running git command: {type(e).__name__}: {e}"


def tool_function(
    command: str,
    path: str | None = None,
    args: str = "",
) -> str:
    """Execute a git command.
    
    Args:
        command: The git command to run (status, diff, log, show)
        path: Path to the git repository (default: allowed root)
        args: Additional arguments for the git command
        
    Returns:
        Git command output or error message
    """
    # Determine the repository path
    if path is None:
        repo_path = _ALLOWED_ROOT or os.getcwd()
    else:
        valid, error = _validate_path(path)
        if not valid:
            return error
        repo_path = path
    
    # Check if it's a git repository
    git_dir = Path(repo_path) / ".git"
    if not git_dir.exists():
        return f"Error: {repo_path} is not a git repository"
    
    return _run_git_command(command, repo_path, args)
