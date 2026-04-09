"""
Git tool: interact with git repositories.

Provides git operations like status, log, diff, and branch information
to help the agent understand version control state.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Interact with git repositories. "
            "Provides operations like status, log, diff, and branch info. "
            "Helps understand version control state before making modifications."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "log", "diff", "branch", "show"],
                    "description": "The git command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the git repository (directory with .git).",
                },
                "args": {
                    "type": "string",
                    "description": "Additional arguments for the git command (e.g., file path, commit hash).",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: 50).",
                    "default": 50,
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


def _run_git_command(repo_path: str, git_args: list[str], max_lines: int = 50) -> str:
    """Run a git command and return formatted output."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path] + git_args,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Git error: {result.stderr.strip()}"
        
        output = result.stdout.strip()
        lines = output.split("\n")
        
        if len(lines) > max_lines:
            output = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        
        return output if output else "(no output)"
        
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out after 30 seconds"
    except FileNotFoundError:
        return "Error: git command not found. Is git installed?"
    except Exception as e:
        return f"Error running git command: {e}"


def tool_function(
    command: str,
    path: str,
    args: str | None = None,
    max_lines: int = 50,
) -> str:
    """Execute a git command.
    
    Args:
        command: The git subcommand (status, log, diff, branch, show)
        path: Path to the git repository
        args: Additional arguments for the command
        max_lines: Maximum lines to return
    
    Returns:
        Git command output
    """
    try:
        p = Path(path)
        
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"
        
        if not p.exists():
            return f"Error: {path} does not exist."
        
        # Check if it's a git repository
        git_dir = p / ".git"
        if not git_dir.exists() and not (p / ".git").is_file():
            # Check if path is inside a git repo
            try:
                result = subprocess.run(
                    ["git", "-C", str(p), "rev-parse", "--git-dir"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    return f"Error: {path} is not a git repository (or any parent)"
            except Exception:
                return f"Error: {path} is not a git repository"
        
        # Build git arguments
        git_args = []
        
        if command == "status":
            git_args = ["status", "--short"]
            if args:
                git_args.extend(args.split())
        
        elif command == "log":
            git_args = ["log", "--oneline", "-20"]
            if args:
                git_args = ["log"] + args.split()
        
        elif command == "diff":
            git_args = ["diff"]
            if args:
                git_args.extend(args.split())
        
        elif command == "branch":
            git_args = ["branch", "-vv"]
            if args:
                git_args = ["branch"] + args.split()
        
        elif command == "show":
            if args:
                git_args = ["show", "--stat"] + args.split()
            else:
                git_args = ["show", "--stat", "HEAD"]
        
        else:
            return f"Error: unknown git command '{command}'"
        
        return _run_git_command(str(p), git_args, max_lines)
        
    except Exception as e:
        return f"Error: {e}"
