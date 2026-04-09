"""
Git tool: perform basic git operations.

Provides safe git operations for version control.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def tool_info() -> dict:
    return {
        "name": "git",
        "description": "Perform basic git operations: status, diff, add, commit, log. Useful for tracking changes and creating checkpoints. All operations are performed in the repository root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "diff", "add", "commit", "log", "show"],
                    "description": "Git command to execute",
                },
                "path": {
                    "type": "string",
                    "description": "File path for add/diff/show commands (optional)",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (required for commit)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of commits to show (for log command, default 10)",
                    "default": 10,
                },
            },
            "required": ["command"],
        },
    }


def _run_git(args: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except FileNotFoundError:
        return False, "Git not found - is it installed?"
    except Exception as e:
        return False, f"Error running git: {e}"


def _find_git_root(start_path: str = ".") -> str | None:
    """Find the git repository root from a starting path."""
    current = os.path.abspath(start_path)
    while current != "/":
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def tool_function(
    command: str,
    path: str | None = None,
    message: str | None = None,
    limit: int = 10,
) -> str:
    """Execute a git command."""
    # Find git root
    git_root = _find_git_root(path or ".")
    if git_root is None:
        return "Error: Not in a git repository"
    
    if command == "status":
        success, output = _run_git(["status"], cwd=git_root)
        if success:
            return f"Git status:\n{output}"
        return f"Error: {output}"
    
    elif command == "diff":
        args = ["diff"]
        if path:
            args.append(path)
        success, output = _run_git(args, cwd=git_root)
        if success:
            if output:
                return f"Git diff:\n{output[:3000]}"  # Limit output
            return "No changes to show"
        return f"Error: {output}"
    
    elif command == "add":
        if not path:
            return "Error: path required for add command"
        success, output = _run_git(["add", path], cwd=git_root)
        if success:
            return f"Added {path} to staging area"
        return f"Error: {output}"
    
    elif command == "commit":
        if not message:
            return "Error: message required for commit command"
        success, output = _run_git(["commit", "-m", message], cwd=git_root)
        if success:
            return f"Committed: {output}"
        return f"Error: {output}"
    
    elif command == "log":
        args = ["log", f"--oneline", f"-n", str(limit)]
        success, output = _run_git(args, cwd=git_root)
        if success:
            return f"Recent commits:\n{output}"
        return f"Error: {output}"
    
    elif command == "show":
        args = ["show", "--stat"]
        if path:
            args.append(path)
        success, output = _run_git(args, cwd=git_root)
        if success:
            return f"Git show:\n{output[:3000]}"
        return f"Error: {output}"
    
    else:
        return f"Error: Unknown git command: {command}"
