"""
Git tool: perform git operations to track changes and understand repository history.

Provides capabilities to:
- Check repository status
- View commit history
- Show diffs between versions
- Track what files have been modified

This helps the meta agent understand the current state of changes and maintain
awareness of the codebase evolution.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Perform git operations to track changes and understand repository history. "
            "Check status, view commit history, show diffs, and track modified files. "
            "Helps maintain awareness of codebase changes and evolution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "log", "diff", "show"],
                    "description": "The git command to execute.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the git repository or file.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: 50).",
                },
                "commit": {
                    "type": "string",
                    "description": "Commit hash for 'show' command (default: HEAD).",
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


def _truncate_output(output: str, max_lines: int) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def _find_git_root(path: Path) -> Path | None:
    """Find the git repository root starting from path."""
    current = path if path.is_dir() else path.parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def tool_function(
    command: str,
    path: str,
    max_lines: int = 50,
    commit: str = "HEAD",
) -> str:
    """Execute git commands."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        
        allowed, error = _check_path(str(p))
        if not allowed:
            return error
        
        # Find git root
        git_root = _find_git_root(p)
        if not git_root:
            return f"Error: No git repository found at or above {path}"
        
        if command == "status":
            return _git_status(git_root, max_lines)
        elif command == "log":
            return _git_log(git_root, max_lines)
        elif command == "diff":
            return _git_diff(git_root, max_lines)
        elif command == "show":
            return _git_show(git_root, commit, max_lines)
        else:
            return f"Error: unknown command {command}"
    except Exception as e:
        return f"Error: {e}"


def _git_status(git_root: Path, max_lines: int) -> str:
    """Get git status."""
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "status", "-s"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: git status failed: {result.stderr}"
        
        output = result.stdout.strip()
        if not output:
            return f"Git Status for {git_root}:\nWorking tree clean - no uncommitted changes."
        
        return f"Git Status for {git_root}:\n{_truncate_output(output, max_lines)}"
    except subprocess.TimeoutExpired:
        return "Error: git status timed out"
    except Exception as e:
        return f"Error running git status: {e}"


def _git_log(git_root: Path, max_lines: int) -> str:
    """Get git log."""
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "log", "--oneline", "-20"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: git log failed: {result.stderr}"
        
        output = result.stdout.strip()
        return f"Git Log for {git_root} (last 20 commits):\n{_truncate_output(output, max_lines)}"
    except subprocess.TimeoutExpired:
        return "Error: git log timed out"
    except Exception as e:
        return f"Error running git log: {e}"


def _git_diff(git_root: Path, max_lines: int) -> str:
    """Get git diff of uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: git diff failed: {result.stderr}"
        
        output = result.stdout.strip()
        if not output:
            return f"Git Diff for {git_root}:\nNo uncommitted changes to show."
        
        return f"Git Diff Stats for {git_root}:\n{_truncate_output(output, max_lines)}"
    except subprocess.TimeoutExpired:
        return "Error: git diff timed out"
    except Exception as e:
        return f"Error running git diff: {e}"


def _git_show(git_root: Path, commit: str, max_lines: int) -> str:
    """Show commit details."""
    try:
        result = subprocess.run(
            ["git", "-C", str(git_root), "show", "--stat", "-s", commit],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return f"Error: git show failed: {result.stderr}"
        
        output = result.stdout.strip()
        return f"Git Show for {commit} in {git_root}:\n{_truncate_output(output, max_lines)}"
    except subprocess.TimeoutExpired:
        return "Error: git show timed out"
    except Exception as e:
        return f"Error running git show: {e}"
