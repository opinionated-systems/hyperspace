"""
Git tool: perform git operations like status, diff, log, and show.

Provides git command execution to inspect repository state,
useful for understanding changes before and after modifications.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Git operations for repository inspection. "
            "Commands: status, diff, log, show. "
            "Useful for checking repository state before and after modifications."
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
                    "description": "Repository path (default: allowed root or current dir).",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (default: 100).",
                },
                "file_path": {
                    "type": "string",
                    "description": "Specific file path for diff or show (optional).",
                },
            },
            "required": ["command"],
        },
    }


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for git operations."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def _is_within_root(path: str) -> bool:
    """Check if path is within allowed root."""
    if _ALLOWED_ROOT is None:
        return True
    try:
        Path(path).resolve().relative_to(Path(_ALLOWED_ROOT).resolve())
        return True
    except ValueError:
        return False


def _run_git_command(args: list[str], cwd: str) -> tuple[str, str, int]:
    """Run a git command and return stdout, stderr, returncode."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Error: Git command timed out after 30 seconds", 1
    except Exception as e:
        return "", f"Error: {e}", 1


def _truncate_output(output: str, max_lines: int) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def tool_function(
    command: str,
    path: str | None = None,
    max_lines: int = 100,
    file_path: str | None = None,
) -> str:
    """Execute a git command.

    Args:
        command: The git command (status, diff, log, show)
        path: Repository path (default: allowed root or current dir)
        max_lines: Maximum lines to return
        file_path: Specific file for diff or show

    Returns:
        Git command output
    """
    repo_path = path or _ALLOWED_ROOT or "."

    if not _is_within_root(repo_path):
        return f"Error: Path '{repo_path}' is outside allowed root."

    # Check if it's a git repository
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        # Try to find git root
        stdout, stderr, rc = _run_git_command(["rev-parse", "--git-dir"], repo_path)
        if rc != 0:
            return f"Error: Not a git repository (or no .git directory found in {repo_path})"

    if command == "status":
        stdout, stderr, rc = _run_git_command(["status", "-sb"], repo_path)
        if rc != 0:
            return f"Error: {stderr}"
        return _truncate_output(stdout or "No changes", max_lines)

    elif command == "diff":
        args = ["diff"]
        if file_path:
            if not _is_within_root(os.path.join(repo_path, file_path)):
                return f"Error: File path '{file_path}' is outside allowed root."
            args.append(file_path)
        args.extend(["--stat"])
        stdout, stderr, rc = _run_git_command(args, repo_path)
        if rc != 0:
            return f"Error: {stderr}"
        return _truncate_output(stdout or "No differences", max_lines)

    elif command == "log":
        args = ["log", "--oneline", "-n", str(max_lines)]
        stdout, stderr, rc = _run_git_command(args, repo_path)
        if rc != 0:
            return f"Error: {stderr}"
        return stdout or "No commits"

    elif command == "show":
        args = ["show", "--stat", "-p"]
        if file_path:
            if not _is_within_root(os.path.join(repo_path, file_path)):
                return f"Error: File path '{file_path}' is outside allowed root."
            args.append(file_path)
        else:
            args.append("HEAD")
        stdout, stderr, rc = _run_git_command(args, repo_path)
        if rc != 0:
            return f"Error: {stderr}"
        return _truncate_output(stdout or "Nothing to show", max_lines)

    else:
        return f"Error: Unknown command '{command}'. Use 'status', 'diff', 'log', or 'show'."
