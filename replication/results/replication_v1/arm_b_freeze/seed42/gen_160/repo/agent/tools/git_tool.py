"""
Git tool: perform git operations like status, log, diff, and show.

Provides git introspection capabilities for the agent to track changes
and understand repository state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Git operations for repository introspection. "
            "Commands: status, log, diff, show, branch. "
            "Helps track changes and understand repository state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "log", "diff", "show", "branch"],
                    "description": "The git command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to git repository root.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (for log/diff).",
                    "default": 50,
                },
                "commit": {
                    "type": "string",
                    "description": "Commit hash for show command.",
                },
            },
            "required": ["command", "path"],
        },
    }


def _run_git(args: list[str], cwd: str, check: bool = True) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.returncode, result.stdout, result.stderr


def _truncate_output(output: str, max_lines: int = 50) -> str:
    """Truncate output to max_lines."""
    lines = output.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return output


def tool_function(
    command: str,
    path: str,
    max_lines: int = 50,
    commit: str | None = None,
) -> str:
    """Execute a git command."""
    try:
        p = Path(path)
        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."
        if not p.exists():
            return f"Error: {path} does not exist."

        # Check if it's a git repository
        rc, _, _ = _run_git(["rev-parse", "--git-dir"], str(p), check=False)
        if rc != 0:
            return f"Error: {path} is not a git repository."

        if command == "status":
            rc, stdout, stderr = _run_git(["status", "-sb"], str(p), check=False)
            if rc != 0:
                return f"Error: {stderr}"
            return f"Git status in {path}:\n{stdout}"

        elif command == "log":
            rc, stdout, stderr = _run_git(
                ["log", "--oneline", "-n", str(max_lines)], str(p), check=False
            )
            if rc != 0:
                return f"Error: {stderr}"
            return f"Git log in {path}:\n{_truncate_output(stdout, max_lines)}"

        elif command == "diff":
            rc, stdout, stderr = _run_git(["diff", "--stat"], str(p), check=False)
            if rc != 0:
                return f"Error: {stderr}"
            if not stdout.strip():
                return f"No uncommitted changes in {path}."
            return f"Git diff in {path}:\n{_truncate_output(stdout, max_lines)}"

        elif command == "show":
            if not commit:
                # Get the latest commit if none specified
                rc, stdout, _ = _run_git(["rev-parse", "HEAD"], str(p), check=False)
                if rc != 0:
                    return f"Error: could not get HEAD commit."
                commit = stdout.strip()
            rc, stdout, stderr = _run_git(
                ["show", "--stat", "-p", commit], str(p), check=False
            )
            if rc != 0:
                return f"Error: {stderr}"
            return f"Git show {commit}:\n{_truncate_output(stdout, max_lines)}"

        elif command == "branch":
            rc, stdout, stderr = _run_git(["branch", "-a"], str(p), check=False)
            if rc != 0:
                return f"Error: {stderr}"
            return f"Git branches in {path}:\n{stdout}"

        else:
            return f"Error: unknown git command {command}"

    except Exception as e:
        return f"Error: {e}"
