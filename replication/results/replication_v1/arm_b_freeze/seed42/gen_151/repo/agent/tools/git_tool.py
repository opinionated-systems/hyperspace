"""
Git tool: status, diff, log operations.

Provides git operations to help the agent understand repository state.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Restrict git operations to paths under this root."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Git operations for repository inspection. "
            "Commands: status, diff, log. "
            "Helps understand repository state and changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "diff", "log"],
                    "description": "The git command to run.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to git repository root.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path for diff command.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to return (for log).",
                    "default": 20,
                },
            },
            "required": ["command", "path"],
        },
    }


def _run_git(args: list[str], cwd: str) -> tuple[str, str, int]:
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
        return "", "Command timed out after 30 seconds", 1
    except Exception as e:
        return "", f"Error running git: {e}", 1


def _truncate(content: str, max_len: int = 5000) -> str:
    """Truncate content if too long."""
    if len(content) > max_len:
        return content[: max_len // 2] + "\n<response clipped>\n" + content[-max_len // 2 :]
    return content


def tool_function(
    command: str,
    path: str,
    file_path: str | None = None,
    max_lines: int = 20,
) -> str:
    """Execute a git command."""
    try:
        p = Path(path)

        if not p.is_absolute():
            return f"Error: {path} is not an absolute path."

        # Scope check
        if _ALLOWED_ROOT is not None:
            resolved = os.path.abspath(str(p))
            if not resolved.startswith(_ALLOWED_ROOT):
                return f"Error: access denied. Operations restricted to {_ALLOWED_ROOT}"

        # Verify it's a git repo
        git_dir = p / ".git"
        if not git_dir.exists() and not (p / ".git").is_dir():
            # Check if path is inside a git repo
            stdout, stderr, rc = _run_git(["rev-parse", "--git-dir"], str(p))
            if rc != 0:
                return f"Error: {path} is not a git repository."

        if command == "status":
            stdout, stderr, rc = _run_git(["status", "-sb"], str(p))
            if rc != 0:
                return f"Error: {stderr}"
            return f"Git status:\n{_truncate(stdout)}"

        elif command == "diff":
            args = ["diff"]
            if file_path:
                args.append(file_path)
            stdout, stderr, rc = _run_git(args, str(p))
            if rc != 0:
                return f"Error: {stderr}"
            if not stdout.strip():
                return "No changes to display."
            return f"Git diff:\n{_truncate(stdout)}"

        elif command == "log":
            stdout, stderr, rc = _run_git(
                ["log", "--oneline", "-n", str(max_lines)], str(p)
            )
            if rc != 0:
                return f"Error: {stderr}"
            return f"Git log (last {max_lines} commits):\n{stdout}"

        else:
            return f"Error: unknown command {command}"

    except Exception as e:
        return f"Error: {e}"
