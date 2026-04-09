"""
Git tool: execute git commands for version control operations.

Provides safe git operations for the agent to track changes,
view diffs, check status, and understand codebase history.
"""

from __future__ import annotations

import subprocess
from typing import Any


def _run_git_command(args: list[str], cwd: str | None = None) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout or "Success (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def git_status(repo_path: str = ".") -> str:
    """Check git status of a repository."""
    return _run_git_command(["-C", repo_path, "status", "--short"])


def git_diff(repo_path: str = ".", staged: bool = False) -> str:
    """Show git diff of uncommitted changes."""
    cmd = ["-C", repo_path, "diff"]
    if staged:
        cmd.append("--staged")
    return _run_git_command(cmd)


def git_log(repo_path: str = ".", max_count: int = 10) -> str:
    """Show recent git commit history."""
    return _run_git_command(
        ["-C", repo_path, "log", f"--max-count={max_count}", "--oneline", "--decorate"]
    )


def git_show(repo_path: str = ".", commit: str = "HEAD") -> str:
    """Show details of a specific commit."""
    return _run_git_command(["-C", repo_path, "show", "--stat", commit])


def git_add(repo_path: str = ".", files: str = ".") -> str:
    """Stage files for commit."""
    return _run_git_command(["-C", repo_path, "add", files])


def git_commit(repo_path: str = ".", message: str = "Agent commit") -> str:
    """Create a git commit with staged changes."""
    return _run_git_command(["-C", repo_path, "commit", "-m", message])


def git_branch(repo_path: str = ".") -> str:
    """List git branches."""
    return _run_git_command(["-C", repo_path, "branch", "-a"])


def tool_info() -> dict[str, Any]:
    return {
        "name": "git",
        "description": "Execute git commands for version control operations including status, diff, log, and commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "show", "add", "commit", "branch"],
                    "description": "The git command to execute",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Path to the git repository (default: current directory)",
                    "default": ".",
                },
                "staged": {
                    "type": "boolean",
                    "description": "For diff command: show staged changes instead of unstaged",
                    "default": False,
                },
                "max_count": {
                    "type": "integer",
                    "description": "For log command: number of commits to show",
                    "default": 10,
                },
                "commit": {
                    "type": "string",
                    "description": "For show command: commit hash or reference to show",
                    "default": "HEAD",
                },
                "files": {
                    "type": "string",
                    "description": "For add command: files to stage (default: all)",
                    "default": ".",
                },
                "message": {
                    "type": "string",
                    "description": "For commit command: commit message",
                    "default": "Agent commit",
                },
            },
            "required": ["command"],
        },
    }


def tool_function(
    command: str,
    repo_path: str = ".",
    staged: bool = False,
    max_count: int = 10,
    commit: str = "HEAD",
    files: str = ".",
    message: str = "Agent commit",
) -> str:
    """Execute a git command."""
    commands = {
        "status": lambda: git_status(repo_path),
        "diff": lambda: git_diff(repo_path, staged),
        "log": lambda: git_log(repo_path, max_count),
        "show": lambda: git_show(repo_path, commit),
        "add": lambda: git_add(repo_path, files),
        "commit": lambda: git_commit(repo_path, message),
        "branch": lambda: git_branch(repo_path),
    }
    
    if command not in commands:
        return f"Error: Unknown git command '{command}'"
    
    return commands[command]()
