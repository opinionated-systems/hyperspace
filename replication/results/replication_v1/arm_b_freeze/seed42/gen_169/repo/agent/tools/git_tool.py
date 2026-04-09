"""
Git tool: interact with git repositories.

Provides commands for checking repository status, viewing commit history,
branches, and other git operations useful for understanding code context.
"""

from __future__ import annotations

import subprocess
import os


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Run git commands to interact with the repository. "
            "Useful for checking status, viewing commit history, branches, and diffs. "
            "Common commands: status, log, diff, branch, show."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git command to run (without 'git' prefix).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory for the git command.",
                }
            },
            "required": ["command"],
        },
    }


def _run_git_command(command: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Execute a git command and return (returncode, stdout, stderr)."""
    working_dir = cwd or os.getcwd()
    
    # Validate that cwd is within allowed bounds if set
    allowed_root = os.environ.get('_GIT_ALLOWED_ROOT')
    if allowed_root and cwd:
        abs_cwd = os.path.abspath(cwd)
        abs_allowed = os.path.abspath(allowed_root)
        if not abs_cwd.startswith(abs_allowed):
            return 1, "", f"Error: Working directory '{cwd}' is outside allowed root '{allowed_root}'"
    
    try:
        result = subprocess.run(
            ["git"] + command.split(),
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Error: Git command timed out after 30s"
    except FileNotFoundError:
        return 1, "", "Error: Git is not installed or not found in PATH"
    except Exception as e:
        return 1, "", f"Error: {type(e).__name__}: {e}"


def tool_function(command: str, cwd: str | None = None) -> str:
    """Execute a git command. Returns output.
    
    Args:
        command: The git command to run (e.g., "status", "log --oneline -10")
        cwd: Optional working directory (defaults to current directory)
    
    Returns:
        Command output or error message
    """
    # Sanitize command - only allow safe git subcommands
    allowed_commands = {
        'status', 'log', 'diff', 'show', 'branch', 'remote', 'config',
        'blame', 'ls-files', 'ls-tree', 'rev-parse', 'describe',
        'tag', 'stash', 'reflog', 'shortlog', 'whatchanged'
    }
    
    # Extract the subcommand (first word)
    subcommand = command.split()[0] if command else ''
    
    if subcommand not in allowed_commands:
        return (
            f"Error: Git subcommand '{subcommand}' is not allowed. "
            f"Allowed commands: {', '.join(sorted(allowed_commands))}"
        )
    
    returncode, stdout, stderr = _run_git_command(command, cwd)
    
    if returncode != 0:
        error_msg = stderr.strip() if stderr else "Unknown error"
        return f"Error (exit code {returncode}): {error_msg}"
    
    output = stdout.strip()
    
    # Truncate very long outputs
    if len(output) > 8000:
        lines = output.split('\n')
        head = '\n'.join(lines[:100])
        tail = '\n'.join(lines[-50:]) if len(lines) > 150 else ''
        output = f"{head}\n... [{len(lines) - 150} lines truncated] ...\n{tail}" if tail else f"{head}\n... [output truncated] ..."
    
    return output if output else "(no output)"


def set_allowed_root(root: str) -> None:
    """Set the allowed root directory for git operations."""
    os.environ['_GIT_ALLOWED_ROOT'] = os.path.abspath(root)
