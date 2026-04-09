"""
Git tool: execute git commands for version control operations.

Provides safe git operations for the meta agent to track changes,
create branches, commit modifications, and view repository status.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def _run_git_command(
    command: list[str],
    cwd: str | None = None,
    timeout: int = 30
) -> tuple[bool, str]:
    """Execute a git command safely.
    
    Args:
        command: List of command arguments (e.g., ["git", "status"])
        cwd: Working directory for the command
        timeout: Maximum time to wait for command completion
        
    Returns:
        Tuple of (success, output_or_error)
    """
    # Validate command starts with git
    if not command or command[0] != "git":
        return False, "Error: Command must start with 'git'"
    
    # Block dangerous operations
    blocked_commands = [
        "push", "fetch", "pull", "clone", "remote", "submodule",
        "filter-branch", "update-ref", "symbolic-ref", "config",
        "credential", "credential-cache", "credential-store"
    ]
    
    for blocked in blocked_commands:
        if blocked in command:
            return False, f"Error: Git command '{blocked}' is blocked for security"
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if not output:
                output = "(command succeeded with no output)"
            return True, output
        else:
            error_msg = result.stderr.strip() or f"Command failed with exit code {result.returncode}"
            return False, f"Error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, f"Error: Command timed out after {timeout} seconds"
    except FileNotFoundError:
        return False, "Error: Git executable not found"
    except Exception as e:
        logger.exception("Git command failed")
        return False, f"Error executing git command: {type(e).__name__}: {e}"


def tool_info() -> dict[str, Any]:
    """Return tool metadata for LLM tool calling."""
    return {
        "name": "git",
        "description": (
            "Execute git commands for version control operations. "
            "Supports: status, log, diff, show, branch, checkout, add, commit, reset, stash. "
            "Blocked for security: push, pull, fetch, clone, remote, submodule, config."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The git command to execute (e.g., 'status', 'log --oneline -5', "
                        "'diff HEAD', 'branch', 'add .', 'commit -m message'). "
                        "Do not include 'git' prefix - it will be added automatically."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory for the command. Defaults to current directory.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum time in seconds to wait for command completion. Default: 30",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    }


def tool_function(
    command: str,
    cwd: str | None = None,
    timeout: int = 30
) -> str:
    """Execute a git command.
    
    Args:
        command: The git command (without 'git' prefix)
        cwd: Working directory
        timeout: Timeout in seconds
        
    Returns:
        Command output or error message
    """
    if not command or not command.strip():
        return "Error: Empty command"
    
    # Parse command into list
    cmd_parts = ["git"] + command.strip().split()
    
    logger.info(f"Executing git command: {' '.join(cmd_parts)}")
    
    success, output = _run_git_command(cmd_parts, cwd=cwd, timeout=timeout)
    
    if success:
        logger.info(f"Git command succeeded: {output[:100]}...")
    else:
        logger.warning(f"Git command failed: {output}")
    
    return output
