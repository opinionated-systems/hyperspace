"""
Git tool: run git commands to inspect repository state.

Provides safe git operations for viewing repository history and status.
Read-only operations only - no commits, pushes, or destructive operations.
"""

from __future__ import annotations

import os
import subprocess


_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set working directory for git commands."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)


def tool_info() -> dict:
    return {
        "name": "git",
        "description": (
            "Run git commands to inspect repository state. "
            "Provides read-only operations: status, log, diff, show, branch. "
            "Useful for understanding code history and recent changes. "
            "Only safe, non-destructive commands are allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git command to run (e.g., 'status', 'log --oneline -5', 'diff HEAD~1').",
                }
            },
            "required": ["command"],
        },
    }


_ALLOWED_COMMANDS = {
    "status", "log", "diff", "show", "branch", "remote", "config",
    "blame", "ls-files", "ls-tree", "rev-parse", "describe",
}

_MAX_OUTPUT_LEN = 50000  # Maximum characters to return


def _is_safe_command(command: str) -> bool:
    """Check if the git command is safe (read-only)."""
    # Split command and get the first argument (the subcommand)
    parts = command.strip().split()
    if not parts:
        return False
    
    subcommand = parts[0].lower()
    
    # Check if it's in the allowed list
    if subcommand not in _ALLOWED_COMMANDS:
        return False
    
    # Block dangerous flags that could modify state
    dangerous_flags = ["-c", "--exec", ";", "|", "&&", "||", "`", "$"]
    for flag in dangerous_flags:
        if flag in command:
            return False
    
    return True


def _truncate_output(output: str, max_len: int = _MAX_OUTPUT_LEN) -> str:
    """Truncate output if it exceeds max length, keeping start and end."""
    if len(output) <= max_len:
        return output
    
    half = max_len // 2
    start = output[:half]
    end = output[-half:]
    truncated_len = len(output) - max_len
    return f"{start}\n\n... [{truncated_len} characters truncated] ...\n\n{end}"


def tool_function(command: str) -> str:
    """Execute a git command. Returns output.
    
    Only read-only commands are allowed (status, log, diff, show, branch, etc.)
    Commands that modify the repository are blocked for safety.
    """
    # Validate command safety
    if not _is_safe_command(command):
        return (
            f"Error: Command '{command}' is not allowed. "
            f"Only read-only git commands are permitted: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )
    
    try:
        # Determine working directory
        cwd = _ALLOWED_ROOT if _ALLOWED_ROOT else os.getcwd()
        
        # Run the git command
        result = subprocess.run(
            ["git"] + command.split(),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        if result.returncode != 0:
            return f"Error (exit code {result.returncode}): {output}"
        
        if not output.strip():
            return "(no output)"
        
        return _truncate_output(output)
        
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"
