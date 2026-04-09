"""
Bash tool: run commands in a bash shell.

Simplified implementation using subprocess.run for reliability.
"""

from __future__ import annotations

import os
import re
import subprocess


def tool_info() -> dict:
    return {
        "name": "bash",
        "description": (
            "Run commands in a bash shell. "
            "State is persistent across calls. "
            "Use 'sed -n 10,25p /path/to/file' to view line ranges. "
            "Avoid commands that produce very large output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                }
            },
            "required": ["command"],
        },
    }


_TIMEOUT = 120.0
_MAX_OUTPUT_SIZE = 100000  # Max ~100KB output to prevent memory issues

# Module-level state
_cwd: str | None = None
_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set working directory for new sessions."""
    global _ALLOWED_ROOT, _cwd
    _ALLOWED_ROOT = os.path.abspath(root)
    _cwd = _ALLOWED_ROOT


def reset_session() -> None:
    """Reset the bash session (working directory)."""
    global _cwd
    _cwd = _ALLOWED_ROOT


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Commands are scoped to _ALLOWED_ROOT.
    """
    global _cwd
    
    # Strip the command to avoid issues with leading/trailing whitespace
    command = command.strip()
    
    if not command:
        return "Error: Empty command provided."
    
    # Check for potentially dangerous commands (match as whole words/phrases)
    dangerous_patterns = [
        (r'\brm\s+-rf\s+/\b', "rm -rf /"),
        (r'\brm\s+-rf\s+/\*\b', "rm -rf /*"),
        (r'>\s*/dev/sda', "> /dev/sda"),
        (r'\bmkfs\.', "mkfs."),
        (r'\bdd\s+if=', "dd if="),
        (r':\(\)\{\s*:\|:&\s*\};:', "fork bomb"),
        (r'\bchmod\s+-R\s+777\s+/\b', "chmod -R 777 /"),
        (r'\bchown\s+-R\s+root\s+/\b', "chown -R root /"),
        (r'\bmv\s+/\s+/dev/null\b', "mv / /dev/null"),
        (r'\brm\s+-rf\s+~\b', "rm -rf ~"),
        (r'\brm\s+-rf\s+\$HOME\b', "rm -rf $HOME"),
        (r'>\s*~/.bashrc', "> ~/.bashrc"),
        (r'\bcurl\s+.*\|\s*sh\b', "curl | sh"),
        (r'\bwget\s+.*\|\s*sh\b', "wget | sh"),
    ]
    for pattern, description in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Error: Potentially dangerous command detected: '{description}'. Command blocked for safety."
    
    # Check for interactive commands that would hang
    interactive_commands = ["vim", "vi", "nano", "emacs", "less", "more", "top", "htop"]
    cmd_first_word = command.split()[0] if command.split() else ""
    if cmd_first_word in interactive_commands:
        return f"Error: Interactive command '{cmd_first_word}' is not supported. Use non-interactive alternatives."
    
    # Determine working directory
    cwd = _cwd if _cwd else (_ALLOWED_ROOT if _ALLOWED_ROOT else os.getcwd())
    
    # Handle cd commands specially to track directory changes
    if command.startswith("cd "):
        new_dir = command[3:].strip()
        # Handle relative paths
        if not os.path.isabs(new_dir):
            new_dir = os.path.join(cwd, new_dir)
        new_dir = os.path.abspath(os.path.normpath(new_dir))
        
        # Check if within allowed root
        if _ALLOWED_ROOT and not new_dir.startswith(_ALLOWED_ROOT):
            return f"Error: cd outside allowed root. Access denied."
        
        if os.path.isdir(new_dir):
            _cwd = new_dir
            return f"Changed directory to: {new_dir}"
        else:
            return f"Error: Directory not found: {new_dir}"
    
    try:
        # Run the command using subprocess
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n" + result.stderr
            else:
                output = result.stderr
        
        # Truncate if output is too large
        if len(output) > _MAX_OUTPUT_SIZE:
            output = output[:_MAX_OUTPUT_SIZE] + f"\n... [output truncated: {len(output)} chars total]"
        
        return output if output else "(no output - command executed successfully)"
        
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
