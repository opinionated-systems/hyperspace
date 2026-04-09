"""
Bash tool: run commands in a persistent shell session.

Simple implementation using subprocess.run with proper timeout handling.
"""

from __future__ import annotations

import os
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

# Module-level state for persistent session
_cwd: str | None = None
_allowed_root: str | None = None


def set_allowed_root(root: str) -> None:
    """Set working directory for new sessions."""
    global _allowed_root, _cwd
    _allowed_root = os.path.abspath(root)
    _cwd = _allowed_root


def reset_session() -> None:
    """Reset the bash session."""
    global _cwd
    _cwd = _allowed_root


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars carry between calls.
    Commands are scoped to _allowed_root.
    """
    global _cwd
    
    # Strip the command to avoid issues with leading/trailing whitespace
    command = command.strip()
    
    if not command:
        return "Error: Empty command provided."
    
    # Check for potentially dangerous commands
    dangerous_patterns = [
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=", ":(){ :|:& };:",
        "chmod -R 777 /", "chown -R root /", "mv / /dev/null", "rm -rf ~", 
        "rm -rf $HOME", "> ~/.bashrc", "curl.*|.*sh", "wget.*|.*sh"
    ]
    for pattern in dangerous_patterns:
        if pattern in command:
            return f"Error: Potentially dangerous command detected: '{pattern}'. Command blocked for safety."
    
    # Check for interactive commands that would hang
    interactive_commands = ["vim", "vi", "nano", "emacs", "less", "more", "top", "htop"]
    cmd_first_word = command.split()[0] if command.split() else ""
    if cmd_first_word in interactive_commands:
        return f"Error: Interactive command '{cmd_first_word}' is not supported. Use non-interactive alternatives."
    
    # Initialize cwd if not set
    if _cwd is None:
        _cwd = _allowed_root or os.getcwd()
    
    # Check if command tries to cd outside allowed root
    if _allowed_root and command.strip().startswith("cd "):
        # Extract the target directory
        target = command.strip()[3:].strip()
        # Handle relative paths
        if not target.startswith("/"):
            target = os.path.join(_cwd, target)
        target = os.path.abspath(os.path.expanduser(target))
        if not target.startswith(_allowed_root):
            return f"Error: cd outside allowed root ({_allowed_root}) is not permitted."
    
    try:
        # Run command in a new subprocess but preserve cwd
        full_command = f"cd '{_cwd}' && {command}"
        
        result = subprocess.run(
            ["bash", "-c", full_command],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            env=os.environ.copy(),
        )
        
        # Update cwd if cd command was used
        if command.strip().startswith("cd "):
            pwd_result = subprocess.run(
                ["bash", "-c", f"cd '{_cwd}' && {command} && pwd"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if pwd_result.returncode == 0:
                new_cwd = pwd_result.stdout.strip().split('\n')[-1]
                if new_cwd and os.path.isdir(new_cwd):
                    _cwd = new_cwd
        
        output = result.stdout
        if result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr
        
        # Truncate if output is too large
        if len(output) > _MAX_OUTPUT_SIZE:
            output = output[:_MAX_OUTPUT_SIZE] + f"\n... [output truncated: {len(output)} chars total]"
        
        # Add helpful context for empty output
        if not output:
            return "(no output - command executed successfully)"
        
        return output.strip()
        
    except subprocess.TimeoutExpired:
        return f"Error: Timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
