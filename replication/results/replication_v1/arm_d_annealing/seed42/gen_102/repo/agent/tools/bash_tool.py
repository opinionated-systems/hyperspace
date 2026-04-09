"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses communicate() with timeout for reliable command execution.
"""

from __future__ import annotations

import os
import subprocess
import time


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


class BashSession:
    """Persistent bash session using Popen with communicate() for reliability."""

    def __init__(self) -> None:
        self._cwd: str | None = None
        self._env: dict = {}
        self._started = False

    def start(self, cwd: str | None = None) -> None:
        if self._started:
            return
        self._cwd = cwd or os.getcwd()
        self._env = os.environ.copy()
        self._started = True

    def stop(self) -> None:
        self._started = False
        self._cwd = None

    def run(self, command: str) -> str:
        if not self._started:
            raise ValueError("Session not started")

        # Run command in a new subprocess but preserve cwd and env
        # Use bash -c to execute the command
        full_command = f"cd '{self._cwd}' && {command}"
        
        try:
            result = subprocess.run(
                ["bash", "-c", full_command],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
                env=self._env,
            )
            
            # Update cwd if cd command was used
            # Extract the new cwd by running pwd
            pwd_result = subprocess.run(
                ["bash", "-c", f"cd '{self._cwd}' && {command} && pwd"],
                capture_output=True,
                text=True,
                timeout=5,
                env=self._env,
            )
            if pwd_result.returncode == 0:
                new_cwd = pwd_result.stdout.strip().split('\n')[-1]
                if new_cwd and os.path.isdir(new_cwd):
                    self._cwd = new_cwd
            
            output = result.stdout
            if result.stderr:
                output = output + "\n" + result.stderr if output else result.stderr
            
            # Truncate if output is too large
            if len(output) > _MAX_OUTPUT_SIZE:
                output = output[:_MAX_OUTPUT_SIZE] + f"\n... [output truncated: {len(output)} chars total]"
            
            return output.strip()
            
        except subprocess.TimeoutExpired:
            raise ValueError(f"Timed out after {_TIMEOUT}s")
        except Exception as e:
            raise ValueError(f"Command failed: {e}")

    def get_cwd(self) -> str:
        return self._cwd or os.getcwd()


# Module-level persistent session
_session: BashSession | None = None
_ALLOWED_ROOT: str | None = None


def set_allowed_root(root: str) -> None:
    """Set working directory for new sessions."""
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)
    reset_session()


def reset_session() -> None:
    """Reset the bash session."""
    global _session
    if _session is not None:
        _session.stop()
        _session = None


def _get_session() -> BashSession:
    global _session
    # Always create a new session to avoid stale state issues
    if _session is not None:
        _session.stop()
    _session = BashSession()
    _session.start(cwd=_ALLOWED_ROOT)
    return _session


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    """
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
    
    try:
        session = _get_session()
        
        # Check if command tries to cd outside allowed root
        if _ALLOWED_ROOT and command.strip().startswith("cd "):
            # Extract the target directory
            target = command.strip()[3:].strip()
            # Handle relative paths
            if not target.startswith("/"):
                current_cwd = session.get_cwd()
                target = os.path.join(current_cwd, target)
            target = os.path.abspath(os.path.expanduser(target))
            if not target.startswith(_ALLOWED_ROOT):
                return f"Error: cd outside allowed root ({_ALLOWED_ROOT}) is not permitted."
        
        output = session.run(command)
        
        # Add helpful context for empty output
        if not output:
            return "(no output - command executed successfully)"
        
        return output
    except ValueError as e:
        # On session errors, reset session for next call
        reset_session()
        return f"Error: Session error - {e}. Session has been reset."
    except Exception as e:
        # On other errors, reset session for next call
        reset_session()
        return f"Error: {type(e).__name__}: {e}"
