"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses threading to avoid blocking on readline() with interactive bash.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Tuple


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
_SENTINEL = "<<SENTINEL_EXIT>>"


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None

    def start(self, cwd: str | None = None) -> None:
        if self._started:
            return
        self._process = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            cwd=cwd or os.getcwd(),
            env=os.environ.copy(),
            bufsize=0,
        )
        self._started = True
        self._output_buffer = []
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Background thread that reads stdout line by line."""
        try:
            while self._process and self._process.poll() is None:
                line = self._process.stdout.readline()
                if not line:
                    break
                with self._output_lock:
                    self._output_buffer.append(line.decode(errors="ignore"))
        except Exception:
            pass

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._started = False
        self._timed_out = False

    def run(self, command: str) -> str:
        if not self._started:
            raise ValueError("Session not started")
        if self._process.poll() is not None:
            raise ValueError(f"Bash exited with code {self._process.returncode}")
        if self._timed_out:
            raise ValueError("Session timed out, must restart")

        # Clear buffer
        with self._output_lock:
            self._output_buffer.clear()

        # Send command with sentinel
        cmd = f"{command}\necho '{_SENTINEL}'\n"
        self._process.stdin.write(cmd.encode())
        self._process.stdin.flush()

        # Wait for sentinel in output
        start = time.time()
        while True:
            if time.time() - start > _TIMEOUT:
                self._timed_out = True
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(0.1)
            with self._output_lock:
                full = "".join(self._output_buffer)
                if _SENTINEL in full:
                    # Extract output before sentinel
                    output = full[: full.index(_SENTINEL)].strip()
                    self._output_buffer.clear()
                    return output

        return ""


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
    if _session is None or not _session._started or _session._timed_out:
        if _session is not None:
            _session.stop()
        _session = BashSession()
        _session.start(cwd=_ALLOWED_ROOT)
    return _session


def _validate_command(command: str) -> tuple[bool, str]:
    """Validate a bash command for safety.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Error: Empty command provided."
    
    # Check for dangerous commands
    dangerous_patterns = [
        ("rm -rf /", "recursive delete of root directory"),
        ("rm -rf /*", "recursive delete of all files"),
        ("mkfs", "filesystem formatting"),
        ("dd if=", "direct disk operations"),
        (":(){ :|:& };:", "fork bomb"),
        ("chmod -R 777 /", "recursive permission change on root"),
        ("chown -R", "recursive ownership change"),
        ("shutdown", "system shutdown"),
        ("reboot", "system reboot"),
        ("halt", "system halt"),
        ("poweroff", "system poweroff"),
    ]
    
    cmd_lower = command.lower()
    for pattern, description in dangerous_patterns:
        if pattern in cmd_lower:
            return False, f"Error: Command blocked - contains potentially dangerous pattern '{pattern}' ({description})."
    
    return True, ""


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    """
    # Validate command
    is_valid, error_msg = _validate_command(command)
    if not is_valid:
        return error_msg
    
    # Track command execution time
    start_time = time.time()
    
    # Pre-process command to handle common issues
    command = command.strip()
    
    # Add safety wrapper for commands that might hang on interactive prompts
    # Wrap common interactive commands with yes/no auto-answers
    interactive_commands = ['apt-get', 'apt', 'pip', 'conda', 'npm', 'yarn']
    cmd_first_word = command.split()[0] if command.split() else ''
    
    if cmd_first_word in interactive_commands and '-y' not in command and '--yes' not in command:
        # Add non-interactive flags for common package managers
        if cmd_first_word in ['apt-get', 'apt']:
            command = command.replace(cmd_first_word, f'{cmd_first_word} -y', 1)
        elif cmd_first_word in ['conda']:
            command = command.replace(cmd_first_word, f'{cmd_first_word} -y', 1)
    
    try:
        session = _get_session()
        # Run the command, then verify cwd is still within allowed root.
        # If the meta agent cd's outside, reset it back.
        if _ALLOWED_ROOT:
            wrapped = (
                f"{command}\n"
                f"_cwd=$(pwd)\n"
                f"case \"$_cwd\" in \"{_ALLOWED_ROOT}\"*) ;; *) cd \"{_ALLOWED_ROOT}\" ; "
                f"echo \"WARNING: cd outside allowed root, reset to {_ALLOWED_ROOT}\" ;; esac"
            )
        else:
            wrapped = command
        output = session.run(wrapped)
        
        # Truncate very long outputs with better context preservation
        max_output_len = 10000
        if len(output) > max_output_len:
            # Keep beginning, middle indicator, and end
            half_len = max_output_len // 2
            output = (
                output[:half_len] + 
                f"\n... [output truncated - {len(output) - max_output_len} chars removed] ...\n" + 
                output[-half_len:]
            )
        
        # Add execution time info for debugging (only for slow commands)
        elapsed = time.time() - start_time
        if elapsed > 5:
            output = f"[Command took {elapsed:.1f}s]\n{output}"
        
        # Return empty output indicator
        if not output or output.strip() == '':
            return "(no output)"
            
        return output
        
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        error_msg = str(e)
        elapsed = time.time() - start_time
        
        # Provide helpful, actionable error messages
        error_lower = error_msg.lower()
        
        if "timed out" in error_lower:
            return (
                f"Error: Command timed out after {_TIMEOUT}s (after {elapsed:.1f}s elapsed).\n"
                f"The command may be:\n"
                f"  - Waiting for interactive input (use flags like -y, --non-interactive)\n"
                f"  - Running an infinite loop or long computation\n"
                f"  - Stuck on a network request\n"
                f"Try: breaking into smaller steps, adding timeout limits, or checking for prompts."
            )
        elif "no such file" in error_lower or "not found" in error_lower:
            return (
                f"Error: {e}\n"
                f"Check:\n"
                f"  - The file/directory path is correct\n"
                f"  - The file exists (use 'ls -la' to verify)\n"
                f"  - You're in the correct directory (use 'pwd')"
            )
        elif "permission denied" in error_lower:
            return (
                f"Error: {e}\n"
                f"You may need to:\n"
                f"  - Check file permissions with 'ls -la'\n"
                f"  - Use appropriate user permissions\n"
                f"  - Check if the resource is locked by another process"
            )
        elif "command not found" in error_lower:
            return (
                f"Error: {e}\n"
                f"The command is not available. Try:\n"
                f"  - Check spelling\n"
                f"  - Install the required package\n"
                f"  - Check if it's in your PATH (use 'which <command>')"
            )
        elif "is a directory" in error_lower:
            return f"Error: {e}. You tried to operate on a directory as if it were a file."
        elif "not a directory" in error_lower:
            return f"Error: {e}. The path is not a directory."
        else:
            return (
                f"Error: {e} (after {elapsed:.1f}s)\n"
                f"Do NOT retry the same command — it may fail again. Try a different approach."
            )
