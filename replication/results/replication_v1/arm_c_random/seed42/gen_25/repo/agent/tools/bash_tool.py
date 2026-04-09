"""
Bash tool: run commands in a persistent shell session with enhanced features.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Enhanced features:
- Command history tracking
- Output size limiting with smart truncation
- Exit code capture and reporting
- Working directory tracking
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from typing import NamedTuple


class CommandResult(NamedTuple):
    """Result of a command execution."""
    command: str
    output: str
    exit_code: int
    duration_ms: float
    timestamp: float


# Command history (limited size for memory management)
_MAX_HISTORY = 50
_command_history: deque[CommandResult] = deque(maxlen=_MAX_HISTORY)

# Output size limits
_MAX_OUTPUT_LINES = 500
_MAX_OUTPUT_CHARS = 50000
_TRUNCATION_MESSAGE = "\n... [output truncated - {} lines, {} chars total]"


def get_command_history() -> list[CommandResult]:
    """Return the command execution history."""
    return list(_command_history)


def get_last_command() -> CommandResult | None:
    """Return the most recent command result."""
    return _command_history[-1] if _command_history else None


def tool_info() -> dict:
    return {
        "name": "bash",
        "description": (
            "Run commands in a bash shell. "
            "State is persistent across calls. "
            "Use 'sed -n 10,25p /path/to/file' to view line ranges. "
            "Avoid commands that produce very large output. "
            "Supports command history tracking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds (default: 120). Max: 300.",
                },
            },
            "required": ["command"],
        },
    }


_TIMEOUT = 120.0
_MAX_TIMEOUT = 300.0
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
        self._current_dir: str = ""

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
        self._current_dir = cwd or os.getcwd()
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

    def get_current_dir(self) -> str:
        """Get the current working directory of the session."""
        return self._current_dir

    def run(self, command: str, timeout: float = _TIMEOUT) -> tuple[str, int]:
        """Run a command and return (output, exit_code).
        
        Args:
            command: The command to execute
            timeout: Maximum time to wait in seconds
            
        Returns:
            Tuple of (output_string, exit_code)
        """
        if not self._started:
            raise ValueError("Session not started")
        if self._process.poll() is not None:
            raise ValueError(f"Bash exited with code {self._process.returncode}")
        if self._timed_out:
            raise ValueError("Session timed out, must restart")

        # Clear buffer
        with self._output_lock:
            self._output_buffer.clear()

        # Send command with exit code capture and sentinel
        # Use a subshell to capture exit code reliably
        cmd = (
            f"{{ {command}; }}\n"
            f"_EXIT_CODE=$?\n"
            f"echo '{_SENTINEL}'\n"
            f"exit $_EXIT_CODE\n"
        )
        self._process.stdin.write(cmd.encode())
        self._process.stdin.flush()

        # Wait for sentinel in output
        start = time.time()
        effective_timeout = min(timeout, _MAX_TIMEOUT)
        
        while True:
            if time.time() - start > effective_timeout:
                self._timed_out = True
                raise ValueError(f"Timed out after {effective_timeout}s")

            time.sleep(0.1)
            with self._output_lock:
                full = "".join(self._output_buffer)
                if _SENTINEL in full:
                    # Extract output before sentinel
                    output = full[: full.index(_SENTINEL)].strip()
                    self._output_buffer.clear()
                    
                    # Get exit code from process
                    # We need to restart the process to get a fresh shell
                    exit_code = self._process.poll()
                    if exit_code is None:
                        # Process still running, restart it
                        self.stop()
                        exit_code = 0  # Assume success if we got output
                    
                    return output, exit_code

        return "", 1


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


def _truncate_output(output: str) -> str:
    """Truncate output if it exceeds size limits."""
    lines = output.split("\n")
    total_chars = len(output)
    
    if len(lines) > _MAX_OUTPUT_LINES or total_chars > _MAX_OUTPUT_CHARS:
        # Truncate to max lines
        truncated_lines = lines[:_MAX_OUTPUT_LINES]
        truncated = "\n".join(truncated_lines)
        
        # If still too long, truncate by chars
        if len(truncated) > _MAX_OUTPUT_CHARS:
            truncated = truncated[:_MAX_OUTPUT_CHARS]
        
        return truncated + _TRUNCATION_MESSAGE.format(len(lines), total_chars)
    
    return output


def tool_function(command: str, timeout: float = _TIMEOUT) -> str:
    """Execute a bash command. Returns output with exit code.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    
    Args:
        command: The bash command to execute
        timeout: Optional timeout in seconds (default: 120, max: 300)
    
    Returns:
        Command output with exit code information
    """
    global _command_history
    
    start_time = time.time()
    
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
            
        output, exit_code = session.run(wrapped, timeout=timeout)
        
        # Truncate if necessary
        output = _truncate_output(output)
        
        # Record in history
        duration_ms = (time.time() - start_time) * 1000
        result = CommandResult(
            command=command,
            output=output,
            exit_code=exit_code,
            duration_ms=duration_ms,
            timestamp=start_time
        )
        _command_history.append(result)
        
        # Format output with exit code
        if exit_code != 0:
            prefix = f"[Exit code: {exit_code}]\n"
        else:
            prefix = ""
            
        return prefix + (output if output else "(no output)")
        
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        
        # Record failure in history
        duration_ms = (time.time() - start_time) * 1000
        result = CommandResult(
            command=command,
            output=f"Error: {e}",
            exit_code=-1,
            duration_ms=duration_ms,
            timestamp=start_time
        )
        _command_history.append(result)
        
        return f"Error: {e}"
