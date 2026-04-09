"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses threading to avoid blocking on readline() with interactive bash.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CommandStats:
    """Statistics for command execution tracking."""
    total_commands: int = 0
    total_execution_time: float = 0.0
    timeouts: int = 0
    errors: int = 0
    command_history: list[dict[str, Any]] = field(default_factory=list)
    
    def record(self, command: str, duration: float, success: bool, error_type: str | None = None) -> None:
        """Record a command execution."""
        self.total_commands += 1
        self.total_execution_time += duration
        if not success:
            if error_type == "timeout":
                self.timeouts += 1
            else:
                self.errors += 1
        
        # Keep last 100 commands in history
        self.command_history.append({
            "command": command[:100],  # Truncate for memory
            "duration": round(duration, 3),
            "success": success,
            "error_type": error_type,
            "timestamp": time.time(),
        })
        if len(self.command_history) > 100:
            self.command_history.pop(0)
    
    def get_summary(self) -> dict[str, Any]:
        """Get execution statistics summary."""
        avg_time = self.total_execution_time / max(1, self.total_commands)
        return {
            "total_commands": self.total_commands,
            "avg_execution_time": round(avg_time, 3),
            "timeouts": self.timeouts,
            "errors": self.errors,
            "success_rate": round((self.total_commands - self.timeouts - self.errors) / max(1, self.total_commands), 3),
        }


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
_command_stats = CommandStats()


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


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    """
    logger.info(f"Executing bash command: {command[:100]}...")
    
    start_time = time.time()
    error_type = None
    
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
        result = output if output else "(no output)"
        duration = time.time() - start_time
        _command_stats.record(command, duration, success=True)
        logger.info(f"Command completed in {duration:.3f}s, output length: {len(result)} chars")
        return result
    except Exception as e:
        # On error, reset session for next call
        duration = time.time() - start_time
        error_str = str(e)
        if "Timed out" in error_str:
            error_type = "timeout"
        else:
            error_type = "error"
        _command_stats.record(command, duration, success=False, error_type=error_type)
        logger.error(f"Command failed after {duration:.3f}s: {e}")
        reset_session()
        return f"Error: {e}"


def validate_command(command: str) -> tuple[bool, str]:
    """Validate a shell command for potentially dangerous operations.

    Args:
        command: The shell command to validate.

    Returns:
        Tuple of (is_safe, warning_message). is_safe is True if the command
        passes validation, False if potentially dangerous patterns are detected.
        warning_message contains details if unsafe.
    """
    dangerous_patterns = [
        ("rm -rf /", "Attempting to recursively delete root directory"),
        ("> /dev/sda", "Direct write to disk device"),
        ("mkfs.", "Filesystem formatting command detected"),
        ("dd if=/dev/zero", "Disk wiping operation detected"),
        (":(){ :|:& };:", "Fork bomb detected"),
        ("chmod -R 777 /", "Dangerous permission change on root"),
        ("mv / /dev/null", "Attempting to move root to null"),
    ]

    cmd_lower = command.lower().strip()

    # Check for empty command
    if not cmd_lower:
        return False, "Empty command"

    # Check for dangerous patterns
    for pattern, warning in dangerous_patterns:
        if pattern in cmd_lower:
            return False, f"Security warning: {warning}"

    return True, ""


def run_with_validation(command: str) -> str:
    """Execute a bash command with pre-validation for dangerous operations.

    Args:
        command: The shell command to run.

    Returns:
        Command output or error message if validation fails.
    """
    is_safe, warning = validate_command(command)
    if not is_safe:
        return f"Command blocked: {warning}"
    return tool_function(command)


def get_command_stats() -> dict[str, Any]:
    """Get command execution statistics.
    
    Returns:
        Dictionary containing execution statistics summary.
    """
    return _command_stats.get_summary()


def reset_command_stats() -> None:
    """Reset command execution statistics."""
    global _command_stats
    _command_stats = CommandStats()
    logger.info("Command statistics reset")
