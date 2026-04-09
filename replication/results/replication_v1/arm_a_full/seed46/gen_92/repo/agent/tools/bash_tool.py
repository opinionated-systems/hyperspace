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
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution with timing and metadata."""
    output: str
    exit_code: int
    duration_ms: float
    command: str
    timed_out: bool = False


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

    def run(self, command: str) -> CommandResult:
        """Execute a command and return detailed result with timing."""
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
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > _TIMEOUT:
                self._timed_out = True
                duration_ms = elapsed * 1000
                logger.warning(f"Command timed out after {duration_ms:.1f}ms: {command[:100]}...")
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(0.1)
            with self._output_lock:
                full = "".join(self._output_buffer)
                if _SENTINEL in full:
                    # Extract output before sentinel
                    output = full[: full.index(_SENTINEL)].strip()
                    self._output_buffer.clear()
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log slow commands for debugging
                    if duration_ms > 5000:  # Log commands taking > 5 seconds
                        logger.info(f"Slow command ({duration_ms:.1f}ms): {command[:100]}...")
                    
                    return CommandResult(
                        output=output,
                        exit_code=0,
                        duration_ms=duration_ms,
                        command=command,
                        timed_out=False
                    )

        # Should never reach here
        duration_ms = (time.time() - start_time) * 1000
        return CommandResult(
            output="",
            exit_code=-1,
            duration_ms=duration_ms,
            command=command,
            timed_out=False
        )


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


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    
    Now includes execution time tracking and detailed logging for debugging.
    """
    # Strip the command to handle accidental whitespace
    command = command.strip()
    
    if not command:
        return "Error: Empty command provided"
    
    start_time = time.time()
    
    try:
        session = _get_session()
        # Run the command, then verify cwd is still within allowed root.
        # If the meta agent cd's outside, reset it back.
        if _ALLOWED_ROOT:
            # Escape special characters in the command for safe shell execution
            escaped_root = _ALLOWED_ROOT.replace('"', '\\"')
            wrapped = (
                f"{command}\n"
                f"_cwd=$(pwd)\n"
                f"case \"$_cwd\" in \"{escaped_root}\"*) ;; *) cd \"{escaped_root}\" ; "
                f"echo \"WARNING: cd outside allowed root, reset to {escaped_root}\" ;; esac"
            )
        else:
            wrapped = command
        
        result = session.run(wrapped)
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Log command execution details
        logger.debug(f"Bash command completed in {result.duration_ms:.1f}ms (total: {total_duration_ms:.1f}ms): {command[:80]}...")
        
        output = result.output if result.output else "(no output)"
        return output
        
    except Exception as e:
        # On error, reset session for next call
        total_duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Bash command failed after {total_duration_ms:.1f}ms: {command[:80]}... - Error: {e}")
        reset_session()
        return f"Error: {e}. Do NOT retry the same command — it will time out again. Try a different approach."
