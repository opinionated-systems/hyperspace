"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses threading to avoid blocking on readline() with interactive bash.
Enhanced with safety checks and better error handling.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Commands/patterns that are potentially dangerous and should be blocked
BLOCKED_PATTERNS = [
    r'rm\s+-rf\s+/\s*$',  # rm -rf /
    r'rm\s+-rf\s+/\*',   # rm -rf /*
    r'rm\s+-rf\s+~',     # rm -rf ~
    r'>\s*/dev/sda',      # Overwrite disk
    r'mkfs\.\w+\s+/',    # Format filesystem
    r'dd\s+if=.*of=/dev/[sh]d\w',  # dd to disk
    r':\s*\(\)\s*\{\s*:\s*\|\s*:&\s*\};',  # Fork bomb
    r'chmod\s+-R\s+777\s*/',  # chmod -R 777 /
    r'chown\s+-R\s+\w+\s*/',  # chown -R user /
    r'mv\s+/\s+',        # mv / 
    r'shutdown',          # shutdown
    r'reboot',            # reboot
    r'halt',              # halt
    r'poweroff',          # poweroff
    r'init\s+0',          # init 0
]

# Maximum output size to prevent memory issues
MAX_OUTPUT_SIZE = 500000  # 500KB


def tool_info() -> dict:
    return {
        "name": "bash",
        "description": (
            "Run commands in a bash shell. "
            "State is persistent across calls. "
            "Use 'sed -n 10,25p /path/to/file' to view line ranges. "
            "Avoid commands that produce very large output. "
            "Dangerous commands are blocked for safety."
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


def _is_command_safe(command: str) -> tuple[bool, str]:
    """Check if a command is safe to execute.
    
    Returns:
        (is_safe, error_message)
    """
    if not command or not isinstance(command, str):
        return False, "Command must be a non-empty string"
    
    cmd_normalized = command.strip().lower()
    
    # Check for blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cmd_normalized, re.IGNORECASE):
            return False, f"Command matches blocked pattern"
    
    return True, ""


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


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    
    Enhanced with safety checks and output size limits.
    """
    # Validate command type
    if not isinstance(command, str):
        return f"Error: Expected string command, got {type(command).__name__}"
    
    # Safety check
    is_safe, error_msg = _is_command_safe(command)
    if not is_safe:
        logger.warning(f"Blocked dangerous command: {command[:100]}...")
        return f"Error: Command blocked for safety - {error_msg}"
    
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
        
        # Truncate if too large
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + f"\n... [output truncated, total length: {len(output)}]"
        
        return output if output else "(no output)"
        
    except subprocess.SubprocessError as e:
        logger.error(f"Subprocess error: {e}")
        reset_session()
        return f"Error: Subprocess failed - {e}"
    except Exception as e:
        # On error, reset session for next call
        logger.error(f"Bash execution error: {e}")
        reset_session()
        return f"Error: {type(e).__name__}: {e}"
