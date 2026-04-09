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
from collections import deque


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
_MAX_HISTORY = 100  # Maximum number of commands to keep in history


class CommandHistory:
    """Track command history for debugging and auditing."""
    
    def __init__(self, max_size: int = _MAX_HISTORY) -> None:
        self._history: deque[tuple[str, str, float]] = deque(maxlen=max_size)
    
    def add(self, command: str, output: str, duration: float) -> None:
        """Add a command entry to history."""
        # Truncate output if too long for history storage
        output_summary = output[:500] + "..." if len(output) > 500 else output
        self._history.append((command, output_summary, duration))
    
    def get_recent(self, n: int = 10) -> list[tuple[str, str, float]]:
        """Get the n most recent commands."""
        return list(self._history)[-n:]
    
    def get_all(self) -> list[tuple[str, str, float]]:
        """Get all commands in history."""
        return list(self._history)
    
    def clear(self) -> None:
        """Clear the command history."""
        self._history.clear()


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._history = CommandHistory()

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
                    # Track command in history
                    duration = time.time() - start
                    self._history.add(command, output, duration)
                    return output

        return ""
    
    def get_history(self, n: int = 10) -> list[tuple[str, str, float]]:
        """Get recent command history."""
        return self._history.get_recent(n)
    
    def clear_history(self) -> None:
        """Clear command history."""
        self._history.clear()


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


def get_command_history(n: int = 10) -> list[tuple[str, str, float]]:
    """Get recent command history from the current session.
    
    Returns a list of tuples (command, output_summary, duration_seconds).
    Returns empty list if no session exists.
    """
    global _session
    if _session is None:
        return []
    return _session.get_history(n)


def clear_command_history() -> None:
    """Clear the command history of the current session."""
    global _session
    if _session is not None:
        _session.clear_history()


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    """
    # Strip the command to handle accidental whitespace
    command = command.strip()
    
    if not command:
        return "Error: Empty command provided"
    
    # Security: Block dangerous commands
    dangerous_patterns = [
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=", ":(){ :|:& };:",
        "chmod -R 777 /", "chown -R root /", "sudo ", "su -", "passwd", "shadow"
    ]
    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in cmd_lower:
            return f"Error: Command blocked for security reasons (contains '{pattern}')"
    
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
        output = session.run(wrapped)
        return output if output else "(no output)"
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        return f"Error: {e}. Do NOT retry the same command — it will time out again. Try a different approach."
