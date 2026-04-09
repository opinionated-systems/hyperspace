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


def tool_info() -> dict:
    return {
        "name": "bash",
        "description": (
            "Run commands in a bash shell. "
            "State is persistent across calls. "
            "Use 'sed -n 10,25p /path/to/file' to view line ranges. "
            "Avoid commands that produce very large output. "
            "Set show_history=True to view recent command history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                },
                "show_history": {
                    "type": "boolean",
                    "description": "If True, return command history instead of running command (default: false).",
                    "default": False,
                },
                "history_count": {
                    "type": "integer",
                    "description": "Number of history entries to show when show_history=True (default: 10).",
                    "default": 10,
                },
            },
            "required": ["command"],
        },
    }


_TIMEOUT = float(os.environ.get("BASH_TIMEOUT", "120.0"))
_SENTINEL = "<<SENTINEL_EXIT>>"
_MAX_OUTPUT_SIZE = 100000  # Max output size to prevent memory issues
_MAX_HISTORY_SIZE = 100  # Maximum number of commands to keep in history


class CommandHistory:
    """Track command history for debugging and auditing."""
    
    def __init__(self, max_size: int = _MAX_HISTORY_SIZE) -> None:
        self._history: list[tuple[str, str, float]] = []  # (command, output_preview, duration)
        self._max_size = max_size
    
    def add(self, command: str, output: str, duration: float) -> None:
        """Add a command to history."""
        # Store truncated output preview (first 200 chars)
        preview = output[:200] + "..." if len(output) > 200 else output
        self._history.append((command, preview, duration))
        
        # Trim history if needed
        if len(self._history) > self._max_size:
            self._history = self._history[-self._max_size:]
    
    def get_history(self, n: int = 10) -> list[tuple[str, str, float]]:
        """Get last n commands from history."""
        return self._history[-n:]
    
    def format_history(self, n: int = 10) -> str:
        """Format command history as a string."""
        if not self._history:
            return "No command history available."
        
        lines = [f"Last {min(n, len(self._history))} commands:"]
        for i, (cmd, preview, duration) in enumerate(self._history[-n:], 1):
            lines.append(f"\n{i}. Command: {cmd[:80]}{'...' if len(cmd) > 80 else ''}")
            lines.append(f"   Duration: {duration:.2f}s")
            lines.append(f"   Output: {preview[:100]}{'...' if len(preview) > 100 else ''}")
        return "\n".join(lines)


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._command_history = CommandHistory()

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
            total_size = 0
            while self._process and self._process.poll() is None:
                line = self._process.stdout.readline()
                if not line:
                    break
                with self._output_lock:
                    decoded = line.decode(errors="ignore")
                    total_size += len(decoded)
                    # Prevent memory issues with huge outputs
                    if total_size > _MAX_OUTPUT_SIZE:
                        if not hasattr(self, '_warned_size'):
                            self._output_buffer.append("\n[WARNING: Output truncated due to size limit]\n")
                            self._warned_size = True
                        continue
                    self._output_buffer.append(decoded)
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
                    self._command_history.add(command, output, duration)
                    return output

        return ""
    
    def get_command_history(self, n: int = 10) -> str:
        """Get formatted command history."""
        return self._command_history.format_history(n)


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


def tool_function(command: str, show_history: bool = False, history_count: int = 10) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    
    Args:
        command: The bash command to run
        show_history: If True, return command history instead of running command
        history_count: Number of history entries to show (default: 10)
    """
    # Handle history request
    if show_history:
        session = _get_session()
        return session.get_command_history(history_count)
    
    # Validate command
    if not command or not command.strip():
        return "Error: Empty command provided."
    
    # Check for dangerous commands
    dangerous_patterns = [
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=/dev/zero",
        ":(){ :|:& };:", "chmod -R 777 /", "chown -R root /",
    ]
    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in cmd_lower:
            return f"Error: Command contains potentially dangerous pattern '{pattern}'. Operation blocked."
    
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
        return output if output else "(no output)"
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        return f"Error: {e}. Do NOT retry the same command — it will time out again. Try a different approach."
