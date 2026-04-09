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
_MAX_OUTPUT_SIZE = 100000  # Max output size to prevent memory issues


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._command_count = 0

    def start(self, cwd: str | None = None) -> None:
        if self._started:
            return
        try:
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
            self._command_count = 0
            # Start reader thread
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()
        except Exception as e:
            raise ValueError(f"Failed to start bash session: {e}")

    def _read_loop(self) -> None:
        """Background thread that reads stdout line by line."""
        try:
            while self._process and self._process.poll() is None:
                try:
                    line = self._process.stdout.readline()
                    if not line:
                        break
                    with self._output_lock:
                        self._output_buffer.append(line.decode(errors="ignore"))
                        # Prevent unbounded growth
                        if len(self._output_buffer) > 10000:
                            self._output_buffer = self._output_buffer[-5000:]
                except Exception:
                    break
        except Exception:
            pass

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass  # Force kill failed, but we tried
            except Exception:
                pass
        self._process = None
        self._started = False
        self._timed_out = False
        self._command_count = 0

    def run(self, command: str) -> str:
        if not self._started:
            raise ValueError("Session not started")
        if self._process.poll() is not None:
            raise ValueError(f"Bash exited with code {self._process.returncode}")
        if self._timed_out:
            raise ValueError("Session timed out, must restart")

        self._command_count += 1
        
        # Clear buffer
        with self._output_lock:
            self._output_buffer.clear()

        # Sanitize command to prevent issues with special characters in sentinel
        safe_command = command.replace("'", "'\"'\"'")
        
        # Send command with sentinel
        cmd = f"{safe_command}\necho '{_SENTINEL}'\n"
        try:
            self._process.stdin.write(cmd.encode())
            self._process.stdin.flush()
        except Exception as e:
            raise ValueError(f"Failed to send command: {e}")

        # Wait for sentinel in output
        start = time.time()
        last_output_size = 0
        stall_count = 0
        
        while True:
            if time.time() - start > _TIMEOUT:
                self._timed_out = True
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(0.1)
            with self._output_lock:
                full = "".join(self._output_buffer)
                
                # Check for stall (no new output for a while)
                current_size = len(full)
                if current_size == last_output_size:
                    stall_count += 1
                    if stall_count > 600:  # 60 seconds of no output
                        self._timed_out = True
                        raise ValueError(f"Command stalled (no output for 60s)")
                else:
                    stall_count = 0
                    last_output_size = current_size
                
                # Check for sentinel
                if _SENTINEL in full:
                    # Extract output before sentinel
                    output = full[: full.index(_SENTINEL)].strip()
                    self._output_buffer.clear()
                    
                    # Truncate if too large
                    if len(output) > _MAX_OUTPUT_SIZE:
                        output = output[:_MAX_OUTPUT_SIZE//2] + "\n... [output truncated] ...\n" + output[-_MAX_OUTPUT_SIZE//2:]
                    
                    return output if output else "(no output)"

        return "(no output)"


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
    """
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
