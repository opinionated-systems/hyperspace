# BEGIN_REPLACE
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


# Bash session management (no class needed)


# Persistent bash session management

# Global variables to hold the subprocess and its I/O threads
_process: subprocess.Popen | None = None
_output_thread: threading.Thread | None = None
_output_buffer: list[str] = []
_output_lock = threading.Lock()
_stop_event = threading.Event()

def _reader_thread():
    """Continuously read from the subprocess stdout and store lines."""
    assert _process is not None
    while not _stop_event.is_set():
        line = _process.stdout.readline()
        if not line:
            break
        decoded = line.decode(errors="ignore")
        with _output_lock:
            _output_buffer.append(decoded)
            # If sentinel appears, we can stop reading further for this command
            if _SENTINEL in decoded:
                break

def _ensure_process():
    global _process, _output_thread, _stop_event
    if _process is None:
        # Start a bash subprocess with unbuffered output
        _process = subprocess.Popen(
            ["bash", "-i"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        # Ensure the process is ready by sending a simple command
        _process.stdin.write(b"echo ready\n")
        _process.stdin.flush()
        # Small pause to let it start
        time.sleep(0.1)

def tool_function(command: str) -> str:
    """Run a bash command in the persistent session.

    The command is executed, and the function returns everything printed
    before the sentinel string. If the command hangs or exceeds the
    timeout, an error message is returned.
    """
    try:
        _ensure_process()
        assert _process is not None and _process.stdin is not None
        # Clear previous output buffer
        with _output_lock:
            _output_buffer.clear()
        # Send the command followed by an echo of the sentinel so we know when it finishes
        full_cmd = f"{command}\n echo {_SENTINEL}\n"
        _process.stdin.write(full_cmd.encode())
        _process.stdin.flush()
        # Start reader thread
        global _output_thread, _stop_event
        _stop_event.clear()
        _output_thread = threading.Thread(target=_reader_thread, daemon=True)
        _output_thread.start()
        # Wait for thread to finish or timeout
        _output_thread.join(_TIMEOUT)
        if _output_thread.is_alive():
            _stop_event.set()
            return f"Error: command timed out after {_TIMEOUT} seconds."
        # Gather output
        with _output_lock:
            output = "".join(_output_buffer)
        # Remove the sentinel line
        output = output.replace(_SENTINEL, "").strip()
        return output
    except Exception as e:
        return f"Error executing bash command: {e}"

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
_session: any = None  # type: ignore
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
