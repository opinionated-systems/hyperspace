"""
Bash tool: run commands in a persistent shell session.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import List

_SENTINEL = "<<SENTINEL_EXIT>>"
_TIMEOUT = 120.0

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

class BashSession:
    """Persistent bash session using subprocess.Popen and a reader thread."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: List[str] = []
        self._reader_thread: threading.Thread | None = None

    def start(self, cwd: str | None = None) -> None:
        if self._started:
            return
        self._process = subprocess.Popen(
            ["bash", "-i"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        self._started = True
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        self._read_until_sentinel()

    def _reader(self) -> None:
        assert self._process is not None
        for line in self._process.stdout:
            with self._output_lock:
                self._output_buffer.append(line)

    def _read_until_sentinel(self) -> str:
        start = time.time()
        while True:
            if time.time() - start > _TIMEOUT:
                self._timed_out = True
                break
            with self._output_lock:
                buffer = "".join(self._output_buffer)
                if _SENTINEL in buffer:
                    idx = buffer.index(_SENTINEL)
                    result = buffer[:idx]
                    self._output_buffer = [buffer[idx + len(_SENTINEL):]]
                    return result.strip()
            time.sleep(0.05)
        return ""

    def run(self, command: str) -> str:
        if not self._started or self._process is None:
            raise RuntimeError("Bash session not started")
        if self._process.poll() is not None:
            raise RuntimeError("Bash process terminated")
        with self._output_lock:
            self._output_buffer.clear()
        full_cmd = f"{command}\n echo '{_SENTINEL}'\n"
        self._process.stdin.write(full_cmd)
        self._process.stdin.flush()
        return self._read_until_sentinel()

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._started = False
        self._timed_out = False
        if self._reader_thread:
            self._reader_thread.join(timeout=1)
            self._reader_thread = None

_ALLOWED_ROOT: str | None = None
_session: BashSession | None = None

def set_allowed_root(root: str) -> None:
    global _ALLOWED_ROOT
    _ALLOWED_ROOT = os.path.abspath(root)
    reset_session()

def reset_session() -> None:
    global _session
    if _session is not None:
        _session.stop()
        _session = None

def _get_session() -> BashSession:
    global _session
    if _session is None or not _session._started or _session._timed_out:
        reset_session()
        _session = BashSession()
        _session.start(cwd=_ALLOWED_ROOT)
    return _session

def tool_function(command: str) -> str:
    try:
        session = _get_session()
        if _ALLOWED_ROOT:
            wrapped = (
                f"{command}\n"
                f"_cwd=$(pwd)\n"
                f"case \"$_cwd\" in \"{_ALLOWED_ROOT}\"*) ;; *) cd \"{_ALLOWED_ROOT}\" ; "
                f"echo \"WARNING: cd outside allowed root, reset to {_ALLOWED_ROOT}\" ;; esac"
            )
        else:
            wrapped = command
        return session.run(wrapped) or "(no output)"
    except Exception as e:
        reset_session()
        return f"Error: {e}."
