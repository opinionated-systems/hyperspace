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

from agent.config import get_config

# Get configuration
_cfg = get_config()


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


_TIMEOUT = _cfg.bash.timeout
_SENTINEL = _cfg.bash.sentinel


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, cwd: str | None = None) -> None:
        if self._started:
            return
        self._stop_event.clear()
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
        self._timed_out = False
        self._output_buffer = []
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Background thread that reads stdout line by line."""
        try:
            while not self._stop_event.is_set() and self._process and self._process.poll() is None:
                try:
                    line = self._process.stdout.readline()
                    if not line:
                        break
                    with self._output_lock:
                        self._output_buffer.append(line.decode(errors="ignore"))
                except (ValueError, OSError):
                    # Stream closed
                    break
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            except Exception:
                pass
        self._process = None
        self._started = False
        self._timed_out = False
        self._output_buffer.clear()

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
        # Escape single quotes in the sentinel to prevent injection
        safe_sentinel = _SENTINEL.replace("'", "'\"'\"'")
        cmd = f"{command}\necho '{safe_sentinel}'\n"
        try:
            self._process.stdin.write(cmd.encode())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ValueError(f"Failed to send command to bash: {e}")

        # Wait for sentinel in output
        start = time.time()
        check_interval = 0.05  # Start with faster polling
        while True:
            if time.time() - start > _TIMEOUT:
                self._timed_out = True
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(check_interval)
            # Gradually increase polling interval to reduce CPU usage
            check_interval = min(check_interval * 1.1, 0.5)
            
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
    """
    # Strip common leading/trailing whitespace and normalize
    command = command.strip()
    
    if not command:
        return "Error: Empty command provided."
    
    # Block dangerous commands
    dangerous_patterns = [
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=/dev/zero",
        ":(){ :|:& };:", "chmod -R 777 /", "chown -R root /",
    ]
    for pattern in dangerous_patterns:
        if pattern in command:
            return f"Error: Potentially dangerous command blocked: {pattern}"
    
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
        # Truncate very long outputs
        limit = _cfg.agent.output_truncation_limit
        if len(output) > limit:
            half = limit // 2
            output = output[:half] + "\n... [output truncated, too long] ...\n" + output[-half:]
        return output if output else "(no output)"
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        return f"Error: {e}. Do NOT retry the same command — it will time out again. Try a different approach."
