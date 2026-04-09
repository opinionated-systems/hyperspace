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


_TIMEOUT = float(os.environ.get("BASH_TIMEOUT", "120.0"))
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
    # Validate command
    if not command or not command.strip():
        return "Error: Empty command provided."
    
    # Strip the command for validation
    cmd_stripped = command.strip()
    cmd_lower = cmd_stripped.lower()
    
    # Check for dangerous commands - expanded list with better pattern matching
    dangerous_patterns = [
        ("rm -rf /", "recursive delete of root directory"),
        ("rm -rf /*", "recursive delete of all files"),
        ("> /dev/sda", "direct disk write to system device"),
        ("mkfs.", "filesystem formatting"),
        ("dd if=/dev/zero", "disk zeroing operation"),
        (":(){ :|:& };:", "fork bomb"),
        ("chmod -R 777 /", "recursive permission change on root"),
        ("chown -R root /", "recursive ownership change on root"),
        ("rm -rf ~", "recursive delete of home directory"),
        ("rm -rf $HOME", "recursive delete of home directory"),
        ("shutdown", "system shutdown"),
        ("reboot", "system reboot"),
        ("halt", "system halt"),
        ("poweroff", "system poweroff"),
        ("init 0", "system shutdown"),
        ("kill -9 -1", "kill all processes"),
    ]
    
    for pattern, description in dangerous_patterns:
        if pattern in cmd_lower:
            return f"Error: Command blocked - contains potentially dangerous pattern '{pattern}' ({description}). Operation blocked for security."
    
    # Additional validation: check for command chaining that might bypass checks
    suspicious_patterns = [
        "; rm ", "&& rm ", "|| rm ", "| rm ",
        "; chmod ", "&& chmod ", "; chown ", "&& chown ",
    ]
    for pattern in suspicious_patterns:
        if pattern in cmd_lower:
            logger.warning(f"Suspicious command pattern detected: {pattern} in command: {cmd_stripped[:100]}")
    
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
        
        # Log successful execution for debugging
        logger.debug(f"Bash command executed successfully: {cmd_stripped[:50]}...")
        
        return output if output else "(no output)"
    except ValueError as e:
        # Specific handling for timeout errors
        error_msg = str(e)
        if "Timed out" in error_msg:
            reset_session()
            return f"Error: Command timed out after {_TIMEOUT}s. The command may be hanging or producing too much output. Try a more specific command or add output limits (e.g., 'head -20')."
        elif "Session timed out" in error_msg:
            reset_session()
            return f"Error: Session was previously timed out. Session has been reset. Please retry your command."
        else:
            reset_session()
            return f"Error: {e}. Session reset. Try a different approach."
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        error_type = type(e).__name__
        logger.error(f"Bash command failed with {error_type}: {e}")
        return f"Error ({error_type}): {e}. Session has been reset. Do NOT retry the same command — it will likely fail again. Try a different approach."
