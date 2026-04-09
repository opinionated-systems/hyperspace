"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses threading to avoid blocking on readline() with interactive bash.
Enhanced with better error handling, logging, and resource management.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


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
            warned_size = False
            line_count = 0
            max_lines = 10000  # Prevent excessive line accumulation
            
            while self._process and self._process.poll() is None:
                try:
                    line = self._process.stdout.readline()
                    if not line:
                        break
                    
                    line_count += 1
                    if line_count > max_lines and not warned_size:
                        with self._output_lock:
                            self._output_buffer.append("\n[WARNING: Output exceeded max line count, truncating...]\n")
                        warned_size = True
                        continue
                    
                    with self._output_lock:
                        decoded = line.decode(errors="ignore")
                        total_size += len(decoded)
                        # Prevent memory issues with huge outputs
                        if total_size > _MAX_OUTPUT_SIZE:
                            if not warned_size:
                                self._output_buffer.append("\n[WARNING: Output truncated due to size limit]\n")
                                warned_size = True
                            continue
                        self._output_buffer.append(decoded)
                except Exception as e:
                    # Log error but continue reading
                    logger.debug(f"BashSession read error: {e}")
                    continue
        except Exception as e:
            logger.debug(f"BashSession read loop error: {e}")

    def stop(self) -> None:
        """Stop the bash session gracefully."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("BashSession: process did not terminate, killing...")
                    self._process.kill()
                    self._process.wait(timeout=2)
            except Exception as e:
                logger.error(f"BashSession: error stopping process: {e}")
        
        # Clean up reader thread
        if self._reader_thread and self._reader_thread.is_alive():
            try:
                self._reader_thread.join(timeout=2)
            except Exception as e:
                logger.debug(f"BashSession: error joining reader thread: {e}")
        
        self._process = None
        self._started = False
        self._timed_out = False
        self._output_buffer.clear()

    def is_healthy(self) -> bool:
        """Check if the session is healthy and ready for commands."""
        if not self._started or self._process is None:
            return False
        if self._timed_out:
            return False
        if self._process.poll() is not None:
            return False
        return True

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
    """Get or create a healthy bash session."""
    global _session
    if _session is None or not _session.is_healthy():
        if _session is not None:
            logger.debug("BashSession: session unhealthy, restarting...")
            _session.stop()
        _session = BashSession()
        _session.start(cwd=_ALLOWED_ROOT)
        logger.debug("BashSession: new session started")
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
    
    # Check for dangerous commands
    dangerous_patterns = [
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=/dev/zero",
        ":(){ :|:& };:", "chmod -R 777 /", "chown -R root /",
        "rm -rf ~", "rm -rf $HOME", "> ~/.bashrc", "> ~/.profile",
    ]
    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in cmd_lower:
            logger.warning(f"Bash tool: blocked dangerous command pattern '{pattern}'")
            return f"Error: Command contains potentially dangerous pattern '{pattern}'. Operation blocked."
    
    # Limit command length to prevent memory issues
    if len(command) > 10000:
        return "Error: Command too long (max 10000 characters)."
    
    # Check for command injection attempts
    injection_patterns = [
        "; rm ", "| rm ", "&& rm ", "; sudo ", "| sudo ", "&& sudo ",
        "$(rm ", "`rm ", "${IFS}rm", "${IFS}sudo",
    ]
    for pattern in injection_patterns:
        if pattern in cmd_lower:
            logger.warning(f"Bash tool: blocked injection pattern '{pattern}'")
            return f"Error: Command contains potentially dangerous pattern '{pattern}'. Operation blocked."
    
    try:
        session = _get_session()
        
        # Verify session is healthy before running
        if not session.is_healthy():
            logger.warning("Bash tool: session unhealthy, resetting...")
            reset_session()
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
        
        # Log command execution for debugging
        logger.debug(f"Bash tool: executed command, output length={len(output) if output else 0}")
        
        return output if output else "(no output)"
    except Exception as e:
        # On error, reset session for next call
        logger.error(f"Bash tool: error executing command: {e}")
        reset_session()
        return f"Error: {e}. Do NOT retry the same command — it will time out again. Try a different approach."
