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

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TIMEOUT = 120.0  # Default command timeout in seconds
MAX_OUTPUT_SIZE = 100000  # Max output size to prevent memory issues
SENTINEL = "<<SENTINEL_EXIT>>"  # Marker for command completion
POLL_INTERVAL = 0.1  # Seconds between output checks
READER_STARTUP_DELAY = 0.5  # Seconds to wait for reader thread startup
MAX_SESSION_AGE = 3600  # Maximum session age in seconds before auto-reset

# Dangerous command patterns to block
DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=/dev/zero",
    ":(){ :|:& };:", "chmod -R 777 /", "chown -R root /",
    "rm -rf ~", "rm -rf ~/*",
]

# Resource-intensive command patterns to warn about
RESOURCE_INTENSIVE_PATTERNS = [
    "find /", "find ~", "grep -r /", "du -sh /", "ls -R /",
]


def tool_info() -> dict:
    """Return tool metadata for the bash tool."""
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


# Get timeout from environment or use default
_TIMEOUT = float(os.environ.get("BASH_TIMEOUT", str(DEFAULT_TIMEOUT)))


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._start_time: float = 0.0

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
        self._start_time = time.time()
        self._output_buffer = []
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def is_expired(self) -> bool:
        """Check if the session has exceeded its maximum age."""
        if not self._started or self._start_time == 0:
            return False
        return (time.time() - self._start_time) > MAX_SESSION_AGE

    def _read_loop(self) -> None:
        """Background thread that reads stdout line by line.
        
        Continuously reads from the process stdout and stores output
        in the buffer, with size limiting to prevent memory issues.
        """
        try:
            total_size = 0
            warned_size = False
            while self._process and self._process.poll() is None:
                line = self._process.stdout.readline()
                if not line:
                    break
                with self._output_lock:
                    decoded = line.decode(errors="ignore")
                    total_size += len(decoded)
                    # Prevent memory issues with huge outputs
                    if total_size > MAX_OUTPUT_SIZE:
                        if not warned_size:
                            self._output_buffer.append("\n[WARNING: Output truncated due to size limit]\n")
                            warned_size = True
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
        """Execute a command in the bash session.
        
        Args:
            command: The bash command to execute.
            
        Returns:
            The command output as a string.
            
        Raises:
            ValueError: If the session is not started, has exited, or timed out.
            ValueError: If the command times out.
        """
        # Validate session state
        if not self._started:
            raise ValueError("Session not started")
        if self._process is None:
            raise ValueError("Process not initialized")
        if self._process.poll() is not None:
            raise ValueError(f"Bash exited with code {self._process.returncode}")
        if self._timed_out:
            raise ValueError("Session timed out, must restart")
        
        # Validate command input
        if not command or not command.strip():
            return "(empty command)"

        # Clear buffer
        with self._output_lock:
            self._output_buffer.clear()

        # Send command with sentinel
        cmd = f"{command}\necho '{SENTINEL}'\n"
        self._process.stdin.write(cmd.encode())
        self._process.stdin.flush()

        # Wait for sentinel in output
        start = time.time()
        while True:
            if time.time() - start > _TIMEOUT:
                self._timed_out = True
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(POLL_INTERVAL)
            with self._output_lock:
                full = "".join(self._output_buffer)
                if SENTINEL in full:
                    # Extract output before sentinel
                    output = full[: full.index(SENTINEL)].strip()
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
    if _session is None or not _session._started or _session._timed_out or _session.is_expired():
        if _session is not None:
            if _session.is_expired():
                logger.debug("Bash session expired, creating new session")
            _session.stop()
        _session = BashSession()
        _session.start(cwd=_ALLOWED_ROOT)
    return _session


def _check_resource_intensive(command: str) -> str | None:
    """Check if a command is resource-intensive and return a warning if so.
    
    Args:
        command: The command to check.
        
    Returns:
        Warning message if the command is resource-intensive, None otherwise.
    """
    cmd_lower = command.lower()
    for pattern in RESOURCE_INTENSIVE_PATTERNS:
        if pattern in cmd_lower:
            return f"Warning: Command '{command}' may be resource-intensive. Consider using more specific paths or filters."
    return None


def tool_function(command: str) -> str:
    """Execute a bash command. Returns output.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    
    Args:
        command: The bash command to execute.
        
    Returns:
        The command output, or an error message if execution fails.
    """
    # Validate command type
    if not isinstance(command, str):
        return f"Error: Command must be a string, got {type(command).__name__}"
    
    # Validate command
    command = command.strip()
    if not command:
        return "Error: Empty command provided."
    
    # Check for dangerous commands
    cmd_lower = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd_lower:
            logger.warning(f"Blocked dangerous command pattern: {pattern}")
            return f"Error: Command contains potentially dangerous pattern '{pattern}'. Operation blocked."
    
    # Check for resource-intensive commands
    resource_warning = _check_resource_intensive(command)
    if resource_warning:
        logger.warning(resource_warning)
    
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
        
        # Prepend resource warning if applicable
        if resource_warning and output:
            output = f"[Note: {resource_warning}]\n{output}"
        
        return output if output else "(no output)"
    except ValueError as e:
        # Specific handling for session errors
        logger.error(f"Bash session error: {e}")
        reset_session()
        return f"Error: Session error - {e}. Session has been reset."
    except Exception as e:
        # On error, reset session for next call
        logger.error(f"Bash command failed: {e}")
        reset_session()
        return f"Error: {e}. Do NOT retry the same command — it will time out again. Try a different approach."
