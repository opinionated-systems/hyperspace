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
    """
    # Validate command is not empty
    if not command or not command.strip():
        return "Error: Empty command provided"
    
    # Strip the command to remove leading/trailing whitespace
    command = command.strip()
    
    # Check for potentially dangerous commands
    dangerous_patterns = [
        ("rm -rf /", "recursive root deletion"),
        ("rm -rf /*", "recursive root deletion"),
        ("> /dev/sda", "disk overwrite"),
        ("mkfs.", "filesystem format"),
        ("dd if=", "disk write"),
        (":(){ :|:& };:", "fork bomb"),
        ("chmod -R 777 /", "recursive permission change"),
        ("chown -R root /", "recursive ownership change"),
        ("rm -rf ~", "home directory deletion"),
        ("rm -rf /home", "home directory deletion"),
    ]
    cmd_lower = command.lower()
    for pattern, description in dangerous_patterns:
        if pattern in cmd_lower:
            return f"Error: Command blocked - contains potentially dangerous pattern ({description}). Operation blocked for security."
    
    # Validate command length to prevent abuse
    if len(command) > 10000:
        return "Error: Command too long (max 10000 characters). Please break into smaller commands."
    
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
        
        # Handle empty output
        if not output or not output.strip():
            return "(no output)"
        
        # Truncate very long output to prevent context overflow
        max_output_len = 50000
        if len(output) > max_output_len:
            output = output[:max_output_len//2] + f"\n... [output truncated, total length: {len(output)} chars] ...\n" + output[-max_output_len//2:]
        
        return output
    except ValueError as e:
        # Handle session errors specifically
        reset_session()
        return f"Error: Session error - {e}. Session has been reset. Please retry your command."
    except subprocess.TimeoutExpired as e:
        # Handle timeout specifically
        reset_session()
        return f"Error: Command timed out after {_TIMEOUT}s. The session has been reset. Try a simpler command or increase timeout."
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        error_msg = str(e)
        # Provide helpful error messages for common issues
        if "No such file or directory" in error_msg:
            return f"Error: File or directory not found - {error_msg}"
        elif "Permission denied" in error_msg:
            return f"Error: Permission denied - {error_msg}"
        elif "command not found" in error_msg.lower():
            return f"Error: Command not found - {error_msg}. Check if the command is installed."
        else:
            return f"Error: {error_msg}. Do NOT retry the same command — it will likely fail again. Try a different approach."
