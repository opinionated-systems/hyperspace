"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses threading to avoid blocking on readline() with interactive bash.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import select
import fcntl
import uuid


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
_MAX_OUTPUT_SIZE = 100000  # Max ~100KB output to prevent memory issues


class BashSession:
    """Persistent bash session using Popen + threaded reader."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._started = False
        self._timed_out = False
        self._output_lock = threading.Lock()
        self._output_buffer: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()

    def start(self, cwd: str | None = None) -> None:
        if self._started:
            return
        self._stop_reader.clear()
        self._process = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            cwd=cwd or os.getcwd(),
            env=os.environ.copy(),
            bufsize=1,  # Line buffered for better performance
            universal_newlines=False,  # Keep binary mode for consistent behavior
        )
        # Make stdout non-blocking
        fd = self._process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        self._started = True
        self._output_buffer = []
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Background thread that reads stdout."""
        try:
            while not self._stop_reader.is_set() and self._process and self._process.poll() is None:
                try:
                    # Use select to check if data is available with timeout
                    fd = self._process.stdout.fileno()
                    ready, _, _ = select.select([fd], [], [], 0.1)
                    if ready:
                        try:
                            data = os.read(fd, 4096).decode(errors="ignore")
                            if data:
                                with self._output_lock:
                                    self._output_buffer.append(data)
                        except (OSError, IOError):
                            # Non-blocking read might fail if no data
                            pass
                except (ValueError, select.error):
                    # File descriptor closed or other error
                    break
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_reader.set()
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    self._process.kill()
                    self._process.wait(timeout=1)
                except Exception:
                    pass
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

        # Send command with sentinel - use unique sentinel to avoid collisions
        unique_sentinel = f"<<SENTINEL_{uuid.uuid4().hex[:8]}>>"
        # Use printf for more reliable output and ensure newline
        cmd = f"{command}\nprintf '%s\\n' '{unique_sentinel}'\n"
        try:
            self._process.stdin.write(cmd.encode())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ValueError(f"Failed to write to bash process: {e}")

        # Wait for sentinel in output
        start = time.time()
        last_output_time = start
        check_interval = 0.05  # 50ms check interval
        
        while True:
            elapsed = time.time() - start
            if elapsed > _TIMEOUT:
                self._timed_out = True
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(check_interval)
            
            with self._output_lock:
                full = "".join(self._output_buffer)
                if unique_sentinel in full:
                    # Extract output before sentinel
                    output = full[: full.index(unique_sentinel)].strip()
                    self._output_buffer.clear()
                    # Truncate if output is too large
                    if len(output) > _MAX_OUTPUT_SIZE:
                        output = output[:_MAX_OUTPUT_SIZE] + f"\n... [output truncated: {len(output)} chars total]"
                    return output if output else "(no output - command executed successfully)"
                
                # Check if we have any new output to update last_output_time
                if full:
                    last_output_time = time.time()
                elif time.time() - last_output_time > 5:
                    # No output for 5 seconds, check if process is still alive
                    if self._process.poll() is not None:
                        # Process died unexpectedly
                        raise ValueError(f"Bash process exited unexpectedly with code {self._process.returncode}")

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
    # Strip the command to avoid issues with leading/trailing whitespace
    command = command.strip()
    
    if not command:
        return "Error: Empty command provided."
    
    # Check for potentially dangerous commands
    dangerous_patterns = [
        "rm -rf /", "rm -rf /*", "> /dev/sda", "mkfs.", "dd if=", ":(){ :|:& };:",
        "chmod -R 777 /", "chown -R root /", "mv / /dev/null", "rm -rf ~", 
        "rm -rf $HOME", "> ~/.bashrc", "curl.*|.*sh", "wget.*|.*sh"
    ]
    for pattern in dangerous_patterns:
        if pattern in command:
            return f"Error: Potentially dangerous command detected: '{pattern}'. Command blocked for safety."
    
    # Check for interactive commands that would hang
    interactive_commands = ["vim", "vi", "nano", "emacs", "less", "more", "top", "htop"]
    cmd_first_word = command.split()[0] if command.split() else ""
    if cmd_first_word in interactive_commands:
        return f"Error: Interactive command '{cmd_first_word}' is not supported. Use non-interactive alternatives."
    
    # Check for commands that might hang or produce excessive output
    problematic_patterns = [
        (r"find\s+.*-name\s+['\"]?\*['\"]?", "find with wildcard name pattern may be slow, consider narrowing the search"),
        (r"grep\s+-r\s+.*\s+/", "recursive grep from root may be slow, consider specifying a directory"),
    ]
    for pattern, warning in problematic_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            # Don't block, just warn - the command might still be useful
            pass
    
    try:
        session = _get_session()
        # Run the command, then verify cwd is still within allowed root.
        # If the meta agent cd's outside, reset it back.
        if _ALLOWED_ROOT:
            wrapped = (
                f"{command}\n"
                f"_exit_code=$?\n"
                f"_cwd=$(pwd)\n"
                f"case \"$_cwd\" in \"{_ALLOWED_ROOT}\"*) ;; *) cd \"{_ALLOWED_ROOT}\" ; "
                f"echo \"WARNING: cd outside allowed root, reset to {_ALLOWED_ROOT}\" ;; esac\n"
                f"(exit $_exit_code)"
            )
        else:
            wrapped = command
        output = session.run(wrapped)
        
        # Add helpful context for empty output
        if not output:
            return "(no output - command executed successfully)"
        
        return output
    except ValueError as e:
        # On session errors, reset session for next call
        reset_session()
        return f"Error: Session error - {e}. Session has been reset."
    except Exception as e:
        # On other errors, reset session for next call
        reset_session()
        return f"Error: {type(e).__name__}: {e}"
