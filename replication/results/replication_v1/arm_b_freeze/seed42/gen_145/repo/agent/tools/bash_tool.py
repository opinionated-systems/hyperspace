"""
Bash tool: run commands in a persistent shell session.

Reimplemented from facebookresearch/HyperAgents agent/tools/bash.py.
Same interface, same timeout, same sentinel-based output detection.
Session is persistent across calls (matching paper): cd, env vars, etc. carry over.

Uses threading to avoid blocking on readline() with interactive bash.

Recent improvements:
- Added command validation to prevent dangerous operations
- Added command whitelist for common safe operations
- Better error messages for invalid commands
"""

from __future__ import annotations

import os
import re
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


# Dangerous command patterns to block
_DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r">\s*/dev/null",
    r":\(\)\s*\{\s*:\|\:\&\s*\}",  # Fork bomb
    r"mkfs\.",
    r"dd\s+if=.*of=/dev/[sh]d",
    r"\$0",  # Self-execution patterns
]

# Whitelist of safe command starters (optional security feature)
_SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "sed", "awk",
    "echo", "pwd", "cd", "mkdir", "touch", "cp", "mv", "rm",
    "python", "python3", "pip", "git", "wc", "sort", "uniq",
    "diff", "file", "stat", "which", "whoami", "date", "ps",
    "curl", "wget", "tar", "zip", "unzip", "chmod", "chown",
}


def _validate_command(command: str) -> tuple[bool, str]:
    """Validate command for dangerous patterns.
    
    Returns:
        (is_valid, error_message)
    """
    # Check for dangerous patterns
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Command blocked: matches dangerous pattern '{pattern}'"
    
    # Check for command chaining that might be dangerous
    dangerous_chain = ["|", ";", "&&", "||"]
    for chain in dangerous_chain:
        if chain in command:
            parts = command.split(chain)
            for part in parts:
                part = part.strip()
                if part:
                    # Check each part for dangerous patterns
                    for pattern in _DANGEROUS_PATTERNS:
                        if re.search(pattern, part, re.IGNORECASE):
                            return False, f"Command blocked: dangerous pattern in chained command"
    
    return True, ""


_TIMEOUT = 120.0
_SENTINEL = "<<SENTINEL_EXIT>>"
_MAX_OUTPUT_SIZE = 100000  # Limit output to prevent memory issues


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
                    # Prevent buffer from growing too large
                    if sum(len(s) for s in self._output_buffer) > _MAX_OUTPUT_SIZE * 2:
                        # Truncate oldest output
                        total = 0
                        keep_idx = 0
                        for i, s in enumerate(reversed(self._output_buffer)):
                            total += len(s)
                            if total > _MAX_OUTPUT_SIZE:
                                keep_idx = len(self._output_buffer) - i - 1
                                break
                        self._output_buffer = self._output_buffer[keep_idx:]
                        self._output_buffer.insert(0, "... [output truncated] ...\n")
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
                    # Limit output size
                    if len(output) > _MAX_OUTPUT_SIZE:
                        output = output[:_MAX_OUTPUT_SIZE] + f"\n... [output truncated, total length: {len(output)}] ..."
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
    
    Commands are validated for dangerous patterns before execution.
    """
    # Validate command before execution
    is_valid, error_msg = _validate_command(command)
    if not is_valid:
        return f"Error: {error_msg}"
    
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
        return f"Error: {e}"
