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
    """Execute a bash command with enhanced logging and safety checks.

    Session is persistent across calls (matching paper).
    cd, env vars, aliases carry between calls.
    Commands are scoped to _ALLOWED_ROOT via a wrapper that checks
    the working directory stays within bounds (matching paper's Docker).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Strip the command to handle accidental whitespace
    command = command.strip()
    
    if not command:
        logger.warning("Empty bash command received")
        return "Error: Empty command provided"
    
    # Log command execution attempt (truncate long commands)
    cmd_preview = command[:100] + "..." if len(command) > 100 else command
    logger.info(f"Executing bash command: {cmd_preview}")
    
    # Check for dangerous commands
    dangerous_patterns = [
        'rm -rf /', 'rm -rf /*', '> /dev/null', ':(){ :|:& };:',
        'mkfs.', 'dd if=/dev/zero', 'chmod -R 777 /', 'rm -rf ~', 'rm -rf $HOME'
    ]
    for pattern in dangerous_patterns:
        if pattern in command:
            logger.error(f"Dangerous command pattern detected: '{pattern}'")
            return f"Error: Potentially dangerous command detected: '{pattern}'. Command blocked for safety."
    
    # Check for interactive commands that would hang
    interactive_patterns = ['vim ', 'vi ', 'nano ', 'emacs ', 'less ', 'more ', 'top ', 'htop ']
    for pattern in interactive_patterns:
        if command.startswith(pattern) or f' {pattern}' in command:
            logger.warning(f"Interactive command detected: '{pattern.strip()}'")
            return f"Error: Interactive command '{pattern.strip()}' detected. Use non-interactive alternatives (e.g., 'cat' instead of 'less', 'sed' instead of 'vim')."
    
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
        
        # Log output summary
        output_len = len(output)
        if output_len > 0:
            preview = output[:80].replace('\n', ' ')
            logger.info(f"Command executed successfully. Output length: {output_len} chars. Preview: {preview}...")
        else:
            logger.info("Command executed successfully. No output.")
        
        # Truncate very long outputs to prevent context overflow
        max_output_len = 50000
        if output_len > max_output_len:
            truncated = output[:max_output_len//2] + "\n... [output truncated - too long] ...\n" + output[-max_output_len//2:]
            logger.info(f"Output truncated from {output_len} to {len(truncated)} chars")
            return truncated
        return output if output else "(no output)"
        
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        error_msg = str(e)
        logger.error(f"Command execution failed: {error_msg}")
        if "timed out" in error_msg.lower():
            return f"Error: Command timed out after {_TIMEOUT}s. The command may be hanging or producing too much output. Try a more specific command or add filters (e.g., 'head', 'tail', 'grep')."
        return f"Error: {e}. Do NOT retry the same command — it may fail again. Try a different approach."
