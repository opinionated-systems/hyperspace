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
    
    # Strip the command for processing
    cmd_stripped = command.strip()
    cmd_lower = cmd_stripped.lower()
    
    # Check for dangerous commands - expanded list
    dangerous_patterns = [
        ("rm -rf /", "recursive delete of root directory"),
        ("rm -rf /*", "recursive delete of all files"),
        ("> /dev/sda", "disk overwrite"),
        ("mkfs.", "filesystem format"),
        ("dd if=", "direct disk write"),
        (":(){ :|:& };:", "fork bomb"),
        ("chmod -R 777 /", "recursive permission change on root"),
        ("chown -R root /", "recursive ownership change on root"),
        ("rm -rf ~", "home directory deletion"),
        ("rm -rf $home", "home directory deletion"),
        ("rm -rf /home", "home directories deletion"),
        ("shutdown", "system shutdown"),
        ("reboot", "system reboot"),
        ("halt", "system halt"),
        ("poweroff", "system poweroff"),
    ]
    
    for pattern, description in dangerous_patterns:
        if pattern in cmd_lower:
            return f"Error: Potentially dangerous command detected ({description}): '{pattern}'. Command blocked for safety."
    
    # Check for interactive commands that might hang - expanded list
    interactive_patterns = [
        ("vim ", "vim"), ("vi ", "vi"), ("nano ", "nano"), ("emacs ", "emacs"),
        ("less ", "less"), ("more ", "more"), ("top", "top"), ("htop", "htop"),
        ("watch ", "watch"), ("tail -f", "tail -f"), ("multitail", "multitail"),
    ]
    
    for pattern, name in interactive_patterns:
        if pattern in cmd_lower:
            return f"Error: Interactive command '{name}' detected. Use non-interactive alternatives (e.g., 'cat' instead of 'less', 'sed' instead of 'vim', 'cat' instead of 'tail -f')."
    
    # Check for commands that might wait for input
    input_patterns = ["read ", "read\t", "select ", "while ", "for "]
    for pattern in input_patterns:
        if cmd_lower.startswith(pattern):
            return f"Error: Command may wait for input (starts with '{pattern.strip()}'). Use commands that don't require interactive input."
    
    # Add timeout wrapper for commands that might hang
    timeout_cmd = f"timeout {_TIMEOUT - 5} {cmd_stripped} 2>&1 || echo \"Command completed or timed out\""
    
    try:
        session = _get_session()
        # Run the command, then verify cwd is still within allowed root.
        # If the meta agent cd's outside, reset it back.
        if _ALLOWED_ROOT:
            wrapped = (
                f"{timeout_cmd}\n"
                f"_cwd=$(pwd)\n"
                f"case \"$_cwd\" in \"{_ALLOWED_ROOT}\"*) ;; *) cd \"{_ALLOWED_ROOT}\" ; "
                f"echo \"WARNING: cd outside allowed root, reset to {_ALLOWED_ROOT}\" ;; esac"
            )
        else:
            wrapped = timeout_cmd
        output = session.run(wrapped)
        
        # Handle empty output
        if not output:
            return "(no output)"
        
        # Check for timeout message
        if "Command completed or timed out" in output:
            output = output.replace("Command completed or timed out", "[Note: Command may have been terminated due to time limit]").strip()
        
        # Truncate very long outputs
        if len(output) > 100000:
            return output[:50000] + "\n... [output truncated, total length: " + str(len(output)) + " chars] ...\n" + output[-50000:]
        
        return output
    except ValueError as e:
        # Specific handling for timeout and session errors
        error_msg = str(e)
        if "timed out" in error_msg.lower():
            reset_session()
            return f"Error: Command timed out after {_TIMEOUT}s. The command may be too slow or interactive. Try a simpler command, add appropriate flags (e.g., -y for non-interactive), or break it into smaller steps."
        elif "exited" in error_msg.lower():
            reset_session()
            return f"Error: Bash session terminated unexpectedly. Session has been reset. Try again with a different command."
        else:
            reset_session()
            return f"Error: {e}. Session reset. Try a different approach."
    except Exception as e:
        # On error, reset session for next call
        reset_session()
        return f"Error: {type(e).__name__}: {e}. Do NOT retry the same command — it will likely fail again. Try a different approach."
