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
import select


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
_SENTINEL = "<<SENTINEL_EXIT_$(date +%s)>>"
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
        self._sentinel_marker = "<<SENTINEL_EXIT_"

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
            universal_newlines=False,
        )
        self._started = True
        self._output_buffer = []
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Background thread that reads stdout using non-blocking I/O."""
        try:
            import fcntl
            # Set stdout to non-blocking mode
            fd = self._process.stdout.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        except Exception:
            pass  # Fallback to blocking mode if fcntl fails
        
        try:
            while self._process and self._process.poll() is None:
                try:
                    # Try to read in non-blocking mode
                    data = self._process.stdout.read(4096)
                    if data:
                        with self._output_lock:
                            self._output_buffer.append(data.decode(errors="ignore"))
                    else:
                        # No data available, sleep briefly
                        time.sleep(0.01)
                except (IOError, OSError):
                    # No data available in non-blocking mode
                    time.sleep(0.01)
                except Exception:
                    break
        except Exception:
            pass

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.close()
            except:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=2)
                except:
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

        # Generate unique sentinel with timestamp
        import random
        unique_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        sentinel = f"<<SENTINEL_EXIT_{unique_id}>>"

        # Send command with sentinel - use printf for more reliable output
        cmd = f"{command}\nprintf '%s\\n' '{sentinel}'\n"
        
        try:
            self._process.stdin.write(cmd.encode())
            self._process.stdin.flush()
        except BrokenPipeError:
            raise ValueError("Bash process pipe broken")

        # Wait for sentinel in output
        start = time.time()
        last_output_time = start
        
        while True:
            current_time = time.time()
            if current_time - start > _TIMEOUT:
                self._timed_out = True
                raise ValueError(f"Timed out after {_TIMEOUT}s")

            time.sleep(0.05)  # Shorter sleep for more responsive checking
            
            with self._output_lock:
                full = "".join(self._output_buffer)
                if sentinel in full:
                    # Extract output before sentinel
                    output = full[: full.index(sentinel)].strip()
                    self._output_buffer.clear()
                    # Truncate if output is too large
                    if len(output) > _MAX_OUTPUT_SIZE:
                        output = output[:_MAX_OUTPUT_SIZE] + f"\n... [output truncated: {len(output)} chars total]"
                    return output if output else "(no output - command executed successfully)"
                
                # Check if we have any new output to update last_output_time
                if full:
                    last_output_time = current_time
                elif current_time - last_output_time > 30:
                    # No output for 30 seconds, might be stuck
                    self._timed_out = True
                    raise ValueError(f"No output for 30 seconds, command may be stuck")

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
    
    try:
        session = _get_session()
        # Run the command, then verify cwd is still within allowed root.
        # If the meta agent cd's outside, reset it back.
        if _ALLOWED_ROOT:
            wrapped = (
                f"{command}\n"
                f"_cwd=$(pwd)\n"
                f"_exit_code=$?\n"
                f"case \"$_cwd\" in \"{_ALLOWED_ROOT}\"*) ;; *) cd \"{_ALLOWED_ROOT}\" ; "
                f"echo \"WARNING: cd outside allowed root, reset to {_ALLOWED_ROOT}\" ;; esac\n"
                f"exit $_exit_code"
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
